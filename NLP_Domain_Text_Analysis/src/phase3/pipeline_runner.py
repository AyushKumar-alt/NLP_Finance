"""Execute a :class:`~src.phase3.pipelines.PipelineSpec` over the corpus.

The runner reuses the *actual* Phase 2 components (hybrid tokenizer, typed
date/number tokenizer, stopword strategies, WordNet lemmatization, Porter
stemming) so that a Phase 3 pipeline is the Phase 2 code path in a different
order, not a re-implementation of it.

Every step is timed with :func:`time.perf_counter`; the timings are reported as
observed measurements and are deliberately excluded from the final pipeline
score. Query-time normalization reuses the same component order, so a query is
transformed exactly like the indexed text.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.phase2.date_number_tokenizer import detect_expressions
from src.phase2.lemmatization import WordNetLemmaCache
from src.phase2.load_corpus import Unit, normalise_for_tokenisation
from src.phase2.statistics import TokenizedCorpus, build_tokenized
from src.phase2.stemming import get_stemmer
from src.phase2.stopwords import StopwordStrategy, build_strategies
from src.phase2.tokenizers import custom_tokenize_batch, hybrid_tokenize, nltk_tokenize

from .config import Phase3Config, log_event
from .pipelines import PipelineSpec

#: Tokenizers available to a pipeline spec.
TOKENIZERS: Dict[str, Callable[[str], List[str]]] = {
    "hybrid": hybrid_tokenize,
    "custom": lambda text: custom_tokenize_batch([text])[0],
    "nltk": nltk_tokenize,
}

_WS = re.compile(r"\s+")


@dataclass
class PipelineResult:
    """Everything one pipeline produced, ready for indexing and comparison."""

    spec: PipelineSpec
    view: TokenizedCorpus
    step_seconds: Dict[str, float]
    total_seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)
    measurements: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.view.total_tokens

    @property
    def vocabulary_size(self) -> int:
        return self.view.vocabulary_size

    @property
    def unique_tokens(self) -> int:
        return self.view.unique_tokens


class PipelineRunner:
    """Runs pipelines and normalizes queries with the same component order."""

    def __init__(self, config: Phase3Config, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger
        self._phase2 = config.phase2_config()
        self._stopwords: Dict[str, StopwordStrategy] = build_strategies(self._phase2, logger)
        self._lemma_cache: Optional[WordNetLemmaCache] = None

    # ------------------------------------------------------------------
    # lazily built components (shared by every pipeline)
    # ------------------------------------------------------------------
    @property
    def lemma_cache(self) -> WordNetLemmaCache:
        if self._lemma_cache is None:
            self._lemma_cache = WordNetLemmaCache(self.logger)
        return self._lemma_cache

    def stopword_words(self, spec: PipelineSpec) -> set:
        strategy = self._stopwords.get(spec.stopword_strategy)
        if strategy is None:
            raise KeyError(
                f"unknown stopword strategy '{spec.stopword_strategy}'; "
                f"available: {sorted(self._stopwords)}"
            )
        return {word.casefold() for word in strategy.words}

    # ------------------------------------------------------------------
    # single-token components (cached: the corpus vocabulary is small)
    # ------------------------------------------------------------------
    def _stemmer(self, spec: PipelineSpec):
        if not hasattr(self, "_stemmer_cache"):
            self._stemmer_cache: Dict[str, Any] = {}
        name = spec.morphology_algorithm or "porter"
        if name not in self._stemmer_cache:
            self._stemmer_cache[name] = get_stemmer(name)
        return self._stemmer_cache[name]

    def transform_token(self, spec: PipelineSpec, token: str, stopwords: set,
                        step: str | None = None) -> str:
        """Apply one pipeline step to one token.

        Exactly one step is applied per call. Applying the whole order here would
        re-apply morphology when the caller's loop later reaches the morphology
        step, and Porter stemming is **not** idempotent (``financial`` ->
        ``financi`` -> ``financ``), so the index and the query side would quietly
        disagree. ``step=None`` applies the full order, for callers that want it.
        """
        current = token
        for name in (spec.order if step is None else (step,)):
            if name == "stopwords":
                if current.casefold() in stopwords:
                    return ""
            elif name == "morphology":
                current = self._morphologize(spec, current)
        return current

    def _morphologize(self, spec: PipelineSpec, token: str) -> str:
        if not any(character.isalpha() for character in token):
            return token
        if spec.morphology == "none":
            return token
        if spec.morphology == "stem":
            try:
                return self._stemmer(spec).stem(token.casefold())
            except Exception:
                return token.casefold()
        if spec.morphology == "lemmatize":
            if token.isupper() and len(token) <= 6:
                return token.casefold()
            if spec.morphology_algorithm == "lemminflect_rulebased":
                from src.phase2.lemmatization import lemminflect_lemmatize
                res = lemminflect_lemmatize(token, "NOUN")
                if res.casefold() == token.casefold() and (token.casefold().endswith("ing") or token.casefold().endswith("ed") or token.casefold().endswith("es") or token.casefold().endswith("s")):
                    res_v = lemminflect_lemmatize(token, "VERB")
                    if res_v and res_v != token:
                        return res_v
                return res
            return self.lemma_cache.lemmatize(token)
        raise KeyError(f"unknown morphology '{spec.morphology}' in pipeline {spec.key}")

    # ------------------------------------------------------------------
    # date/number protection
    # ------------------------------------------------------------------
    def protect_expressions(self, spec: PipelineSpec, text: str, tokens: Sequence[str]) -> List[str]:
        """Merge tokens that form one typed financial expression into one term.

        Phase 2 measured that standard tokenization leaves only 12.85% of the
        typed expressions intact. This step locates the Phase 2 expressions in
        the pipeline's own token stream and re-joins them, so ``7.4 per cent``,
        ``FY2025-26`` and ``Rs 1.25 lakh crore`` survive as single index terms.
        """
        if spec.date_number_strategy == "none":
            return list(tokens)
        surface = normalise_for_tokenisation(text)
        expressions = detect_expressions(surface)
        if not expressions:
            return list(tokens)

        tokenizer = TOKENIZERS[spec.tokenizer]
        # longest expression first, so a range beats its parts
        targets: List[List[str]] = []
        for expression in expressions:
            pieces = tokenizer(expression["text"])
            if len(pieces) > 1:
                targets.append([p.casefold() for p in pieces])
        targets.sort(key=len, reverse=True)

        lowered = [token.casefold() for token in tokens]
        consumed = [False] * len(tokens)
        merged: List[Optional[str]] = [None] * len(tokens)

        for target in targets:
            width = len(target)
            for start in range(0, len(tokens) - width + 1):
                if any(consumed[start : start + width]):
                    continue
                if lowered[start : start + width] == target:
                    merged[start] = " ".join(tokens[start : start + width]).casefold()
                    for offset in range(width):
                        consumed[start + offset] = True
                    break

        out: List[str] = []
        for index, token in enumerate(tokens):
            if consumed[index]:
                if merged[index] is not None:
                    out.append(merged[index])
                continue
            out.append(token)
        return out

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def run(self, spec: PipelineSpec, units: Sequence[Unit]) -> PipelineResult:
        """Execute ``spec`` over ``units`` and return the token view plus timings."""
        texts = [unit.text for unit in units]
        stopwords = self.stopword_words(spec)
        tokenizer = TOKENIZERS.get(spec.tokenizer)
        if tokenizer is None:
            raise KeyError(
                f"unknown tokenizer '{spec.tokenizer}' for pipeline {spec.key}; "
                f"available: {sorted(TOKENIZERS)}"
            )

        timings: Dict[str, float] = {}
        tokens: List[List[str]] = [[] for _ in units]
        expression_counts: Dict[str, int] = {}
        started = time.perf_counter()

        for step in spec.order:
            step_start = time.perf_counter()
            if step == "tokenize":
                tokens = [tokenizer(normalise_for_tokenisation(text)) for text in texts]
            elif step == "date_number":
                protected: List[List[str]] = []
                for text, unit_tokens in zip(texts, tokens):
                    before = len(unit_tokens)
                    merged = self.protect_expressions(spec, text, unit_tokens)
                    if len(merged) < before:
                        expression_counts[spec.key] = expression_counts.get(spec.key, 0) + before - len(merged)
                    protected.append(merged)
                tokens = protected
            elif step in ("stopwords", "morphology"):
                tokens = [
                    [
                        token
                        for token in (
                            self.transform_token(spec, token, stopwords, step) for token in unit_tokens
                        )
                        if token
                    ]
                    for unit_tokens in tokens
                ]
            elif step in ("pos", "ner", "ngrams"):
                # POS/NER metadata and n-gram materialisation are produced once,
                # in the index builder; a pipeline only declares them as part of
                # its design, so they are not timed twice.
                pass
            timings[step] = time.perf_counter() - step_start
            if self.logger is not None:
                log_event(
                    self.logger,
                    "INFO",
                    spec.key,
                    f"  {step:12} {timings[step]:6.2f}s  tokens={sum(len(t) for t in tokens)}",
                )

        view = build_tokenized(spec.key, list(units), tokens)
        total_seconds = time.perf_counter() - started
        result = PipelineResult(
            spec=spec,
            view=view,
            step_seconds=timings,
            total_seconds=total_seconds,
        )
        result.stats = {
            "pipeline": spec.key,
            "pipeline_name": spec.name,
            "tokenizer": spec.tokenizer,
            "date_number_strategy": spec.date_number_strategy,
            "stopword_strategy": spec.stopword_strategy,
            "morphology_strategy": spec.morphology_label,
            "morphology_kind": spec.term_kind,
            "pos_strategy": spec.pos_strategy,
            "ner_strategy": spec.ner_strategy,
            "ngram_strategy": f"1..{spec.ngram_max} generated from the pipeline token stream",
            "index_representation": spec.index_representation,
            "order": " -> ".join(spec.order),
            "total_tokens": view.total_tokens,
            "unique_tokens": view.unique_tokens,
            "vocabulary_size": view.vocabulary_size,
            "units": len(view.units),
            "documents": view.document_count,
            "avg_tokens_per_unit": view.avg_tokens_per_unit(),
            "type_token_ratio": view.type_token_ratio(),
            "expressions_rejoined": expression_counts.get(spec.key, 0),
            "processing_time_seconds": round(total_seconds, 3),
            "step_seconds": {k: round(v, 3) for k, v in timings.items()},
        }
        return result

    # ------------------------------------------------------------------
    # query-side normalization (identical component order)
    # ------------------------------------------------------------------
    def normalize_query(self, spec: PipelineSpec, query: str) -> List[str]:
        """Apply the pipeline's own components to a query string.

        ``"GDP growth in FY26"`` becomes ``['gdp', 'growth', 'fy26']`` under
        Pipeline A (WordNet lemmas) and ``['gdp', 'grow', 'fy26']`` under
        Pipeline B (Porter stems) - exactly how the two indexes were built.
        """
        stopwords = self.stopword_words(spec)
        tokens = self.tokenize_text(spec, query)
        for step in spec.order:
            if step == "date_number":
                tokens = self.protect_expressions(spec, query, tokens)
            elif step in ("stopwords", "morphology"):
                tokens = [
                    token
                    for token in (
                        self.transform_token(spec, token, stopwords, step) for token in tokens
                    )
                    if token
                ]
        return [self.term_for(spec, token) for token in tokens]

    def tokenize_text(self, spec: PipelineSpec, text: str) -> List[str]:
        tokenizer = TOKENIZERS.get(spec.tokenizer)
        if tokenizer is None:
            raise KeyError(f"unknown tokenizer '{spec.tokenizer}'")
        return tokenizer(normalise_for_tokenisation(text))

    # ------------------------------------------------------------------
    # phrase-side token stream (stopwords kept)
    # ------------------------------------------------------------------
    def phrase_stream(self, spec: PipelineSpec, text: str) -> List[str]:
        """Token stream used for *phrase* matching, with stopwords retained.

        The index vocabulary excludes stopwords, which is right for keyword
        retrieval and wrong for phrases: ``current account deficit`` must still
        be locatable, and it is only locatable if ``current`` is somewhere in
        the stream. The same tokenizer, date/number protection and morphology
        are applied, in the same order as the pipeline, so the terms are
        identical to the indexed ones - only the stopwords are kept.
        """
        tokens = self.tokenize_text(spec, text)
        for step in spec.order:
            if step == "date_number":
                tokens = self.protect_expressions(spec, text, tokens)
            elif step == "morphology":
                tokens = [self._morphologize(spec, token) for token in tokens]
        terms = [self.term_for(spec, token) for token in tokens]
        return [term for term in terms if term]

    def phrase_stream_terms(self, spec: PipelineSpec, phrase: str) -> List[str]:
        """The phrase's own term sequence, in order, stopwords included."""
        return self.phrase_stream(spec, phrase)

    def term_for(self, spec: PipelineSpec, token: str) -> str:
        """Canonical index-term form of a single token (case folding only)."""
        if self.config.get("retrieval.case_sensitive", False):
            term = token
        else:
            term = token.casefold()
        return _WS.sub(" ", term).strip()
