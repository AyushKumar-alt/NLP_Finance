"""Lemmatization experiments.

Four real approaches are compared, all available in this environment:

``wordnet_lookup_pos_agnostic``
    Princeton WordNet, noun-first lookup with no POS information (the classic
    WordNetLemmatizer behaviour).
``wordnet_lookup_pos_aware``
    The same WordNet table, but the part of speech comes from the spaCy pass.
    This is the documented POS dependency: without POS, WordNet returns the
    noun lemma for ``grew`` -> ``growth``-style ambiguities are frequent.
``spacy_rule_based``
    spaCy's rule-based lemmatizer (inflectional rules + exception lists).
``lemminflect_rulebased``
    ``lemminflect``'s rule-based inflectional morphology, POS driven.

Environment note (recorded in every artefact): the NLTK ``wordnet`` corpus could
not be downloaded here (network policy blocks ``raw.githubusercontent.com``), so
the WordNet table is read from the ``spacy-lookups-data`` package, which ships
the same WordNet-derived lemma table. ``nltk`` is still used for stemming and
tokenization, where its resources are available.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .spacy_pipeline import SpacyAnnotations
from .statistics import TokenizedCorpus, percent_reduction

#: Universal POS -> WordNet part of speech, used by the POS-aware lookups.
UNIVERSAL_TO_WORDNET = {
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}

_WORDNET_TABLE: Optional[Dict[str, str]] = None
_WORDNET_SOURCE = ""


def _read_spacy_lookups(table_name: str):
    """Read one gzipped JSON table from ``spacy-lookups-data`` (or a plain JSON)."""
    import gzip
    import json

    import spacy_lookups_data

    path = spacy_lookups_data.en[table_name]
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if not gz.exists():
            raise FileNotFoundError(f"{table_name} not found at {path} or {gz}")
        with gzip.open(gz, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_lookup_tables() -> Tuple[Dict[str, str], Dict[str, set]]:
    """Load the WordNet-derived lemma tables shipped with ``spacy-lookups-data``.

    * ``lemma_lookup`` maps an inflected surface form to its lemma (~41k forms).
    * ``lemma_index`` maps a WordNet part of speech to the surface forms WordNet
      knows in that category. It is what makes the POS-aware variant a genuine
      WordNet lookup rather than a plain dictionary lookup.
    """
    try:
        raw = _read_spacy_lookups("lemma_lookup")
    except Exception:  # pragma: no cover
        raw = {}
    lookup: Dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            lookup[key.casefold()] = value
        elif isinstance(value, list) and value:
            lookup[key.casefold()] = value[0]
    try:
        raw_index = _read_spacy_lookups("lemma_index")
    except Exception:  # pragma: no cover
        raw_index = {}
    index: Dict[str, set] = {}
    for pos, surfaces in raw_index.items():
        if isinstance(surfaces, list):
            index[pos] = {str(s).casefold() for s in surfaces}
    return lookup, index


def load_wordnet_table(logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """Expose ``surface_form -> lemma`` for reporting, examples and tests."""
    global _WORDNET_TABLE, _WORDNET_SOURCE
    if _WORDNET_TABLE is None:
        lookup, index = _load_lookup_tables()
        _WORDNET_TABLE = lookup
        _WORDNET_SOURCE = "spacy-lookups-data en lemma_lookup + en lemma_index (WordNet-derived)"
        if logger is not None:
            log_event(
                logger,
                "INFO",
                "lemmatization",
                f"WordNet lemma table: {len(lookup)} inflected forms, "
                f"{sum(len(v) for v in index.values())} POS-indexed surface forms",
            )
    return _WORDNET_TABLE


def wordnet_source() -> str:
    return _WORDNET_SOURCE


class WordNetLemmaCache:
    """WordNet lookups, with and without a POS hint.

    ``pos_agnostic`` reproduces the default behaviour of
    ``nltk.WordNetLemmatizer()``: a noun-first lookup that ignores the tag.
    ``POS-aware`` consults the WordNet surface index for the requested part of
    speech first, so a form that is not in WordNet's category for that tag is
    left untouched (this is exactly where a POS error would propagate).
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._lookup, self._index = _load_lookup_tables()
        load_wordnet_table(logger)

    def lemmatize(self, token: str, pos: Optional[str] = None) -> str:
        key = token.casefold()
        lemma = self._lookup.get(key)
        if lemma is None:
            return token
        if pos:
            wn_pos = UNIVERSAL_TO_WORDNET.get(pos.upper())
            surfaces = self._index.get(wn_pos, set()) if wn_pos else set()
            if surfaces and key not in surfaces:
                # WordNet does not know this form in the requested category:
                # keep the surface form (the NLTK behaviour for a wrong POS).
                return token
        return lemma


def lemmatize_view(
    tokenized: TokenizedCorpus,
    method: str = "wordnet_lookup_pos_agnostic",
    logger: Optional[logging.Logger] = None,
) -> TokenizedCorpus:
    """Lemmatize an arbitrary token view (used by the pipeline comparison).

    Only the POS-agnostic WordNet lookup is available here because the other
    methods need spaCy tokens; the multi-method comparison in
    :func:`run_lemmatization` uses the spaCy token stream instead.
    """
    cache = WordNetLemmaCache(logger)
    lemmas = [
        [
            cache.lemmatize(token) if any(ch.isalpha() for ch in token) else token
            for token in unit_tokens
        ]
        for unit_tokens in tokenized.tokens
    ]
    return TokenizedCorpus(name=f"lemma_{method}", units=tokenized.units, tokens=lemmas)


def lemminflect_lemmatize(token: str, upos: Optional[str] = None) -> str:
    """Rule-based lemmatization restricted to the POS lemminflect supports.

    ``lemminflect`` only handles NOUN/VERB/ADJ/ADV; any other universal tag
    (ADP, PUNCT, NUM, ...) would raise inside the library, so the surface form
    is returned unchanged and that limitation is recorded in the comparison row.
    """
    if upos and upos.upper() not in ("NOUN", "VERB", "ADJ", "ADV"):
        return token
    try:
        import lemminflect

        result = lemminflect.getLemma(token.casefold(), (upos or "NOUN").upper())
        if isinstance(result, (tuple, list)):
            return result[0] if result else token
        return result
    except Exception:
        return token


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def run_lemmatization(
    tokenized: TokenizedCorpus,
    annotations: SpacyAnnotations,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Compare the four approaches on the same token stream."""
    methods = list(
        config.get(
            "lemmatization.methods",
            [
                "wordnet_lookup_pos_agnostic",
                "wordnet_lookup_pos_aware",
                "spacy_rule_based",
                "lemminflect_rulebased",
            ],
        )
    )
    # spaCy lemmas come from the shared pass, aligned index-for-index.
    spacy_lemmas = [a.lemmas for a in annotations.annotations]
    spacy_pos = [a.pos for a in annotations.annotations]

    wordnet_agnostic = WordNetLemmaCache(logger)
    wordnet_aware = WordNetLemmaCache(logger)

    def transform(
        method: str,
        unit_index: int,
        token_index: int,
        token: str,
    ) -> str:
        if not any(ch.isalpha() for ch in token):
            return token
        upos = None
        tags = spacy_pos[unit_index]
        if token_index < len(tags):
            upos = tags[token_index]
        if method == "wordnet_lookup_pos_agnostic":
            return wordnet_agnostic.lemmatize(token, None)
        if method == "wordnet_lookup_pos_aware":
            return wordnet_aware.lemmatize(token, upos)
        if method == "spacy_rule_based":
            lemmas = spacy_lemmas[unit_index]
            if token_index < len(lemmas) and lemmas[token_index]:
                return lemmas[token_index]
            return token
        if method == "lemminflect_rulebased":
            return lemminflect_lemmatize(token, upos)
        raise KeyError(f"unknown lemmatization method '{method}'")

    views: Dict[str, TokenizedCorpus] = {}
    comparison_rows: List[Dict[str, object]] = []
    baseline_vocab = tokenized.vocabulary_size

    for method in methods:
        lemmas: List[List[str]] = []
        for unit_index, unit_tokens in enumerate(tokenized.tokens):
            lemmas.append(
                [transform(method, unit_index, index, token) for index, token in enumerate(unit_tokens)]
            )
        view = TokenizedCorpus(
            name=f"lemma_{method}", units=tokenized.units, tokens=lemmas
        )
        views[method] = view
        comparison_rows.append(
            {
                "method": method,
                "pos_required": method in ("wordnet_lookup_pos_aware", "lemminflect_rulebased"),
                "pos_source": "spaCy universal POS from the shared pass" if method in ("wordnet_lookup_pos_aware", "lemminflect_rulebased") else "none",
                "lemma_resource": wordnet_source() if method.startswith("wordnet") else (
                    "spaCy en_core_web_sm rule-based lemmatizer" if method == "spacy_rule_based" else "lemminflect rule tables"
                ),
                "total_tokens": view.total_tokens,
                "unique_tokens": view.unique_tokens,
                "unique_lemmas": view.vocabulary_size,
                "vocabulary_before": baseline_vocab,
                "vocabulary_reduction": baseline_vocab - view.vocabulary_size,
                "vocabulary_reduction_percent": percent_reduction(baseline_vocab, view.vocabulary_size),
                "avg_tokens_per_document": view.avg_tokens_per_document(),
                "type_token_ratio": view.type_token_ratio(),
                "limitations": _limitation(method),
            }
        )
        if logger is not None:
            log_event(
                logger,
                "INFO",
                "lemmatization",
                f"{method}: vocabulary {baseline_vocab} -> {view.vocabulary_size} "
                f"({percent_reduction(baseline_vocab, view.vocabulary_size)}% reduction)",
            )

    # ---- examples: observed corpus vocabulary + required probe words ----
    observed = sorted({t.casefold() for t in tokenized.flat if any(c.isalpha() for c in t)})
    probe = [w.casefold() for w in config.get("lemmatization.probe_words", []) or []]
    example_words: List[str] = []
    seen: set = set()
    for word in probe:
        if word not in seen:
            seen.add(word)
            example_words.append(word)
    for word in observed:
        if len(example_words) >= 60:
            break
        if word not in seen:
            seen.add(word)
            example_words.append(word)

    example_rows: List[Dict[str, object]] = []
    for word in example_words:
        cells = {method: transform(method, 0, 0, word) for method in methods}
        # Observe the POS the shared spaCy pass actually assigned in the corpus.
        pos_seen = _observed_pos(tokenized, annotations, word)
        example_rows.append(
            {
                "original_word": word,
                "observed_pos_in_corpus": pos_seen,
                "wordnet_lookup_pos_agnostic": cells.get("wordnet_lookup_pos_agnostic", ""),
                "wordnet_lookup_pos_aware": cells.get("wordnet_lookup_pos_aware", ""),
                "spacy_rule_based": cells.get("spacy_rule_based", ""),
                "lemminflect_rulebased": cells.get("lemminflect_rulebased", ""),
                "from_probe_list": word in set(probe),
                "notes": _lemma_note(word, cells),
            }
        )

    return {"comparison": comparison_rows, "examples": example_rows, "views": views}


def _observed_pos(tokenized: TokenizedCorpus, annotations: SpacyAnnotations, word: str, limit: int = 5) -> str:
    for index, unit_tokens in enumerate(tokenized.tokens):
        for token_index, token in enumerate(unit_tokens):
            if token.casefold() == word:
                pos_list = annotations.annotations[index].pos
                if token_index < len(pos_list) and pos_list[token_index]:
                    return pos_list[token_index]
    return ""


def _limitation(method: str) -> str:
    return {
        "wordnet_lookup_pos_agnostic": "no POS input, so verbs and adjectives are lemmatized as nouns; domain coinages absent from WordNet are returned unchanged",
        "wordnet_lookup_pos_aware": "needs reliable POS, so a POS-tagging error propagates into the lemma; domain coinages still unchanged",
        "spacy_rule_based": "trained on general English; financial coinages (e.g. disinflation) are usually left unchanged",
        "lemminflect_rulebased": "rule tables cover standard English inflections only; irregular financial nouns are unchanged",
    }.get(method, "")


def _lemma_note(word: str, cells: Dict[str, str]) -> str:
    values = set(cells.values())
    if len(values) == 1:
        return "all methods agree"
    unchanged = [name for name, value in cells.items() if value == word]
    if unchanged:
        return f"unchanged by: {', '.join(sorted(unchanged))}"
    return "methods diverge"
