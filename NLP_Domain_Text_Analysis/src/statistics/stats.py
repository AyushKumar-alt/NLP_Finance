"""Phase 1 baseline corpus statistics.

Scope warning (important for the report)
----------------------------------------
Everything here is a **baseline** measurement taken on the Phase 1 corpus. It
is deliberately *not* the tokenizer experiment of Phase 2 and must never be
presented as one:

``sentences``
    NLTK ``punkt_tab`` (trained, abbreviation-aware) sentence segmentation.
    Falls back to a documented abbreviation-aware regular expression when the
    NLTK data package is unavailable. The original text is never modified -
    only the segment boundaries are counted.
``baseline_token_count``
    Tokens produced by the single baseline regex declared in
    ``config: statistics.baseline_token_pattern``. The pattern is written to
    KEEP numbers (``7.4``, ``2025``), percentages (``7.4 per cent``),
    currency symbols (``₹``, ``US$``) and fiscal-period abbreviations
    (``FY26``, ``Q3``, ``H1``).
``vocabulary``
    Unique NFKC + casefolded surface forms. **No** stemming and **no**
    lemmatization - those belong to Phase 2.

Economic-text hazards that the sentence counter must respect:
``7.4 per cent``, ``U.S.``, ``IMF.``, ``RBI.``, ``FY26.``, ``Rs.``, ``etc.``
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.config import Phase1Config
from ..core.utils import count_words, normalise_key

ABBREVIATIONS = {
    "u.s.", "u.k.", "u.n.", "eu", "imf", "rbi", "sebi", "gdp", "gva", "gfcf",
    "pfce", "fdi", "cpi", "wpi", "iip", "mospi", "unctad", "ngo.", "s.", "p.",
    "no.", "pp.", "fig.", "chart", "table", "box", "sec.", "vol.", "ed.",
    "rs.", "mr.", "cr.", "lakh", "crore", "mn.", "bn.", "tr.",
}

SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])                 # sentence-final punctuation
    \s+                       # followed by whitespace
    (?=
        (?:[A-Z0-9₹$€£¥"'(\[]) # next token starts a new sentence
    )
    """,
    re.VERBOSE,
)

INITIAL = re.compile(r"\b[A-Z]\.")


def _merge_abbreviation_fragments(sentences: Sequence[str]) -> List[str]:
    """Re-join sentences that a trained model split after a domain abbreviation.

    ``punkt`` is trained on general English and does split ``RBI.``/``IMF.``/
    ``GDP.`` in economic prose. Those are abbreviations, not sentence ends, so a
    documented post-guard merges them again. The method is reported as
    ``nltk:punkt_tab+abbrev_guard`` so the experiment stays honest.
    """
    guard = {a.rstrip(".") for a in ABBREVIATIONS}
    merged: List[str] = []
    for sentence in sentences:
        stripped = sentence.rstrip()
        if merged:
            previous = merged[-1].rstrip()
            tail = re.split(r"[\s(]", previous)[-1].rstrip(".:;,")
            if tail.casefold() in guard and len(previous.split()) <= 12:
                merged[-1] = f"{previous} {stripped.lstrip()}"
                continue
        merged.append(stripped)
    return merged


class SentenceCounter:
    """Counts sentences without ever altering the source text."""

    def __init__(self, config: Phase1Config):
        self.config = config
        self.preferred = str(config.get("statistics.sentence_tokenizer", "nltk_punkt_tab"))
        self.method = "unavailable"
        self._tokenizer = None
        self.method = "regex_abbreviation_aware"
        self._init_nltk()

    def _init_nltk(self) -> None:
        try:
            import nltk  # type: ignore
        except Exception:  # pragma: no cover - nltk missing
            self._tokenizer = None
            return

        for path in self.config.get("statistics.nltk_search_paths", []) or []:
            candidate = str(self.config.project_root / path)
            if candidate not in nltk.data.path:
                nltk.data.path.append(candidate)

        if not self.preferred.startswith("nltk"):
            return

        # ``punkt_tab`` (NLTK >= 3.9) is the preferred trained model; the older
        # ``punkt`` model is used as a fallback. Both are abbreviation aware,
        # which matters for ``U.S.``, ``RBI.``, ``FY26.`` and ``7.4 per cent``.
        loaders = (
            ("tokenizers/punkt_tab", None),
            ("tokenizers/punkt", None),
        )
        if self.preferred == "nltk_punkt":
            loaders = loaders[::-1]

        for resource, _fmt in loaders:
            try:
                nltk.data.find(resource)
            except LookupError:
                continue
            try:
                tokenizer = nltk.data.load(resource)
                if hasattr(tokenizer, "tokenize_sents"):
                    self._tokenizer = tokenizer.tokenize_sents
                    self.method = f"nltk:{resource.split('/')[-1]}"
                    return
            except Exception:  # punkt_tab needs the PunktTokenizer driver
                pass
            try:
                from nltk.tokenize import PunktTokenizer  # type: ignore

                tokenizer = None
                for language in ("english", resource.split("/")[-1]):
                    try:
                        tokenizer = PunktTokenizer(lang=language)
                        break
                    except Exception:  # pragma: no cover - unknown language pack
                        continue
                if tokenizer is None:
                    continue
                self._tokenizer = tokenizer.tokenize
                self.method = f"nltk:{resource.split('/')[-1]}"
                if bool(self.config.get("statistics.abbreviation_post_guard", True)):
                    self.method += "+abbrev_guard"
                return
            except Exception:  # pragma: no cover
                continue

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(self.split(text))

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if self._tokenizer is not None:
            try:
                sentences = self._tokenizer(text)
            except Exception:  # pragma: no cover - defensive
                sentences = None
            if sentences:
                if bool(self.config.get("statistics.abbreviation_post_guard", True)):
                    sentences = _merge_abbreviation_fragments(sentences)
                return [s for s in sentences if s and s.strip()]
        return self._segment_regex(text)

    # ------------------------------------------------------------------
    @staticmethod
    def _segment_regex(text: str) -> List[str]:
        """Abbreviation-aware fallback segmentation.

        Protects the hazards listed in the module docstring before splitting on
        sentence-final punctuation.
        """
        protected = text
        for abbreviation in sorted(ABBREVIATIONS, key=len, reverse=True):
            if not abbreviation.endswith("."):
                continue
            protected = re.sub(
                rf"(?i)(?<=\b{re.escape(abbreviation[:-1])})\.(?=\s+[A-Z0-9\"'(\[])",
                "\u0001",
                protected,
            )
        # ``7.4 per cent`` - the dot is a decimal point, not a full stop.
        protected = re.sub(r"(?<=\d)\.(?=\s*(?:per\s+cent|percent|%))", "\u0001", protected)
        # Single capital letter initials such as ``A.``.
        protected = INITIAL.sub(lambda m: m.group(0).replace(".", "\u0001"), protected)
        pieces = SENTENCE_BOUNDARY.split(protected)
        return [p.replace("\u0001", ".").strip() for p in pieces if p.strip()]


@dataclass
class Tokenizer:
    """Baseline word tokenizer - explicitly *not* the Phase 2 experiment."""

    pattern: re.Pattern
    keep_original_case: bool = False

    def tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return self.pattern.findall(text)

    def normalised_tokens(self, text: str) -> List[str]:
        return [normalise_key(token) for token in self.tokens(text)]


def build_tokenizer(config: Phase1Config) -> Tokenizer:
    pattern = str(config.get("statistics.baseline_token_pattern"))
    compiled = re.compile(pattern, re.UNICODE)
    return Tokenizer(pattern=compiled)


@dataclass
class DocumentStats:
    document_id: str
    source_id: str
    filename: str
    page_count: int
    sentences: int = 0
    words: int = 0
    characters: int = 0
    baseline_token_count: int = 0
    unique_words: int = 0
    vocabulary_size: int = 0
    average_document_length: float = 0.0
    sentences_per_page: float = 0.0
    paragraphs: int = 0
    tables: int = 0
    figures: int = 0
    footnotes: int = 0
    sections: int = 0
    units: int = 0
    raw_characters: int = 0
    cleaned_characters: int = 0
    raw_vocabulary_size: int = 0
    rows: Dict[str, int] = field(default_factory=dict)


class StatisticsCollector:
    """Accumulates per-document and corpus-level baseline statistics."""

    def __init__(self, config: Phase1Config):
        self.config = config
        self.sentences = SentenceCounter(config)
        self.tokenizer = build_tokenizer(config)
        self.min_length = int(config.get("statistics.vocabulary_min_length", 1))
        self.term_min_length = int(config.get("statistics.term_dictionary_min_length", 1))

        self.term_frequency: Counter = Counter()
        self.document_frequency: Counter = Counter()
        self.term_documents: Dict[str, set] = defaultdict(set)
        self.vocab_frequency: Counter = Counter()
        self.raw_vocab: set = set()
        self._corpus_sentences = 0
        self._corpus_baseline_tokens = 0
        self._corpus_raw_tokens = 0
        self._corpus_words = 0
        self._corpus_characters = 0
        self._corpus_raw_characters = 0
        self._corpus_cleaned_characters = 0
        self._corpus_vocab: set = set()
        self._corpus_raw_vocab: set = set()

    # ------------------------------------------------------------------
    def add_document(self, result, record) -> DocumentStats:
        stats = DocumentStats(
            document_id=result.document_id,
            source_id=result.source_id,
            filename=record.filename,
            page_count=result.page_count,
        )
        units_by_type: Counter = Counter(unit.unit_type for unit in result.units)

        stats.paragraphs = units_by_type.get("paragraph", 0)
        stats.tables = units_by_type.get("table", 0)
        stats.figures = units_by_type.get("figure", 0)
        stats.footnotes = units_by_type.get("footnote", 0) + units_by_type.get("reference", 0)
        stats.sections = len(result.sections)
        stats.units = len(result.units)
        stats.raw_characters = len(result.raw_text)
        stats.cleaned_characters = len(result.cleaned_text)

        document_terms: Counter = Counter()
        document_raw_forms: set = set()
        for unit in result.units:
            text = unit.text
            if not text:
                continue
            stats.sentences += self.sentences.count(text)
            stats.characters += len(text)
            stats.words += count_words(text)

            baseline = self.tokenizer.tokens(text)
            stats.baseline_token_count += len(baseline)
            self._corpus_raw_tokens += len(baseline)

            # raw vocabulary = case-sensitive surface forms, as extracted.
            document_raw_forms.update(baseline)
            self._corpus_raw_vocab.update(baseline)

            for token in baseline:
                term = normalise_key(token)
                if len(term) < self.term_min_length:
                    continue
                document_terms[term] += 1
                self._corpus_vocab.add(term)

        stats.raw_vocabulary_size = len(document_raw_forms)
        stats.vocabulary_size = len(document_terms)
        stats.unique_words = stats.words
        stats.average_document_length = (
            round(stats.baseline_token_count / stats.units, 4) if stats.units else 0.0
        )
        stats.sentences_per_page = (
            round(stats.sentences / result.page_count, 4) if result.page_count else 0.0
        )

        for term, frequency in document_terms.items():
            self.term_frequency[term] += frequency
            self.document_frequency[term] += 1
            self.term_documents[term].add(result.document_id)

        self._corpus_sentences += stats.sentences
        self._corpus_baseline_tokens += stats.baseline_token_count
        self._corpus_words += stats.words
        self._corpus_characters += stats.characters
        self._corpus_raw_characters += stats.raw_characters
        self._corpus_cleaned_characters += stats.cleaned_characters
        return stats

    # ------------------------------------------------------------------
    def corpus_totals(self, documents: Sequence[DocumentStats]) -> Dict[str, object]:
        total_units = sum(d.units for d in documents)
        total_pages = sum(d.page_count for d in documents)
        return {
            "documents": len(documents),
            "pages": total_pages,
            "sentences": self._corpus_sentences,
            "words": self._corpus_words,
            "characters": self._corpus_characters,
            "baseline_token_count": self._corpus_baseline_tokens,
            "baseline_vocabulary_size": len(self._corpus_vocab),
            "raw_vocabulary_size": len(self._corpus_raw_vocab),            "units": total_units,
            "raw_characters": self._corpus_raw_characters,
            "cleaned_characters": self._corpus_cleaned_characters,
            "average_tokens_per_unit": (
                round(self._corpus_baseline_tokens / total_units, 4) if total_units else 0.0
            ),
            "average_tokens_per_document": (
                round(self._corpus_baseline_tokens / len(documents), 4) if documents else 0.0
            ),
            "average_sentences_per_document": (
                round(self._corpus_sentences / len(documents), 4) if documents else 0.0
            ),
            "sentence_tokenizer_method": self.sentences.method,
            "baseline_token_pattern": self.tokenizer.pattern.pattern,
            "baseline_normalization": str(
                self.config.get("statistics.baseline_normalization", "")
            ),
        }

    # ------------------------------------------------------------------
    def vocabulary_rows(self, limit: int) -> List[Dict[str, object]]:
        rows = []
        for term, frequency in self.term_frequency.most_common(limit):
            rows.append(
                {
                    "term": term,
                    "frequency": frequency,
                    "document_frequency": self.document_frequency[term],
                    "documents": sorted(self.term_documents[term]),
                }
            )
        return rows

    def term_dictionary_rows(self, limit: int) -> List[Dict[str, object]]:
        return self.vocabulary_rows(limit)
