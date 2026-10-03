"""Tokenizer implementations and the tokenization experiment driver.

Four tokenizers are compared on **one identical text sample** (see
``load_corpus.build_experiment_sample``):

``nltk``
    ``nltk.tokenize.word_tokenize`` (TreebankWordTokenizer + Punkt sentence
    splitting). The standard academic baseline.
``spacy``
    ``en_core_web_sm``'s tokenizer. Rule-based, trained on English orthography,
    keeps ``6.50`` and ``repo-rate`` as single tokens.
``custom``
    The financial rule set in :mod:`src.phase2.custom_tokenizer`.
``hybrid``
    NLTK word tokenization **plus** a domain layer: domain expressions are held
    out before tokenization and re-inserted afterwards, and orphaned currency /
    percent symbols are merged back into their numbers.

Every tokenizer is measured with the same metric functions so the comparison is
apples-to-apples.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from .config import Phase2Config, log_event
from .custom_tokenizer import custom_tokenize, preprocess
from .date_number_tokenizer import EXPRESSION_PATTERNS, date_number_tokenize
from .load_corpus import ExperimentSample, Unit, normalise_for_tokenisation

# ----------------------------------------------------------------------
# spaCy loading
# ----------------------------------------------------------------------
_SPACY_NLP = None
_SPACY_STATUS: Dict[str, object] = {}


def load_spacy(model_name: str, logger: Optional[logging.Logger] = None):
    """Load the configured spaCy model once, with a clear failure message."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "spaCy is not installed. Install it with:\n"
            "    pip install spacy\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc
    try:
        _SPACY_NLP = spacy.load(model_name)
    except OSError as exc:
        _SPACY_STATUS["error"] = str(exc)
        raise RuntimeError(
            f"The spaCy model '{model_name}' is not installed. Install it with:\n"
            f"    python -m spacy download {model_name}\n"
            "No substitute model is loaded, because a different pipeline would "
            "silently change the token/POS/NER results being measured."
        ) from exc
    _SPACY_STATUS["model"] = model_name
    _SPACY_STATUS["version"] = getattr(_SPACY_NLP, "meta", {}).get("version", "unknown")
    if logger is not None:
        log_event(logger, "INFO", "tokenization", f"spaCy model '{model_name}' loaded")
    return _SPACY_NLP


def spacy_status() -> Dict[str, object]:
    return dict(_SPACY_STATUS)


# ----------------------------------------------------------------------
# Token classifiers (shared by every tokenizer so counts are comparable)
# ----------------------------------------------------------------------
RE_NUMERIC = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?$")
RE_HAS_DIGIT = re.compile(r"\d")
RE_CURRENCY = re.compile(
    r"^(?:\u20b9|US\$|US\s*\$|Rs\.?|rs\.?|INR|inr|USD|usd|\$|\u20ac|EUR|eur|GBP|gbp|\u00a3)"
)
RE_PERCENT = re.compile(r"^(?:[+-]?\d[\d,]*(?:\.\d+)?\s*(?:%|per\s+cent|percent|pct\.?))$", re.I)
RE_ABBREV = re.compile(r"^(?:[A-Z]{2,6}|(?:[A-Za-z]\.){2,}[A-Za-z]{1,6}|[A-Za-z]{1,6}\.(?:[A-Za-z]\.)+)$")
RE_HYPHEN = re.compile(r"[-/]")
RE_DATE = re.compile(
    r"(?i)^(?:"
    r"\d{1,2}\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s*\d{0,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s*\d{0,4}"
    r"|fy\s*\d{2,4}(?:\s*[-/]\s*\d{2,4})?"
    r"|\d{4}\s*[-/]\s*\d{2,4}"
    r"|q[1-4]\s*(?:fy\s*\d{2,4}|\d{2,4})?"
    r"|(?:19|20)\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r")$"
)
RE_FISCAL_YEAR = re.compile(r"(?i)^(?:fy\s*\d{2,4}(?:\s*[-/]\s*\d{2,4})?|\d{4}\s*[-/]\s*\d{2,4})$")

#: Small, explicit financial lexicon used only for *counting* how many tokens a
#: tokenizer considers to be special financial tokens. It is derived from the
#: corpus terminology study, not from any external labelled resource.
SPECIAL_FINANCIAL_TOKENS = frozenset(
    {
        "gdp", "gva", "cpi", "wpi", "fdi", "fpi", "rbi", "sebi", "imf", "wto", "npa",
        "repo", "repo-rate", "reverse-repo", "msf", "crr", "slr", "g-sec", "gsec",
        "lakh", "crore", "billion", "trillion", "million", "per", "cent", "fiscal",
        "monetary", "inflation", "deflation", "disinflation", "budget", "niti", "gst",
        "esop", "emi", "npa-ratio", "twrs", "dpi", "plfs", "nre", "capex", "g-secs",
    }
)


def classify_token(token: str) -> Dict[str, bool]:
    """Category flags for one token, using the same tests for every tokenizer."""
    has_digit = bool(RE_HAS_DIGIT.search(token))
    is_alpha = any(ch.isalpha() for ch in token)
    flags = {
        "numeric": bool(RE_NUMERIC.match(token)) or (has_digit and not RE_DATE.match(token)),
        "date": bool(RE_DATE.match(token)),
        "fiscal_year": bool(RE_FISCAL_YEAR.match(token)),
        "percentage": bool(RE_PERCENT.match(token)) or token.strip() == "%",
        "currency": bool(RE_CURRENCY.match(token)),
        "abbreviation": bool(RE_ABBREV.match(token)),
        "hyphenated": bool(RE_HYPHEN.search(token)) and is_alpha,
        "special_financial": token.casefold() in SPECIAL_FINANCIAL_TOKENS,
    }
    return flags


CATEGORY_KEYS = (
    "numeric",
    "date",
    "fiscal_year",
    "percentage",
    "currency",
    "abbreviation",
    "hyphenated",
    "special_financial",
)


# ----------------------------------------------------------------------
# Tokenizer implementations
# ----------------------------------------------------------------------
def nltk_tokenize(text: str) -> List[str]:
    """Baseline: NLTK ``word_tokenize`` (Treebank + Punkt)."""
    return word_tokenize(text)


def spacy_tokenize_batch(texts: Sequence[str], nlp, batch_size: int = 128) -> List[List[str]]:
    """spaCy tokenizer, batched.

    Only the tokenizer runs: the parser, NER and lemmatizer are disabled because
    this function needs token boundaries only, and skipping them keeps the pass
    fast and inside the model's memory limit on very long table units.
    """
    docs = nlp.pipe(
        list(texts),
        batch_size=batch_size,
        disable=["parser", "ner", "senter", "lemmatizer"],
    )
    return [[tok.text for tok in doc] for doc in docs]


def custom_tokenize_batch(texts: Sequence[str], keep_punctuation: bool = True) -> List[List[str]]:
    return [custom_tokenize(t, keep_punctuation=keep_punctuation) for t in texts]


# ---- hybrid -------------------------------------------------------
#: Domain expressions are lifted out before the standard tokenizer runs so it
#: cannot fragment them, then re-inserted afterwards.
_HYBRID_PROTECT = re.compile(
    "|".join(
        f"(?:{pattern})"
        for _name, pattern, _desc in sorted(
            EXPRESSION_PATTERNS, key=lambda row: -len(row[1])
        )
    ),
    re.IGNORECASE | re.UNICODE,
)
_PLACEHOLDER = "\uE000{}\uE001"  # private-use sentinel, removed by the cleaner


def hybrid_tokenize(text: str) -> List[str]:
    """NLTK word tokenization with a financial-domain layer.

    Three steps, each of which fixes a concrete failure of the plain baseline:

    1. *protect*   - domain expressions (dates, fiscal years, percentages,
       currency amounts, magnitudes) are replaced by sentinels;
    2. *tokenize*  - NLTK tokenizes what is left;
    3. *repair*    - sentinels are expanded back into whole tokens, and orphan
       symbols that NLTK detached (``6.50`` + ``%``, ``$`` + ``2.5``) are merged.
    """
    if not text:
        return []
    surface = preprocess(text)
    protected: List[str] = []

    def _protect(match: re.Match) -> str:
        protected.append(match.group().strip())
        return _PLACEHOLDER.format(len(protected) - 1)

    staged = _HYBRID_PROTECT.sub(_protect, surface)
    tokens = word_tokenize(staged)

    # NLTK can split *inside* a sentinel, so the expansion is done on a regex
    # over each token rather than with a startswith/endswith test.
    expanded: List[str] = []
    for token in tokens:
        if "\ue000" not in token:
            expanded.append(token)
            continue
        cursor = 0
        for sentinel in _SENTINEL_RE.finditer(token):
            prefix = token[cursor : sentinel.start()].strip()
            if prefix:
                expanded.append(prefix)
            index = sentinel.group(1)
            expanded.append(
                protected[int(index)] if index.isdigit() and int(index) < len(protected) else sentinel.group(0)
            )
            cursor = sentinel.end()
        tail = token[cursor:].strip()
        if tail:
            expanded.append(tail)

    return _repair_orphan_symbols(expanded)


_SENTINEL_RE = re.compile("\uE000(\\d+)\uE001")


_ORPHAN_MERGE = (
    (re.compile(r"^([-+]?\d[\d,]*(?:\.\d+)?)$"), re.compile(r"^(?:%|per\s+cent|percent)$", re.I)),
    (re.compile(r"^(?:\u20b9|US\$|Rs\.?|INR|\$|\u20ac|GBP)$", re.I), re.compile(r"^\d")),
    (re.compile(r"^/$"), re.compile(r"^\d")),
)


def _repair_orphan_symbols(tokens: List[str]) -> List[str]:
    """Merge symbols NLTK detached from the number they belong to."""
    out: List[str] = []
    for token in tokens:
        merged = False
        for number_re, symbol_re in _ORPHAN_MERGE:
            if symbol_re.match(token) and out:
                if number_re.match(out[-1]):
                    out[-1] = f"{out[-1]}{token}"
                    merged = True
                    break
        if not merged:
            out.append(token)
    return out


# ----------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------
@dataclass
class TokenizerRun:
    name: str
    tokens_per_unit: List[List[str]]
    sentences: int
    documents_processed: int
    seconds: float
    description: str

    def metrics(self) -> Dict[str, object]:
        flat: List[str] = [t for unit_tokens in self.tokens_per_unit for t in unit_tokens]
        lowered = [t.casefold() for t in flat]
        word_like = {t for t in lowered if any(ch.isalpha() for ch in t)}
        category_counts = {key: 0 for key in CATEGORY_KEYS}
        for token in flat:
            for key, value in classify_token(token).items():
                if value:
                    category_counts[key] += 1
        return {
            "tokenizer": self.name,
            "description": self.description,
            "documents_processed": self.documents_processed,
            "units_processed": len(self.tokens_per_unit),
            "sentences": self.sentences,
            "total_tokens": len(flat),
            "unique_tokens": len(set(flat)),
            "vocabulary_size": len(word_like),
            "avg_tokens_per_document": round(len(flat) / self.documents_processed, 2)
            if self.documents_processed
            else 0.0,
            "avg_tokens_per_sentence": round(len(flat) / self.sentences, 2) if self.sentences else 0.0,
            "avg_tokens_per_unit": round(len(flat) / len(self.tokens_per_unit), 2)
            if self.tokens_per_unit
            else 0.0,
            "numeric_tokens": category_counts["numeric"],
            "date_tokens": category_counts["date"],
            "fiscal_year_tokens": category_counts["fiscal_year"],
            "percentage_tokens": category_counts["percentage"],
            "currency_tokens": category_counts["currency"],
            "abbreviation_tokens": category_counts["abbreviation"],
            "hyphenated_tokens": category_counts["hyphenated"],
            "special_financial_tokens": category_counts["special_financial"],
            "execution_time_seconds": round(self.seconds, 3),
        }


TOKENIZER_DESCRIPTIONS = {
    "nltk": "NLTK word_tokenize (Treebank word tokenizer + Punkt sentence splitter)",
    "spacy": f"spaCy en_core_web_sm tokenizer",
    "custom": "Phase 2 financial rule-based tokenizer (17 ordered rules)",
    "hybrid": "NLTK word_tokenize + domain protection/repair layer",
    "bpe": "Byte-level BPE trained on this corpus (see bpe.py)",
}


def _prepare_texts(sample: ExperimentSample) -> List[str]:
    """One identical normalised surface text per unit, shared by all tokenizers."""
    return [normalise_for_tokenisation(unit.text) for unit in sample.units]


def count_sentences(texts: Sequence[str]) -> int:
    total = 0
    for text in texts:
        stripped = text.strip()
        if not stripped:
            continue
        try:
            total += len(sent_tokenize(stripped))
        except LookupError:  # pragma: no cover - punkt ships with the environment
            total += max(1, stripped.count(".") + stripped.count("!") + stripped.count("?"))
    return total


def run_tokenizers(
    sample: ExperimentSample,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
    include_spacy: bool = True,
) -> List[TokenizerRun]:
    """Run all four tokenizers over the shared sample and time each one."""
    texts = _prepare_texts(sample)
    documents = len({unit.document_id for unit in sample.units})
    sentence_total = count_sentences(texts)
    runs: List[TokenizerRun] = []

    def timed(name: str, fn: Callable[[], List[List[str]]]) -> None:
        start = time.perf_counter()
        tokens = fn()
        elapsed = time.perf_counter() - start
        runs.append(
            TokenizerRun(
                name=name,
                tokens_per_unit=tokens,
                sentences=sentence_total,
                documents_processed=documents,
                seconds=elapsed,
                description=TOKENIZER_DESCRIPTIONS[name],
            )
        )
        if logger is not None:
            log_event(
                logger,
                "INFO",
                "tokenization",
                f"{name}: {sum(len(t) for t in tokens)} tokens over {len(tokens)} units in {elapsed:.2f}s",
            )

    timed("nltk", lambda: [nltk_tokenize(t) for t in texts])
    timed("custom", lambda: custom_tokenize_batch(texts))

    if include_spacy:
        nlp = load_spacy(str(config.get("language.spacy_model", "en_core_web_sm")), logger)
        batch = int(config.get("tokenization.spacy.batch_size", 128))
        timed("spacy", lambda: spacy_tokenize_batch(texts, nlp, batch))
    else:  # pragma: no cover - only used when spaCy is unavailable
        if logger is not None:
            log_event(logger, "WARNING", "tokenization", "spaCy unavailable; skipped (no substitute loaded)")

    timed("hybrid", lambda: [hybrid_tokenize(t) for t in texts])
    return runs


# ----------------------------------------------------------------------
# Example extraction (traceable evidence for the report)
# ----------------------------------------------------------------------
def example_rows(
    run: TokenizerRun,
    sample: ExperimentSample,
    limit: int,
    min_chars: int = 40,
    max_chars: int = 400,
) -> List[Dict[str, object]]:
    """Median-length examples for one tokenizer, fully traceable."""
    candidates: List[Tuple[int, int, Unit, List[str]]] = []
    for unit, tokens in zip(sample.units, run.tokens_per_unit):
        length = len(unit.text)
        if min_chars <= length <= max_chars and 8 <= len(tokens) <= 60:
            candidates.append((length, unit.unit_index, unit, tokens))
    candidates.sort(key=lambda row: (row[0], row[2].unit_id))
    if not candidates:
        return []
    step = max(1, len(candidates) // max(1, limit))
    picked = candidates[::step][:limit]
    rows: List[Dict[str, object]] = []
    for _length, _index, unit, tokens in picked:
        row: Dict[str, object] = dict(unit.to_trace())
        row.update(
            {
                "tokenizer": run.name,
                "original_text": unit.text.replace("\n", " | "),
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )
        rows.append(row)
    return rows


def domain_expression_examples(sample: ExperimentSample, limit: int = 12) -> List[Dict[str, object]]:
    """Pick real corpus sentences that contain a financial/date/number expression."""
    wanted = (
        "percentage",
        "currency_amount",
        "fiscal_year_range",
        "fiscal_year",
        "quarter",
        "day_month_year",
        "month_year",
        "scaled_magnitude",
    )
    buckets: Dict[str, List[Tuple[int, Unit, List[str]]]] = {name: [] for name in wanted}
    for unit in sample.units:
        if not (60 <= len(unit.text) <= 400):
            continue
        expressions = date_number_tokenize(unit.text).expressions
        for name in wanted:
            if any(expr["category"] == name for expr in expressions) and len(buckets[name]) < 40:
                buckets[name].append((len(unit.text), unit, [e["text"] for e in expressions]))

    rows: List[Dict[str, object]] = []
    for name in wanted:
        pool = sorted(buckets[name], key=lambda row: (row[0], row[1].unit_id))
        if not pool:
            continue
        length, unit, exprs = pool[len(pool) // 2]
        row = dict(unit.to_trace())
        row.update(
            {
                "example_category": name,
                "original_text": unit.text.replace("\n", " | "),
                "detected_expressions": sorted(set(exprs))[:10],
            }
        )
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


# ----------------------------------------------------------------------
# Date / number comparison
# ----------------------------------------------------------------------
def date_number_comparison(sample: ExperimentSample, logger: Optional[logging.Logger] = None) -> List[Dict[str, object]]:
    """Standard tokenization vs the typed date/number tokenizer.

    Measures how many domain expressions survive as *single tokens* (good for
    indexing) and how many typed components are available (good for numeric
    analysis). Every count is derived from the corpus, not assumed.
    """
    texts = _prepare_texts(sample)
    standard_tokens: List[str] = []
    standard_expression_survival: Dict[str, int] = {}
    typed_counts: Dict[str, int] = {}
    expression_totals: Dict[str, int] = {}
    whole_expression_tokens: Dict[str, int] = {}

    date_names = {
        "fiscal_year_range",
        "fiscal_year",
        "year_range",
        "quarter",
        "day_month_year",
        "month_year",
        "day_month",
        "slash_date",
        "year",
    }
    number_names = {"percentage", "currency_amount", "scaled_magnitude", "grouped_number", "decimal_number", "integer"}

    for unit, text in zip(sample.units, texts):
        std = nltk_tokenize(text)
        standard_tokens.extend(std)
        std_set = set(std)
        typed = date_number_tokenize(text)
        seen: set = set()
        for expr in typed.expressions:
            category = expr["category"]
            expression_totals[category] = expression_totals.get(category, 0) + 1
            if expr["text"] in std_set:
                seen.add(category)
        for category in seen:
            standard_expression_survival[category] = standard_expression_survival.get(category, 0) + 1

        for typed_token in typed.typed_tokens:
            if typed_token.category in ("word", "punct"):
                continue
            typed_counts[typed_token.category] = typed_counts.get(typed_token.category, 0) + 1
            if typed_token.token == typed_token.expression.strip():
                whole_expression_tokens[typed_token.category] = (
                    whole_expression_tokens.get(typed_token.category, 0) + 1
                )

    rows: List[Dict[str, object]] = []
    for category in sorted(set(expression_totals) | set(standard_expression_survival)):
        is_date = category in date_names
        rows.append(
            {
                "category": category,
                "class": "date" if is_date else ("number" if category in number_names else "other"),
                "expressions_detected": expression_totals.get(category, 0),
                "expressions_intact_after_standard_tokenization": standard_expression_survival.get(category, 0),
                "intact_rate_percent": round(
                    100.0 * standard_expression_survival.get(category, 0) / expression_totals[category], 2
                )
                if expression_totals.get(category)
                else 0.0,
                "typed_component_tokens": typed_counts.get(category, 0),
                "tokens_equal_to_whole_expression": whole_expression_tokens.get(category, 0),
            }
        )

    totals = {
        "category": "TOTAL",
        "class": "summary",
        "expressions_detected": sum(expression_totals.values()),
        "expressions_intact_after_standard_tokenization": sum(standard_expression_survival.values()),
        "intact_rate_percent": round(
            100.0 * sum(standard_expression_survival.values()) / max(1, sum(expression_totals.values())), 2
        ),
        "typed_component_tokens": sum(typed_counts.values()),
        "tokens_equal_to_whole_expression": sum(whole_expression_tokens.values()),
    }
    rows.append(totals)
    if logger is not None:
        log_event(
            logger,
            "INFO",
            "date_number",
            f"date/number comparison: {totals['expressions_detected']} typed expressions, "
            f"{totals['expressions_intact_after_standard_tokenization']} survive standard tokenization "
            f"({totals['intact_rate_percent']}%)",
        )
    return rows
