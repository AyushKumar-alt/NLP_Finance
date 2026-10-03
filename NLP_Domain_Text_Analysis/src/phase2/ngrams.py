"""N-gram analysis (n = 1..5) with document frequency and phrase mining.

The n-gram stream is built from the **hybrid tokenizer** (configurable) so that
financial expressions such as ``FY2025-26`` and ``7.4%`` stay intact, which
changes the resulting phrases substantially compared with a plain word tokenizer.
The comparison with the stopword-removed stream is reported as well, because it
is the single biggest lever on n-gram quality.

Domain phrases are *mined*, not hard-coded: a candidate phrase must clear a
frequency threshold, a document-frequency threshold, and a function-word filter,
and its domain relevance is justified by the domain vocabulary it contains.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .statistics import TokenizedCorpus

#: Function words that dominate grammatical (non-informative) bigrams/trigrams.
FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
        "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "that", "this", "these", "those",
        "it", "its", "which", "who", "whom", "whose", "than", "then", "so", "such",
        "not", "no", "nor", "also", "can", "could", "may", "might", "will", "would",
        "shall", "should", "must", "there", "here", "when", "while", "where", "how",
        "all", "any", "each", "more", "most", "other", "some", "such", "both", "few",
        "into", "over", "under", "about", "between", "through", "during", "after",
        "before", "above", "below", "up", "down", "out", "off", "again", "further",
        "one", "two", "us", "we", "you", "they", "he", "she", "his", "her", "their",
    }
)

#: Domain vocabulary used only to *justify* a mined phrase as domain-relevant.
DOMAIN_VOCABULARY = frozenset(
    {
        "gdp", "gva", "cpi", "wpi", "inflation", "deflation", "disinflation", "fiscal",
        "monetary", "policy", "repo", "rate", "rates", "repo-rate", "reverse", "msf",
        "crr", "slr", "bank", "banks", "banking", "credit", "deposit", "deposits",
        "loan", "loans", "lending", "lender", "lenders", "borrowers", "npa", "gnpa",
        "equity", "equities", "market", "markets", "capital", "investment", "investments",
        "investor", "investors", "finance", "financial", "fiscal", "budget", "tax",
        "taxes", "gst", "trade", "exports", "imports", "export", "import", "current",
        "account", "deficit", "surplus", "balance", "payments", "reserve", "foreign",
        "fdi", "fpi", "rupee", "currency", "exchange", "rate", "yields", "yield",
        "bond", "bonds", "securities", "equity", "commodities", "oil", "energy",
        "employment", "unemployment", "wage", "wages", "income", "consumption", "savings",
        "saving", "savings", "growth", "demand", "supply", "sector", "sectors", "industry",
        "industries", "output", "productivity", "employment", "poverty", "employment",
        "expenditure", "revenue", "profit", "profits", "loss", "losses", "returns",
        "securitisation", "disinvestment", "regulation", "regulatory", "supervision",
        "liquidity", "money", "debt", "liabilities", "assets", "stable", "stability",
        "financialisation", "macroeconomic", "microeconomic", "monetary", "fiscal",
    }
)


def ngram_stream(tokenized: TokenizedCorpus, n: int, lowercase: bool = True) -> List[Tuple[str, ...]]:
    """All n-grams of the sample, in document order."""
    stream: List[Tuple[str, ...]] = []
    for unit_tokens in tokenized.tokens:
        tokens = [t.casefold() for t in unit_tokens] if lowercase else list(unit_tokens)
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            stream.append(tuple(tokens[index: index + n]))
    return stream


def ngram_counts(
    tokenized: TokenizedCorpus,
    n: int,
    lowercase: bool = True,
) -> Tuple[Counter, Counter, Dict[Tuple[str, ...], Tuple[str, int, str]]]:
    """Frequency, document frequency and first occurrence of every n-gram."""
    frequencies: Counter = Counter()
    document_frequency: Counter = Counter()
    first_seen: Dict[Tuple[str, ...], Tuple[str, int, str]] = {}

    for unit, unit_tokens in zip(tokenized.units, tokenized.tokens):
        tokens = [t.casefold() for t in unit_tokens] if lowercase else list(unit_tokens)
        if len(tokens) < n:
            continue
        unit_grams = {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i: i + n])
            frequencies[gram] += 1
            if gram not in first_seen:
                first_seen[gram] = (unit.document_id, unit.page_number, unit.unit_id)
        document_frequency.update(unit_grams)
    return frequencies, document_frequency, first_seen


def ngram_rows(
    tokenized: TokenizedCorpus,
    n: int,
    config: Phase2Config,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Rows for ``<n>gram_results.csv`` plus a summary dictionary for the charts."""
    lowercase = bool(config.get("ngrams.lowercase", True))
    write_top = int(config.get("ngrams.full_vocabulary_written_top", 5000))
    limit = write_top if limit is None else limit

    frequencies, document_frequency, first_seen = ngram_counts(tokenized, n, lowercase)
    total = sum(frequencies.values())

    rows: List[Dict[str, object]] = []
    for gram, frequency in frequencies.most_common(limit):
        document_id, page, unit_id = first_seen[gram]
        rows.append(
            {
                "ngram": " ".join(gram),
                "n": n,
                "frequency": frequency,
                "document_frequency": document_frequency.get(gram, 0),
                "example_document": document_id,
                "example_page": page,
                "example_unit_id": unit_id,
                "contains_domain_term": _domain_terms(gram),
                "is_function_word_phrase": _function_ratio(gram) >= float(
                    config.get("ngrams.phrase_filter_blocklist_ratio", 0.85)
                ),
                "has_content_token": any(_has_word_character(token) for token in gram),
                "is_repeated_token": len(gram) > 1 and len(set(gram)) == 1,
            }
        )

    content = [gram for gram in frequencies if any(_has_word_character(t) for t in gram)]
    top_content = content[0] if content else None
    summary = {
        "n": n,
        "total_ngrams": total,
        "unique_ngrams": len(frequencies),
        "top_ngram": " ".join(frequencies.most_common(1)[0][0]) if frequencies else "",
        "top_ngram_frequency": frequencies.most_common(1)[0][1] if frequencies else 0,
        "top_content_ngram": " ".join(top_content) if top_content else "",
        "top_content_ngram_frequency": frequencies[top_content] if top_content else 0,
        "singletons": sum(1 for f in frequencies.values() if f == 1),
    }
    return rows, summary


def _has_word_character(token: str) -> bool:
    """True when the token carries at least one letter or digit."""
    return any(character.isalnum() for character in token)


def _domain_terms(gram: Sequence[str]) -> str:
    return ";".join(sorted({t for t in gram if t in DOMAIN_VOCABULARY}))


def _function_ratio(gram: Sequence[str]) -> float:
    if not gram:
        return 1.0
    return sum(1 for token in gram if token in FUNCTION_WORDS) / len(gram)


# ----------------------------------------------------------------------
# Domain phrase mining
# ----------------------------------------------------------------------
def mine_domain_phrases(
    tokenized: TokenizedCorpus,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, object]]:
    """Mine meaningful financial phrases from the corpus statistics.

    A phrase qualifies when it

    1. reaches ``domain_phrase_min_frequency`` occurrences,
    2. appears in at least ``domain_phrase_min_documents`` documents,
    3. is not mostly made of function words, and
    4. contains at least one term from the domain vocabulary.

    The justification column states *which* domain term caused the phrase to be
    selected, so nothing is claimed without evidence.
    """
    min_frequency = int(config.get("ngrams.domain_phrase_min_frequency", 5))
    min_documents = int(config.get("ngrams.domain_phrase_min_documents", 3))
    max_n = int(config.get("ngrams.domain_phrase_max_n", 4))
    function_threshold = float(config.get("ngrams.phrase_filter_blocklist_ratio", 0.85))
    lowercase = bool(config.get("ngrams.lowercase", True))

    rows: List[Dict[str, object]] = []
    for n in range(2, max_n + 1):
        frequencies, document_frequency, first_seen = ngram_counts(tokenized, n, lowercase)
        for gram, frequency in frequencies.items():
            if frequency < min_frequency or document_frequency.get(gram, 0) < min_documents:
                continue
            if _function_ratio(gram) >= function_threshold:
                continue
            domain_terms = _domain_terms(gram)
            if not domain_terms:
                continue
            document_id, page, unit_id = first_seen[gram]
            rows.append(
                {
                    "ngram": " ".join(gram),
                    "n": n,
                    "frequency": frequency,
                    "document_frequency": document_frequency.get(gram, 0),
                    "domain_terms_in_phrase": domain_terms,
                    "domain_relevance_reason": _relevance_reason(gram, domain_terms),
                    "function_word_ratio": round(_function_ratio(gram), 3),
                    "example_context": _context_for(gram, unit_id, document_id, page),
                    "example_document": document_id,
                    "example_page": page,
                    "example_unit_id": unit_id,
                }
            )
            if len(rows) > 4000:  # keep the file manageable
                break
    rows.sort(key=lambda row: (-int(row["frequency"]), -int(row["n"]), str(row["ngram"])))
    if logger is not None:
        log_event(
            logger,
            "INFO",
            "ngrams",
            f"mined {len(rows)} domain phrases (freq>={min_frequency}, docs>={min_documents}, n<= {max_n})",
        )
    return rows[:1200]


def _relevance_reason(gram: Sequence[str], domain_terms: str) -> str:
    parts = [f"contains domain term '{term}'" for term in domain_terms.split(";")]
    joined = " ".join(gram)
    if len(gram) == 2:
        parts.append(f"'{joined}' is a noun phrase / collocation, not a function-word pair")
    else:
        parts.append(f"'{joined}' is a fixed multi-word expression, not a grammatical template")
    return "; ".join(parts)


def _context_for(gram: Sequence[str], unit_id: str, document_id: str, page: int) -> str:
    return f"see {unit_id} ({document_id}, page {page}) - contains '{' '.join(gram)}'"


# ----------------------------------------------------------------------
# Preprocessing effect on n-grams
# ----------------------------------------------------------------------
def preprocessing_effect_rows(
    tokenized: TokenizedCorpus,
    stopword_view: TokenizedCorpus,
    config: Phase2Config,
) -> List[Dict[str, object]]:
    """How much stopword removal changes the n-gram vocabulary."""
    rows: List[Dict[str, object]] = []
    for n in range(1, int(max(config.get("ngrams.sizes", [1, 2, 3, 4, 5]))) + 1):
        raw_freq, _raw_df, _raw_first = ngram_counts(tokenized, n)
        stop_freq, stop_df, _stop_first = ngram_counts(stopword_view, n)
        raw_multi = {g: f for g, f in raw_freq.items() if len(g) > 1}
        stop_multi = {g: f for g, f in stop_freq.items() if len(g) > 1}
        shared = set(raw_multi) & set(stop_multi)
        rows.append(
            {
                "n": n,
                "raw_unique_ngrams": len(raw_freq),
                "stopword_removed_unique_ngrams": len(stop_freq),
                "unique_ngrams_after_removal": len(stop_multi),
                "multiword_phrases_raw": len(raw_multi),
                "multiword_phrases_after_removal": len(stop_multi),
                "shared_multiword_phrases": len(shared),
                "phrases_lost_after_stopword_removal": len(set(raw_multi) - set(stop_multi)),
                "phrases_introduced_after_stopword_removal": len(set(stop_multi) - set(raw_multi)),
                "examples_lost": [" ".join(g) for g in sorted(set(raw_multi) - set(stop_multi))[:8]],
                "examples_introduced": [" ".join(g) for g in sorted(set(stop_multi) - set(raw_multi))[:8]],
            }
        )
    return rows
