"""Stopword strategies: none / standard English / domain-aware.

Why stopword removal is dangerous in this corpus
------------------------------------------------
A standard English stopword list removes *modality and domain-noun* words that
carry retrieval signal in economic writing::

    "The rate may increase if growth is weak and banks remain cautious."
      -> standard removal: ['increase', 'growth', 'weak', 'banks', 'cautious']
                           ('may' is removed, and so is nothing else here, but in
                            a query like "how will inflation affect bank lending"
                            'will' and 'affect' carry the temporal/causal frame)

The words at risk are exactly the ones listed in
``config.phase2_config.yaml: stopwords.protected_financial_terms``. This module:

1. builds three token views of the same corpus,
2. measures the actual reduction for each,
3. and reports, from the corpus itself, which protected financial terms a
   standard list would delete and how often they occur.

No claim about a term's importance is asserted without a corpus count.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import Phase2Config, log_event
from .load_corpus import ExperimentSample
from .statistics import TokenizedCorpus, build_tokenized, percent_reduction, top_terms

_STOPWORD_SOURCE: Dict[str, object] = {}


def _load_stopwords(config: Phase2Config, logger: Optional[logging.Logger] = None) -> Set[str]:
    """Resolve a *real* standard English stopword list, recording which source.

    Priority: ``nltk.corpus.stopwords`` -> ``spacy.lang.en.stop_words`` ->
    ``sklearn.feature_extraction.text.ENGLISH_STOP_WORDS``. Which one was used is
    written into the artefacts so the experiment is reproducible.
    """
    if _STOPWORD_SOURCE.get("words"):
        return set(_STOPWORD_SOURCE["words"])  # type: ignore[arg-type]

    words: Set[str] = set()
    source = ""
    try:  # 1. NLTK (preferred: it is the list the assignment names)
        from nltk.corpus import stopwords as nltk_stopwords

        words = {w.casefold() for w in nltk_stopwords.words("english")}
        source = "nltk.corpus.stopwords:english"
    except Exception as exc:  # 2. spaCy's published English list
        try:
            from spacy.lang.en.stop_words import STOP_WORDS

            words = {w.casefold() for w in STOP_WORDS}
            source = "spacy.lang.en.stop_words.STOP_WORDS"
            _STOPWORD_SOURCE["nltk_error"] = str(exc)[:200]
        except Exception:  # 3. scikit-learn's list
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            words = {w.casefold() for w in ENGLISH_STOP_WORDS}
            source = "sklearn.feature_extraction.text.ENGLISH_STOP_WORDS"

    _STOPWORD_SOURCE["words"] = words
    _STOPWORD_SOURCE["source"] = source
    _STOPWORD_SOURCE["size"] = len(words)
    if logger is not None:
        log_event(logger, "INFO", "stopwords", f"standard stopword list source: {source} ({len(words)} terms)")
    return words


def stopword_source() -> Dict[str, object]:
    return {k: v for k, v in _STOPWORD_SOURCE.items() if k != "words"}


@dataclass
class StopwordStrategy:
    name: str
    words: Set[str]
    description: str
    protected: Set[str]


def build_strategies(config: Phase2Config, logger: Optional[logging.Logger] = None) -> Dict[str, StopwordStrategy]:
    standard = _load_stopwords(config, logger)
    protected = {w.casefold() for w in config.get("stopwords.protected_financial_terms", []) or []}
    extra = {w.casefold() for w in config.get("stopwords.domain_function_words", []) or []}

    domain = (standard - protected) | extra
    return {
        "none": StopwordStrategy("none", set(), "no removal (baseline)", set()),
        "standard_english": StopwordStrategy(
            "standard_english",
            standard,
            "standard English stopword list applied unchanged",
            set(),
        ),
        "domain_aware": StopwordStrategy(
            "domain_aware",
            domain,
            "standard list minus protected financial terms, plus domain function words",
            protected,
        ),
    }


def apply_strategy(tokenized: TokenizedCorpus, strategy: StopwordStrategy) -> TokenizedCorpus:
    if not strategy.words:
        return tokenized
    kept: List[List[str]] = []
    for unit_tokens in tokenized.tokens:
        kept.append([t for t in unit_tokens if t.casefold() not in strategy.words])
    return TokenizedCorpus(name=f"{tokenized.name}__{strategy.name}", units=tokenized.units, tokens=kept)


def removed_terms(tokenized: TokenizedCorpus, strategy: StopwordStrategy) -> Counter:
    counter: Counter = Counter()
    for unit_tokens in tokenized.tokens:
        for token in unit_tokens:
            if token.casefold() in strategy.words:
                counter[token.casefold()] += 1
    return counter


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def run_stopword_experiments(
    tokenized: TokenizedCorpus,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Produce stopword_comparison.csv rows, domain analysis rows and the final list."""
    strategies = build_strategies(config, logger)
    top_n = int(config.get("stopwords.top_terms_to_report", 25))
    baseline = tokenized
    baseline_freq = baseline.freq()

    comparison_rows: List[Dict[str, object]] = []
    analysis_rows: List[Dict[str, object]] = []
    final_words: Set[str] = set()
    views: Dict[str, TokenizedCorpus] = {"none": baseline}

    for name, strategy in strategies.items():
        view = apply_strategy(tokenized, strategy)
        views[name] = view
        removed = removed_terms(tokenized, strategy)
        removed_total = sum(removed.values())
        kept_freq = view.freq()
        top_kept = top_terms(kept_freq, top_n)
        comparison_rows.append(
            {
                "strategy": name,
                "description": strategy.description,
                "stopword_list_size": len(strategy.words),
                "stopword_list_source": stopword_source().get("source", "none (no removal)"),
                "total_tokens": view.total_tokens,
                "tokens_removed": removed_total,
                "unique_tokens": view.unique_tokens,
                "vocabulary_size": view.vocabulary_size,
                "token_reduction_percent": percent_reduction(baseline.total_tokens, view.total_tokens),
                "vocabulary_reduction_percent": percent_reduction(baseline.vocabulary_size, view.vocabulary_size),
                "avg_tokens_per_document": view.avg_tokens_per_document(),
                "top_remaining_terms": [f"{term}:{count}" for term, count in top_kept],
            }
        )

    # ---- domain-specific analysis, driven by real corpus counts ----
    standard = strategies["standard_english"]
    domain = strategies["domain_aware"]
    standard_removed = removed_terms(tokenized, standard)
    domain_removed = removed_terms(tokenized, domain)
    protected = sorted(domain.protected)

    for term in protected:
        occurrences = baseline_freq.get(term, 0)
        if occurrences == 0:
            continue
        removed_by_standard = standard_removed.get(term, 0)
        removed_by_domain = domain_removed.get(term, 0)
        analysis_rows.append(
            {
                "term": term,
                "in_standard_stopword_list": term in standard.words,
                "in_domain_aware_stopword_list": term in domain.words,
                "corpus_occurrences": occurrences,
                "occurrences_removed_by_standard": removed_by_standard,
                "occurrences_removed_by_domain_aware": removed_by_domain,
                "documents_affected": tokenized.doc_freq().get(term, 0),
                "outcome": (
                    "removed_by_standard_only"
                    if removed_by_standard and not removed_by_domain
                    else "removed_by_both"
                    if removed_by_domain
                    else "preserved"
                ),
                "rationale": (
                    "protected: expresses modality/time/causality or is a financial "
                    "domain noun used as a query term"
                    if term in domain.protected
                    else "domain function word: safe to drop for this domain"
                ),
            }
        )

    final_words = domain.words

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "stopwords",
            "standard list removed "
            f"{sum(standard_removed.values())} tokens; domain-aware removed {sum(domain_removed.values())} "
            f"({len(analysis_rows)} protected financial terms checked against corpus counts)",
        )

    return {
        "comparison": comparison_rows,
        "domain_analysis": analysis_rows,
        "final_stopwords": sorted(final_words),
        "views": views,
        "strategies": strategies,
    }


def stopword_intersection_report(views: Dict[str, TokenizedCorpus]) -> List[Dict[str, object]]:
    """Terms that survive the standard list but not the domain-aware list."""
    standard_view = views["standard_english"]
    domain_view = views["domain_aware"]
    standard_vocab = {t for t in standard_view.lower_flat if any(c.isalpha() for c in t)}
    domain_vocab = {t for t in domain_view.lower_flat if any(c.isalpha() for c in t)}
    lost = sorted(standard_vocab - domain_vocab)
    return [{"term": term} for term in lost]
