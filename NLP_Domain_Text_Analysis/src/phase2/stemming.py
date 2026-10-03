"""Stemming experiments: Porter, Snowball, Lancaster, and pipeline ordering.

Observed behaviour only. For every algorithm this module reports vocabulary
reduction plus the concrete financial/economic word families whose members
collapse onto one stem, so the report can *show* over-stemming instead of
asserting that it happens.

Also implements the ordering experiment:

    Pipeline A: tokenize -> stopword removal -> stemming
    Pipeline B: tokenize -> stemming       -> stopword removal

The order matters because the two filters do not commute: a stopword list
contains surface forms (``rates``, ``policies``), while a stemmer maps forms to
roots (``rate``, ``polici``). Removing stopwords first can delete a surface form
that the stemmer would have collapsed, and stemming first can map a protected
stopword onto a stem that is not in the stopword list, so it survives.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

from .config import Phase2Config, log_event
from .statistics import TokenizedCorpus, percent_reduction, top_terms

STEMMER_FACTORIES: Dict[str, Callable[[], object]] = {
    "porter": lambda: PorterStemmer(),
    "snowball_english": lambda: SnowballStemmer("english"),
    "lancaster": lambda: LancasterStemmer(),
    "snowball_porter2": lambda: SnowballStemmer("porter"),
}

STEMMER_DESCRIPTIONS = {
    "porter": "Porter (1980) - original Porter stemmer, the classic baseline",
    "snowball_english": "Snowball 'english' (Porter2) - improved Porter algorithm",
    "lancaster": "Lancaster (Paice/Husk) - aggressive, most aggressive of the three",
    "snowball_porter2": "Snowball 'porter' - Porter as implemented inside Snowball",
}


def get_stemmer(name: str):
    if name not in STEMMER_FACTORIES:
        raise KeyError(f"unknown stemmer '{name}'; available: {sorted(STEMMER_FACTORIES)}")
    return STEMMER_FACTORIES[name]()


def stem_view(tokenized: TokenizedCorpus, algorithm: str) -> TokenizedCorpus:
    """Stem an arbitrary token view (used by the pipeline-order experiments)."""
    stemmer = get_stemmer(algorithm)
    cache: Dict[str, str] = {}

    def transform(token: str) -> str:
        if not any(ch.isalpha() for ch in token):
            return token
        key = token.casefold()
        if key not in cache:
            try:
                cache[key] = stemmer.stem(key)
            except Exception:
                cache[key] = key
        return cache[key]

    return _apply(tokenized, transform, f"stem_{algorithm}")


def _apply(corpus: TokenizedCorpus, transform: Callable[[str], str], name: str) -> TokenizedCorpus:
    return TokenizedCorpus(
        name=name,
        units=corpus.units,
        tokens=[[transform(token) for token in unit_tokens] for unit_tokens in corpus.tokens],
    )


# ----------------------------------------------------------------------
# Word families: evidence of over/under-stemming
# ----------------------------------------------------------------------
def word_families(words: Sequence[str], stemmer_name: str) -> List[Dict[str, object]]:
    """Group probe words by the stem they map to.

    A stem produced by *more than one* probe word is a collision (evidence of
    over-stemming); a probe word that keeps a full suffix such as ``-ing`` or
    ``-ation`` after stemming is evidence of under-stemming for that family.
    """
    stemmer = get_stemmer(stemmer_name)
    mapping: Dict[str, str] = {word: stemmer.stem(word) for word in words}
    by_stem: Dict[str, List[str]] = {}
    for word, stem in mapping.items():
        by_stem.setdefault(stem, []).append(word)
    rows: List[Dict[str, object]] = []
    for word in words:
        stem = mapping[word]
        collision = sorted(by_stem[stem])
        rows.append(
            {
                "original_word": word,
                "stemmer": stemmer_name,
                "stem": stem,
                "changed": stem != word,
                "collision_group": ";".join(collision) if len(collision) > 1 else "",
                "collision_size": len(collision),
                "residual_suffixes": _residual_suffixes(word, stem),
                "observation": _observe(word, stem, collision),
            }
        )
    return rows


def _residual_suffixes(word: str, stem: str) -> str:
    if stem == word or not stem:
        return ""
    index = word.find(stem)
    if index < 0:
        return ""
    return word[index + len(stem) :]


def _observe(word: str, stem: str, collision: List[str]) -> str:
    if stem == word:
        return "unchanged (under-stemmed: inflection left in place)"
    if len(collision) > 1 and not all(w.startswith(stem) for w in collision):
        return f"over-stemmed: collides {sorted(collision)} onto '{stem}'"
    suffix = _residual_suffixes(word, stem)
    if suffix:
        return f"partially stemmed, residual '{suffix}'"
    return "clean reduction"


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def run_stemming(
    tokenized: TokenizedCorpus,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    algorithms = list(config.get("stemming.algorithms", ["porter", "snowball_english", "lancaster"]))
    probe_words = list(config.get("stemming.probe_words", []) or [])
    baseline_vocab = tokenized.vocabulary_size
    baseline_tokens = tokenized.total_tokens
    baseline_unique = tokenized.unique_tokens

    comparison_rows: List[Dict[str, object]] = []
    example_rows: List[Dict[str, object]] = []
    views: Dict[str, TokenizedCorpus] = {}
    family_rows: List[Dict[str, object]] = []

    for algorithm in algorithms:
        stemmer = get_stemmer(algorithm)
        # Word-like tokens only: stemming punctuation or a pure number is
        # meaningless and would pollute the vocabulary statistics.
        cache: Dict[str, str] = {}

        def transform(token: str, _stemmer=stemmer, _cache=cache) -> str:
            if not any(ch.isalpha() for ch in token):
                return token
            key = token.casefold()
            if key not in _cache:
                try:
                    _cache[key] = _stemmer.stem(key)
                except Exception:  # a stemmer must never abort a corpus run
                    _cache[key] = key
            return _cache[key]

        start = time.perf_counter()
        stemmed = _apply(tokenized, transform, f"stem_{algorithm}")
        elapsed = time.perf_counter() - start
        views[algorithm] = stemmed

        collisions = [
            stem
            for stem, count in Counter(
                t for t in stemmed.lower_flat if any(c.isalpha() for c in t)
            ).items()
            if count > 1
        ]
        comparison_rows.append(
            {
                "algorithm": algorithm,
                "description": STEMMER_DESCRIPTIONS.get(algorithm, algorithm),
                "total_tokens_before": baseline_tokens,
                "unique_tokens_before": baseline_unique,
                "vocabulary_before": baseline_vocab,
                "total_tokens_after": stemmed.total_tokens,
                "unique_tokens_after": stemmed.unique_tokens,
                "unique_stems": stemmed.vocabulary_size,
                "vocabulary_reduction": baseline_vocab - stemmed.vocabulary_size,
                "vocabulary_reduction_percent": percent_reduction(baseline_vocab, stemmed.vocabulary_size),
                "unique_token_reduction_percent": percent_reduction(baseline_unique, stemmed.unique_tokens),
                "avg_tokens_per_document": stemmed.avg_tokens_per_document(),
                "type_token_ratio": stemmed.type_token_ratio(),
                "distinct_stems": len(set(stemmed.lower_flat)),
                "colliding_stems": len(collisions),
                "execution_time_seconds": round(elapsed, 3),
            }
        )

        for row in word_families(probe_words, algorithm):
            family_rows.append(row)

        if logger is not None:
            log_event(
                logger,
                "INFO",
                "stemming",
                f"{algorithm}: vocabulary {baseline_vocab} -> {stemmed.vocabulary_size} "
                f"({percent_reduction(baseline_vocab, stemmed.vocabulary_size)}% reduction) in {elapsed:.2f}s",
            )

    # ---- side-by-side example table (provenance: observed corpus vocabulary) ----
    observed_vocab = sorted({t for t in tokenized.lower_flat if any(c.isalpha() for c in t)})
    example_limit = int(config.get("stemming.example_vocabulary_size", 4000))
    for word in observed_vocab[:example_limit]:
        cells = {algorithm: get_stemmer(algorithm).stem(word) for algorithm in algorithms}
        distinct = sorted(set(cells.values()))
        example_rows.append(
            {
                "original_word": word,
                "porter": cells.get("porter", ""),
                "snowball": cells.get("snowball_english", ""),
                "lancaster": cells.get("lancaster", ""),
                "algorithms_agreeing": len(distinct) == 1,
                "distinct_stems": ";".join(distinct),
                "notes": _example_note(word, cells),
            }
        )

    # ---- probe-word families (assignment's required example table) ----
    return {
        "comparison": comparison_rows,
        "examples": example_rows,
        "families": family_rows,
        "views": views,
    }


def _example_note(word: str, cells: Dict[str, str]) -> str:
    values = list(cells.values())
    if len(set(values)) == 1:
        return "all algorithms agree"
    aggressive = [name for name, value in cells.items() if value != word]
    if len(set(values)) == 1:
        return "identical reduction"
    return f"algorithms diverge ({', '.join(sorted(aggressive))} reduce this form)"


# ----------------------------------------------------------------------
# Ordering experiment
# ----------------------------------------------------------------------
def run_ordering_experiment(
    tokenized: TokenizedCorpus,
    stopwords: set,
    algorithm: str,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, object]]:
    """Compare stopword-then-stem against stem-then-stopword."""
    stemmer = get_stemmer(algorithm)
    rows: List[Dict[str, object]] = []

    def stem(token: str) -> str:
        if not any(ch.isalpha() for ch in token):
            return token
        try:
            return stemmer.stem(token.casefold())
        except Exception:
            return token.casefold()

    # Pipeline A: stopword removal -> stemming
    start_a = time.perf_counter()
    filtered_a = TokenizedCorpus(
        name="stopword_filtered",
        units=tokenized.units,
        tokens=[[t for t in unit if t.casefold() not in stopwords] for unit in tokenized.tokens],
    )
    view_a = _apply(filtered_a, stem, "stopword_then_stem")
    time_a = time.perf_counter() - start_a

    # Pipeline B: stemming -> stopword removal
    start_b = time.perf_counter()
    stemmed_all = _apply(tokenized, stem, "stemmed_all")
    view_b = TokenizedCorpus(
        name="stem_then_stopword",
        units=stemmed_all.units,
        tokens=[[t for t in unit if t.casefold() not in stopwords] for unit in stemmed_all.tokens],
    )
    time_b = time.perf_counter() - start_b

    set_a = {t for t in view_a.lower_flat}
    set_b = {t for t in view_b.lower_flat}
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    rows.append(
        {
            "pipeline": "A: stopword_removal_then_stemming",
            "stemmer": algorithm,
            "stopwords_applied": len(stopwords),
            "total_tokens_before": tokenized.total_tokens,
            "total_tokens": view_a.total_tokens,
            "tokens_removed": tokenized.total_tokens - view_a.total_tokens,
            "unique_tokens": view_a.unique_tokens,
            "vocabulary_size": view_a.vocabulary_size,
            "vocabulary_reduction_percent": percent_reduction(tokenized.vocabulary_size, view_a.vocabulary_size),
            "processing_time_seconds": round(time_a, 3),
            "tokens_only_in_this_pipeline": ";".join(only_a[:40]),
            "count_only_in_this_pipeline": len(only_a),
        }
    )
    rows.append(
        {
            "pipeline": "B: stemming_then_stopword_removal",
            "stemmer": algorithm,
            "stopwords_applied": len(stopwords),
            "total_tokens_before": tokenized.total_tokens,
            "total_tokens": view_b.total_tokens,
            "tokens_removed": tokenized.total_tokens - view_b.total_tokens,
            "unique_tokens": view_b.unique_tokens,
            "vocabulary_size": view_b.vocabulary_size,
            "vocabulary_reduction_percent": percent_reduction(tokenized.vocabulary_size, view_b.vocabulary_size),
            "processing_time_seconds": round(time_b, 3),
            "tokens_only_in_this_pipeline": ";".join(only_b[:40]),
            "count_only_in_this_pipeline": len(only_b),
        }
    )

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "stemming",
            f"order experiment: A vocab={view_a.vocabulary_size} tokens={view_a.total_tokens} | "
            f"B vocab={view_b.vocabulary_size} tokens={view_b.total_tokens}",
        )
    return rows
