"""Build an :class:`~src.phase3.inverted_index.InvertedIndex` from a pipeline run.

The builder owns three decisions that must be identical for Pipeline A and
Pipeline B so that the comparison is fair:

* terms are the pipeline's tokens, case folded and whitespace normalised;
* only tokens that carry a word character are indexed (bare numbers and
  punctuation would flood the index, exactly as Phase 2 found when measuring
  vocabulary size), *except* financial expressions such as ``7.4%`` and
  ``FY2025-26`` which do carry word or numeric signal and are protected;
* postings store token positions so phrase search is exact.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.phase2.load_corpus import Corpus, Unit

from .config import Phase3Config, log_event
from .inverted_index import DocumentRecord, InvertedIndex, UnitRecord
from .pipeline_runner import PipelineResult, PipelineRunner
from .pipelines import PipelineSpec

#: A token is indexable when it contains a word character or is a protected
#: financial expression (numbers, percentages, fiscal years, currencies).
_WORD = re.compile(r"[^\W\d_]", re.UNICODE)
_DIGIT = re.compile(r"\d", re.UNICODE)
#: Financial shapes that must stay retrievable even without letters.
_FINANCIAL_SHAPE = re.compile(
    r"^(?=.*\d)(?:%|per\s*cent|percent|fy\d|fiscal|\d{1,2}\s+[a-z]{3,}|\d)",
    re.IGNORECASE,
)


def is_indexable(term: str, protect_financial: bool = True) -> bool:
    """Whether a normalized token becomes an index term."""
    if not term:
        return False
    if _WORD.search(term):
        return True
    if protect_financial and _DIGIT.search(term) and _FINANCIAL_SHAPE.search(term):
        return True
    return False


@dataclass
class BuildResult:
    """Index plus the measurements the pipeline comparison needs."""

    index: InvertedIndex
    spec: PipelineSpec
    build_seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)
    unit_terms: Dict[str, List[str]] = field(default_factory=dict)
    ngram_terms: Set[str] = field(default_factory=set)


class IndexBuilder:
    """Turns a :class:`PipelineResult` into an inverted index."""

    def __init__(self, config: Phase3Config, runner: PipelineRunner,
                 logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.runner = runner
        self.logger = logger
        self.protect_financial = bool(config.get("ngrams.filter_all_punctuation", True))
        self.join_with = str(config.get("ngrams.join_with", "_"))
        self.min_frequency = int(config.get("ngrams.min_frequency", 5))
        self.max_phrases = int(config.get("ngrams.max_indexed_phrases", 20000))

    # ------------------------------------------------------------------
    def build(
        self,
        result: PipelineResult,
        corpus: Corpus,
        entities_by_unit: Optional[Dict[str, List[Dict[str, str]]]] = None,
        domain_entities_by_unit: Optional[Dict[str, List[Dict[str, str]]]] = None,
        pipeline_metadata: Optional[Dict[str, Any]] = None,
    ) -> BuildResult:
        """Build the index for one pipeline run."""
        started = time.perf_counter()
        spec = result.spec
        index = InvertedIndex(metadata={"pipeline": spec.key, "pipeline_name": spec.name,
                                        "index_representation": spec.index_representation})
        entities_by_unit = entities_by_unit or {}
        domain_entities_by_unit = domain_entities_by_unit or {}

        # ---- provenance stores ----
        for document in corpus.documents:
            index.add_document(
                DocumentRecord(
                    document_id=document.document_id,
                    source_id=document.source_id,
                    filename=document.filename,
                    title=document.title,
                    publisher=document.publisher,
                    year=document.year,
                    document_type=document.document_type,
                )
            )

        unit_terms: Dict[str, List[str]] = {}
        term_counter: Counter = Counter()
        skipped_empty = 0

        for unit, tokens in zip(result.view.units, result.view.tokens):
            index.add_unit(
                UnitRecord(
                    unit_id=unit.unit_id,
                    document_id=unit.document_id,
                    source_id=unit.source_id,
                    page_number=unit.page_number,
                    section_id=unit.section_id,
                    section_number=unit.section_number,
                    section_title=unit.section_title,
                    unit_type=unit.unit_type,
                    unit_index=unit.unit_index,
                    char_count=len(unit.text),
                )
            )
            terms = [self.runner.term_for(spec, token) for token in tokens]
            unit_terms[unit.unit_id] = terms

            positions: Dict[str, List[int]] = {}
            for position, term in enumerate(terms):
                if not is_indexable(term, self.protect_financial):
                    continue
                positions.setdefault(term, []).append(position)
            for term, term_positions in positions.items():
                index.add_posting(term, unit.unit_id, tf=len(term_positions), positions=term_positions)
                term_counter[term] += len(term_positions)
            if not terms:
                skipped_empty += 1

        # ---- n-gram phrase terms (document frequency filtered) ----
        phrase_counter: Counter = Counter()
        phrase_positions: Dict[str, Dict[str, List[int]]] = {}
        unit_document_pairs: Dict[str, Set[str]] = {}
        sizes = [int(n) for n in (self.config.get("ngrams.sizes", [2, 3]) or [2, 3])]
        sizes = [n for n in sizes if 1 < n <= spec.ngram_max]
        for unit_id, terms in unit_terms.items():
            document_id = index.units[unit_id].document_id
            for n in sizes:
                for start in range(0, max(0, len(terms) - n + 1)):
                    window = terms[start : start + n]
                    if not all(is_indexable(term, self.protect_financial) for term in window):
                        continue
                    phrase = self.join_with.join(window)
                    phrase_counter[phrase] += 1
                    phrase_positions.setdefault(phrase, {}).setdefault(unit_id, []).append(start)
                    unit_document_pairs.setdefault(phrase, set()).add(document_id)

        kept_phrases = [
            phrase for phrase, count in phrase_counter.items()
            if count >= self.min_frequency and len(unit_document_pairs.get(phrase, ())) >= 2
        ]
        kept_phrases.sort(key=lambda phrase: (-phrase_counter[phrase], phrase))
        dropped_by_cap = 0
        if len(kept_phrases) > self.max_phrases:
            dropped_by_cap = len(kept_phrases) - self.max_phrases
            kept_phrases = kept_phrases[: self.max_phrases]

        ngram_terms: Set[str] = set()
        for phrase in kept_phrases:
            for unit_id, starts in phrase_positions.get(phrase, {}).items():
                starts = sorted(starts)
                index.add_posting(phrase, unit_id, tf=len(starts), positions=starts, is_phrase=True)
                term_counter[phrase] += len(starts)
            ngram_terms.add(phrase)

        index.finalise()
        elapsed = time.perf_counter() - started

        document_count = index.document_count()
        stats: Dict[str, Any] = {
            "index_terms": index.term_count,
            "unigram_terms": index.term_count - len(ngram_terms),
            "phrase_terms": len(ngram_terms),
            "indexed_units": index.unit_count,
            "documents": document_count,
            "total_postings": index.posting_count,
            "unigram_postings": index.posting_count - sum(
                len(index.terms[phrase].postings) for phrase in ngram_terms
            ),
            "posting_count": index.posting_count,
            "mean_postings_per_term": round(index.posting_count / index.term_count, 4)
            if index.term_count else 0.0,
            "mean_units_indexed": round(index.unit_count / document_count, 2) if document_count else 0.0,
            "meaningful_ngram_count": len(ngram_terms),
            "candidate_ngrams_before_filter": len(phrase_counter),
            "phrases_dropped_by_cap": dropped_by_cap,
            "units_without_tokens": skipped_empty,
            "index_build_seconds": round(elapsed, 3),
            "total_postings_unigram": index.posting_count - sum(
                len(index.terms[phrase].postings) for phrase in ngram_terms
            ),
            "entity_units_from_phase2": len(entities_by_unit),
            "domain_entity_units_from_phase2": len(domain_entities_by_unit),
        }
        if pipeline_metadata:
            index.metadata.update(pipeline_metadata)

        if self.logger is not None:
            log_event(
                self.logger,
                "INFO",
                spec.key,
                f"  index built in {elapsed:.2f}s: {index.term_count} terms "
                f"({len(ngram_terms)} phrases), {index.posting_count} postings",
            )

        return BuildResult(
            index=index,
            spec=spec,
            build_seconds=elapsed,
            stats=stats,
            unit_terms=unit_terms,
            ngram_terms=ngram_terms,
        )
