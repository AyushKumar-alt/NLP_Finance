"""Phase 3 measurements.

Every number written to a Phase 3 output file is produced here and only here, so
the tables, the JSON summary and the validation report cannot disagree.
"""

from __future__ import annotations

import math
import statistics as _statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .inverted_index import InvertedIndex
from .pipeline_comparison import PipelineComparison, PipelineMeasurement
from .query_registry import QueryRecord


@dataclass
class IndexStatistics:
    """Size and distribution measurements of one inverted index."""

    index_terms: int = 0
    unigram_terms: int = 0
    phrase_terms: int = 0
    total_postings: int = 0
    unigram_postings: int = 0
    phrase_postings: int = 0
    postings_per_term: float = 0.0
    indexed_units: int = 0
    indexed_documents: int = 0
    units_per_document: float = 0.0
    total_positions: int = 0
    mean_positions_per_posting: float = 0.0
    postings_per_unit: float = 0.0
    singleton_terms: int = 0
    terms_in_90_percent_of_postings: int = 0
    mean_term_frequency: float = 0.0
    mean_document_frequency: float = 0.0
    longest_posting_list: int = 0
    longest_posting_term: str = ""
    hapax_ratio: float = 0.0
    top_terms: List[Dict[str, Any]] = field(default_factory=list)
    top_phrases: List[Dict[str, Any]] = field(default_factory=list)

    def as_rows(self) -> List[Dict[str, Any]]:
        flat = {
            "index_terms": self.index_terms,
            "unigram_terms": self.unigram_terms,
            "phrase_terms": self.phrase_terms,
            "total_postings": self.total_postings,
            "unigram_postings": self.unigram_postings,
            "phrase_postings": self.phrase_postings,
            "postings_per_term": self.postings_per_term,
            "indexed_units": self.indexed_units,
            "indexed_documents": self.indexed_documents,
            "units_per_document": self.units_per_document,
            "total_positions": self.total_positions,
            "mean_positions_per_posting": self.mean_positions_per_posting,
            "postings_per_unit": self.postings_per_unit,
            "singleton_terms": self.singleton_terms,
            "terms_in_90_percent_of_postings": self.terms_in_90_percent_of_postings,
            "mean_term_frequency": self.mean_term_frequency,
            "mean_document_frequency": self.mean_document_frequency,
            "longest_posting_list": self.longest_posting_list,
            "longest_posting_term": self.longest_posting_term,
            "hapax_ratio": self.hapax_ratio,
            "top_terms": ";".join(
                f"{row['term']}({row['term_frequency']})" for row in self.top_terms
            ),
            "top_phrases": ";".join(
                f"{row['term']}({row['term_frequency']})" for row in self.top_phrases
            ),
        }
        return [{"metric": key, "value": value} for key, value in flat.items()]


def compute_index_statistics(index: InvertedIndex, top_n: int = 25) -> IndexStatistics:
    """Measure one index. No Phase 2 value is re-used or re-estimated here."""
    stats = IndexStatistics()
    stats.index_terms = index.term_count
    stats.indexed_units = index.unit_count
    stats.indexed_documents = index.document_count()
    stats.units_per_document = (
        round(stats.indexed_units / stats.indexed_documents, 2) if stats.indexed_documents else 0.0
    )

    posting_counts: List[int] = []
    term_frequencies: List[int] = []
    document_frequencies: List[int] = []
    total_positions = 0
    longest = 0
    longest_term = ""
    singletons = 0

    for term, entry in index.terms.items():
        count = entry.posting_frequency
        posting_counts.append(count)
        term_frequencies.append(entry.term_frequency)
        document_frequencies.append(index.document_frequency(term))
        positions = sum(len(posting.positions) for posting in entry.postings)
        total_positions += positions
        if entry.is_phrase:
            stats.phrase_terms += 1
            stats.phrase_postings += count
        else:
            stats.unigram_terms += 1
            stats.unigram_postings += count
        if count == 1:
            singletons += 1
        if count > longest:
            longest, longest_term = count, term

    stats.total_postings = sum(posting_counts)
    stats.postings_per_term = (
        round(stats.total_postings / stats.index_terms, 4) if stats.index_terms else 0.0
    )
    stats.total_positions = total_positions
    stats.mean_positions_per_posting = (
        round(total_positions / stats.total_postings, 4) if stats.total_postings else 0.0
    )
    stats.postings_per_unit = (
        round(stats.total_postings / stats.indexed_units, 4) if stats.indexed_units else 0.0
    )
    stats.singleton_terms = singletons
    stats.hapax_ratio = round(singletons / stats.index_terms, 4) if stats.index_terms else 0.0
    stats.mean_term_frequency = (
        round(_statistics.fmean(term_frequencies), 4) if term_frequencies else 0.0
    )
    stats.mean_document_frequency = (
        round(_statistics.fmean(document_frequencies), 4) if document_frequencies else 0.0
    )
    stats.longest_posting_list = longest
    stats.longest_posting_term = longest_term
    stats.terms_in_90_percent_of_postings = _terms_covering(index, 0.9)

    stats.top_terms = [
        row for row in index.term_statistics() if not row["is_phrase"]
    ]
    stats.top_terms.sort(key=lambda row: (-int(row["term_frequency"]), row["term"]))
    stats.top_terms = stats.top_terms[:top_n]
    stats.top_phrases = [row for row in index.term_statistics() if row["is_phrase"]]
    stats.top_phrases.sort(key=lambda row: (-int(row["term_frequency"]), row["term"]))
    stats.top_phrases = stats.top_phrases[:top_n]
    return stats


def _terms_covering(index: InvertedIndex, fraction: float) -> int:
    """Smallest number of terms accounting for ``fraction`` of all postings."""
    counts = sorted(
        (entry.posting_frequency for entry in index.terms.values()), reverse=True
    )
    target = sum(counts) * fraction
    running = 0
    for position, count in enumerate(counts, start=1):
        running += count
        if running >= target:
            return position
    return len(counts)


# ----------------------------------------------------------------------
# Retrieval measurements
# ----------------------------------------------------------------------
@dataclass
class RetrievalStatistics:
    """Query-side measurements (no quality claims - that is Phase 4)."""

    queries: int = 0
    answered: int = 0
    answerability: float = 0.0
    total_units_returned: int = 0
    total_documents_returned: int = 0
    mean_units_per_query: float = 0.0
    median_units_per_query: float = 0.0
    mean_documents_per_query: float = 0.0
    execution_time_ms: List[float] = field(default_factory=list)
    mean_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "queries": self.queries,
            "answered": self.answered,
            "answerability": self.answerability,
            "total_units_returned": self.total_units_returned,
            "total_documents_returned": self.total_documents_returned,
            "mean_units_per_query": self.mean_units_per_query,
            "median_units_per_query": self.median_units_per_query,
            "mean_documents_per_query": self.mean_documents_per_query,
            "mean_execution_time_ms": self.mean_execution_time_ms,
            "max_execution_time_ms": self.max_execution_time_ms,
            "total_execution_time_ms": self.total_execution_time_ms,
            "by_query_type": self.by_type,
        }


def compute_retrieval_statistics(records: Sequence[QueryRecord]) -> RetrievalStatistics:
    stats = RetrievalStatistics()
    stats.queries = len(records)
    answered = [record for record in records if record.answered]
    stats.answered = len(answered)
    stats.answerability = round(len(answered) / len(records), 4) if records else 0.0
    stats.total_units_returned = sum(record.unit_count for record in records)
    stats.total_documents_returned = sum(record.document_count for record in records)
    unit_counts = [record.unit_count for record in records]
    document_counts = [record.document_count for record in records]
    stats.mean_units_per_query = round(_statistics.fmean(unit_counts), 4) if unit_counts else 0.0
    stats.median_units_per_query = (
        round(_statistics.median(unit_counts), 4) if unit_counts else 0.0
    )
    stats.mean_documents_per_query = (
        round(_statistics.fmean(document_counts), 4) if document_counts else 0.0
    )
    stats.execution_time_ms = [round(record.execution_time_ms, 3) for record in records]
    stats.mean_execution_time_ms = (
        round(_statistics.fmean(stats.execution_time_ms), 3)
        if stats.execution_time_ms else 0.0
    )
    stats.max_execution_time_ms = max(stats.execution_time_ms, default=0.0)
    stats.total_execution_time_ms = round(sum(stats.execution_time_ms), 3)
    for record in records:
        bucket = stats.by_type.setdefault(
            record.query_type, {"queries": 0, "answered": 0, "units": 0}
        )
        bucket["queries"] += 1
        bucket["answered"] += 1 if record.answered else 0
        bucket["units"] += record.unit_count
    return stats


# ----------------------------------------------------------------------
# Selection measurements
# ----------------------------------------------------------------------
def selection_payload(comparison: PipelineComparison) -> Dict[str, Any]:
    """The decision, as data, for ``final_pipeline.json``."""
    return {
        "final_pipeline": comparison.winner_key,
        "final_pipeline_name": comparison.winner.name,
        "score": comparison.winner.total_score,
        "margin_over_runner_up": comparison.margin,
        "reason": comparison.reason,
        "criteria_weights": comparison.criteria_weights,
        "tie_breakers": comparison.tie_breakers,
        "excluded_from_score": comparison.excluded_from_score,
        "scores": {
            key: {
                "pipeline_name": measurement.name,
                "total_score": measurement.total_score,
                "criteria": {
                    criterion.name: {
                        "rate": criterion.rate,
                        "weight": criterion.weight,
                        "weighted": criterion.weighted,
                        "numerator": criterion.numerator,
                        "denominator": criterion.denominator,
                    }
                    for criterion in measurement.criteria
                },
                "index_terms": measurement.index_stats.get("index_terms", 0),
                "total_postings": measurement.index_stats.get("total_postings", 0),
                "processing_time_seconds": measurement.pipeline_stats.get(
                    "processing_time_seconds", 0.0
                ),
            }
            for key, measurement in sorted(comparison.measurements.items())
        },
    }
