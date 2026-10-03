"""Turn search outcomes into the files a reader can audit.

Three tables are produced for the final pipeline:

``retrieval_results.csv``   one row per retrieved content unit, ranked, with the
                            full Phase 1 provenance chain and a snippet;
``document_results.csv``    the same queries aggregated to document level;
``retrieval_summary.csv``   one row per query: answerability, coverage, timing.

Every row is traceable back to ``source_id -> document_id -> page_number ->
section_id -> unit_id``. Snippets are truncated on a word boundary so a row
never contains half a word.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import Phase3Config, log_event
from .keyword_search import SearchOutcome, UnitHit
from .query_registry import QueryRecord

RETRIEVAL_FIELDS = (
    "query_id",
    "query",
    "query_type",
    "retrieval_method",
    "pipeline",
    "rank",
    "score",
    "unit_id",
    "document_id",
    "source_id",
    "page_number",
    "section_id",
    "section_number",
    "section_title",
    "unit_type",
    "matched_terms",
    "matched_term_count",
    "matched_term_frequency",
    "phrase_match",
    "snippet",
    "citation",
)

DOCUMENT_FIELDS = (
    "query_id",
    "query",
    "query_type",
    "rank",
    "document_id",
    "source_id",
    "unit_count",
    "best_score",
    "best_unit_id",
    "pages",
    "sections",
    "matched_terms",
)

SUMMARY_FIELDS = (
    "query_id",
    "query",
    "query_type",
    "description",
    "expected_domain",
    "configured_terms",
    "normalized_terms",
    "matched_terms",
    "missing_terms",
    "syntax_valid",
    "answered",
    "result_units",
    "result_documents",
    "top10_score_sum",
    "execution_time_ms",
)


def truncate_snippet(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, on a word boundary."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return "..."[:limit]
    cut = cleaned[: limit - 3]
    space = cut.rfind(" ")
    if space > (limit - 3) // 2:
        cut = cut[:space]
    return cut.rstrip() + "..."


def citation(document_id: str, page_number: int, section_number: str, unit_id: str) -> str:
    """The human readable citation used in every result row."""
    section = f" | {section_number}" if section_number else ""
    return f"{document_id} | p{page_number}{section} | {unit_id}"


@dataclass
class ResultFormatter:
    """Builds the retrieval tables for one pipeline."""

    config: Phase3Config
    snippet_chars: int = 220

    def __post_init__(self) -> None:
        self.snippet_chars = int(self.config.get("retrieval.snippet_chars", 220))

    # ------------------------------------------------------------------
    def unit_rows(
        self, record: QueryRecord, outcome: SearchOutcome
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for hit in outcome.unit_hits:
            row = hit.to_row()
            rows.append(
                {
                    "query_id": record.query_id,
                    "query": outcome.query,
                    "query_type": outcome.query_type,
                    "retrieval_method": outcome.retrieval_method,
                    "pipeline": outcome.pipeline,
                    "rank": hit.rank,
                    "score": row["score"],
                    "unit_id": hit.unit_id,
                    "document_id": hit.document_id,
                    "source_id": hit.source_id,
                    "page_number": hit.page_number,
                    "section_id": hit.section_id,
                    "section_number": hit.section_number,
                    "section_title": hit.section_title,
                    "unit_type": hit.unit_type,
                    "matched_terms": row["matched_terms"],
                    "matched_term_count": row["matched_term_count"],
                    "matched_term_frequency": row["matched_term_frequency"],
                    "phrase_match": row["phrase_match"],
                    "snippet": truncate_snippet(hit.snippet, self.snippet_chars),
                    "citation": citation(
                        hit.document_id, hit.page_number, hit.section_number, hit.unit_id
                    ),
                }
            )
        return rows

    def document_rows(
        self, record: QueryRecord, outcome: SearchOutcome
    ) -> List[Dict[str, Any]]:
        return [
            {
                "query_id": record.query_id,
                "query": outcome.query,
                "query_type": outcome.query_type,
                "rank": row["rank"],
                "document_id": row["document_id"],
                "source_id": row["source_id"],
                "unit_count": row["unit_count"],
                "best_score": row["best_score"],
                "best_unit_id": row["best_unit_id"],
                "pages": row["pages"],
                "sections": row["sections"],
                "matched_terms": row["matched_terms"],
            }
            for row in outcome.document_hits()
        ]

    def summary_row(self, record: QueryRecord) -> Dict[str, Any]:
        outcome = record.outcome
        top_scores = [hit.score for hit in (outcome.unit_hits[:10] if outcome else [])]
        return {
            "query_id": record.query_id,
            "query": outcome.query if outcome else record.query,
            "query_type": record.query_type,
            "description": record.description,
            "expected_domain": record.expected_domain,
            "configured_terms": ";".join(record.configured_terms),
            "normalized_terms": ";".join(record.normalized_terms),
            "matched_terms": ";".join(record.matched_terms),
            "missing_terms": ";".join(record.missing_terms),
            "syntax_valid": record.syntax_ok,
            "answered": record.answered,
            "result_units": record.unit_count,
            "result_documents": record.document_count,
            "top10_score_sum": round(sum(top_scores), 4),
            "execution_time_ms": round(record.execution_time_ms, 3),
        }

    # ------------------------------------------------------------------
    def all_rows(
        self, records: Sequence[QueryRecord]
    ) -> Dict[str, List[Dict[str, Any]]]:
        unit_rows: List[Dict[str, Any]] = []
        document_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        for record in records:
            summary_rows.append(self.summary_row(record))
            if record.outcome is None:
                continue
            unit_rows.extend(self.unit_rows(record, record.outcome))
            document_rows.extend(self.document_rows(record, record.outcome))
        return {
            "units": unit_rows,
            "documents": document_rows,
            "summary": summary_rows,
        }

    # ------------------------------------------------------------------
    def query_example(self, record: QueryRecord) -> str:
        """A readable per-query example file (text, not CSV)."""
        lines: List[str] = [
            f"Query {record.query_id}: {record.query}",
            f"Type: {record.query_type} ({record.description})",
            f"Expected domain: {record.expected_domain}",
            f"Pipeline-normalized terms: {', '.join(record.normalized_terms) or '(none)'}",
            f"Matched terms: {', '.join(record.matched_terms) or '(none)'}",
            f"Missing terms: {', '.join(record.missing_terms) or '(none)'}",
            f"Results: {record.unit_count} content unit(s) in {record.document_count} document(s)",
            f"Execution time: {record.execution_time_ms:.3f} ms",
            "",
        ]
        if not record.syntax_ok:
            lines.append(f"REJECTED: {record.error}")
            return "\n".join(lines)
        if record.outcome is None:
            return "\n".join(lines)
        for hit in record.outcome.unit_hits[:10]:
            lines.append(
                f"  {hit.rank:>2}. score={hit.score:.4f} "
                f"{citation(hit.document_id, hit.page_number, hit.section_number, hit.unit_id)}"
            )
            lines.append(f"      matched: {', '.join(hit.matched_terms)}")
            if hit.snippet:
                lines.append(f"      {truncate_snippet(hit.snippet, self.snippet_chars)}")
        if record.unit_count > 10:
            lines.append(f"  ... {record.unit_count - 10} further results in retrieval_results.csv")
        return "\n".join(lines)
