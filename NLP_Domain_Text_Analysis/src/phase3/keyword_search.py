"""Single-term (keyword) search over the inverted index.

A keyword query is normalized with the *same* pipeline that built the index, so
``"Inflation"`` and ``"inflation"`` behave identically while ``FY26``,
``FY2025-26`` and ``7.4%`` survive as single terms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .inverted_index import InvertedIndex
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec
from .posting_list import Posting, PostingList


@dataclass
class UnitHit:
    """One retrieved content unit, with everything needed for display."""

    unit_id: str
    document_id: str
    source_id: str
    page_number: int
    section_id: str
    section_number: str
    section_title: str
    unit_type: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    matched_positions: Dict[str, List[int]] = field(default_factory=dict)
    matched_term_frequency: Dict[str, int] = field(default_factory=dict)
    snippet: str = ""
    phrase_match: bool = False
    rank: int = 0

    def to_row(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "page_number": self.page_number,
            "section_id": self.section_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "unit_type": self.unit_type,
            "score": round(self.score, 4),
            "matched_terms": ";".join(self.matched_terms),
            "matched_term_count": len(self.matched_terms),
            "matched_term_frequency": ";".join(
                f"{term}:{count}" for term, count in sorted(self.matched_term_frequency.items())
            ),
            "phrase_match": self.phrase_match,
            "snippet": self.snippet,
        }


@dataclass
class SearchOutcome:
    """Result of one query: unit hits, document aggregation and timing."""

    query: str
    query_type: str
    matched_terms: List[str]
    missing_terms: List[str]
    unit_hits: List[UnitHit]
    execution_time_ms: float
    pipeline: str
    retrieval_method: str

    @property
    def unit_count(self) -> int:
        return len(self.unit_hits)

    @property
    def document_count(self) -> int:
        return len({hit.document_id for hit in self.unit_hits})

    def document_hits(self) -> List[Dict[str, Any]]:
        """Aggregate unit hits into ranked document rows."""
        grouped: Dict[str, List[UnitHit]] = {}
        for hit in self.unit_hits:
            grouped.setdefault(hit.document_id, []).append(hit)
        rows: List[Dict[str, Any]] = []
        for document_id in sorted(
            grouped,
            key=lambda doc: (-max(h.score for h in grouped[doc]), doc),
        ):
            hits = sorted(grouped[document_id], key=lambda h: (-h.score, h.unit_id))
            rows.append(
                {
                    "document_id": document_id,
                    "source_id": hits[0].source_id,
                    "unit_count": len(hits),
                    "best_score": round(max(h.score for h in hits), 4),
                    "best_unit_id": hits[0].unit_id,
                    "pages": ";".join(str(h.page_number) for h in hits),
                    "sections": ";".join(
                        f"{h.section_number} {h.section_title}".strip() for h in hits
                    ),
                    "matched_terms": ";".join(
                        sorted({term for h in hits for term in h.matched_terms})
                    ),
                }
            )
        for position, row in enumerate(rows, start=1):
            row["rank"] = position
        return rows


class KeywordSearch:
    """Resolve one normalized query term to its posting list."""

    def __init__(self, index: InvertedIndex, spec: PipelineSpec, runner: PipelineRunner,
                 logger: Optional[logging.Logger] = None) -> None:
        self.index = index
        self.spec = spec
        self.runner = runner
        self.logger = logger

    def normalize(self, query: str) -> List[str]:
        return self.runner.normalize_query(self.spec, query)

    def postings_for(self, term: str) -> PostingList:
        return self.index.posting_list(term)

    def search(self, query: str) -> Dict[str, Any]:
        """Return ``{'terms': [...], 'missing': [...], 'postings': PostingList}``."""
        started = time.perf_counter()
        normalized = self.normalize(query)
        missing = [term for term in normalized if not self.index.has_term(term)]
        present = [term for term in normalized if self.index.has_term(term)]
        postings = self.index.posting_list(present[0]) if present else PostingList(term=query)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if self.logger is not None and missing:
            from .config import log_event

            log_event(
                self.logger, "WARNING", "keyword_search",
                f"term(s) not in index for '{query}': {missing}",
            )
        return {
            "terms": present,
            "missing": missing,
            "postings": postings,
            "execution_time_ms": round(elapsed_ms, 3),
        }
