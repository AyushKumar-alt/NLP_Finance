"""Build the candidate pool that relevance judgments are made on.

Judging every one of the 6,005 indexable units for 15 queries is not feasible,
and is not what IR evaluation does either. The standard approach is to *pool*:
retrieve with a known-good system, judge the top of each ranking, and measure
against what was judged. This module makes that choice explicit and records it,
rather than hiding it behind a number that looks absolute.

The pool is deterministic: it is the first ``pool_depth`` units of each query's
ranking, in rank order, for the query set declared in the Phase 3 configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.phase3.config import Phase3Config
from src.phase3.retrieval import RetrievalEngine


@dataclass
class PoolItem:
    """One (query, unit) pair awaiting a human decision."""

    query_id: str
    query: str
    query_type: str
    rank: int
    unit_id: str
    document_id: str
    source_id: str
    page_number: int
    section_number: str
    section_title: str
    unit_type: str
    score: float
    snippet: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type,
            "rank": self.rank,
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "page_number": self.page_number,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "unit_type": self.unit_type,
            "score": self.score,
            "snippet": self.snippet,
        }


@dataclass
class Pool:
    items: List[PoolItem] = field(default_factory=list)
    depth: int = 0
    queries: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    def keys(self) -> set:
        return {(item.query_id, item.unit_id) for item in self.items}

    def by_query(self) -> Dict[str, List[PoolItem]]:
        grouped: Dict[str, List[PoolItem]] = {}
        for item in self.items:
            grouped.setdefault(item.query_id, []).append(item)
        return grouped

    def unjudged(self, judged_keys: set) -> List[PoolItem]:
        return [item for item in self.items if (item.query_id, item.unit_id) not in judged_keys]

    def summary(self) -> Dict[str, object]:
        grouped = self.by_query()
        return {
            "pool_depth": self.depth,
            "queries": len(self.queries),
            "candidate_pairs": len(self.items),
            "pairs_per_query": {q: len(v) for q, v in sorted(grouped.items())},
            "distinct_units": len({item.unit_id for item in self.items}),
        }


def build_pool(
    engine: RetrievalEngine,
    queries: Sequence[Dict[str, str]],
    depth: int,
    logger=None,
) -> Pool:
    """Run every query and keep the first ``depth`` hits as candidates."""
    pool = Pool(depth=depth)
    for query in queries:
        query_id = str(query.get("id", "")).strip()
        text = str(query.get("query", "")).strip()
        query_type = str(query.get("type", "")).strip()
        if not query_id or not text:
            pool.problems.append(f"skipping malformed query entry: {query!r}")
            continue
        pool.queries.append(query_id)
        try:
            outcome = engine.search(text, query_type or None)
        except Exception as error:                    # noqa: BLE001 - recorded, not raised
            pool.problems.append(f"{query_id} {text!r}: {type(error).__name__}: {error}")
            continue
        for hit in outcome.unit_hits[:depth]:
            pool.items.append(
                PoolItem(
                    query_id=query_id,
                    query=outcome.query,
                    query_type=outcome.query_type,
                    rank=hit.rank,
                    unit_id=hit.unit_id,
                    document_id=hit.document_id,
                    source_id=hit.source_id,
                    page_number=hit.page_number,
                    section_number=hit.section_number,
                    section_title=hit.section_title,
                    unit_type=hit.unit_type,
                    score=hit.score,
                    snippet=hit.snippet,
                )
            )
        if logger is not None:
            from src.phase3.config import log_event

            log_event(
                logger,
                "INFO",
                "pooling",
                f"{query_id} {text!r} ({outcome.query_type}): {min(len(outcome.unit_hits), depth)}"
                f" candidate(s) from {outcome.unit_count()} retrieved",
            )
    return pool


def pool_from_phase3_config(config: Phase3Config) -> List[Dict[str, str]]:
    """The 15 configured domain queries, in declared order."""
    return [dict(entry) for entry in (config.get("queries") or [])]
