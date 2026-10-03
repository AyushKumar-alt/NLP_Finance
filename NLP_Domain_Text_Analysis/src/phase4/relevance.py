"""Relevance judgment storage.

Judgments are *human* decisions. This module only reads, validates, joins and
writes them; it never derives a label from a similarity score, because a metric
computed from the ranker it is meant to evaluate would measure nothing.

File format (``results/phase4/relevance_judgments.csv``)::

    query_id, query, unit_id, rank, document_id, page_number, section_number,
    relevance, notes, annotator, judgment_method, judged_at

``relevance`` is ``1`` (relevant) or ``0`` (not relevant). A pair that is absent
from the file is **unjudged**, which is not the same as ``0``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.utils import read_csv_rows

FIELDNAMES: Tuple[str, ...] = (
    "query_id",
    "query",
    "unit_id",
    "rank",
    "document_id",
    "page_number",
    "section_number",
    "relevance",
    "notes",
    "annotator",
    "judgment_method",
    "judged_at",
)

RELEVANT = 1
NOT_RELEVANT = 0
VALID_LABELS = (RELEVANT, NOT_RELEVANT)


class RelevanceValidationError(ValueError):
    """A judgment file violates its own contract."""


@dataclass
class Judgment:
    query_id: str
    unit_id: str
    relevance: int
    query: str = ""
    rank: str = ""
    document_id: str = ""
    page_number: str = ""
    section_number: str = ""
    notes: str = ""
    annotator: str = ""
    judgment_method: str = ""
    judged_at: str = ""

    @property
    def is_relevant(self) -> bool:
        return self.relevance == RELEVANT

    def as_row(self) -> Dict[str, str]:
        return {name: str(getattr(self, name)) for name in FIELDNAMES}

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "Judgment":
        raw = str(row.get("relevance", "")).strip()
        try:
            label = int(raw)
        except ValueError as error:
            raise RelevanceValidationError(
                f"relevance must be 0 or 1, found {raw!r} for "
                f"{row.get('query_id')} / {row.get('unit_id')}"
            ) from error
        if label not in VALID_LABELS:
            raise RelevanceValidationError(
                f"relevance must be 0 or 1, found {label} for "
                f"{row.get('query_id')} / {row.get('unit_id')}"
            )
        return cls(
            query_id=str(row.get("query_id", "")).strip(),
            unit_id=str(row.get("unit_id", "")).strip(),
            relevance=label,
            query=str(row.get("query", "")).strip(),
            rank=str(row.get("rank", "")).strip(),
            document_id=str(row.get("document_id", "")).strip(),
            page_number=str(row.get("page_number", "")).strip(),
            section_number=str(row.get("section_number", "")).strip(),
            notes=str(row.get("notes", "")).strip(),
            annotator=str(row.get("annotator", "")).strip(),
            judgment_method=str(row.get("judgment_method", "")).strip(),
            judged_at=str(row.get("judged_at", "")).strip(),
        )


@dataclass
class RelevanceStore:
    """In-memory view of the judgment file, keyed by ``(query_id, unit_id)``."""

    path: Path
    judgments: Dict[Tuple[str, str], Judgment] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path, strict: bool = False) -> "RelevanceStore":
        store = cls(path=Path(path))
        if not store.path.is_file():
            store.problems.append(f"judgment file does not exist yet: {path.name}")
            return store

        rows = read_csv_rows(store.path)
        seen: set = set()
        for index, row in enumerate(rows, start=2):   # line 1 is the header
            try:
                judgment = Judgment.from_row(row)
            except RelevanceValidationError as error:
                message = f"line {index}: {error}"
                if strict:
                    raise
                store.problems.append(message)
                continue
            if not judgment.query_id or not judgment.unit_id:
                message = f"line {index}: query_id and unit_id are both required"
                if strict:
                    raise RelevanceValidationError(message)
                store.problems.append(message)
                continue
            key = (judgment.query_id, judgment.unit_id)
            if key in seen:
                message = (
                    f"line {index}: duplicate judgment for "
                    f"{judgment.query_id} / {judgment.unit_id}; the last one wins"
                )
                if strict:
                    raise RelevanceValidationError(message)
                store.problems.append(message)
            seen.add(key)
            store.judgments[key] = judgment
        return store

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.judgments)

    def get(self, query_id: str, unit_id: str) -> Optional[Judgment]:
        return self.judgments.get((query_id, unit_id))

    def label(self, query_id: str, unit_id: str) -> Optional[int]:
        """``1``, ``0`` or ``None`` when the pair has not been judged."""
        judgment = self.judgments.get((query_id, unit_id))
        return None if judgment is None else judgment.relevance

    def query_ids(self) -> List[str]:
        return sorted({query_id for query_id, _ in self.judgments})

    def relevant_units(self, query_id: str) -> set:
        return {
            unit_id
            for (qid, unit_id), judgment in self.judgments.items()
            if qid == query_id and judgment.relevance == RELEVANT
        }

    def coverage(self, query_ids: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
        wanted = list(query_ids) if query_ids is not None else self.query_ids()
        coverage: Dict[str, Dict[str, int]] = {}
        for query_id in wanted:
            rows = [j for (qid, _), j in self.judgments.items() if qid == query_id]
            coverage[query_id] = {
                "judged": len(rows),
                "relevant": sum(1 for j in rows if j.relevance == RELEVANT),
                "not_relevant": sum(1 for j in rows if j.relevance == NOT_RELEVANT),
            }
        return coverage

    # ------------------------------------------------------------------
    def upsert(
        self,
        query_id: str,
        unit_id: str,
        relevance: int,
        notes: str = "",
        annotator: str = "",
        judgment_method: str = "",
        judged_at: str = "",
        **provenance: str,
    ) -> Judgment:
        """Record or replace one judgment. This is what the API POST calls."""
        if relevance not in VALID_LABELS:
            raise RelevanceValidationError(f"relevance must be 0 or 1, got {relevance!r}")
        if not query_id or not unit_id:
            raise RelevanceValidationError("query_id and unit_id are both required")
        existing = self.judgments.get((query_id, unit_id))
        judgment = Judgment(
            query_id=query_id,
            unit_id=unit_id,
            relevance=int(relevance),
            query=str(provenance.get("query", existing.query if existing else "")),
            rank=str(provenance.get("rank", existing.rank if existing else "")),
            document_id=str(provenance.get("document_id", existing.document_id if existing else "")),
            page_number=str(provenance.get("page_number", existing.page_number if existing else "")),
            section_number=str(
                provenance.get("section_number", existing.section_number if existing else "")
            ),
            notes=notes or (existing.notes if existing else ""),
            annotator=annotator or (existing.annotator if existing else ""),
            judgment_method=judgment_method or (existing.judgment_method if existing else ""),
            judged_at=judged_at or (existing.judged_at if existing else ""),
        )
        self.judgments[(query_id, unit_id)] = judgment
        return judgment

    def save(self, annotator: str = "", judgment_method: str = "") -> Path:
        """Write the whole store back, deterministically ordered."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered: Iterable[Judgment] = sorted(
            self.judgments.values(), key=lambda j: (j.query_id, j.unit_id)
        )
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
            writer.writeheader()
            for judgment in ordered:
                row = judgment.as_row()
                if annotator and not row["annotator"]:
                    row["annotator"] = annotator
                if judgment_method and not row["judgment_method"]:
                    row["judgment_method"] = judgment_method
                writer.writerow(row)
        return self.path
