"""The inverted index: terms -> posting lists, plus the provenance store.

Layout
------
``terms``
    ``term -> TermEntry`` with ``df`` (document frequency), ``tf`` (term
    frequency), the sorted posting list and a flag for n-gram phrase terms.
``units``
    one record per indexed content unit, holding the full Phase 1 chain
    (``source_id -> document_id -> page_number -> section_id -> unit_id``).
    Keeping metadata in its own store avoids repeating it in every posting.
``documents``
    document level metadata (title, publisher, year, source) so results can be
    displayed and aggregated without touching the corpus file again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .posting_list import Posting, PostingList


@dataclass
class TermEntry:
    """One index term and its posting list."""

    term: str
    postings: List[Posting] = field(default_factory=list)
    is_phrase: bool = False

    @property
    def term_frequency(self) -> int:
        return sum(posting.tf for posting in self.postings)

    @property
    def posting_frequency(self) -> int:
        return len(self.postings)


@dataclass
class UnitRecord:
    """Provenance of one indexed content unit."""

    unit_id: str
    document_id: str
    source_id: str
    page_number: int
    section_id: str
    section_number: str
    section_title: str
    unit_type: str
    unit_index: int
    char_count: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "page_number": self.page_number,
            "section_id": self.section_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "unit_type": self.unit_type,
            "unit_index": self.unit_index,
            "char_count": self.char_count,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "UnitRecord":
        return cls(
            unit_id=str(payload["unit_id"]),
            document_id=str(payload["document_id"]),
            source_id=str(payload["source_id"]),
            page_number=int(payload.get("page_number", 0)),
            section_id=str(payload.get("section_id", "")),
            section_number=str(payload.get("section_number", "")),
            section_title=str(payload.get("section_title", "")),
            unit_type=str(payload.get("unit_type", "")),
            unit_index=int(payload.get("unit_index", 0)),
            char_count=int(payload.get("char_count", 0)),
        )


@dataclass
class DocumentRecord:
    """Document level metadata used for aggregation and display."""

    document_id: str
    source_id: str
    filename: str
    title: str = ""
    publisher: str = ""
    year: str = ""
    document_type: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "filename": self.filename,
            "title": self.title,
            "publisher": self.publisher,
            "year": self.year,
            "document_type": self.document_type,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "DocumentRecord":
        return cls(
            document_id=str(payload["document_id"]),
            source_id=str(payload.get("source_id", "")),
            filename=str(payload.get("filename", "")),
            title=str(payload.get("title", "")),
            publisher=str(payload.get("publisher", "")),
            year=str(payload.get("year", "")),
            document_type=str(payload.get("document_type", "")),
        )


class InvertedIndex:
    """Searchable index over content units."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.terms: Dict[str, TermEntry] = {}
        self.units: Dict[str, UnitRecord] = {}
        self.documents: Dict[str, DocumentRecord] = {}
        self.metadata: Dict[str, Any] = dict(metadata or {})

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_unit(self, record: UnitRecord) -> None:
        self.units[record.unit_id] = record

    def add_document(self, record: DocumentRecord) -> None:
        self.documents[record.document_id] = record

    def add_posting(self, term: str, unit_id: str, tf: int, positions: Sequence[int],
                    is_phrase: bool = False) -> None:
        entry = self.terms.get(term)
        if entry is None:
            entry = TermEntry(term=term, is_phrase=is_phrase)
            self.terms[term] = entry
        entry.postings.append(
            Posting(unit_id=unit_id, tf=int(tf), positions=tuple(int(p) for p in positions))
        )

    def finalise(self) -> None:
        """Sort every posting list so serialization is deterministic."""
        for entry in self.terms.values():
            entry.postings.sort(key=lambda posting: (posting.unit_id, posting.positions))

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def __contains__(self, term: str) -> bool:
        return term in self.terms

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def posting_count(self) -> int:
        return sum(entry.posting_frequency for entry in self.terms.values())

    @property
    def unit_count(self) -> int:
        return len(self.units)

    def document_count(self) -> int:
        return len(self.documents)

    def posting_list(self, term: str) -> PostingList:
        """Posting list for ``term``; an empty list when the term is unknown."""
        entry = self.terms.get(term)
        if entry is None:
            return PostingList(term=term, postings=[], is_phrase=False)
        return PostingList(term=term, postings=list(entry.postings), is_phrase=entry.is_phrase)

    def has_term(self, term: str) -> bool:
        return term in self.terms

    def unit(self, unit_id: str) -> Optional[UnitRecord]:
        return self.units.get(unit_id)

    def document(self, document_id: str) -> Optional[DocumentRecord]:
        return self.documents.get(document_id)

    def unit_to_document(self) -> Dict[str, str]:
        return {unit_id: record.document_id for unit_id, record in self.units.items()}

    def term_statistics(self) -> List[Dict[str, Any]]:
        unit_to_document = self.unit_to_document()
        rows: List[Dict[str, Any]] = []
        for term in sorted(self.terms):
            entry = self.terms[term]
            documents = sorted(
                {
                    unit_to_document[posting.unit_id]
                    for posting in entry.postings
                    if posting.unit_id in unit_to_document
                }
            )
            rows.append(
                {
                    "term": term,
                    "is_phrase": entry.is_phrase,
                    "document_frequency": len(documents),
                    "posting_frequency": entry.posting_frequency,
                    "term_frequency": entry.term_frequency,
                    "unit_count": entry.posting_frequency,
                    "document_ids": ";".join(documents),
                    "first_unit_id": entry.postings[0].unit_id if entry.postings else "",
                }
            )
        return rows

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "documents": {
                document_id: record.to_json() for document_id, record in sorted(self.documents.items())
            },
            "units": {
                unit_id: record.to_json() for unit_id, record in sorted(self.units.items())
            },
            "terms": {
                term: {
                    "df": self._document_frequency(term),
                    "tf": entry.term_frequency,
                    "posting_frequency": entry.posting_frequency,
                    "is_phrase": entry.is_phrase,
                    "postings": [posting.to_json() for posting in entry.postings],
                }
                for term, entry in sorted(self.terms.items())
            },
        }

    def _document_frequency(self, term: str) -> int:
        return self.document_frequency(term)

    def document_frequency(self, term: str) -> int:
        """Number of distinct documents containing ``term`` (0 when unknown)."""
        entry = self.terms.get(term)
        if entry is None:
            return 0
        unit_to_document = self.unit_to_document()
        return len(
            {
                unit_to_document[posting.unit_id]
                for posting in entry.postings
                if posting.unit_id in unit_to_document
            }
        )

    def save(self, path: Path) -> Dict[str, Any]:
        payload = self.to_json()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        return {
            "file": str(path.name),
            "terms": len(payload["terms"]),
            "units": len(payload["units"]),
            "documents": len(payload["documents"]),
            "bytes": path.stat().st_size,
        }

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        index = cls(metadata=payload.get("metadata", {}))
        for document_id, record in payload.get("documents", {}).items():
            index.add_document(DocumentRecord.from_json(record))
        for unit_id, record in payload.get("units", {}).items():
            index.add_unit(UnitRecord.from_json(record))
        for term, entry in payload.get("terms", {}).items():
            index.terms[term] = TermEntry(
                term=term,
                postings=[Posting.from_json(item) for item in entry.get("postings", [])],
                is_phrase=bool(entry.get("is_phrase", False)),
            )
        return index

    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Structural checks used by the Phase 3 validation report."""
        problems: List[str] = []
        for term, entry in self.terms.items():
            unit_ids = [posting.unit_id for posting in entry.postings]
            if len(unit_ids) != len(set(unit_ids)):
                duplicates = sorted({u for u in unit_ids if unit_ids.count(u) > 1})
                problems.append(f"term '{term}' has duplicate postings: {duplicates[:5]}")
            if unit_ids != sorted(unit_ids):
                problems.append(f"term '{term}' postings are not sorted by unit_id")
            for posting in entry.postings:
                if posting.unit_id not in self.units:
                    problems.append(f"term '{term}' references unknown unit '{posting.unit_id}'")
                    break
        for entry in self.terms.values():
            if entry.term_frequency <= 0:
                problems.append(f"term '{entry.term}' has non-positive term frequency")
        return problems
