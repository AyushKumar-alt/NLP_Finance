"""Step 3-5: stable document ids, source (parent document) identification, registries.

Identity policy
---------------
``SRCxx``  a *source* - the real parent report (Economic Survey, SEBI Annual
           Report, Financial Stability Report).
``Dxx``    a *document* - one ingested PDF file. A single source owns many
           documents (``echap01.pdf`` .. ``echap16-2.pdf``).
``Dxx_Sxx_NNN`` / ``Dxx_Pxxx_*`` sections and pages, and
``..._PAR_001`` / ``..._T001`` content units, live inside a document.

Ids are assigned from a deterministic sorted order and persisted in
``data/corpus/metadata/document_registry.csv``. On a re-run the registry is
re-loaded so previously issued ids are never re-issued differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.config import Phase1Config
from ..core.utils import format_id, read_csv_rows, sort_key_for, write_csv
from .discovery import DiscoveredFile

DOCUMENT_REGISTRY_COLUMNS = [
    "document_id",
    "source_id",
    "filename",
    "relative_path",
    "title",
    "publisher",
    "year",
    "document_type",
    "parent_folder",
    "page_count",
    "file_size_bytes",
    "sha256",
    "ingestion_status",
    "extraction_status",
    "extraction_quality",
    "title_provenance",
    "publisher_provenance",
    "year_provenance",
]

SOURCE_REGISTRY_COLUMNS = [
    "source_id",
    "source_title",
    "publisher",
    "publisher_provenance",
    "year",
    "year_provenance",
    "source_type",
    "description",
    "files_count",
    "total_pages",
    "notes",
]


@dataclass
class SourceRecord:
    source_id: str
    source_title: str
    publisher: str
    year: str
    source_type: str
    description: str = ""
    files_count: int = 0
    total_pages: int = 0
    notes: str = ""
    publisher_provenance: str = "inferred_from_folder"
    year_provenance: str = "inferred_from_folder"
    document_ids: List[str] = field(default_factory=list)

    def row(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "publisher": self.publisher,
            "publisher_provenance": self.publisher_provenance,
            "year": self.year,
            "year_provenance": self.year_provenance,
            "source_type": self.source_type,
            "description": self.description,
            "files_count": self.files_count,
            "total_pages": self.total_pages,
            "notes": self.notes,
        }


def _metadata_value(meta: Dict[str, str], key: str) -> str:
    value = (meta or {}).get(key) or ""
    return re.sub(r"\s+", " ", str(value)).strip()


class SourceResolver:
    """Maps an ingested PDF to its parent report using explicit config rules."""

    def __init__(self, config: Phase1Config):
        self.config = config
        self.rules = config.get("sources.rules", []) or []
        self.trust_pdf_title = bool(config.get("sources.trust_pdf_metadata_title", True))
        self.infer_year = bool(config.get("sources.infer_year_from_filename", True))
        self.unknown = config.unknown_label

    def resolve(self, record: DiscoveredFile) -> Tuple[str, SourceRecord]:
        for rule in self.rules:
            folder_pattern = rule.get("match_parent_folder", "")
            file_pattern = rule.get("match_filename", ".*")
            if not folder_pattern or not file_pattern:
                continue
            folder_ok = self._match(folder_pattern, record.parent_folder)
            file_ok = self._match(file_pattern, record.filename)
            # A rule must not fire on an empty folder match for named folders.
            if folder_ok and file_ok:
                source = SourceRecord(
                    source_id=str(rule.get("source_id", self.unknown)),
                    source_title=str(rule.get("source_title", self.unknown)),
                    publisher=str(rule.get("publisher", self.unknown)),
                    year=str(rule.get("year", self.unknown)),
                    source_type=str(rule.get("source_type", "unknown")),
                    description=str(rule.get("description", "")),
                    notes=str(rule.get("notes", "")),
                )
                return source.source_id, source

        source = SourceRecord(
            source_id=self.unknown,
            source_title=self.unknown,
            publisher=self.unknown,
            year=self.unknown,
            source_type="unknown",
            description="Parent report could not be determined confidently.",
            notes="No configuration rule matched; left UNKNOWN rather than guessed.",
        )
        return self.unknown, source


    @staticmethod
    def _match(pattern: str, value: str) -> bool:
        if pattern in {"", None}:
            return False
        if pattern == ".*":
            return True
        try:
            return bool(re.search(pattern, value))
        except re.error:
            return pattern.casefold() in str(value).casefold()


def _derive_title(record: DiscoveredFile, source: SourceRecord, config: Phase1Config) -> Tuple[str, str]:
    """Title with explicit provenance - never hallucinated."""
    stem = Path(record.filename).stem
    pdf_title = _metadata_value(record.pdf_metadata, "title")
    if pdf_title and config.get("sources.trust_pdf_metadata_title", True):
        provenance = "registry_previous_run" if "title" in getattr(record, "restored_keys", set()) else "pdf_metadata"
        return pdf_title, provenance
    if source.source_title != config.unknown_label:
        # A file inside a known parent report: title = stem + parent context.
        return stem, "inferred_from_filename"
    if stem:
        return stem, "inferred_from_filename"
    return config.unknown_label, "missing"


def _derive_publisher_year(record: DiscoveredFile, source: SourceRecord, config: Phase1Config) -> Tuple[str, str, str, str]:
    pdf_author = _metadata_value(record.pdf_metadata, "author")
    publisher = pdf_author if pdf_author else source.publisher
    publisher_prov = "pdf_metadata" if pdf_author else (
        "inferred_from_folder" if source.publisher != config.unknown_label else "missing"
    )
    if "publisher" in getattr(record, "restored_keys", set()):
        publisher_prov = "registry_previous_run"

    year = config.unknown_label
    year_prov = "missing"
    if source.year != config.unknown_label:
        year = source.year
        year_prov = "inferred_from_folder"
    pdf_date = _metadata_value(record.pdf_metadata, "creationDate")
    pdf_year_match = re.search(r"(19|20)\d{2}", pdf_date) if pdf_date else None
    if config.get("sources.infer_year_from_filename", True):
        for candidate in (record.filename, record.parent_folder):
            match = re.search(r"(19|20)\d{2}(?:-?\d{2})?", candidate)
            if match:
                year = match.group(0)
                year_prov = "inferred_from_filename"
                break
    if pdf_year_match and year_prov == "missing":
        year = pdf_year_match.group(0)
        year_prov = "pdf_metadata"
    if "year" in getattr(record, "restored_keys", set()):
        year_prov = "registry_previous_run"
    return publisher, publisher_prov, year, year_prov


def _derive_document_type(record: DiscoveredFile, source: SourceRecord, config: Phase1Config) -> str:
    if re.match(r"(?i)^echap", record.filename):
        return "economic_survey_chapter"
    if re.match(r"(?i)^chapter\s*\d+", record.filename):
        return "annual_report_chapter"
    if re.search(r"(?i)takeaway|briefing|snapshot", record.filename):
        return "briefing_note"
    if re.search(r"(?i)budget", record.filename):
        return "budget_analysis_note"
    if source.source_type != "unknown":
        return f"{source.source_type}_file"
    return "unknown"


class RegistryBuilder:
    """Assigns and persists stable ids for documents and sources."""

    def __init__(self, config: Phase1Config):
        self.config = config
        self.resolver = SourceResolver(config)
        self.documents: Dict[str, DiscoveredFile] = {}
        self.sources: Dict[str, SourceRecord] = {}
        self.meta: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    def _load_existing(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        """(relative_path -> document_id, document_id -> metadata row)."""
        path = self.config.metadata_path("document_registry")
        if not path.exists():
            return {}, {}
        rows = read_csv_rows(path)
        path_map = {row.get("relative_path", ""): row.get("document_id", "") for row in rows if row.get("relative_path")}
        return path_map, {row.get("document_id", ""): row for row in rows if row.get("document_id")}

    def assign(self, records: List[DiscoveredFile]) -> None:
        existing_ids, existing_meta = self._load_existing()
        next_doc_number = 1

        # Sources are keyed by config order so SRC ids stay stable too.
        for index, rule in enumerate(self.config.get("sources.rules", []) or [], start=1):
            sid = str(rule.get("source_id") or format_id(
                self.config.source_id_prefix, index, self.config.source_id_padding
            ))
            self.sources[sid] = SourceRecord(
                source_id=sid,
                source_title=str(rule.get("source_title", self.config.unknown_label)),
                publisher=str(rule.get("publisher", self.config.unknown_label)),
                year=str(rule.get("year", self.config.unknown_label)),
                source_type=str(rule.get("source_type", "unknown")),
                description=str(rule.get("description", "")),
                notes=str(rule.get("notes", "")),
            )

        # Deterministic order regardless of filesystem enumeration.
        ordered = sorted(records, key=lambda r: sort_key_for(r.relative_path, r.filename))

        for record in ordered:
            existing_id = existing_ids.get(record.relative_path, "")
            if existing_id and existing_id in existing_meta:
                record.document_id = existing_id
                match = re.search(r"(\d+)$", existing_id)
                if match:
                    next_doc_number = max(next_doc_number, int(match.group(1)) + 1)
            else:
                record.document_id = format_id(
                    self.config.id_prefix, next_doc_number, self.config.id_padding
                )
                next_doc_number += 1

            # Metadata restored from a previous run (reproducibility).
            stored = dict(existing_meta.get(record.document_id, {}))
            for key in ("title", "publisher", "year"):
                if stored.get(key):
                    record.pdf_metadata[key] = stored[key]
                    record.restored_keys.add(key)
            self.meta[record.document_id] = stored

            source_id, source = self.resolver.resolve(record)
            record.source_id = source_id
            record.detected_source = source.source_title
            if source_id != self.config.unknown_label and source_id in self.sources:
                self.sources[source_id].document_ids.append(record.document_id)
            else:
                if source_id in self.sources:
                    self.sources[source_id].document_ids.append(record.document_id)
                else:
                    source.document_ids.append(record.document_id)
                    self.sources[source_id] = source

            self.documents[record.document_id] = record

    # ------------------------------------------------------------------
    def document_rows(self, extraction_status: Dict[str, str], quality: Dict[str, str]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for doc_id in sorted(self.documents):
            record = self.documents[doc_id]
            source = self.sources.get(record.source_id)
            title, title_prov = _derive_title(record, source, self.config) if source else (
                Path(record.filename).stem, "inferred_from_filename"
            )
            publisher, pub_prov, year, year_prov = (
                _derive_publisher_year(record, source, self.config) if source else ("", "", "", "")
            )
            rows.append(
                {
                    "document_id": doc_id,
                    "source_id": record.source_id,
                    "filename": record.filename,
                    "relative_path": record.relative_path,
                    "title": title,
                    "publisher": publisher,
                    "year": year,
                    "document_type": _derive_document_type(record, source, self.config) if source else "unknown",
                    "parent_folder": record.parent_folder,
                    "page_count": record.page_count,
                    "file_size_bytes": record.file_size_bytes,
                    "sha256": record.sha256,
                    "ingestion_status": record.status,
                    "extraction_status": extraction_status.get(doc_id, "NOT_RUN"),
                    "extraction_quality": quality.get(doc_id, "NOT_RUN"),
                    "title_provenance": title_prov,
                    "publisher_provenance": pub_prov,
                    "year_provenance": year_prov,
                }
            )
        return rows

    def source_rows(self) -> List[Dict[str, object]]:
        rows = []
        for sid in sorted(self.sources):
            source = self.sources[sid]
            source.files_count = len(source.document_ids)
            source.total_pages = sum(
                self.documents[d].page_count
                for d in source.document_ids
                if d in self.documents
            )
            rows.append(source.row())
        return rows

    def persist(self, extraction_status: Dict[str, str], quality: Dict[str, str]) -> None:
        write_csv(
            self.config.metadata_path("document_registry"),
            self.document_rows(extraction_status, quality),
            DOCUMENT_REGISTRY_COLUMNS,
        )
        write_csv(
            self.config.metadata_path("source_registry"),
            self.source_rows(),
            SOURCE_REGISTRY_COLUMNS,
        )
