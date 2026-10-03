"""Corpus and Phase 1 document services.

Phase 1 is the source of truth for documents, pages, sections and content units.
This service reads its registries and the structured corpus, and never recomputes
a statistic that Phase 1 already measured.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.config.settings import ID_PATTERN, Settings
from backend.services import artifacts

_ID_RE = re.compile(ID_PATTERN)


def validate_id(value: str, field: str = "id") -> str:
    """Reject anything that is not a plain project identifier.

    This is the boundary that stops a request such as ``../../etc/passwd`` from
    ever reaching the filesystem: the value is matched against a whitelist
    pattern before it is used to build any path or key.
    """
    if not value or not _ID_RE.match(value):
        raise ValueError(
            f"invalid {field}: expected letters, digits and _ . : + - only "
            f"(max 120 characters), got {value!r}"
        )
    return value


# ----------------------------------------------------------------------
def _settings_paths(settings: Settings) -> Dict[str, Any]:
    return {
        "document_registry": settings.path(f"{settings.corpus_dir}/metadata/document_registry.csv"),
        "source_registry": settings.path(f"{settings.corpus_dir}/metadata/source_registry.csv"),
        "sections_index": settings.path(f"{settings.corpus_dir}/metadata/sections_index.csv"),
        "corpus_jsonl": settings.path(f"{settings.corpus_dir}/structured/corpus.jsonl"),
        "document_statistics": settings.path(f"{settings.phase1_results}/document_statistics.csv"),
        "extraction_quality": settings.path(f"{settings.phase1_results}/extraction_quality_report.csv"),
        "file_inventory": settings.path(f"{settings.phase1_results}/file_inventory.csv"),
        "pdf_validation": settings.path(f"{settings.phase1_results}/pdf_validation.csv"),
        "validation_rules": settings.path(f"{settings.phase1_results}/validation_rules.csv"),
        "duplicate_report": settings.path(f"{settings.phase1_results}/duplicate_report.csv"),
        "errors": settings.path(f"{settings.phase1_results}/errors.csv"),
    }


def document_registry(settings: Settings) -> List[Dict[str, str]]:
    return artifacts.read_csv_rows(_settings_paths(settings)["document_registry"])


def source_registry(settings: Settings) -> List[Dict[str, str]]:
    return artifacts.read_csv_rows(_settings_paths(settings)["source_registry"])


def document_statistics(settings: Settings) -> List[Dict[str, str]]:
    return artifacts.read_csv_rows(_settings_paths(settings)["document_statistics"])


def phase1_validation(settings: Settings) -> Dict[str, Any]:
    return artifacts.read_json(settings.path(f"{settings.phase1_results}/validation_report.json"))


def phase1_rules(settings: Settings) -> List[Dict[str, str]]:
    return artifacts.read_csv_rows(_settings_paths(settings)["validation_rules"])


def phase1_inventory(settings: Settings) -> Dict[str, Any]:
    paths = _settings_paths(settings)
    return {
        "file_inventory": artifacts.read_csv_rows(paths["file_inventory"]),
        "extraction_quality": artifacts.read_csv_rows(paths["extraction_quality"]),
        "pdf_validation": artifacts.read_csv_rows(paths["pdf_validation"]),
        "validation_rules": artifacts.read_csv_rows(paths["validation_rules"]),
        "duplicate_report": artifacts.read_csv_rows(paths["duplicate_report"], required=False),
        "errors": artifacts.read_csv_rows(paths["errors"], required=False),
    }


# ----------------------------------------------------------------------
_UNIT_INDEX: Dict[str, Dict[str, Any]] = {}
_SECTIONS: Dict[str, List[Dict[str, str]]] = {}


def _unit_index(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """``unit_id -> unit record``, built once from the Phase 1 corpus."""
    key = f"units::{settings.corpus_dir}"
    if key in artifacts._CACHE:
        return artifacts._CACHE[key]
    path = _settings_paths(settings)["corpus_jsonl"]
    index: Dict[str, Dict[str, Any]] = {}
    for record in artifacts.read_jsonl(path):
        unit_id = str(record.get("unit_id", ""))
        if unit_id:
            index[unit_id] = record
    artifacts._CACHE[key] = index
    return index


def _sections(settings: Settings) -> Dict[str, List[Dict[str, str]]]:
    if _SECTIONS:
        return _SECTIONS
    for row in artifacts.read_csv_rows(_settings_paths(settings)["sections_index"]):
        _SECTIONS.setdefault(row.get("document_id", ""), []).append(row)
    return _SECTIONS


def unit(settings: Settings, unit_id: str) -> Optional[Dict[str, Any]]:
    validate_id(unit_id, "unit_id")
    return _unit_index(settings).get(unit_id)


def units_of_document(settings: Settings, document_id: str) -> List[Dict[str, Any]]:
    validate_id(document_id, "document_id")
    return [
        record
        for record in _unit_index(settings).values()
        if record.get("document_id") == document_id
    ]


def sections_of_document(settings: Settings, document_id: str) -> List[Dict[str, str]]:
    validate_id(document_id, "document_id")
    return _sections(settings).get(document_id, [])


# ----------------------------------------------------------------------
def document_summary(settings: Settings) -> List[Dict[str, Any]]:
    """One row per document: registry metadata joined with Phase 1 statistics."""
    stats = {row["document_id"]: row for row in document_statistics(settings)}
    quality = {
        row["document_id"]: row
        for row in artifacts.read_csv_rows(_settings_paths(settings)["extraction_quality"])
    }
    files = {row["document_id"]: row for row in
             artifacts.read_csv_rows(_settings_paths(settings)["file_inventory"])}

    rows: List[Dict[str, Any]] = []
    for entry in document_registry(settings):
        document_id = entry.get("document_id", "")
        stat = stats.get(document_id, {})
        qual = quality.get(document_id, {})
        file_row = files.get(document_id, {})
        rows.append(
            {
                "document_id": document_id,
                "source_id": entry.get("source_id"),
                "title": entry.get("title"),
                "filename": entry.get("filename"),
                "publisher": entry.get("publisher"),
                "year": entry.get("year"),
                "document_type": entry.get("document_type"),
                "page_count": artifacts.as_int(entry.get("page_count"), 0),
                "file_size_bytes": artifacts.as_int(entry.get("file_size_bytes")),
                "sha256": entry.get("sha256"),
                "ingestion_status": entry.get("ingestion_status"),
                "extraction_status": entry.get("extraction_status"),
                "extraction_quality": entry.get("extraction_quality"),
                "title_provenance": entry.get("title_provenance"),
                "publisher_provenance": entry.get("publisher_provenance"),
                "year_provenance": entry.get("year_provenance"),
                "detected_source": file_row.get("detected_source"),
                "sentences": artifacts.as_int(stat.get("sentences")),
                "words": artifacts.as_int(stat.get("words")),
                "characters": artifacts.as_int(stat.get("characters")),
                "baseline_token_count": artifacts.as_int(stat.get("baseline_token_count")),
                "unique_words": artifacts.as_int(stat.get("unique_words")),
                "vocabulary_size": artifacts.as_int(stat.get("vocabulary_size")),
                "raw_vocabulary_size": artifacts.as_int(stat.get("raw_vocabulary_size")),
                "average_document_length": artifacts.as_float(stat.get("average_document_length")),
                "sentences_per_page": artifacts.as_float(stat.get("sentences_per_page")),
                "sections": artifacts.as_int(stat.get("sections")),
                "units": artifacts.as_int(stat.get("units")),
                "paragraphs": artifacts.as_int(stat.get("paragraphs")),
                "tables": artifacts.as_int(stat.get("tables")),
                "figures": artifacts.as_int(stat.get("figures")),
                "footnotes": artifacts.as_int(stat.get("footnotes")),
                "raw_characters": artifacts.as_int(stat.get("raw_characters")),
                "cleaned_characters": artifacts.as_int(stat.get("cleaned_characters")),
                "quality_score": artifacts.as_float(qual.get("quality_score")),
                "quality_grade": qual.get("quality_grade"),
                "pages_with_low_text": artifacts.as_int(qual.get("pages_with_low_text")),
                "pages_likely_scanned": artifacts.as_int(qual.get("pages_likely_scanned")),
                "warnings": qual.get("warnings", ""),
            }
        )
    rows.sort(key=lambda r: r["document_id"])
    return rows


def source_summary(settings: Settings) -> List[Dict[str, Any]]:
    summaries = {row["source_id"]: row for row in document_summary(settings)}
    out: List[Dict[str, Any]] = []
    for entry in source_registry(settings):
        source_id = entry.get("source_id", "")
        members = [d for d in summaries.values() if d["source_id"] == source_id]
        out.append(
            {
                "source_id": source_id,
                "source_title": entry.get("source_title"),
                "publisher": entry.get("publisher"),
                "publisher_provenance": entry.get("publisher_provenance"),
                "year": entry.get("year"),
                "year_provenance": entry.get("year_provenance"),
                "source_type": entry.get("source_type"),
                "description": entry.get("description"),
                "notes": entry.get("notes"),
                "declared_files_count": artifacts.as_int(entry.get("files_count")),
                "declared_total_pages": artifacts.as_int(entry.get("total_pages")),
                "documents": len(members),
                "observed_pages": sum(d.get("page_count") or 0 for d in members),
                "observed_units": sum(d.get("units") or 0 for d in members),
                "observed_baseline_tokens": sum(d.get("baseline_token_count") or 0 for d in members),
                "document_ids": [d["document_id"] for d in members],
            }
        )
    out.sort(key=lambda r: r["source_id"])
    return out


def document_detail(settings: Settings, document_id: str) -> Dict[str, Any]:
    """Everything the document page needs, in one response."""
    validate_id(document_id, "document_id")
    summary = next(
        (row for row in document_summary(settings) if row["document_id"] == document_id), None
    )
    if summary is None:
        return {}
    unit_records = units_of_document(settings, document_id)
    sections = sections_of_document(settings, document_id)
    pages = sorted(
        {
            (
                artifacts.as_int(record.get("page_number"), 0),
                str(record.get("page_id", "")),
            )
            for record in unit_records
        }
    )
    units_by_type: Dict[str, int] = {}
    for record in unit_records:
        key = str(record.get("unit_type", "other"))
        units_by_type[key] = units_by_type.get(key, 0) + 1
    return {
        **summary,
        "source": next(
            (
                s
                for s in source_summary(settings)
                if s["source_id"] == summary.get("source_id")
            ),
            None,
        ),
        "pages": [
            {"page_id": page_id, "page_number": number}
            for number, page_id in pages
            if page_id
        ],
        "sections": sections,
        "unit_count": len(unit_records),
        "units_by_type": dict(sorted(units_by_type.items())),
        "unit_characters": sum(artifacts.as_int(r.get("char_count"), 0) or 0 for r in unit_records),
        "unit_words": sum(artifacts.as_int(r.get("word_count"), 0) or 0 for r in unit_records),
    }
