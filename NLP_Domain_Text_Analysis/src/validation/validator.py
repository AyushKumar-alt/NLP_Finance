"""Phase 1 validation.

Fifteen rules, all executed against the artefacts that were actually written to
disk. Nothing is hard-coded: every number in ``validation_report.json`` is
computed here.

1.  Every discovered PDF has a document id.
2.  Every PDF appears in ``document_registry.csv``.
3.  Every page has a page record.
4.  Every structured unit has a ``unit_id``.
5.  No ``unit_id`` is duplicated.
6.  Every unit references an existing ``document_id``.
7.  Every unit references a valid page (page_id and page_number).
8.  Raw text exists for every successfully processed document.
9.  Cleaned text exists for every successfully processed document.
10. ``corpus.jsonl`` loads without errors.
11. Character counts are non-negative.
12. Word counts are non-negative.
13. Duplicate detection has been executed.
14. Extraction errors are logged.
15. The original PDFs were not modified (sha256 re-verified).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..core.config import Phase1Config
from ..core.utils import read_csv_rows, read_jsonl, sha256_file
from ..ingestion.discovery import DiscoveredFile

VALIDATION_COLUMNS = [
    "rule_id",
    "rule",
    "status",
    "checked_items",
    "failures",
    "detail",
]


@dataclass
class ValidationContext:
    config: Phase1Config
    files: Sequence[DiscoveredFile]
    results: Dict[str, Any]
    unit_counts: Dict[str, int]
    page_counts: Dict[str, int]
    errors: List[Dict[str, Any]]
    duplicate_summary: Dict[str, Any]
    processed_ids: Sequence[str]
    failed_ids: Sequence[str]


@dataclass
class ValidationResult:
    rules: List[Dict[str, Any]] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0

    def add(self, rule_id: str, rule: str, failures: int, checked: int, detail: str) -> None:
        status = "PASS" if failures == 0 else "FAIL"
        self.rules.append(
            {
                "rule_id": rule_id,
                "rule": rule,
                "status": status,
                "checked_items": checked,
                "failures": failures,
                "detail": detail,
            }
        )
        if failures:
            self.errors += 1


class Phase1Validator:
    def __init__(self, config: Phase1Config):
        self.config = config
        self.unknown = config.unknown_label

    # ------------------------------------------------------------------
    def run(self, context: ValidationContext) -> Dict[str, Any]:
        result = ValidationResult()
        files = list(context.files)

        # 1 ----------------------------------------------------------------
        missing_id = [f.relative_path for f in files if not f.document_id]
        result.add(
            "R01",
            "every discovered PDF has a document id",
            len(missing_id),
            len(files),
            f"{len(files) - len(missing_id)}/{len(files)} files carry a document id",
        )

        # 2 ----------------------------------------------------------------
        registry = read_csv_rows(self.config.metadata_path("document_registry"))
        registered = {row.get("relative_path", "") for row in registry}
        not_registered = [f.relative_path for f in files if f.relative_path not in registered]
        result.add(
            "R02",
            "every PDF appears in document_registry.csv",
            len(not_registered),
            len(files),
            f"{len(registry)} registry rows",
        )

        # 3 ----------------------------------------------------------------
        page_dir = self.config.out_path("pages_dir")
        page_failures = 0
        page_checked = 0
        for doc_id, expected in context.page_counts.items():
            if expected <= 0:
                continue
            for page_number in range(1, expected + 1):
                page_checked += 1
                page_file = page_dir / f"{doc_id}_P{page_number:03d}.json"
                if not page_file.exists():
                    page_failures += 1
        result.add(
            "R03",
            "every page has a page record",
            page_failures,
            page_checked,
            f"{page_checked - page_failures}/{page_checked} page json files present",
        )

        # 4/5/6/7/11/12 -----------------------------------------------------
        corpus_path = self.config.out_path("structured_dir") / str(
            self.config.get("artifacts.corpus_jsonl", "corpus.jsonl")
        )
        units = read_jsonl(corpus_path) if corpus_path.exists() else []
        jsonl_error = None
        if not corpus_path.exists():
            jsonl_error = "corpus.jsonl not found"
        else:
            try:
                with open(corpus_path, "r", encoding="utf-8") as handle:
                    for number, line in enumerate(handle, start=1):
                        if line.strip():
                            json.loads(line)
            except Exception as exc:
                jsonl_error = f"line-level JSON error: {exc}"

        missing_unit_id = [i for i, unit in enumerate(units) if not unit.get("unit_id")]
        result.add(
            "R04",
            "every structured unit has a unit_id",
            len(missing_unit_id),
            len(units),
            f"{len(units) - len(missing_unit_id)}/{len(units)} units carry a unit_id",
        )

        seen: Dict[str, int] = {}
        for unit in units:
            key = unit.get("unit_id")
            if key:
                seen[key] = seen.get(key, 0) + 1
        duplicates = [key for key, count in seen.items() if count > 1]
        result.add(
            "R05",
            "no unit_id is duplicated",
            len(duplicates),
            len(units),
            f"{len(duplicates)} duplicated unit ids" + (f": {duplicates[:5]}" if duplicates else ""),
        )

        known_docs = set(context.results.keys())
        orphan_docs = sorted(
            {
                str(unit.get("document_id"))
                for unit in units
                if unit.get("document_id") not in known_docs
            }
        )
        result.add(
            "R06",
            "every unit references an existing document_id",
            len(orphan_docs),
            len(units),
            f"{len(known_docs)} known documents",
        )

        orphan_pages = 0
        for unit in units:
            doc_id = unit.get("document_id")
            page_number = unit.get("page_number")
            page_id = unit.get("page_id")
            if not page_number or not page_id:
                orphan_pages += 1
                continue
            expected_page_id = f"{doc_id}_P{int(page_number):03d}"
            if page_id != expected_page_id:
                orphan_pages += 1
            elif int(page_number) > context.page_counts.get(doc_id, 0):
                orphan_pages += 1
        result.add(
            "R07",
            "every unit references a valid page",
            orphan_pages,
            len(units),
            "page_id must match <document_id>_P<page_number> and exist in the document",
        )

        # 8/9 ---------------------------------------------------------------
        raw_dir = self.config.out_path("raw_text_dir")
        cleaned_dir = self.config.out_path("cleaned_text_dir")
        raw_missing = [doc for doc in context.processed_ids if not (raw_dir / f"{doc}.txt").exists()]
        result.add(
            "R08",
            "raw text exists for every successfully processed document",
            len(raw_missing),
            len(context.processed_ids),
            f"{len(context.processed_ids) - len(raw_missing)}/{len(context.processed_ids)} raw files present",
        )
        cleaned_missing = [
            doc for doc in context.processed_ids if not (cleaned_dir / f"{doc}.txt").exists()
        ]
        result.add(
            "R09",
            "cleaned text exists for every successfully processed document",
            len(cleaned_missing),
            len(context.processed_ids),
            f"{len(context.processed_ids) - len(cleaned_missing)}/{len(context.processed_ids)} cleaned files present",
        )

        # 10 ----------------------------------------------------------------
        result.add(
            "R10",
            "corpus.jsonl can be loaded without errors",
            1 if jsonl_error else 0,
            1,
            jsonl_error or f"{len(units)} records parsed cleanly",
        )

        # 11/12 -------------------------------------------------------------
        negative_chars = [u.get("unit_id") for u in units if int(u.get("char_count", 0) or 0) < 0]
        result.add(
            "R11",
            "character counts are non-negative",
            len(negative_chars),
            len(units),
            f"{len(negative_chars)} negative char counts",
        )
        negative_words = [u.get("unit_id") for u in units if int(u.get("word_count", 0) or 0) < 0]
        result.add(
            "R12",
            "word counts are non-negative",
            len(negative_words),
            len(units),
            f"{len(negative_words)} negative word counts",
        )

        # 13 ----------------------------------------------------------------
        duplicate_executed = "exact_duplicate_count" in context.duplicate_summary
        inventory = read_csv_rows(self.config.results_path("file_inventory"))
        duplicates_flagged = sum(1 for row in inventory if str(row.get("is_duplicate", "")).lower() == "true")
        result.add(
            "R13",
            "duplicate detection has been executed",
            0 if duplicate_executed else 1,
            len(inventory),
            f"{duplicates_flagged} duplicate files flagged out of {len(inventory)}",
        )

        # 14 ----------------------------------------------------------------
        error_rows = read_csv_rows(self.config.results_path("errors"))
        result.add(
            "R14",
            "extraction errors are logged",
            0 if error_rows or not context.failed_ids else 1,
            len(context.errors),
            f"{len(error_rows)} error rows in errors.csv for {len(context.failed_ids)} failed documents",
        )

        # 15 ----------------------------------------------------------------
        modified = []
        for record in files:
            if not record.sha256 or record.status in {"ERROR", "INVALID"}:
                continue
            path = Path(record.source_path)
            if not path.exists():
                modified.append(f"{record.document_id}:missing")
                continue
            chunk = int(self.config.get("duplicates.hash_chunk_bytes", 1_048_576))
            current = sha256_file(path, chunk)
            if current != record.sha256:
                modified.append(f"{record.document_id}:hash_changed")
        result.add(
            "R15",
            "original PDFs were not modified",
            len(modified),
            len(files),
            f"{len(files) - len(modified)}/{len(files)} files byte-identical to their inventory hash",
        )

        # warnings -----------------------------------------------------------
        warnings = 0
        for row in context.results.values():
            warnings += len(getattr(row, "warnings", []) or [])
        low_text_pages = 0
        for doc in context.results.values():
            for page in getattr(doc, "page_validation", []) or []:
                if page.get("extraction_warning"):
                    low_text_pages += 1
        warnings += low_text_pages

        status = "PASS" if result.errors == 0 else "FAIL"
        documents_found = len(files)
        documents_processed = len(context.processed_ids)
        documents_failed = len(context.failed_ids)

        return {
            "status": status,
            "generated_at": None,  # filled by the runner
            "documents_found": documents_found,
            "documents_processed": documents_processed,
            "documents_failed": documents_failed,
            "validation_errors": result.errors,
            "warnings": warnings,
            "documents_discovered_total": documents_found,
            "failed_document_ids": list(context.failed_ids),
            "processed_document_ids": list(context.processed_ids),
            "duplicate_summary": context.duplicate_summary,
            "rules": result.rules,
            "rule_counts": {
                "passed": sum(1 for r in result.rules if r["status"] == "PASS"),
                "failed": sum(1 for r in result.rules if r["status"] == "FAIL"),
                "total": len(result.rules),
            },
            "pages_validated": page_checked,
            "units_validated": len(units),
        }


def write_rule_table(path: Path, rules: Sequence[Dict[str, Any]]) -> None:
    from ..core.utils import write_csv

    write_csv(path, list(rules), VALIDATION_COLUMNS)
