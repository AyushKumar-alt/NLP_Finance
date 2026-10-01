"""Phase 1 orchestrator.

Single entry point::

    python -m src.phase1.run

Pipeline order (each stage is independently recoverable):

1.  discover + inventory the input PDFs
2.  assign stable document ids and identify parent sources
3.  open + validate every PDF (page count, text extractability, scanned pages)
4.  extract pages, detect sections, paragraphs, tables, figures, boxes, footnotes
5.  write raw / cleaned / page / table / figure artefacts
6.  build the master corpus ``corpus.jsonl`` + flattened TXT corpora
7.  write metadata registries
8.  compute Phase 1 baseline statistics and dictionaries
9.  run the 15 validation rules
10. print the Phase 1 summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # allow `python src/phase1/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import load_config
from src.core.utils import (
    log_event,
    setup_logging,
    sha256_text,
    utc_now,
    write_csv,
    write_json,
    write_jsonl,
)
from src.ingestion.discovery import INVENTORY_COLUMNS, build_inventory, discover_files, mark_duplicates
from src.ingestion.registry import RegistryBuilder
from src.statistics.stats import StatisticsCollector
from src.structure.document_processor import DocumentProcessor
from src.validation.validator import Phase1Validator, ValidationContext, write_rule_table

DOCUMENT_STATISTICS_COLUMNS = [
    "document_id",
    "source_id",
    "filename",
    "page_count",
    "sentences",
    "words",
    "characters",
    "baseline_token_count",
    "unique_words",
    "vocabulary_size",
    "raw_vocabulary_size",
    "average_document_length",
    "sentences_per_page",
    "sections",
    "units",
    "paragraphs",
    "tables",
    "figures",
    "footnotes",
    "raw_characters",
    "cleaned_characters",
]

QUALITY_COLUMNS = [
    "document_id",
    "source_id",
    "filename",
    "page_count",
    "pages_with_text",
    "pages_with_low_text",
    "pages_likely_scanned",
    "tables_detected",
    "figures_detected",
    "boxes_detected",
    "paragraphs_detected",
    "footnotes_detected",
    "sections_detected",
    "total_units",
    "total_characters",
    "total_words",
    "component_text_coverage",
    "component_page_completeness",
    "component_unit_density",
    "component_structure",
    "component_structural_fidelity",
    "quality_score",
    "quality_grade",
    "warnings",
]

PDF_VALIDATION_COLUMNS = [
    "document_id",
    "source_id",
    "filename",
    "page_number",
    "text_char_count",
    "word_count",
    "is_text_extractable",
    "is_likely_scanned",
    "image_count",
    "drawing_count",
    "unit_count",
    "extraction_warning",
]

ERROR_COLUMNS = [
    "document_id",
    "file",
    "stage",
    "error_type",
    "error_message",
    "recoverable",
]

DUPLICATE_COLUMNS = [
    "document_id",
    "filename",
    "relative_path",
    "sha256",
    "duplicate_kind",
    "duplicate_of",
    "similarity",
    "note",
]

UNIT_REGISTRY_COLUMNS = [
    "unit_id",
    "document_id",
    "source_id",
    "filename",
    "page_id",
    "page_number",
    "section_id",
    "section_number",
    "section_title",
    "unit_type",
    "unit_index",
    "char_count",
    "word_count",
]

CORPUS_INDEX_COLUMNS = [
    "unit_id",
    "document_id",
    "source_id",
    "filename",
    "page_number",
    "section_id",
    "unit_type",
    "char_count",
    "word_count",
    "jsonl_line",
]

SECTION_INDEX_COLUMNS = [
    "document_id",
    "source_id",
    "section_id",
    "section_number",
    "section_title",
    "page_start",
    "page_end",
    "detection_method",
    "confidence",
    "is_confirmed",
]

DOCUMENT_MARKER = "[DOCUMENT_ID={doc}]"
SOURCE_MARKER = "[SOURCE_ID={src}]"
PAGE_MARKER = "[PAGE={page}]"
SECTION_MARKER = "[SECTION={number} | {title}]"
UNIT_MARKER = "[UNIT={unit}]"
TYPE_MARKER = "[TYPE={kind}]"


class Phase1Runner:
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        self.config = load_config(config_path)
        self.logger = setup_logging(
            self.config.project_root / str(self.config.get("output.log_file", "logs/phase1.log")),
            verbose=verbose,
        )
        self.errors: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _record_error(
        self,
        document_id: str,
        file_path: str,
        stage: str,
        exc: BaseException,
        recoverable: bool = True,
    ) -> None:
        self.errors.append(
            {
                "document_id": document_id,
                "file": file_path,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "recoverable": recoverable,
            }
        )
        log_event(
            self.logger,
            "WARNING",
            document_id,
            stage,
            f"{type(exc).__name__}: {exc}",
        )

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        started = time.time()
        config = self.config
        log_event(self.logger, "INFO", "PIPELINE", "Phase 1 started", f"input={config.input_directory}")

        # ---- 0. clear the generated artefacts -------------------------
        # Only the directories this pipeline owns are purged, so a re-run can
        # never leave a stale unit behind when a document changes. The input
        # directory is never touched.
        if bool(config.get("output.clean_before_write", True)):
            self._clean_outputs()

        # ---- 1. discovery -------------------------------------------
        paths = discover_files(config)
        log_event(self.logger, "INFO", "PIPELINE", "Discovery", f"{len(paths)} PDF files discovered")
        hash_bytes = int(config.get("duplicates.hash_chunk_bytes", 1_048_576))
        try:
            files = build_inventory(config, paths, hash_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            self._record_error("PIPELINE", str(config.input_directory), "inventory", exc, recoverable=False)
            raise

        # ---- 2. document ids + sources ------------------------------
        registry = RegistryBuilder(config)
        registry.assign(files)
        duplicate_summary = mark_duplicates(files)
        log_event(
            self.logger,
            "INFO",
            "PIPELINE",
            "Source identification",
            f"{len(registry.sources)} source groups, {duplicate_summary['exact_duplicate_count']} exact duplicates",
        )

        # ---- 3/4. extraction ----------------------------------------
        processor = DocumentProcessor(config, self.logger)
        results: Dict[str, Any] = {}
        for record in files:
            if record.status == "INVALID":
                results[record.document_id] = _empty_result(record.document_id, record.source_id)
                self.errors.append(
                    {
                        "document_id": record.document_id,
                        "file": record.source_path,
                        "stage": "validation",
                        "error_type": "InvalidFile",
                        "error_message": record.error_message,
                        "recoverable": True,
                    }
                )
                continue
            try:
                result = processor.process(record)
            except Exception as exc:  # one bad PDF must never stop the run
                self._record_error(record.document_id, record.source_path, "document_processing", exc)
                result = _empty_result(record.document_id, record.source_id)
                result.status = "FAILED"
                result.warnings.append(f"{type(exc).__name__}: {exc}")
            results[record.document_id] = result

        processed_ids = [doc for doc, res in results.items() if res.status == "SUCCESS"]
        failed_ids = [doc for doc, res in results.items() if res.status != "SUCCESS"]
        for doc in failed_ids:
            log_event(self.logger, "WARNING", doc, "Extraction status", f"status={results[doc].status}")

        # ---- 5. artefacts -------------------------------------------
        self._write_text_artifacts(files, results)
        self._write_page_artifacts(results)
        self._write_table_artifacts(results)
        self._write_figure_artifacts(results)
        self._write_results_csvs(files, results)

        # ---- 6. master corpus + flattened TXT -----------------------
        unit_records = self._write_corpus(files, results)
        self._write_flattened_txt(files, results)

        # ---- 7. metadata registries ---------------------------------
        quality_map = self._quality_map(files, results)
        extraction_status = {doc: res.status for doc, res in results.items()}
        registry.persist(extraction_status, quality_map)
        self._write_unit_registry(files, results)

        # ---- 8. statistics ------------------------------------------
        collector = StatisticsCollector(config)
        document_stats = []
        for record in files:
            result = results.get(record.document_id)
            if result is None or result.status != "SUCCESS":
                continue
            document_stats.append(collector.add_document(result, record))
        corpus_totals = collector.corpus_totals(document_stats)
        write_csv(
            config.results_path("document_statistics"),
            [vars(stats) for stats in document_stats],
            DOCUMENT_STATISTICS_COLUMNS,
        )
        vocab_limit = int(config.get("statistics.vocabulary_max_rows", 200_000))
        write_csv(config.metadata_path("vocabulary_baseline"), collector.vocabulary_rows(vocab_limit),
                  ["term", "frequency", "document_frequency", "documents"])
        term_limit = int(config.get("statistics.term_dictionary_max_rows", 200_000))
        write_csv(config.metadata_path("baseline_term_dictionary"), collector.term_dictionary_rows(term_limit),
                  ["term", "frequency", "document_frequency", "documents"])
        self._write_duplicate_report(files, results, duplicate_summary)

        # ---- 9. validation ------------------------------------------
        validator = Phase1Validator(config)
        context = ValidationContext(
            config=config,
            files=files,
            results=results,
            unit_counts={},
            page_counts={doc: len(res.pages) for doc, res in results.items()},
            errors=self.errors,
            duplicate_summary=duplicate_summary,
            processed_ids=processed_ids,
            failed_ids=failed_ids,
        )
        report = validator.run(context)
        report["generated_at"] = utc_now()
        report["input_directory"] = str(config.input_directory)
        report["extraction_library"] = str(config.get("extraction.extraction_library"))
        report["quality_score_formula"] = (
            "100 * (0.30*text_coverage + 0.20*page_completeness + 0.15*unit_density "
            "+ 0.15*structure + 0.20*structural_fidelity)"
        )
        write_json(config.results_path("validation_report"), report)
        write_rule_table(config.results_path("validation_report").with_name("validation_rules.csv"), report["rules"])

        # ---- 10. summary + manifest ---------------------------------
        summary = self._summary(
            files=files,
            results=results,
            registry=registry,
            duplicate_summary=duplicate_summary,
            corpus_totals=corpus_totals,
            unit_records=unit_records,
            report=report,
            elapsed=time.time() - started,
        )
        write_json(config.out_path("manifests_dir") / str(config.get("artifacts.run_manifest", "run_manifest.json")), summary["manifest"])
        write_json(config.out_path("manifests_dir") / str(config.get("artifacts.source_manifest", "source_manifest.json")), summary["source_manifest"])
        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # artefact writers
    # ------------------------------------------------------------------
    def _clean_outputs(self) -> None:
        import shutil

        managed = (
            "raw_text_dir", "cleaned_text_dir", "structured_dir", "pages_dir",
            "tables_dir", "figures_dir", "metadata_dir", "manifests_dir",
            "results_dir",
        )
        input_dir = self.config.input_directory
        for key in managed:
            directory = self.config.out_path(key)
            # Safety net: never purge the read-only input directory.
            if directory == input_dir or input_dir in directory.parents:
                raise RuntimeError(f"refusing to clean {directory}: inside the input directory")
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        log_event(self.logger, "INFO", "PIPELINE", "Clean outputs", "generated artefact folders cleared")

    def _write_text_artifacts(self, files, results) -> None:
        raw_dir = self.config.out_path("raw_text_dir")
        cleaned_dir = self.config.out_path("cleaned_text_dir")
        for record in files:
            result = results.get(record.document_id)
            if result is None or result.status != "SUCCESS":
                continue
            (raw_dir / f"{record.document_id}.txt").write_text(result.raw_text, encoding="utf-8")
            (cleaned_dir / f"{record.document_id}.txt").write_text(result.cleaned_text, encoding="utf-8")

    def _write_page_artifacts(self, results) -> None:
        page_dir = self.config.out_path("pages_dir")
        for result in results.values():
            for page in result.pages:
                write_json(page_dir / f"{page.page_id}.json", page.to_dict(include_text=True))

    def _write_table_artifacts(self, results) -> None:
        table_dir = self.config.out_path("tables_dir")
        for result in results.values():
            for table in result.tables:
                write_json(
                    table_dir / f"{table.table_id}.json",
                    {
                        "table_id": table.table_id,
                        "document_id": table.document_id,
                        "source_id": table.source_id,
                        "page_number": table.page_number,
                        "page_start": table.page_start,
                        "page_end": table.page_end,
                        "section_id": table.section_id,
                        "label": table.label,
                        "caption": table.caption,
                        "source_note": table.source_note,
                        "rows": table.rows,
                        "columns": table.columns,
                        "data_extracted": table.data_extracted,
                        "table_extraction_quality": table.quality,
                        "quality_reason": table.quality_reason,
                        "body_continues_on_next_page": table.page_end > table.page_start,
                        "bbox": table.bbox,
                        "raw_text": table.raw_text,
                        "table_data": table.table_data,
                    },
                )

    def _write_figure_artifacts(self, results) -> None:
        figure_dir = self.config.out_path("figures_dir")
        for result in results.values():
            for figure in result.figures:
                write_json(
                    figure_dir / f"{figure.figure_id}.json",
                    {
                        "figure_id": figure.figure_id,
                        "document_id": figure.document_id,
                        "source_id": figure.source_id,
                        "page_number": figure.page_number,
                        "section_id": figure.section_id,
                        "label": figure.label,
                        "kind": figure.kind,
                        "caption": figure.caption,
                        "source_note": figure.source_note,
                        "surrounding_text": figure.surrounding_text,
                        "axis_labels": figure.axis_labels,
                        "data_extracted": figure.data_extracted,
                        "data_note": figure.data_note,
                        "bbox": figure.bbox,
                    },
                )

    def _write_results_csvs(self, files, results) -> None:
        config = self.config

        # file inventory ------------------------------------------------
        write_csv(config.results_path("file_inventory"), [f.row() for f in files], INVENTORY_COLUMNS)

        # per page validation -------------------------------------------
        validation_rows: List[Dict[str, Any]] = []
        for record in files:
            result = results.get(record.document_id)
            if result is None:
                continue
            for row in result.page_validation:
                validation_rows.append(
                    {
                        "document_id": record.document_id,
                        "source_id": record.source_id,
                        "filename": record.filename,
                        **row,
                    }
                )
        write_csv(config.results_path("pdf_validation"), validation_rows, PDF_VALIDATION_COLUMNS)

        # errors ---------------------------------------------------------
        for record in files:
            result = results.get(record.document_id)
            if result is None:
                continue
            for warning in result.warnings:
                self.errors.append(
                    {
                        "document_id": record.document_id,
                        "file": record.source_path,
                        "stage": "extraction_warning",
                        "error_type": "ExtractionWarning",
                        "error_message": warning,
                        "recoverable": True,
                    }
                )
        write_csv(config.results_path("errors"), self.errors, ERROR_COLUMNS)

        # extraction quality report --------------------------------------
        quality_rows = self._quality_rows(files, results)
        write_csv(config.results_path("extraction_quality_report"), quality_rows, QUALITY_COLUMNS)

        # sections index --------------------------------------------------
        section_rows: List[Dict[str, Any]] = []
        for result in results.values():
            for section in result.sections:
                section_rows.append(section.to_dict())
        section_rows.sort(key=lambda row: (str(row["document_id"]), int(row["page_start"]), float(row["start_y"])))
        write_csv(
            config.metadata_path("sections_index"),
            section_rows,
            SECTION_INDEX_COLUMNS,
        )

    # ------------------------------------------------------------------
    def _quality_rows(self, files, results) -> List[Dict[str, Any]]:
        weights = self.config.section("quality").get("weights", {})
        w_text = float(weights.get("text_coverage", 0.30))
        w_page = float(weights.get("page_completeness", 0.20))
        w_unit = float(weights.get("unit_density", 0.15))
        w_struct = float(weights.get("structure", 0.15))
        w_fidelity = float(weights.get("structural_fidelity", 0.20))
        min_units_per_page = int(self.config.get("quality.min_units_per_page", 1))
        low_char = int(self.config.get("validation.low_text_char_threshold", 120))
        thresholds = self.config.section("quality").get("grade_thresholds", {}) or {}

        rows: List[Dict[str, Any]] = []
        for record in files:
            result = results.get(record.document_id)
            if result is None:
                continue
            page_count = max(1, len(result.pages))
            pages_with_text = sum(
                1 for p in result.page_validation
                if int(p.get("text_char_count", 0)) >= low_char
            )
            pages_low_text = sum(
                1 for p in result.page_validation
                if 0 < int(p.get("text_char_count", 0)) < low_char
            )
            pages_scanned = sum(1 for p in result.page_validation if p.get("is_likely_scanned"))
            total_units = len(result.units)

            # Component 1 - text coverage: share of pages that carry a real
            # block of prose (not merely any character).
            text_coverage = pages_with_text / page_count
            # Component 2 - page completeness: share of pages that yielded at
            # least one typed content unit.
            page_completeness = sum(
                1 for page in result.pages if page.unit_count >= min_units_per_page
            ) / page_count
            # Component 3 - unit density: content units per page, saturating at
            # ``quality_unit_density_target`` units per page.
            target = float(self.config.get("quality.unit_density_target", 6.0))
            per_page = total_units / page_count if page_count else 0.0
            unit_density = min(1.0, per_page / target) if target else 0.0
            # Component 4 - structure: one structural element (table, figure,
            # box or section) per ``quality_structure_per_element_pages`` pages.
            per_element_pages = float(
                self.config.get("quality.structure_per_element_pages", 4.0)
            )
            structural = (
                len(result.tables)
                + len(result.figures)
                + result.counters.get("box", 0)
                + len(result.sections) / 2.0
            )
            structure = min(1.0, structural / (page_count / per_element_pages))
            # Component 5 - structural fidelity: how many detected tables had
            # their numeric cell layout reconstructed at HIGH quality. A
            # document without tables is neutral (1.0) rather than penalised.
            if result.tables:
                high = sum(1 for t in result.tables if t.quality == "HIGH")
                medium = sum(1 for t in result.tables if t.quality == "MEDIUM")
                fidelity = (high + 0.5 * medium) / len(result.tables)
            else:
                fidelity = 1.0
            score = 100.0 * (
                w_text * text_coverage
                + w_page * page_completeness
                + w_unit * unit_density
                + w_struct * structure
                + w_fidelity * fidelity
            )
            grade = "D"
            for label, limit in sorted(thresholds.items(), key=lambda kv: -float(kv[1])):
                if score >= float(limit):
                    grade = label
                    break
            counters = result.counters
            rows.append(
                {
                    "document_id": record.document_id,
                    "source_id": record.source_id,
                    "filename": record.filename,
                    "page_count": len(result.pages),
                    "pages_with_text": pages_with_text,
                    "pages_with_low_text": pages_low_text,
                    "pages_likely_scanned": pages_scanned,
                    "tables_detected": len(result.tables),
                    "figures_detected": len(result.figures),
                    "boxes_detected": counters.get("box", 0),
                    "paragraphs_detected": counters.get("paragraph", 0),
                    "footnotes_detected": counters.get("footnote", 0) + counters.get("reference", 0),
                    "sections_detected": len(result.sections),
                    "total_units": total_units,
                    "total_characters": sum(p.char_count for p in result.pages),
                    "total_words": sum(p.word_count for p in result.pages),
                    "component_text_coverage": round(text_coverage, 4),
                    "component_page_completeness": round(page_completeness, 4),
                    "component_unit_density": round(unit_density, 4),
                    "component_structure": round(structure, 4),
                    "component_structural_fidelity": round(fidelity, 4),
                    "quality_score": round(score, 2),
                    "quality_grade": grade,
                    "warnings": "; ".join(result.warnings) if result.warnings else "",
                }
            )
        return rows

    def _quality_map(self, files, results) -> Dict[str, str]:
        return {row["document_id"]: f"{row['quality_grade']} ({row['quality_score']})" for row in self._quality_rows(files, results)}

    # ------------------------------------------------------------------
    def _write_corpus(self, files, results) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        index_rows: List[Dict[str, Any]] = []
        for record in files:
            result = results.get(record.document_id)
            if result is None or result.status != "SUCCESS":
                continue
            for unit in result.units:
                payload = unit.to_jsonl_record()
                records.append(payload)
                index_rows.append(
                    {
                        "unit_id": payload["unit_id"],
                        "document_id": payload["document_id"],
                        "source_id": payload["source_id"],
                        "filename": payload.get("filename", ""),
                        "page_number": payload["page_number"],
                        "section_id": payload["section_id"],
                        "unit_type": payload["unit_type"],
                        "char_count": payload["char_count"],
                        "word_count": payload["word_count"],
                        "jsonl_line": len(records),
                    }
                )
        path = self.config.out_path("structured_dir") / str(self.config.get("artifacts.corpus_jsonl", "corpus.jsonl"))
        count = write_jsonl(path, records)
        log_event(self.logger, "INFO", "PIPELINE", "Corpus build", f"{count} content units written to corpus.jsonl")
        write_csv(self.config.metadata_path("corpus_index"), index_rows, CORPUS_INDEX_COLUMNS)
        return records

    def _write_unit_registry(self, files, results) -> None:
        rows: List[Dict[str, Any]] = []
        for record in files:
            result = results.get(record.document_id)
            if result is None or result.status != "SUCCESS":
                continue
            for unit in result.units:
                rows.append(
                    {
                        "unit_id": unit.unit_id,
                        "document_id": unit.document_id,
                        "source_id": unit.source_id,
                        "filename": record.filename,
                        "page_id": unit.page_id,
                        "page_number": unit.page_number,
                        "section_id": unit.section_id,
                        "section_number": unit.section_number,
                        "section_title": unit.section_title,
                        "unit_type": unit.unit_type,
                        "unit_index": unit.unit_index,
                        "char_count": unit.char_count,
                        "word_count": unit.word_count,
                    }
                )
        write_csv(self.config.metadata_path("unit_registry"), rows, UNIT_REGISTRY_COLUMNS)

    def _write_flattened_txt(self, files, results) -> None:
        """Combined TXT corpus with full traceability markers."""
        cleaned_dir = self.config.out_path("cleaned_text_dir")
        combined_path = cleaned_dir / str(self.config.get("artifacts.all_documents_txt", "all_documents.txt"))
        sections: List[str] = []
        for record in files:
            result = results.get(record.document_id)
            if result is None or result.status != "SUCCESS":
                continue
            sections.append(self._document_markdown(result))
        combined_path.write_text("\n\n".join(sections), encoding="utf-8")

    def _document_markdown(self, result) -> str:
        lines: List[str] = [
            "=" * 78,
            DOCUMENT_MARKER.format(doc=result.document_id),
            SOURCE_MARKER.format(src=result.source_id),
            f"[FILENAME={next((u.filename for u in result.units if u.filename), '')}]",
            "=" * 78,
            "",
        ]
        current_page = None
        for unit in result.units:
            if unit.page_number != current_page:
                current_page = unit.page_number
                lines.append("")
                lines.append("-" * 78)
                lines.append(PAGE_MARKER.format(page=current_page))
            lines.append(SECTION_MARKER.format(number=unit.section_number, title=unit.section_title))
            lines.append(UNIT_MARKER.format(unit=unit.unit_id))
            lines.append(TYPE_MARKER.format(kind=unit.unit_type))
            if unit.caption:
                lines.append(f"[CAPTION={unit.caption}]")
            if unit.source_note:
                lines.append(f"[SOURCE_NOTE={unit.source_note}]")
            if unit.label:
                lines.append(f"[LABEL={unit.label}]")
            if unit.table_data:
                for row in unit.table_data:
                    lines.append(" | ".join(str(cell) for cell in row))
                if unit.table_extraction_quality:
                    lines.append(f"[TABLE_QUALITY={unit.table_extraction_quality}]")
            if unit.axis_labels:
                lines.append(f"[AXIS_LABELS] {' ; '.join(unit.axis_labels)}")
            lines.append(unit.text)
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _write_duplicate_report(self, files, results, duplicate_summary) -> None:
        rows: List[Dict[str, Any]] = []
        if bool(self.config.get("duplicates.detect_extracted_text_duplicates", True)):
            by_hash: Dict[str, str] = {}
            min_chars = int(self.config.get("duplicates.near_duplicate_min_chars", 5000))
            ratio_threshold = float(self.config.get("duplicates.near_duplicate_min_ratio", 0.995))
            signatures: List[tuple] = []
            for record in files:
                result = results.get(record.document_id)
                if result is None or result.status != "SUCCESS" or len(result.raw_text) < min_chars:
                    continue
                fingerprint = sha256_text(result.raw_text[:200_000])
                signatures.append((record.document_id, result.raw_text, fingerprint))
                if fingerprint in by_hash and by_hash[fingerprint] != record.document_id:
                    rows.append(
                        {
                            "document_id": record.document_id,
                            "filename": record.filename,
                            "relative_path": record.relative_path,
                            "sha256": record.sha256,
                            "duplicate_kind": "extracted_text_identical",
                            "duplicate_of": by_hash[fingerprint],
                            "similarity": 1.0,
                            "note": "extracted raw text is byte-identical to another document",
                        }
                    )
                else:
                    by_hash[fingerprint] = record.document_id
            # cheap shingle based near-duplicate probe over document openings
            shingle_len = 200
            index: Dict[str, str] = {}
            for doc_id, text, _fingerprint in signatures:
                head = text[:20000]
                for start in range(0, max(1, len(head) - shingle_len), shingle_len):
                    key = sha256_text(head[start:start + shingle_len])
                    index.setdefault(key, doc_id)
            for doc_id, text, _fingerprint in signatures:
                head = text[:20000]
                shingles = [head[i:i + shingle_len] for i in range(0, max(1, len(head) - shingle_len), shingle_len)]
                if not shingles:
                    continue
                owners: Dict[str, int] = {}
                for shingle in shingles:
                    owner = index.get(sha256_text(shingle))
                    if owner and owner != doc_id:
                        owners[owner] = owners.get(owner, 0) + 1
                for owner, hits in owners.items():
                    similarity = hits / len(shingles)
                    if similarity >= ratio_threshold:
                        rows.append(
                            {
                                "document_id": doc_id,
                                "filename": next(f.filename for f in files if f.document_id == doc_id),
                                "relative_path": next(f.relative_path for f in files if f.document_id == doc_id),
                                "sha256": next(f.sha256 for f in files if f.document_id == doc_id),
                                "duplicate_kind": "extracted_text_near_duplicate",
                                "duplicate_of": owner,
                                "similarity": round(similarity, 4),
                                "note": "opening 20000 characters share the same 200-char shingles",
                            }
                        )
        write_csv(self.config.results_path("duplicate_report"), rows, DUPLICATE_COLUMNS)

    # ------------------------------------------------------------------
    def _summary(self, files, results, registry, duplicate_summary, corpus_totals, unit_records, report, elapsed) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        for unit in unit_records:
            type_counts[unit["unit_type"]] = type_counts.get(unit["unit_type"], 0) + 1
        total_tables = sum(1 for u in unit_records if u["unit_type"] == "table")
        total_figures = sum(1 for u in unit_records if u["unit_type"] == "figure")
        warning_count = sum(len(r.warnings) for r in results.values())
        warning_count += sum(1 for e in self.errors if e["error_type"] == "ExtractionWarning")
        # Page-level extraction warnings (low text density, likely scanned pages,
        # typed-but-unusable pages) are warnings too and must be reported.
        page_warnings = sum(
            1
            for r in results.values()
            for page in r.page_validation
            if page.get("extraction_warning")
        )
        warning_count += page_warnings

        manifest = {
            "phase": "phase1",
            "project": str(self.config.get("run.project_name")),
            "input_directory": str(self.config.input_directory),
            "config_path": str(self.config.config_path),
            "extraction_library": str(self.config.get("extraction.extraction_library")),
            "generated_at": utc_now(),
            "elapsed_seconds": round(elapsed, 2),
            "documents_found": len(files),
            "documents_processed": report["documents_processed"],
            "documents_failed": report["documents_failed"],
            "sources_detected": len([s for s in registry.sources.values()]),
            "source_groups": {sid: rec.files_count for sid, rec in registry.sources.items()},
            "exact_duplicates": duplicate_summary.get("exact_duplicate_count", 0),
            "total_pages": sum(len(r.pages) for r in results.values()),
            "total_units": len(unit_records),
            "unit_type_counts": type_counts,
            "total_tables": total_tables,
            "total_figures": total_figures,
            "total_sections": sum(len(r.sections) for r in results.values()),
            "corpus_totals": corpus_totals,
            "extraction_warnings": warning_count,
            "validation_status": report["status"],
            "validation_errors": report["validation_errors"],
            "sentence_tokenizer": corpus_totals.get("sentence_tokenizer_method"),
        }
        source_manifest = {
            "generated_at": utc_now(),
            "sources": [
                {
                    **source.row(),
                    "documents": [
                        {
                            "document_id": doc_id,
                            "filename": next(f.filename for f in files if f.document_id == doc_id),
                            "page_count": next(f.page_count for f in files if f.document_id == doc_id),
                        }
                        for doc_id in source.document_ids
                        if doc_id in results
                    ],
                }
                for source in registry.sources.values()
            ],
        }
        summary = {
            "manifest": manifest,
            "source_manifest": source_manifest,
            "files": files,
            "results": results,
            "corpus_totals": corpus_totals,
            "report": report,
            "duplicate_summary": duplicate_summary,
            "type_counts": type_counts,
            "warning_count": warning_count,
        }
        return summary

    # ------------------------------------------------------------------
    def _print_summary(self, summary) -> None:
        manifest = summary["manifest"]
        totals = summary["corpus_totals"]
        report = summary["report"]
        results = summary["results"]
        table_qualities: Dict[str, int] = {}
        figure_data = 0
        for result in results.values():
            for table in result.tables:
                table_qualities[table.quality] = table_qualities.get(table.quality, 0) + 1
            figure_data += sum(1 for f in result.figures if f.data_extracted)

        lines = [
            "",
            "=" * 74,
            "PHASE 1 COMPLETE",
            "=" * 74,
            f"Input directory:        {manifest['input_directory']}",
            f"Extraction library:     {manifest['extraction_library']}",
            f"PDF files discovered:   {manifest['documents_found']}",
            f"Successfully processed: {manifest['documents_processed']}",
            f"Failed:                 {manifest['documents_failed']}",
            f"Exact duplicates:       {manifest['exact_duplicates']}",
            f"Sources detected:       {manifest['sources_detected']}",
            f"Total pages:            {manifest['total_pages']}",
            f"Total sections:         {manifest['total_sections']}",
            f"Total paragraphs:       {summary['type_counts'].get('paragraph', 0)}",
            f"Total tables:           {summary['type_counts'].get('table', 0)}",
            f"Total figures:          {summary['type_counts'].get('figure', 0)}",
            f"Total boxes:            {summary['type_counts'].get('box', 0)}",
            f"Total footnotes:        {summary['type_counts'].get('footnote', 0)}",
            f"Total references:       {summary['type_counts'].get('reference', 0)}",
            f"Total headings:         {summary['type_counts'].get('heading', 0)}",
            f"Structured units:       {manifest['total_units']}",
            f"Total characters:       {totals['characters']}",
            f"Total words:            {totals['words']}",
            f"Total sentences:        {totals['sentences']} (via {totals['sentence_tokenizer_method']})",
            f"Baseline tokens:        {totals['baseline_token_count']}",
            f"Baseline vocabulary:    {totals['baseline_vocabulary_size']}",
            f"Raw vocabulary:         {totals['raw_vocabulary_size']}",
            f"Extraction warnings:    {summary['warning_count']}",
            f"Table quality:          {table_qualities or '{}'}",
            f"Figures with data:      {figure_data}",
            f"Validation:             {report['status']} "
            f"({report['rule_counts']['passed']}/{report['rule_counts']['total']} rules passed, "
            f"{report['validation_errors']} failures)",
            "",
            "Master corpus:          data/corpus/structured/corpus.jsonl",
            "Flat TXT corpus:        data/corpus/cleaned/all_documents.txt",
            "Validation report:      results/phase1/validation_report.json",
            f"Elapsed:                {manifest['elapsed_seconds']}s",
            "=" * 74,
            "",
        ]
        print("\n".join(lines))


def _empty_result(document_id: str, source_id: str):
    from src.structure.document_processor import DocumentResult

    return DocumentResult(document_id=document_id, source_id=source_id, status="NOT_PROCESSED")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.phase1.run",
        description="Phase 1: PDF corpus engineering for the NLP domain-specific corpus.",
    )
    parser.add_argument("--config", default=None, help="path to phase1_config.yaml")
    parser.add_argument("--input", default=None, help="override the configured input directory")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging on the console")
    args = parser.parse_args(argv)

    if args.input:
        import os

        os.environ["NLP_PHASE1_INPUT_DIR"] = args.input

    runner = Phase1Runner(config_path=args.config, verbose=args.verbose)
    summary = runner.run()
    return 0 if summary["report"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
