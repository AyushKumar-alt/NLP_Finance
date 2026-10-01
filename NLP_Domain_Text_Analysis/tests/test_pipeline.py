"""End-to-end pipeline + validation tests."""

from __future__ import annotations

import csv
import json


def _rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pipeline_produces_every_expected_artifact(pipeline):
    summary = pipeline["runner"].run()
    config = pipeline["config"]

    for key in ("file_inventory", "pdf_validation", "document_statistics",
                "extraction_quality_report", "errors", "validation_report"):
        assert config.results_path(key).exists(), f"missing results artefact: {key}"
    for key in ("document_registry", "source_registry", "unit_registry",
                "baseline_term_dictionary", "vocabulary_baseline"):
        assert config.metadata_path(key).exists(), f"missing metadata artefact: {key}"
    assert (config.out_path("structured_dir") / "corpus.jsonl").exists()
    assert (config.out_path("cleaned_text_dir") / "all_documents.txt").exists()
    assert summary["manifest"]["documents_found"] == 3


def test_page_table_and_raw_artefacts_exist_per_document(pipeline):
    pipeline["runner"].run()
    config = pipeline["config"]
    registry = _rows(config.metadata_path("document_registry"))
    for row in registry:
        doc_id = row["document_id"]
        assert (config.out_path("raw_text_dir") / f"{doc_id}.txt").exists()
        assert (config.out_path("cleaned_text_dir") / f"{doc_id}.txt").exists()
        for page_number in range(1, int(row["page_count"]) + 1):
            assert (config.out_path("pages_dir") / f"{doc_id}_P{page_number:03d}.json").exists()


def test_validation_report_is_computed_not_hard_coded(pipeline):
    summary = pipeline["runner"].run()
    report = json.loads(
        (pipeline["config"].results_path("validation_report")).read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["documents_found"] == 3
    assert report["documents_processed"] == 3
    assert report["documents_failed"] == 0
    assert report["validation_errors"] == 0
    assert report["rule_counts"]["total"] == 15
    assert report["rule_counts"]["failed"] == 0
    # Every number must be traceable back to the run.
    assert report["documents_found"] == summary["manifest"]["documents_found"]


def test_validation_proves_the_pdfs_were_not_modified(pipeline, synthetic_corpus):
    pipeline["runner"].run()
    report = json.loads(
        (pipeline["config"].results_path("validation_report")).read_text(encoding="utf-8")
    )
    rule15 = next(r for r in report["rules"] if r["rule_id"] == "R15")
    assert rule15["status"] == "PASS"
    assert rule15["checked_items"] == 3
    assert all(p.exists() for p in synthetic_corpus.rglob("*.pdf"))


def test_inventory_records_every_discovered_file(pipeline, synthetic_corpus):
    pipeline["runner"].run()
    rows = _rows(pipeline["config"].results_path("file_inventory"))
    assert len(rows) == len(list(synthetic_corpus.rglob("*.pdf")))
    for row in rows:
        assert row["document_id"] and row["sha256"] and row["status"]
        assert row["source_id"] in {"SRC01", "SRC02", "SRC03"}


def test_flattened_txt_keeps_traceability_markers(pipeline):
    pipeline["runner"].run()
    text = (pipeline["config"].out_path("cleaned_text_dir") / "all_documents.txt").read_text(
        encoding="utf-8"
    )
    assert "[DOCUMENT_ID=" in text
    assert "[SOURCE_ID=" in text
    assert "[PAGE=" in text
    assert "[SECTION=" in text
    assert "[UNIT=" in text


def test_a_single_bad_pdf_does_not_stop_the_pipeline(pipeline, synthetic_corpus, monkeypatch):
    from src.extraction import pdf_reader
    from src.extraction.pdf_reader import PdfOpenError

    real_open = pdf_reader.open_document
    broken = synthetic_corpus / "Sebi annual report" / "Chapter 02.pdf"

    def flaky(path):
        if str(path) == str(broken):
            raise PdfOpenError("simulated corrupt document")
        return real_open(path)

    monkeypatch.setattr(pdf_reader, "open_document", flaky)
    monkeypatch.setattr(
        "src.structure.document_processor.open_document", flaky
    )

    summary = pipeline["runner"].run()
    assert summary["manifest"]["documents_failed"] == 1
    assert summary["manifest"]["documents_processed"] == 2
    errors = _rows(pipeline["config"].results_path("errors"))
    assert any("simulated corrupt document" in row["error_message"] for row in errors)
    # The corpus of the surviving documents is still usable.
    corpus = (pipeline["config"].out_path("structured_dir") / "corpus.jsonl").read_text(
        encoding="utf-8"
    )
    assert corpus.strip()
