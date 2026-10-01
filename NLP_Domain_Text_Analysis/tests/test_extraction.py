"""Extraction / structure tests: typed units, traceability, table recovery."""

from __future__ import annotations

import json

from src.extraction.page_analyzer import PageAnalyzer
from src.extraction.pdf_reader import iter_page_geometry, open_document


def test_page_geometry_exposes_spans_and_tokens(pipeline, synthetic_corpus):
    from src.extraction.pdf_reader import iter_page_geometry, open_document

    doc = open_document(synthetic_corpus / "echap01.pdf")
    try:
        geometry = iter_page_geometry(doc, 0)
    finally:
        doc.close()
    assert geometry.raw_text.strip()
    assert geometry.tokens, "character-accurate tokens are required for table recovery"
    assert geometry.body_font_size > 0


def _unused() -> None:  # pragma: no cover - placeholder removed
    return None


def test_table_caption_and_grid_are_recovered(pipeline, synthetic_corpus):
    from src.extraction.page_analyzer import PageAnalyzer

    analyzer = PageAnalyzer(pipeline["config"])
    doc = open_document(synthetic_corpus / "echap01.pdf")
    try:
        geometry = iter_page_geometry(doc, 0)
    finally:
        doc.close()

    elements = analyzer.analyze(geometry)
    tables = [e for e in elements if e.kind == "table"]
    assert tables, "the Table I.1 caption must produce a table element"
    table = tables[0]
    assert table.caption.startswith("Table I.1")
    assert table.source_note.startswith("Source:")
    assert table.table_data, "a grid must be reconstructed"
    row = next(r for r in table.table_data if r and r[0] == "EMDE")
    assert row[0] == "EMDE"
    assert "4.2" in row and "3.7" in row


def test_numbers_percentages_and_currency_survive_the_pipeline(pipeline):
    summary = pipeline["runner"].run()
    corpus = pipeline["config"].out_path("structured_dir") / "corpus.jsonl"
    text = corpus.read_text(encoding="utf-8")
    assert "7.4 per cent" in text
    assert "FY26" in text
    assert "2 per cent" in text
    assert "PFCE" in text
    assert "RBI" in text
    assert "₹" in text or "Rs" in text or "%" in text


def test_footnote_and_url_are_preserved(pipeline):
    pipeline["runner"].run()
    corpus = pipeline["config"].out_path("structured_dir") / "corpus.jsonl"
    records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    footnotes = [r for r in records if r["unit_type"] == "footnote"]
    assert footnotes, "small type in the lower band must become a footnote unit"
    assert any("https://tinyurl.com/5n5ku8rm" in r["text"] for r in footnotes)


def test_units_are_traceable_to_document_page_and_section(pipeline):
    import csv

    pipeline["runner"].run()
    config = pipeline["config"]
    corpus = config.out_path("structured_dir") / "corpus.jsonl"
    records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    with open(config.metadata_path("document_registry"), encoding="utf-8-sig", newline="") as handle:
        registry = {row["document_id"] for row in csv.DictReader(handle)}

    assert records
    for record in records:
        assert record["unit_id"].startswith(record["document_id"])
        assert record["document_id"] in registry
        assert record["page_id"] == f"{record['document_id']}_P{record['page_number']:03d}"
        assert record["section_id"]
        assert record["char_count"] >= 0 and record["word_count"] >= 0
        assert record["text"], "a content unit must never carry empty text"


def test_unit_ids_are_unique(pipeline):
    pipeline["runner"].run()
    corpus = pipeline["config"].out_path("structured_dir") / "corpus.jsonl"
    ids = [
        json.loads(line)["unit_id"]
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(ids) == len(set(ids))


def test_section_detection_reads_numbered_headings(pipeline):
    pipeline["runner"].run()
    index = pipeline["config"].metadata_path("sections_index")
    rows = index.read_text(encoding="utf-8").splitlines()
    assert len(rows) > 1
    body = "\n".join(rows)
    assert "GLOBAL ECONOMIC GROWTH" in body
