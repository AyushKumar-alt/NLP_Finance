"""Ingestion tests: discovery determinism, ids, source identification."""

from __future__ import annotations

from src.ingestion.discovery import build_inventory, discover_files
from src.ingestion.registry import RegistryBuilder


def test_discovery_finds_every_pdf_recursively(pipeline, synthetic_corpus):
    config = pipeline["config"]
    paths = discover_files(config)
    assert len(paths) == 3
    assert {p.name for p in paths} == {"echap01.pdf", "takeaways.pdf", "Chapter 02.pdf"}


def test_discovery_order_is_deterministic(pipeline):
    from pathlib import Path

    config = pipeline["config"]
    root = config.input_directory

    def relpaths():
        return [
            str(Path(p).relative_to(root)).replace("\\", "/").casefold()
            for p in discover_files(config)
        ]

    first, second = relpaths(), relpaths()
    assert first == second
    # Ordering key is the case-folded normalised relative path, then filename.
    assert first == sorted(first)


def test_ids_are_stable_and_deterministic(pipeline, synthetic_corpus):
    config = pipeline["config"]
    paths = discover_files(config)
    records = build_inventory(config, paths, 1 << 20)
    registry = RegistryBuilder(config)
    registry.assign(records)
    ids = {r.relative_path: r.document_id for r in records}

    assert set(ids.values()) == {"D01", "D02", "D03"}
    # A second, independent build must produce exactly the same mapping.
    again = build_inventory(config, discover_files(config), 1 << 20)
    RegistryBuilder(config).assign(again)
    assert {r.relative_path: r.document_id for r in again} == ids


def test_source_identification_groups_chapters_under_one_parent(pipeline, synthetic_corpus):
    config = pipeline["config"]
    records = build_inventory(config, discover_files(config), 1 << 20)
    registry = RegistryBuilder(config)
    registry.assign(records)
    by_name = {r.filename: r.source_id for r in records}

    assert by_name["echap01.pdf"] == "SRC01"
    assert by_name["takeaways.pdf"] == "SRC02"
    assert by_name["Chapter 02.pdf"] == "SRC03"

    sources = registry.source_rows()
    assert {row["source_id"] for row in sources} == {"SRC01", "SRC02", "SRC03"}
    es = next(row for row in sources if row["source_id"] == "SRC01")
    assert es["source_title"] == "Economic Survey 2025-26"


def test_every_file_gets_a_sha256_and_a_status(pipeline, synthetic_corpus):
    config = pipeline["config"]
    records = build_inventory(config, discover_files(config), 1 << 20)
    assert all(len(r.sha256) == 64 for r in records)
    assert all(r.status in {"OK", "INVALID", "ERROR"} for r in records)
