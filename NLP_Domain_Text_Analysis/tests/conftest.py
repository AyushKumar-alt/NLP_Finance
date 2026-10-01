"""Shared pytest fixtures: a synthetic financial PDF corpus + a temp config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "phase1_config.yaml"

BODY = (
    "1.1.\t India\u2019s real GDP growth is estimated at 7.4 per cent in FY26, with "
    "private final consumption expenditure (PFCE) remaining the largest contributor. "
    "The Reserve Bank of India (RBI) reported 2 per cent CPI inflation."
)

FOOTNOTE = (
    "11 National Bank for Agriculture and Rural Development (NABARD). (November, 2025). "
    "Rural Economic Conditions and Sentiments Survey (RECSS). (https://tinyurl.com/5n5ku8rm)"
)


def _wrap(text: str, width: int = 95) -> list:
    import textwrap

    return textwrap.wrap(text, width=width)


def _build_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((70, 60), "GLOBAL ECONOMIC GROWTH", fontsize=14, fontname="hebo")

    y = 90.0
    for line in _wrap(BODY):
        page.insert_text((70, y), line, fontsize=10, fontname="helv")
        y += 13.0

    page.insert_text((70, y + 10), "Table I.1: Revisions in global growth", fontsize=10)
    rows = [
        ("AE", "1.9", "1.4", "1.6"),
        ("EMDE", "4.2", "3.7", "4.2"),
        ("Global", "3.3", "3.2", "3.4"),
        ("India", "6.5", "7.1", "7.4"),
    ]
    y += 25.0
    for row in rows:
        page.insert_text((75, y), row[0], fontsize=9)
        page.insert_text((220, y), row[1], fontsize=9)
        page.insert_text((300, y), row[2], fontsize=9)
        page.insert_text((380, y), row[3], fontsize=9)
        y += 14.0
    page.insert_text((75, y + 10), "Source: IMF World Economic Outlook", fontsize=8)

    fy = 745.0
    for line in _wrap(FOOTNOTE, width=115):
        page.insert_text((70, fy), line, fontsize=7.5)
        fy += 10.0
    page.insert_text((540, 815), "2", fontsize=9)

    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory) -> Path:
    """``echap01.pdf`` (Economic Survey look) + a SEBI-style chapter."""
    import shutil

    root = tmp_path_factory.mktemp("pdfs")
    (root / "FINANCIAL STABILITY REPORT").mkdir()
    (root / "Sebi annual report").mkdir()
    _build_pdf(root / "echap01.pdf")
    _build_pdf(root / "FINANCIAL STABILITY REPORT" / "takeaways.pdf")
    _build_pdf(root / "Sebi annual report" / "Chapter 02.pdf")
    return root


@pytest.fixture()
def pipeline(tmp_path, monkeypatch, synthetic_corpus):
    """A :class:`Phase1Runner` wired to the synthetic corpus and a temp output.

    A copy of ``config/phase1_config.yaml`` is written into a throw-away project
    root, so ``Phase1Config.project_root`` - and therefore every output path -
    resolves inside ``tmp_path``. The real repository output is never touched.
    """
    import shutil

    from src.phase1.run import Phase1Runner

    project_root = tmp_path / "project"
    (project_root / "config").mkdir(parents=True)
    config_path = project_root / "config" / "phase1_config.yaml"
    shutil.copyfile(CONFIG_PATH, config_path)
    monkeypatch.setenv("NLP_PHASE1_INPUT_DIR", str(synthetic_corpus))

    runner = Phase1Runner(config_path=str(config_path))
    return {"runner": runner, "config": runner.config, "root": project_root}
