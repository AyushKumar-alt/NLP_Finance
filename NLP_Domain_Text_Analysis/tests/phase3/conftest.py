"""Fixtures for the Phase 3 tests: a synthetic corpus and an in-memory config.

Nothing here touches the repository's ``results`` directories. The Phase 3
configuration is loaded from the real file and overridden in memory (smaller n-gram
threshold, smaller result cap), while ``project_root`` still points at the
repository so that the read-only Phase 2 configuration can be found.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2.load_corpus import Corpus, Document, Unit  # noqa: E402
from src.phase3.config import load_config  # noqa: E402

UNIT_TEXTS: List[tuple] = [
    (
        "D01_P001_SEC_1_PAR001",
        "D01",
        "SRC01",
        1,
        "1",
        "Overview",
        "Real GDP growth is estimated at 7.4 per cent in FY26 as inflation moderated. "
        "The Reserve Bank of India kept the repo rate unchanged.",
    ),
    (
        "D01_P002_SEC_2_PAR002",
        "D01",
        "SRC01",
        2,
        "2",
        "Prices",
        "CPI inflation eased to 4.1 per cent in FY26. Monetary policy remained accommodative "
        "and the Reserve Bank of India reviewed the repo rate and the policy rate.",
    ),
    (
        "D01_P003_SEC_3_PAR003",
        "D01",
        "SRC01",
        3,
        "3",
        "Savings",
        "Household savings and investments rose, supporting financial stability in the "
        "banking sector.",
    ),
    (
        "D02_P004_SEC_1_PAR004",
        "D02",
        "SRC02",
        4,
        "1",
        "Overview",
        "Banking sector credit growth picked up. Insurance premiums and lending to "
        "industry also increased during the year.",
    ),
    (
        "D02_P005_SEC_2_PAR005",
        "D02",
        "SRC02",
        5,
        "2",
        "External",
        "The current account deficit narrowed to 1.2 per cent of GDP while foreign direct "
        "investment inflows improved.",
    ),
    (
        "D03_P006_SEC_1_PAR006",
        "D03",
        "SRC03",
        6,
        "1",
        "Markets",
        "Disinflation continues. Inflation expectations are anchored and the financial "
        "stability of institutions remains sound.",
    ),
]


def make_unit(
    unit_id: str,
    document_id: str,
    source_id: str,
    page_number: int,
    section_number: str,
    section_title: str,
    text: str,
) -> Unit:
    return Unit(
        unit_id=unit_id,
        document_id=document_id,
        source_id=source_id,
        filename=f"{document_id}.pdf",
        page_id=unit_id.split("_PAR")[0],
        page_number=page_number,
        section_id=f"{unit_id.split('_PAR')[0]}_SEC_{section_number}",
        section_number=section_number,
        section_title=section_title,
        unit_type="paragraph",
        unit_index=1,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


@pytest.fixture(scope="session")
def synthetic_units() -> List[Unit]:
    return [make_unit(*entry) for entry in UNIT_TEXTS]


@pytest.fixture(scope="session")
def synthetic_corpus(synthetic_units: List[Unit]) -> Corpus:
    documents = [
        Document("D01", "SRC01", "D01.pdf", title="Economic Survey", publisher="Government",
                 year="2025", document_type="economic_survey"),
        Document("D02", "SRC02", "D02.pdf", title="Financial Stability Report",
                 publisher="RBI", year="2025", document_type="financial_stability_report"),
        Document("D03", "SRC03", "D03.pdf", title="Annual Report", publisher="SEBI",
                 year="2024", document_type="regulator_annual_report"),
    ]
    return Corpus(documents=documents, units=synthetic_units, policy="prose_tables",
                  policy_unit_types=["paragraph"])


@pytest.fixture(scope="session")
def phase3_config():
    """Real configuration, overridden for a six-unit corpus."""
    config = load_config(PROJECT_ROOT / "config" / "phase3_config.yaml")
    data = copy.deepcopy(config.as_dict())
    data["ngrams"]["min_frequency"] = 1
    data["ngrams"]["max_indexed_phrases"] = 500
    data["retrieval"]["max_results_per_query"] = 25
    data["retrieval"]["top_terms_to_report"] = 5
    config._data = data
    return config


@pytest.fixture(scope="session")
def specs(phase3_config):
    from src.phase3.pipelines import load_pipeline_specs

    return load_pipeline_specs(phase3_config)


@pytest.fixture(scope="session")
def runner(phase3_config):
    from src.phase3.pipeline_runner import PipelineRunner

    return PipelineRunner(phase3_config)


@pytest.fixture(scope="session")
def builds(phase3_config, runner, specs, synthetic_corpus):
    """Both pipelines executed and both indexes built over the synthetic corpus."""
    from src.phase3.index_builder import IndexBuilder

    builder = IndexBuilder(phase3_config, runner)
    units = synthetic_corpus.selected_units()
    out = {}
    for key in sorted(specs):
        result = runner.run(specs[key], units)
        out[key] = builder.build(result, synthetic_corpus)
    return out


@pytest.fixture(scope="session")
def engines(phase3_config, runner, specs, builds, synthetic_corpus):
    from src.phase3.retrieval import build_engine

    units = synthetic_corpus.selected_units()
    return {
        key: build_engine(builds[key].index, specs[key], runner, phase3_config, units=units)
        for key in sorted(builds)
    }


@pytest.fixture(scope="session")
def engine(engines):
    """The Pipeline A engine (lemma terms)."""
    return engines["pipeline_a"]
