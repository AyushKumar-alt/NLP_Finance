"""Focused unit test for Phase 2 Pipeline A vs Pipeline B controlled normalization comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase2.config import load_config
from src.phase2.load_corpus import ExperimentSample, Unit
from src.phase2.comparisons import run_pipeline_ab_comparison, CONTROLLED_FINANCE_TERMS


@pytest.fixture
def synthetic_sample() -> ExperimentSample:
    """Create synthetic financial units for controlled pipeline testing."""
    u1 = Unit(
        unit_id="D01_P001_U001",
        document_id="D01",
        source_id="SRC01",
        filename="echap01.pdf",
        page_id="D01_P001",
        page_number=1,
        section_id="SEC_1",
        section_number="1.1",
        section_title="Macroeconomic Growth",
        unit_type="paragraph",
        unit_index=1,
        text="The repo rate was set at 6.5% in FY2025-26 by RBI. Real GDP growth remained strong at ₹1,000 lakh crore.",
        char_count=104,
        word_count=18,
    )
    u2 = Unit(
        unit_id="D01_P001_U002",
        document_id="D01",
        source_id="SRC01",
        filename="echap01.pdf",
        page_id="D01_P001",
        page_number=1,
        section_id="SEC_1",
        section_number="1.1",
        section_title="Macroeconomic Growth",
        unit_type="paragraph",
        unit_index=2,
        text="Inflation dynamics and non-performing assets were monitored by SEBI and commercial banks.",
        char_count=90,
        word_count=12,
    )
    units = [u1, u2]
    return ExperimentSample(policy="prose_tables", units=units, documents={})


def test_pipeline_ab_comparison_structure(synthetic_sample: ExperimentSample):
    config = load_config()
    results = run_pipeline_ab_comparison(synthetic_sample, config)

    # 1. Exactly two primary pipelines exist
    assert "rows" in results
    rows = results["rows"]
    assert len(rows) == 2, f"Expected exactly 2 pipeline rows, got {len(rows)}"

    row_a = rows[0]
    row_b = rows[1]

    # 2. Pipeline names & normalization strategies
    assert row_a["pipeline"] == "Pipeline A"
    assert "Lemmatization" in row_a["normalization"]
    assert row_b["pipeline"] == "Pipeline B"
    assert "Stemming" in row_b["normalization"]

    # 3. Both receive identical starting units
    assert row_a["processed_units"] == len(synthetic_sample.units)
    assert row_b["processed_units"] == len(synthetic_sample.units)

    # 4. Metrics exist and are non-negative
    for row in (row_a, row_b):
        assert row["token_count"] > 0
        assert row["vocabulary_size"] > 0
        assert row["type_token_ratio"] > 0
        assert row["avg_tokens_per_unit"] > 0
        assert 0.0 <= row["domain_preservation_pct"] <= 100.0
        assert row["domain_terms_total"] == len(CONTROLLED_FINANCE_TERMS)
        assert row["runtime_seconds"] >= 0.0

    # 5. Example transformations exist
    assert "examples" in results
    assert len(results["examples"]) > 0
    ex = results["examples"][0]
    assert "expression" in ex
    assert "pipeline_a_lemmatized" in ex
    assert "pipeline_b_stemmed" in ex


def test_ml_pos_sparse_index_compatibility():
    """Verify that _ensure_int32_indices converts 64-bit sparse matrix indices to 32-bit for scikit-learn."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import SGDClassifier
    from src.phase2.ml_pos import _ensure_int32_indices

    # Create a CSR matrix with explicit int64 indices
    indptr = np.array([0, 2, 3], dtype=np.int32)
    indices = np.array([0, 2, 1], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    m = csr_matrix((data, indices, indptr), shape=(2, 3))
    m.indices = m.indices.astype(np.int64)
    m.indptr = m.indptr.astype(np.int64)

    assert m.indices.dtype == np.int64
    assert m.indptr.dtype == np.int64

    # Apply the fix
    _ensure_int32_indices(m)

    assert m.indices.dtype == np.int32
    assert m.indptr.dtype == np.int32

    # Verify SGDClassifier fit succeeds cleanly without raising ValueError
    clf = SGDClassifier(loss="log_loss", random_state=42)
    clf.fit(m, [0, 1])
    preds = clf.predict(m)
    assert len(preds) == 2

