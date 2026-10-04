"""Regression tests for Phase 4 runtime index isolation."""

import pytest
from src.phase4.runtime import load_phase3_runtime


def test_runtime_engine_specs_and_indexes():
    """Verify that runtime engines for Pipeline A and Pipeline B have distinct specs and indexes."""
    runtime = load_phase3_runtime()
    eng_a = runtime.engine("pipeline_a_lemma")
    eng_b = runtime.engine("pipeline_b_stem")

    # A. engine("pipeline_a_lemma").spec.key is Pipeline A
    assert eng_a.spec.key == "pipeline_a_lemma"

    # B. engine("pipeline_b_stem").spec.key is Pipeline B
    assert eng_b.spec.key == "pipeline_b_stem"

    # C. A and B do not share the same index object
    assert eng_a.index is not eng_b.index
    assert eng_a.index.term_count != eng_b.index.term_count
    assert eng_a.index.term_count > 20000
    assert eng_b.index.term_count > 20000


def test_runtime_engine_divergence_query():
    """Verify that a known divergence query produces different top-10 rankings for A and B."""
    runtime = load_phase3_runtime()
    eng_a = runtime.engine("pipeline_a_lemma")
    eng_b = runtime.engine("pipeline_b_stem")

    res_a = eng_a.search("banking AND credit")
    res_b = eng_b.search("banking AND credit")

    units_a = [h.unit_id for h in res_a.unit_hits[:10]]
    units_b = [h.unit_id for h in res_b.unit_hits[:10]]

    # Ordered top-10 lists must differ for this query
    assert units_a != units_b
    assert units_a[0] != units_b[0]
