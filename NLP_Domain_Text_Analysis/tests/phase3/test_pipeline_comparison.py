"""Pipeline comparison, criteria measurement and the final selection."""

from __future__ import annotations

import pytest

from src.phase3.pipeline_comparison import (
    CriterionScore,
    PipelineMeasurement,
    compare_pipelines,
    find_financial_expressions,
    load_domain_terms,
    measure_domain_terms,
    measure_financial_expressions,
    measure_query_answerability,
    measure_variant_collapse,
)


class FakeOutcome:
    """Minimal stand-in for a SearchOutcome (only unit_count is used)."""

    def __init__(self, query: str, unit_count: int) -> None:
        self.query = query
        self.unit_count = unit_count


# ----------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------
def test_financial_expressions_are_found_in_the_raw_text(synthetic_units):
    found = find_financial_expressions(synthetic_units)
    categories = {category for category, _ in found}
    assert "percentage" in categories
    assert "fiscal_year" in categories
    surfaces = {surface for _, surface in found}
    assert "7.4 per cent" in surfaces
    assert "FY26" in surfaces


def test_both_pipelines_score_the_same_expression_occurrences(synthetic_units, runner, builds):
    found = find_financial_expressions(synthetic_units)
    totals = {}
    for key, spec in ((k, b.spec) for k, b in builds.items()):
        preserved, total, _examples, _lost, _by_category = measure_financial_expressions(
            found, spec, runner, builds[key].index
        )
        totals[key] = (preserved, total)
    assert len({total for _preserved, total in totals.values()}) == 1
    assert all(preserved == total for preserved, total in totals.values())


def test_domain_terms_include_the_configured_and_phase2_sources(phase3_config):
    terms = load_domain_terms(phase3_config, evidence=None)
    assert "gdp" in terms
    assert "monetary policy" in terms
    assert terms == list(dict.fromkeys(terms)), "terms must be unique and order-preserving"


def test_domain_term_recall_is_measured_against_the_index(runner, specs, builds):
    terms = ["gdp", "inflation", "definitely_not_a_term"]
    found, total, missing = measure_domain_terms(
        terms, specs["pipeline_a"], runner, builds["pipeline_a"].index
    )
    assert total == 3
    assert found == 2
    assert missing == ["definitely_not_a_term"]


def test_variant_collapse_counts_fully_collapsed_groups(runner, specs, builds):
    collapsed, total, names, detail = measure_variant_collapse(
        specs["pipeline_a"], runner, builds["pipeline_a"].index,
        _config_with_variant_groups(
            [["deficit", "deficits"], ["inflation", "inflated"]]
        ),
    )
    assert total == 2
    # WordNet maps the plural to the singular lemma ('deficits' -> 'deficit'); it does
    # *not* map the adjective 'inflated' onto the noun 'inflation', so only one
    # group collapses - which is exactly the limitation the criterion measures.
    assert collapsed == 1
    assert any("deficit" in name for name in names)
    assert len(detail) == 2
    assert {row["collapsed"] for row in detail} == {True, False}


def test_variant_collapse_favours_stemming_on_this_corpus(runner, specs, builds):
    """The measured difference between the two pipelines, on one example group."""
    groups = _config_with_variant_groups([["saving", "savings"]])
    a, *_ = measure_variant_collapse(
        specs["pipeline_a"], runner, builds["pipeline_a"].index, groups
    )
    b, *_ = measure_variant_collapse(
        specs["pipeline_b"], runner, builds["pipeline_b"].index, groups
    )
    # Porter maps both forms to 'save'; WordNet maps 'savings' to 'saving'
    assert a == 0
    assert b == 1


def test_query_answerability_counts_non_empty_results():
    answered, total, unanswered = measure_query_answerability(
        [FakeOutcome("a", 3), FakeOutcome("b", 0), FakeOutcome("c", 1)]
    )
    assert (answered, total) == (2, 3)
    assert unanswered == ["b"]


# ----------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------
def _config_with_variant_groups(groups):
    from src.phase3.config import Phase3Config
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    data = {
        "selection": {
            "criteria": {
                "financial_expression_preservation": 0.35,
                "domain_term_recall": 0.25,
                "variant_collapse_rate": 0.25,
                "query_answerability": 0.15,
            },
            "tie_breakers": ["smaller index vocabulary"],
            "exclude_from_score": ["processing_time_seconds"],
        },
        "variant_groups": groups,
        "ngrams": {"sizes": [2, 3], "min_frequency": 1, "join_with": "_"},
        "retrieval": {"case_sensitive": False},
        "financial_expressions": {"reference_terms": ["gdp", "inflation"]},
    }
    return Phase3Config(data, root / "config" / "phase3_config.yaml", root)


def _measurement(name: str, score: float, index_terms: int = 10) -> PipelineMeasurement:
    criteria = [
        CriterionScore("financial_expression_preservation", 0.35, 9, 10, 0.9),
        CriterionScore("domain_term_recall", 0.25, 8, 10, 0.8),
        CriterionScore("variant_collapse_rate", 0.25, 7, 10, 0.7),
        CriterionScore("query_answerability", 0.15, 6, 10, 0.6),
    ]
    from src.phase3.pipelines import PipelineSpec

    spec = PipelineSpec(
        key=name.lower().replace(" ", "_"),
        name=name,
        tokenizer="hybrid",
        date_number_strategy="typed_protect",
        stopword_strategy="domain_aware",
        morphology="lemmatize",
        morphology_algorithm="wordnet",
        order=("tokenize", "stopwords", "morphology"),
        pos_strategy="",
        ner_strategy="",
        ngram_max=3,
        index_representation="",
    )
    measurement = PipelineMeasurement(
        spec=spec, criteria=criteria, index_stats={"index_terms": index_terms}
    )
    measurement.criteria[0].normalized = score / 0.315 if score else 0.0
    return measurement


def test_highest_weighted_score_wins():
    from src.phase3.pipeline_comparison import _decide

    a = _measurement("Pipeline A", 0.9)
    b = _measurement("Pipeline B", 0.5)
    winner, margin, reason = _decide({"pipeline_a": a, "pipeline_b": b}, ["smaller index vocabulary"])
    assert winner == "pipeline_a"
    assert margin > 0
    assert "wins on the weighted score" in reason


def test_tie_is_broken_by_the_first_declared_tie_breaker():
    from src.phase3.pipeline_comparison import _decide

    a = _measurement("Pipeline A", 0.9, index_terms=500)
    b = _measurement("Pipeline B", 0.9, index_terms=900)
    winner, margin, reason = _decide({"pipeline_a": a, "pipeline_b": b}, ["smaller index vocabulary"])
    assert winner == "pipeline_a"
    assert margin == 0.0
    assert "tie broken by 'smaller index vocabulary'" in reason
    assert "500 vs 900" in reason


def test_weights_and_criteria_are_reported(phase3_config, runner, builds, synthetic_units):
    """The real comparison runs end to end on the synthetic corpus."""
    from src.phase3.query_registry import execute_query_set

    pipeline_stats = {key: {"total_tokens": 10, "processing_time_seconds": 0.01}
                      for key in builds}
    query_outcomes = {}
    for key, spec in ((k, builds[k].spec) for k in builds):
        from src.phase3.retrieval import build_engine

        engine = build_engine(builds[key].index, spec, runner, phase3_config,
                              units=synthetic_units)
        records = execute_query_set(phase3_config, engine)
        query_outcomes[key] = [r.outcome for r in records if r.outcome]

    comparison = compare_pipelines(
        config=phase3_config,
        runner=runner,
        builds=builds,
        pipeline_stats=pipeline_stats,
        query_outcomes=query_outcomes,
        units=synthetic_units,
        evidence=None,
    )
    assert comparison.winner_key in builds
    assert 0.0 < comparison.winner.total_score <= 1.0
    assert sum(comparison.criteria_weights.values()) == pytest.approx(1.0)
    for measurement in comparison.measurements.values():
        for criterion in measurement.criteria:
            assert 0.0 <= criterion.rate <= 1.0
    rows = comparison.rows()
    assert {row["pipeline"] for row in rows} == set(builds)
    assert all("total_score" in row for row in rows)
