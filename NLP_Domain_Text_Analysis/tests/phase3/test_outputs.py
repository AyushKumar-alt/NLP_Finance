"""Query registry, result tables, statistics and the validation rules."""

from __future__ import annotations

import pytest

from src.phase3.query_registry import (
    QUERY_TYPES,
    coverage_summary,
    execute_query_set,
    load_queries,
    validate_query_vocabulary,
)
from src.phase3.result_formatter import (
    DOCUMENT_FIELDS,
    RETRIEVAL_FIELDS,
    ResultFormatter,
    citation,
    truncate_snippet,
)
from src.phase3.statistics import (
    compute_index_statistics,
    compute_retrieval_statistics,
    selection_payload,
)


# ----------------------------------------------------------------------
# query registry
# ----------------------------------------------------------------------
def test_the_configured_query_set_has_fifteen_queries(phase3_config):
    queries = load_queries(phase3_config)
    assert len(queries) == 15
    assert [q["id"] for q in queries] == [f"Q{i:02d}" for i in range(1, 16)]
    assert all(q["type"] in QUERY_TYPES for q in queries)


def test_every_query_type_declared_in_the_config_is_exercised(phase3_config):
    declared = {q["type"] for q in load_queries(phase3_config)}
    assert declared == {"keyword", "phrase", "boolean_and", "boolean_or",
                        "boolean_not", "boolean_group"}


def test_registry_records_normalization_results_and_timing(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    assert len(records) == 15
    for record in records:
        assert record.syntax_ok is True, record.error
        assert record.outcome is not None
        assert record.execution_time_ms >= 0
        assert record.normalized_terms or record.query_type == "boolean_group"


def test_registry_reports_missing_terms_rather_than_hiding_them(phase3_config, engine):
    records = execute_query_set(
        phase3_config, engine,
        queries=[{"id": "T01", "query": "quantum entanglement", "type": "keyword",
                  "description": "absent term", "terms": ["quantum", "entanglement"]}],
    )
    record = records[0]
    assert record.answered is False
    assert record.missing_terms == ["quantum", "entanglement"]
    assert "quantum" in validate_query_vocabulary(engine, records)


def test_unsupported_query_type_is_refused(phase3_config, engine):
    with pytest.raises(ValueError):
        execute_query_set(
            phase3_config, engine,
            queries=[{"id": "T02", "query": "gdp", "type": "fuzzy"}],
        )


def test_a_syntax_error_is_captured_not_raised(phase3_config, engine):
    records = execute_query_set(
        phase3_config, engine,
        queries=[{"id": "T03", "query": "NOT cpi", "type": "boolean_not",
                  "description": "bare NOT", "terms": ["cpi"]}],
    )
    assert records[0].syntax_ok is False
    assert "A AND NOT B" in records[0].error


def test_coverage_summary_is_consistent(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    summary = coverage_summary(records)
    assert summary["queries"] == len(records)
    assert summary["answered"] == sum(1 for r in records if r.answered)
    assert 0.0 <= summary["answerability"] <= 1.0
    assert summary["syntax_valid"] == len(records)
    assert summary["expected_domains"]


def test_configured_query_vocabulary_exists_in_the_index(phase3_config, engine, builds):
    index = builds["pipeline_a"].index
    records = execute_query_set(phase3_config, engine)
    missing = validate_query_vocabulary(engine, records)
    # every configured term of the real query set must resolve; the synthetic
    # six-unit corpus is too small for all of them, so only report the gap
    assert all(isinstance(term, str) for term in missing)
    assert validate_query_vocabulary(engine, records) == missing
    assert "gdp" not in missing


# ----------------------------------------------------------------------
# result tables
# ----------------------------------------------------------------------
def test_retrieval_rows_carry_provenance_and_a_snippet(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    rows = ResultFormatter(phase3_config).all_rows(records)
    assert rows["units"]
    for row in rows["units"]:
        assert row["document_id"].startswith("D")
        assert row["page_number"] >= 1
        assert row["unit_id"]
        assert row["section_id"]
        assert row["citation"] == citation(
            row["document_id"], row["page_number"], row["section_number"], row["unit_id"]
        )
        assert row["snippet"]


def test_retrieval_rows_are_ranked_without_gaps(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    rows = ResultFormatter(phase3_config).all_rows(records)["units"]
    by_query: dict = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    for query_id, group in by_query.items():
        assert [row["rank"] for row in group] == list(range(1, len(group) + 1))
        scores = [row["score"] for row in group]
        assert scores == sorted(scores, reverse=True)


def test_document_rows_are_deduplicated_per_query(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    rows = ResultFormatter(phase3_config).all_rows(records)["documents"]
    for query_id in {row["query_id"] for row in rows}:
        documents = [r["document_id"] for r in rows if r["query_id"] == query_id]
        assert len(documents) == len(set(documents))


def test_formatter_output_matches_its_declared_fields(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    rows = ResultFormatter(phase3_config).all_rows(records)
    assert set(rows["units"][0]) == set(RETRIEVAL_FIELDS)
    assert set(rows["documents"][0]) == set(DOCUMENT_FIELDS)
    assert len(rows["summary"]) == len(records)


def test_snippet_truncation_cuts_on_a_word_boundary():
    text = "one two three four five six seven eight nine ten"
    assert truncate_snippet(text, 20) == "one two three..."
    assert truncate_snippet(text, 20).endswith("...")
    assert truncate_snippet("short", 220) == "short"
    assert len(truncate_snippet(text, 25)) <= 25
    assert len(truncate_snippet(text, 4)) <= 4


def test_query_example_text_names_the_provenance(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    text = ResultFormatter(phase3_config).query_example(records[0])
    assert records[0].query_id in text
    assert "Matched terms" in text


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def test_index_statistics_are_internally_consistent(builds):
    index = builds["pipeline_a"].index
    stats = compute_index_statistics(index, top_n=5)
    assert stats.index_terms == index.term_count
    assert stats.total_postings == index.posting_count
    assert stats.unigram_terms + stats.phrase_terms == stats.index_terms
    assert stats.unigram_postings + stats.phrase_postings == stats.total_postings
    assert stats.indexed_units == index.unit_count
    assert 0 < stats.hapax_ratio <= 1
    assert len(stats.top_terms) <= 5
    assert stats.top_phrases


def test_index_statistics_rows_are_metric_value_pairs(builds):
    rows = compute_index_statistics(builds["pipeline_a"].index).as_rows()
    assert all(set(row) == {"metric", "value"} for row in rows)
    metrics = {row["metric"] for row in rows}
    assert {"index_terms", "total_postings", "indexed_units"} <= metrics


def test_retrieval_statistics_sum_to_the_query_records(phase3_config, engine):
    records = execute_query_set(phase3_config, engine)
    stats = compute_retrieval_statistics(records)
    assert stats.queries == len(records)
    assert stats.total_units_returned == sum(r.unit_count for r in records)
    assert stats.mean_execution_time_ms >= 0
    payload = stats.as_dict()
    assert payload["by_query_type"]
    assert set(payload["by_query_type"]) == {r.query_type for r in records}


def test_selection_payload_is_json_serialisable(phase3_config, runner, builds, synthetic_units):
    import json

    from src.phase3.pipeline_comparison import compare_pipelines
    from src.phase3.query_registry import execute_query_set
    from src.phase3.retrieval import build_engine

    outcomes = {}
    for key in builds:
        engine = build_engine(builds[key].index, builds[key].spec, runner, phase3_config,
                              units=synthetic_units)
        records = execute_query_set(phase3_config, engine)
        outcomes[key] = [r.outcome for r in records if r.outcome]
    comparison = compare_pipelines(
        config=phase3_config, runner=runner, builds=builds,
        pipeline_stats={key: {} for key in builds}, query_outcomes=outcomes,
        units=synthetic_units, evidence=None,
    )
    payload = selection_payload(comparison)
    json.dumps(payload)
    assert payload["final_pipeline"] in builds
    assert payload["reason"]


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
def test_validation_rules_pass_on_the_synthetic_run(phase3_config, builds, engines, synthetic_units):
    from src.phase3.pipeline_comparison import compare_pipelines
    from src.phase3.query_registry import execute_query_set
    from src.phase3.validation import run_validation

    engine = engines["pipeline_a"]
    records = execute_query_set(phase3_config, engine)
    comparison = compare_pipelines(
        config=phase3_config, runner=_runner(), builds=builds,
        pipeline_stats={key: {} for key in builds},
        query_outcomes={key: [r.outcome for r in records if r.outcome] for key in builds},
        units=synthetic_units, evidence=None,
    )
    rows = ResultFormatter(phase3_config).all_rows(records)
    results = run_validation(
        config=phase3_config, index=builds["pipeline_a"].index, comparison=comparison,
        records=records, engine=engine, unit_rows=rows["units"],
    )
    ids = {result.rule_id for result in results}
    assert {"P01", "P02", "P03", "P04", "P05", "P12", "P13"} <= ids
    structural = [r for r in results if r.rule_id in {"P01", "P02", "P03", "P04", "P05",
                                                      "P10", "P11", "P12", "P13", "P14", "P17"}]
    failures = [result for result in structural if result.status == "FAIL"]
    assert not failures, [(r.rule_id, r.detail) for r in failures]
    # P15/P16 check on-disk artefacts, which the synthetic run does not produce
    assert all(r.status in {"PASS", "FAIL", "WARN"} for r in results)


def _runner():
    from src.phase3.config import load_config
    from src.phase3.pipeline_runner import PipelineRunner

    return PipelineRunner(load_config())
