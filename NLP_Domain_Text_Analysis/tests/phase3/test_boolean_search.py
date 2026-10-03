"""Boolean retrieval: AND, OR, A AND NOT B, grouped expressions."""

from __future__ import annotations

import pytest

from src.phase3.query_parser import QuerySyntaxError


def ids(outcome):
    return {hit.unit_id for hit in outcome.unit_hits}


def test_and_returns_the_intersection(engine, builds):
    outcome = engine.search_boolean_and(["gdp", "inflation"])
    left = ids(engine.search_keyword("gdp"))
    right = ids(engine.search_keyword("inflation"))
    assert outcome.unit_count > 0
    assert ids(outcome) == left & right
    assert set(outcome.matched_terms) == {"gdp", "inflation"}


def test_and_of_unrelated_terms_is_empty(engine):
    outcome = engine.search_boolean_and(["gdp", "insurance"])
    assert outcome.unit_count == 0


def test_or_returns_the_union(engine):
    outcome = engine.search_boolean_or(["gdp", "deficit"])
    left = ids(engine.search_keyword("gdp"))
    right = ids(engine.search_keyword("deficit"))
    assert ids(outcome) == left | right


def test_or_results_are_deduplicated(engine):
    outcome = engine.search_boolean_or(["inflation", "cpi"])
    unit_ids = [hit.unit_id for hit in outcome.unit_hits]
    assert len(unit_ids) == len(set(unit_ids))


def test_or_of_a_term_with_itself_is_idempotent(engine):
    once = engine.search_boolean_or(["inflation"])
    twice = engine.search_boolean_or(["inflation", "inflation"])
    assert ids(once) == ids(twice)


def test_not_removes_units_containing_the_excluded_term(engine):
    positive = engine.search_keyword("inflation")
    negative = engine.search_boolean_not(["inflation"], ["cpi"])
    excluded = ids(engine.search_keyword("cpi"))
    assert ids(negative) == ids(positive) - excluded
    assert ids(negative) <= ids(positive)
    assert ids(negative)


def test_not_does_not_return_the_whole_corpus(engine, builds):
    negative = engine.search_boolean_not(["inflation"], ["cpi"])
    assert negative.unit_count < builds["pipeline_a"].index.unit_count


def test_not_with_an_unknown_excluded_term_keeps_everything(engine):
    positive = engine.search_keyword("inflation")
    negative = engine.search_boolean_not(["inflation"], ["unheardofterm"])
    assert ids(negative) == ids(positive)
    assert "unheardofterm" in negative.missing_terms


def test_not_with_an_unknown_include_term_is_empty(engine):
    outcome = engine.search_boolean_not(["unheardofterm"], ["cpi"])
    assert outcome.unit_count == 0


def test_grouped_expression(engine):
    outcome = engine.search("(gdp OR deficit) AND inflation")
    left = ids(engine.search_boolean_or(["gdp", "deficit"]))
    right = ids(engine.search_keyword("inflation"))
    assert ids(outcome) == left & right


def test_boolean_query_types_are_labelled(engine):
    assert engine.search("gdp AND inflation").query_type == "boolean_and"
    assert engine.search("gdp OR deficit").query_type == "boolean_or"
    assert engine.search("inflation AND NOT cpi").query_type == "boolean_not"
    assert engine.search("(gdp OR deficit) AND inflation").query_type == "boolean_group"


def test_grouped_query_text_keeps_its_parentheses(engine):
    outcome = engine.search("(gdp OR deficit) AND inflation")
    assert outcome.query == "(gdp OR deficit) AND inflation"


def test_search_routes_phrase_queries_to_phrase_matching(engine):
    outcome = engine.search('"repo rate"', "phrase")
    assert outcome.query_type == "phrase"
    assert all(hit.phrase_match for hit in outcome.unit_hits)


def test_boolean_search_with_a_phrase_operand(engine):
    outcome = engine.search('"monetary policy" AND rbi')
    phrase_ids = ids(engine.search_phrase("monetary policy"))
    rbi_ids = ids(engine.search_keyword("rbi"))
    assert ids(outcome) == phrase_ids & rbi_ids


def test_bare_not_is_refused_at_the_engine(engine):
    with pytest.raises(QuerySyntaxError):
        engine.search("NOT cpi")


def test_unknown_query_type_is_refused(engine):
    with pytest.raises(ValueError):
        engine.search("gdp", "not_a_type")


def test_search_reports_a_syntax_error_instead_of_returning_wrong_rows(engine):
    with pytest.raises(QuerySyntaxError):
        engine.search("gdp AND (", "boolean_and")


def test_boolean_results_carry_matched_terms_and_scores(engine):
    outcome = engine.search_boolean_and(["gdp", "inflation"])
    for hit in outcome.unit_hits:
        assert set(hit.matched_terms) >= {"gdp", "inflation"}
        assert hit.score > 0
        assert hit.matched_term_frequency["gdp"] >= 1
        assert hit.matched_term_frequency["inflation"] >= 1


def test_pipeline_a_and_b_agree_on_document_coverage(engines):
    a = engines["pipeline_a"].search_keyword("inflation")
    b = engines["pipeline_b"].search_keyword("inflation")
    assert {hit.unit_id for hit in a.unit_hits} == {hit.unit_id for hit in b.unit_hits}
    assert "inflation" in a.matched_terms
    assert "inflat" in b.matched_terms
