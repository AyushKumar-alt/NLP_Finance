"""Keyword and phrase retrieval, including snippets and traceability."""

from __future__ import annotations

import pytest

from src.phase3.keyword_search import UnitHit


# ----------------------------------------------------------------------
# keyword search
# ----------------------------------------------------------------------
def test_keyword_search_finds_units(engine):
    outcome = engine.search_keyword("inflation")
    assert outcome.unit_count > 0
    assert outcome.query_type == "keyword"
    assert outcome.retrieval_method == "keyword"
    assert outcome.pipeline == "pipeline_a"
    assert "inflation" in outcome.matched_terms


def test_keyword_search_is_case_insensitive_but_normalized(engine):
    lower = engine.search_keyword("inflation")
    upper = engine.search_keyword("INFLATION")
    assert [hit.unit_id for hit in lower.unit_hits] == [hit.unit_id for hit in upper.unit_hits]


def test_keyword_search_of_an_unknown_term_returns_nothing(engine):
    outcome = engine.search_keyword("supercalifragilistic")
    assert outcome.unit_count == 0
    assert outcome.matched_terms == []
    assert outcome.missing_terms == ["supercalifragilistic"]
    assert outcome.execution_time_ms >= 0


def test_keyword_search_snippet_contains_the_matched_word(engine, synthetic_units):
    outcome = engine.search_keyword("inflation")
    texts = {unit.unit_id: unit.text for unit in synthetic_units}
    assert outcome.unit_hits
    for hit in outcome.unit_hits:
        assert hit.snippet
        assert len(hit.snippet) <= 260
        assert "inflation" in texts[hit.unit_id].casefold()


def test_keyword_search_respects_the_result_cap(phase3_config, engine):
    cap = int(phase3_config.get("retrieval.max_results_per_query"))
    assert len(engine.search_keyword("inflation").unit_hits) <= cap


def test_keyword_search_ranks_deterministically(engine):
    first = engine.search_keyword("inflation")
    second = engine.search_keyword("inflation")
    assert [(hit.rank, hit.unit_id, hit.score) for hit in first.unit_hits] == [
        (hit.rank, hit.unit_id, hit.score) for hit in second.unit_hits
    ]
    scores = [hit.score for hit in first.unit_hits]
    assert scores == sorted(scores, reverse=True)


def test_document_aggregation_is_deduplicated_and_ordered(engine):
    outcome = engine.search_keyword("inflation")
    rows = outcome.document_hits()
    documents = [row["document_id"] for row in rows]
    assert len(documents) == len(set(documents))
    assert documents == [row["document_id"] for row in rows]
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    best = [row["best_score"] for row in rows]
    assert best == sorted(best, reverse=True)


def test_hits_carry_the_full_provenance_chain(engine, builds):
    index = builds["pipeline_a"].index
    for hit in engine.search_keyword("inflation").unit_hits:
        record = index.unit(hit.unit_id)
        assert record is not None
        assert hit.document_id == record.document_id
        assert hit.source_id == record.source_id
        assert hit.page_number == record.page_number
        assert hit.section_id == record.section_id
        assert hit.section_number == record.section_number
        assert hit.section_title == record.section_title


# ----------------------------------------------------------------------
# phrase search
# ----------------------------------------------------------------------
def test_phrase_search_requires_adjacency(engine, builds):
    """'policy monetary' must not match a unit that says 'monetary policy'."""
    index = builds["pipeline_a"].index
    forward = engine.search_phrase("monetary policy")
    backward = engine.search_phrase("policy monetary")
    forward_ids = {hit.unit_id for hit in forward.unit_hits}
    backward_ids = {hit.unit_id for hit in backward.unit_hits}
    assert forward_ids, "the ordered phrase should match its own unit"
    assert forward_ids != backward_ids


def test_phrase_search_matches_only_units_containing_the_sequence(engine, synthetic_units):
    outcome = engine.search_phrase("monetary policy")
    texts = {unit.unit_id: unit.text for unit in synthetic_units}
    for hit in outcome.unit_hits:
        assert "monetary policy" in texts[hit.unit_id].casefold()
        assert hit.phrase_match is True


def test_phrase_search_of_an_absent_phrase_returns_nothing(engine):
    outcome = engine.search_phrase("quantum entanglement")
    assert outcome.unit_count == 0


def test_phrase_search_marks_phrase_bonus(engine):
    outcome = engine.search_phrase("repo rate")
    assert outcome.unit_hits
    assert all(hit.phrase_match for hit in outcome.unit_hits)
    assert all(hit.score >= 2.0 for hit in outcome.unit_hits)


def test_phrase_search_normalizes_like_the_index(engine, runner, specs):
    """The query is reduced with the same pipeline that built the index."""
    assert runner.normalize_query(specs["pipeline_a"], "monetary policy") == [
        "monetary", "policy"
    ]
    outcome = engine.search_phrase("monetary policy")
    assert outcome.matched_terms == ["monetary", "policy"]


def test_phrase_positions_are_adjacent_in_the_stopword_preserving_stream(engine, builds):
    """Positions come from the stream that keeps stopwords, not the index stream.

    A phrase such as 'current account deficit' cannot be located in the index
    vocabulary (stopwords are not indexed), so the positional claim is checked
    against the pipeline's phrase stream.
    """
    phrase = engine.phrase_search
    terms = ["monetary", "policy"]
    found = phrase.search("monetary policy")
    assert found["unit_ids"]
    for unit_id in found["unit_ids"]:
        match = phrase.match_unit(unit_id, terms)
        assert match is not None
        stream = engine._stream_for_unit(unit_id)
        for position in match.positions:
            assert stream[position : position + len(terms)] == terms


def test_phrase_containing_a_stopword_is_still_locatable(engine):
    """A phrase keeps its stopwords, even though the index vocabulary drops them.

    'bank credit and the repo rate' can only be located if 'and' and 'the' are
    present in the stream used for phrase matching; the keyword pipeline removes
    them, which is correct for keyword retrieval and fatal for phrase retrieval.
    """
    from src.phase3.pipeline_runner import PipelineRunner
    from src.phase3.pipelines import PipelineSpec

    text = "Bank credit and the repo rate were steady."
    spec = PipelineSpec(
        key="t", name="t", tokenizer="hybrid", date_number_strategy="typed_protect",
        stopword_strategy="standard_english", morphology="stem",
        morphology_algorithm="porter",
        order=("tokenize", "date_number", "morphology", "stopwords"),
        pos_strategy="", ner_strategy="", ngram_max=3, index_representation="",
    )
    runner = PipelineRunner(engine.config)
    stopwords = runner.stopword_words(spec)
    assert {"and", "the"} <= stopwords

    indexed_view = runner.normalize_query(spec, text)
    assert "and" not in indexed_view and "the" not in indexed_view

    stream = runner.phrase_stream(spec, text)
    assert "and" in stream and "the" in stream
    terms = runner.phrase_stream_terms(spec, "credit and the repo rate")
    assert terms[:5] == ["credit", "and", "the", "repo", "rate"]
