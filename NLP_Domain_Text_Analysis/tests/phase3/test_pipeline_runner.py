"""The pipeline runner: component order, single application, query/index parity."""

from __future__ import annotations

import pytest

from src.phase3.pipeline_runner import PipelineRunner
from src.phase3.pipelines import SUPPORTED_STEPS


def test_every_step_is_applied_exactly_once(runner, specs, monkeypatch):
    """Regression: morphology used to run twice per token.

    Porter stemming is not idempotent ('financial' -> 'financi' -> 'financ'), so a
    second application silently produced a different index vocabulary from the one
    the query side was normalized with.
    """
    spec = specs["pipeline_b"]
    calls = []
    original = PipelineRunner._morphologize

    def counting(self, pipeline_spec, token):
        calls.append(token)
        return original(self, pipeline_spec, token)

    monkeypatch.setattr(PipelineRunner, "_morphologize", counting)
    runner.normalize_query(spec, "financial stability")
    assert calls == ["financial", "stability"]


def test_stemming_is_not_applied_twice(runner, specs):
    spec = specs["pipeline_b"]
    once = runner.normalize_query(spec, "financial stability")
    assert once == ["financi", "stabil"]
    # the double-stemmed form must not appear
    assert "financ" not in once


def test_index_and_query_produce_the_same_terms(runner, specs, synthetic_units):
    """Under Pipeline B, a query for a surface form finds that form's index term.

    Pipeline B stems before matching stopwords, so the index and the query side
    agree term for term; Pipeline A is checked separately below, because its
    declared order cannot give that guarantee.
    """
    from src.phase2.load_corpus import Corpus, Document

    from src.phase3.config import load_config
    from src.phase3.index_builder import IndexBuilder

    corpus = Corpus(
        documents=[Document("D01", "SRC01", "D01.pdf")],
        units=synthetic_units[:1],
        policy="prose_tables",
        policy_unit_types=["paragraph"],
    )
    config = load_config()
    spec = specs["pipeline_b"]
    result = runner.run(spec, corpus.units)
    build = IndexBuilder(config, runner).build(result, corpus)
    terms = build.unit_terms[corpus.units[0].unit_id]
    content_words = [t for t in terms if any(c.isalpha() for c in t)]
    assert content_words
    for term in content_words:
        assert runner.normalize_query(spec, term) == [term], (
            f"'{term}' is in the index but the query side maps it elsewhere"
        )


def test_pipeline_a_cannot_round_trip_stopword_derived_lemmas(runner, specs):
    """Documented consequence of Pipeline A's declared order, not an accident.

    Stopwords are removed *before* lemmatization, so a non-stopword whose lemma
    happens to be a stopword ('kept' -> 'keep') enters the index, while a query
    for 'keep' is removed as a stopword. Phase 2 measured the same class of
    effect for stemming. The test records it so the asymmetry cannot be mistaken
    for a regression if the pipeline order ever changes.
    """
    spec = specs["pipeline_a"]
    stopwords = runner.stopword_words(spec)
    assert "keep" in stopwords          # 'kept' is not a stopword
    assert "kept" not in stopwords
    result = runner.run(spec, [])
    assert runner.normalize_query(spec, "kept") == ["keep"]
    assert runner.normalize_query(spec, "keep") == []


def test_query_normalization_uses_the_configured_component_order(runner, specs):
    """Stopword-then-lemmatization and stemming-then-stopword must differ."""
    a = runner.normalize_query(specs["pipeline_a"], "monetary policies")
    b = runner.normalize_query(specs["pipeline_b"], "monetary policies")
    assert a == ["monetary", "policy"]
    assert b == ["monetari", "polici"]


def test_stopword_removal_drops_function_words(runner, specs):
    spec = specs["pipeline_b"]
    assert runner.normalize_query(spec, "the rate of the repo rate") == ["rate", "repo", "rate"]


def test_both_pipelines_agree_on_financial_expressions(runner, specs):
    for key in sorted(specs):
        terms = runner.normalize_query(specs[key], "7.4 per cent in FY26")
        assert "7.4 per cent" in terms, f"{key} split the percentage expression"
        assert "fy26" in terms, f"{key} changed the fiscal year"


def test_phrase_stream_keeps_stopwords_the_index_drops(runner, specs):
    spec = specs["pipeline_b"]
    indexed = runner.normalize_query(spec, "the repo rate")
    stream = runner.phrase_stream(spec, "the repo rate")
    assert indexed == ["repo", "rate"]
    assert stream == ["the", "repo", "rate"]


def test_phrase_stream_and_index_share_the_same_morphology(runner, specs):
    """Both sides must stem once, or positional matching silently misaligns."""
    for key in sorted(specs):
        spec = specs[key]
        stream = runner.phrase_stream(spec, "financial stability")
        query = runner.normalize_query(spec, "financial stability")
        # the query drops stopwords; whatever is left must appear in the stream
        assert [term for term in stream if term in query] == query


def test_every_configured_step_is_supported(specs):
    for key, spec in specs.items():
        for step in spec.order:
            assert step in SUPPORTED_STEPS, f"{key} uses unsupported step {step}"


def test_pipelines_must_differ_in_order(specs):
    assert specs["pipeline_a"].order != specs["pipeline_b"].order


def test_runner_reports_step_timings(runner, specs, synthetic_units):
    result = runner.run(specs["pipeline_a"], synthetic_units)
    assert set(result.step_seconds) >= {"tokenize", "stopwords", "morphology"}
    assert result.total_seconds >= sum(
        value for key, value in result.step_seconds.items() if key in ("tokenize",)
    )
    assert result.stats["total_tokens"] == result.total_tokens
    assert result.stats["order"].startswith("tokenize")
