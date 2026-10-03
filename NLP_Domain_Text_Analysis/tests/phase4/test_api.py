"""The FastAPI layer: contracts, the retrieval bridge, and input rejection.

These tests read the real Phase 1-4 artefacts on purpose. The backend exists to
serve exactly what those phases produced, so a test against a synthetic corpus
would not prove the thing that matters. The only write test redirects the
judgment file to a temporary copy.
"""

from __future__ import annotations

import csv

import pytest

from src.phase4.relevance import FIELDNAMES


# ----------------------------------------------------------------------
# system
# ----------------------------------------------------------------------
class TestHealth:
    def test_reports_ok_and_loads_the_retrieval_runtime(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["retrieval_available"] is True
        assert body["retrieval_detail"]["final_pipeline"] == "pipeline_b"
        assert body["retrieval_detail"]["index_terms"] == 21305
        assert body["retrieval_detail"]["indexed_units"] == 6005

    def test_lists_which_phases_have_been_run(self, client):
        body = client.get("/api/health").json()
        assert body["phases"] == {
            "phase1": True, "phase2": True, "phase3": True, "phase4": True
        }

    def test_matches_the_openapi_document(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestStatistics:
    @pytest.fixture(scope="class")
    @classmethod
    def body(cls, client):
        return client.get("/api/statistics").json()

    def test_corpus_totals_come_from_the_document_registry(self, body):
        corpus = body["corpus"]
        assert corpus["documents"] == 31
        assert corpus["pages"] == 885
        assert corpus["units"] == 8104
        assert corpus["selected_units"] == 6005
        assert corpus["text_selection_policy"] == "prose_tables"

    def test_index_block_matches_the_phase3_manifest(self, body):
        index = body["index"]
        assert index["terms"] == 21305
        assert index["postings"] == 191817
        assert index["final_pipeline"] == "pipeline_b"

    def test_pipeline_block_reports_the_winner_and_the_margin(self, body):
        pipelines = body["pipelines"]
        assert pipelines["final_pipeline"] == "pipeline_b"
        assert pipelines["score"] == pytest.approx(0.827162)
        assert pipelines["margin_over_runner_up"] == pytest.approx(0.041666)

    def test_phase4_block_carries_the_judgment_counts(self, body):
        judgments = body["phase4"]["judgments"]
        assert judgments["pairs"] == 139
        assert judgments["relevant"] == 130
        assert judgments["not_relevant"] == 9
        assert judgments["pool_depth"] == 10

    def test_phase4_metrics_are_numbers_not_strings(self, body):
        unit = body["phase4"]["unit_level"]
        assert isinstance(unit["precision_macro"], float)
        assert unit["precision_macro"] == pytest.approx(0.951111, rel=1e-4)

    def test_validation_reports_every_phase(self, body):
        validation = body["validation"]
        assert set(validation["phases"]) == {"phase1", "phase2", "phase3", "phase4"}
        for name, block in validation["phases"].items():
            assert block["available"] is True, name
            assert block["all_passed"] is True, f"{name}: {block['failed']} rule(s) failed"

    def test_states_that_missing_phases_are_not_reported_as_zero(self, body):
        assert "never as zero" in body["sources_note"]


# ----------------------------------------------------------------------
# corpus
# ----------------------------------------------------------------------
class TestDocuments:
    def test_lists_documents_with_pagination(self, client):
        body = client.get("/api/documents", params={"limit": 5}).json()
        assert len(body["items"]) == 5
        assert body["total"] == 31
        assert body["has_more"] is True
        assert body["returned"] == 5

    def test_last_page_reports_no_more(self, client):
        body = client.get("/api/documents", params={"limit": 5, "offset": 30}).json()
        assert body["returned"] == 1
        assert body["has_more"] is False

    def test_filters_by_source(self, client):
        body = client.get("/api/documents", params={"source_id": "SRC01"}).json()
        assert body["total"] > 0
        assert all(row["source_id"] == "SRC01" for row in body["items"])

    def test_search_matches_the_title(self, client):
        body = client.get("/api/documents", params={"search": "economic"}).json()
        assert body["total"] >= 1
        assert all(
            "economic" in (row["title"] or "").lower()
            or "economic" in (row["filename"] or "").lower()
            for row in body["items"]
        )

    def test_one_document_includes_its_sections_and_unit_census(self, client):
        body = client.get("/api/documents/D01").json()["document"]
        assert body["document_id"] == "D01"
        assert body["unit_count"] > 0
        assert body["pages"]
        assert body["sections"]
        assert body["source"]["source_id"] == body["source_id"]

    def test_unknown_document_is_404(self, client):
        response = client.get("/api/documents/D99")
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    def test_unit_lookup_returns_provenance(self, client):
        registry = client.get("/api/queries/Q01/results").json()
        unit_id = registry["items"][0]["unit_id"]
        body = client.get(f"/api/units/{unit_id}").json()["unit"]
        assert body["unit_id"] == unit_id
        assert body["document_id"].startswith("D")
        assert body["text"]


# ----------------------------------------------------------------------
# phase 2
# ----------------------------------------------------------------------
class TestExperiments:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/experiments/tokenization",
            "/api/experiments/preprocessing",
            "/api/experiments/stemming",
            "/api/experiments/lemmatization",
            "/api/experiments/pos",
            "/api/experiments/bpe",
        ],
    )
    def test_table_endpoint_returns_rows(self, client, path):
        body = client.get(path).json()
        assert body, f"{path} returned an empty object"

    def test_tokenization_compares_several_tokenizers(self, client):
        tokenizers = client.get("/api/experiments/tokenization").json()["tokenizers"]
        assert len(tokenizers) >= 3
        assert all(t["total_tokens"] for t in tokenizers)
        names = {t["tokenizer"] for t in tokenizers}
        assert {"nltk", "spacy", "custom", "hybrid"} <= names

    def test_ner_filters_by_label(self, client):
        sidebar = client.get("/api/experiments/ner/sidebar").json()
        label = sidebar["general_labels"][0]["label"]
        body = client.get(
            "/api/experiments/ner", params={"label": label, "limit": 10}
        ).json()
        assert body["total"] == sidebar["general_labels"][0]["count"]
        assert all(row["entity_label"] == label for row in body["items"])

    def test_ner_reports_an_accurate_total_before_paging(self, client):
        body = client.get("/api/experiments/ner", params={"limit": 3}).json()
        assert len(body["items"]) == 3
        assert body["total"] > 3
        assert body["has_more"] is True

    def test_ngrams_respects_the_n_range(self, client):
        body = client.get("/api/experiments/ngrams", params={"n": 2, "limit": 5}).json()
        assert body["n"] == 2
        assert len({row["ngram"] for row in body["items"]}) == 5
        assert [row["frequency"] for row in body["items"]] == sorted(
            (row["frequency"] for row in body["items"]), reverse=True
        )

    def test_ngrams_rejects_n_outside_one_to_five(self, client):
        assert client.get("/api/experiments/ngrams", params={"n": 9}).status_code == 422
        assert client.get("/api/experiments/ngrams", params={"n": 0}).status_code == 422


# ----------------------------------------------------------------------
# phase 3 retrieval
# ----------------------------------------------------------------------
@pytest.mark.slow
class TestSearch:
    def test_keyword_query_returns_ranked_units(self, client):
        body = client.post(
            "/api/search", json={"query": "inflation", "top_k": 5}
        ).json()
        assert body["query_type"] == "keyword"
        assert body["pipeline"] == "pipeline_b"
        assert body["retrieval_method"] == "keyword"
        assert len(body["results"]) == 5
        assert body["total_results"] >= 5
        scores = [row["score"] for row in body["results"]]
        assert scores == sorted(scores, reverse=True), "results are not in score order"

    def test_each_hit_carries_its_citation(self, client):
        body = client.post("/api/search", json={"query": "GDP", "top_k": 1}).json()
        hit = body["results"][0]
        assert hit["citation"].startswith(hit["document_id"])
        assert hit["unit_id"] in hit["citation"]
        assert hit["snippet"]

    def test_phrase_query_reports_the_phrase_method(self, client):
        body = client.post(
            "/api/search", json={"query": "monetary policy", "query_type": "phrase", "top_k": 3}
        ).json()
        assert body["retrieval_method"] == "phrase"

    def test_boolean_not_excludes_the_second_term(self, client):
        body = client.post(
            "/api/search", json={"query": "GDP AND NOT inflation", "top_k": 10}
        ).json()
        assert body["query_type"] == "boolean_not"
        assert body["matched_terms"] == ["gdp"]
        # The excluded term is in the index and was removed on purpose, so it is
        # neither "missing" nor "matched": it gets its own field.
        assert body["excluded_terms"] == ["inflation"]
        assert body["missing_terms"] == []
        assert all(
            "inflat" not in row["matched_terms_list"] for row in body["results"]
        ), "a NOT-excluded term appeared in a hit"

    def test_excluded_terms_are_empty_without_a_not(self, client):
        body = client.post("/api/search", json={"query": "GDP", "top_k": 3}).json()
        assert body["excluded_terms"] == []

    def test_missing_terms_are_reported_not_hidden(self, client):
        body = client.post(
            "/api/search", json={"query": "zzzqqqxyzzy", "top_k": 3}
        ).json()
        assert body["total_results"] == 0
        # The pipeline stems before it looks anything up, so the term reported as
        # missing is the stemmed form, not the raw surface string.
        assert body["missing_terms"] == ["zzzqqqxyzzi"]
        assert body["results"] == []

    def test_results_agree_with_the_stored_phase3_run(self, client):
        """The API and the Phase 3 CLI must return the same ranking.

        If these ever diverge, the GUI is showing something the artefacts do not
        support, which is the failure mode this whole design is meant to avoid.
        """
        live = client.post("/api/search", json={"query": "inflation", "top_k": 10}).json()
        stored = client.get("/api/queries/Q01/results", params={"limit": 10}).json()
        assert [row["unit_id"] for row in live["results"]] == [
            row["unit_id"] for row in stored["items"]
        ]

    def test_top_k_is_bounded_by_the_configured_ceiling(self, client):
        body = client.post("/api/search", json={"query": "GDP", "top_k": 200}).json()
        assert body["top_k"] == 200
        # Phase 3 caps a single query at 50 results; the API does not pretend
        # otherwise, so the UI can say why fewer rows came back.
        assert body["max_results_available"] == 50
        assert body["total_results"] <= 50
        assert len(body["results"]) <= 50

    def test_pipeline_can_be_overridden(self, client):
        body = client.post(
            "/api/search", json={"query": "GDP", "top_k": 3, "pipeline": "pipeline_a"}
        ).json()
        assert body["pipeline"] == "pipeline_a"

    def test_an_unknown_pipeline_is_rejected(self, client):
        response = client.post(
            "/api/search", json={"query": "GDP", "pipeline": "pipeline_z"}
        )
        assert response.status_code in (400, 422)

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": ""},
            {"query": "   "},
            {"query": "x", "top_k": 0},
            {"query": "x", "top_k": 5000},
            {"query": "x", "query_type": "telepathy"},
            {"query": "x", "unknown_field": 1},
        ],
    )
    def test_malformed_requests_are_rejected(self, client, payload):
        assert client.post("/api/search", json=payload).status_code == 422


@pytest.mark.slow
class TestQueryParsing:
    def test_reports_the_inferred_type(self, client):
        body = client.post("/api/search/parse", json={"query": "RBI AND credit"}).json()
        assert body["query_type"] == "boolean_and"
        assert body["valid"] is True
        assert body["error"] is None

    def test_normalizes_a_boolean_expression(self, client):
        body = client.post(
            "/api/search/parse", json={"query": "(GDP OR GVA) AND NOT inflation"}
        ).json()
        assert body["valid"] is True
        assert body["query_type"] == "boolean_group"
        assert body["normalized"] == "(GDP OR GVA) AND NOT inflation"

    def test_reports_a_syntax_error_instead_of_guessing(self, client):
        for broken in ("GDP AND", "GDP AND (inflation", "GDP OR OR inflation"):
            body = client.post("/api/search/parse", json={"query": broken}).json()
            assert body["valid"] is False, broken
            assert body["error"], broken
            assert body["normalized"] is None

    def test_a_truncated_boolean_is_not_mistaken_for_a_keyword(self, client):
        # "GDP AND" classifies as boolean, so it must be parsed and rejected
        # rather than waved through as a two-word keyword search.
        body = client.post("/api/search/parse", json={"query": "GDP AND"}).json()
        assert body["query_type"] == "boolean_and"
        assert body["valid"] is False

    def test_an_unrestricted_not_is_rejected_by_the_phase3_parser(self, client):
        body = client.post("/api/search/parse", json={"query": "NOT inflation"}).json()
        assert body["valid"] is False
        assert "A AND NOT B" in body["error"]

    def test_a_not_over_an_or_is_rejected(self, client):
        body = client.post(
            "/api/search/parse", json={"query": "GDP AND NOT (inflation OR GVA)"}
        ).json()
        assert body["valid"] is False

    def test_a_bare_multi_word_query_is_a_keyword_search(self, client):
        # Without quotes, two words are a keyword search over both terms. A
        # phrase search has to be asked for explicitly.
        body = client.post("/api/search/parse", json={"query": "monetary policy"}).json()
        assert body["query_type"] == "keyword"
        assert body["valid"] is True
        assert body["normalized"] == "monetary policy"

    def test_a_quoted_query_is_a_phrase_search(self, client):
        body = client.post("/api/search/parse", json={"query": '"monetary policy"'}).json()
        assert body["query_type"] == "phrase"
        assert body["valid"] is True


@pytest.mark.slow
class TestIndex:
    def test_term_lookup_normalizes_to_the_indexed_stem(self, client):
        body = client.get("/api/index/term", params={"term": "inflation"}).json()
        assert body["normalized_terms"] == ["inflat"]
        assert body["matched_terms"] == ["inflat"]
        assert body["searchable"] is True
        assert body["posting_frequency"] > 0
        assert body["postings"]

    def test_each_posting_resolves_to_a_unit_with_a_snippet(self, client):
        body = client.get("/api/index/term", params={"term": "inflation", "limit": 3}).json()
        assert len(body["postings"]) == 3
        for posting in body["postings"]:
            assert posting["unit_id"]
            assert posting["document_id"].startswith("D")
            assert posting["term_frequency"] >= 1
            assert posting["snippet"]

    def test_an_unknown_term_is_reported_as_unsearchable(self, client):
        body = client.get("/api/index/term", params={"term": "zzzqqqxyzzy"}).json()
        assert body["searchable"] is False
        assert body["missing_terms"]
        assert body["postings"] == []

    def test_term_statistics_report_both_orders(self, client):
        first = client.get(
            "/api/index/terms", params={"order": "posting_frequency", "limit": 5}
        ).json()
        second = client.get(
            "/api/index/terms", params={"order": "document_frequency", "limit": 5}
        ).json()
        assert first["available_orders"] == ["posting_frequency", "document_frequency"]
        assert first["items"][0]["posting_frequency"] >= first["items"][-1]["posting_frequency"]
        assert second["items"] != first["items"] or len(first["items"]) == 1

    def test_statistics_match_the_manifest(self, client):
        body = client.get("/api/index/statistics").json()
        assert body["statistics"]["index_terms"] == 21305
        assert body["statistics"]["total_postings"] == 191817
        assert body["manifest"]["indexed_units"] == 6005


class TestPipelines:
    def test_lists_both_with_their_scores(self, client):
        rows = client.get("/api/pipelines").json()
        assert len(rows) == 2
        assert rows[0]["pipeline"] == "pipeline_b"
        assert rows[0]["is_final"] is True
        assert rows[0]["total_score"] == pytest.approx(0.827162)
        assert rows[1]["total_score"] == pytest.approx(0.785496)

    def test_exposes_the_criteria_breakdown(self, client):
        rows = client.get("/api/pipelines").json()
        criteria = rows[0]["criteria"]
        assert set(criteria) == {
            "financial_expression_preservation", "domain_term_recall",
            "variant_collapse_rate", "query_answerability",
        }
        assert criteria["variant_collapse_rate"]["numerator"] == 7
        assert criteria["variant_collapse_rate"]["denominator"] == 12

    def test_the_weights_are_the_phase3_weights(self, client):
        body = client.get("/api/pipelines/selection").json()
        assert body["criteria_weights"] == {
            "financial_expression_preservation": 0.35,
            "domain_term_recall": 0.25,
            "variant_collapse_rate": 0.25,
            "query_answerability": 0.15,
        }

    def test_explains_why_the_runner_up_lost(self, client):
        body = client.get("/api/pipelines/pipeline_a").json()
        assert body["margin_over_runner_up"] == pytest.approx(0.041666)
        assert "variant_collapse_rate" in body["selection_reason"]


# ----------------------------------------------------------------------
# phase 4 evaluation
# ----------------------------------------------------------------------
class TestEvaluation:
    @pytest.fixture(scope="class")
    @classmethod
    def report(cls, client):
        return client.get("/api/evaluation").json()

    def test_reports_all_fifteen_queries(self, report):
        assert report["queries_evaluated"] == 15
        assert report["queries_failed"] == 0
        assert len(report["per_query"]["items"]) == 15

    def test_unit_level_metrics_match_the_summary_csv(self, report):
        unit = report["unit_level"]
        assert unit["precision_macro"] == pytest.approx(0.951111, rel=1e-4)
        assert unit["recall_macro"] == pytest.approx(1.0)
        assert unit["f1_macro"] == pytest.approx(0.972930, rel=1e-5)
        assert unit["judged_pairs"] == 139
        assert unit["judged_relevant"] == 130
        assert unit["judged_not_relevant"] == 9

    def test_precision_at_10_stays_strict_for_short_rankings(self, report):
        # Q06 returned 6 units and Q08 returned 3, yet P@10 still divides by 10.
        # The macro value must therefore be below 1.0.
        unit = report["unit_level"]
        assert unit["precision_at_10_macro"] == pytest.approx(0.8667, rel=1e-3)
        assert unit["precision_at_10_macro"] < 1.0

    def test_document_level_is_reported_as_a_secondary_view(self, report):
        document = report["document_level"]
        assert document["f1_macro"] == pytest.approx(0.724151, rel=1e-5)
        assert document["f1_macro"] < report["unit_level"]["f1_macro"]
        assert "secondary view" in document["notes"]

    def test_per_query_rows_carry_real_numbers(self, report):
        row = report["per_query"]["items"][0]
        assert row["query_id"] == "Q01"
        assert row["evaluable"] is True
        assert isinstance(row["precision"], float)
        assert isinstance(row["average_precision"], float)
        assert row["recall_is_pool_bounded"] is True

    def test_states_that_recall_is_pool_bounded(self, report):
        assert any("pool-bounded" in note for note in report["notes"])

    def test_states_why_the_runner_up_has_no_relevance_metrics(self, report):
        runner_up = report["runner_up"]["pipeline_a"]
        assert runner_up["metrics_available"] is False
        assert "not a fair pool" in runner_up["note"]
        assert runner_up["queries"] == 15
        assert runner_up["answered"] == 15

    def test_comparison_marks_the_unavailable_rows(self, client):
        comparison = client.get("/api/evaluation/comparison").json()["comparison"]
        unavailable = [row for row in comparison if not row["available"]]
        assert unavailable, "the runner-up relevance rows should be N/A"
        assert all(row["pipeline_a"] == "N/A" for row in unavailable)
        assert all(row["selected"] == "pipeline_b" for row in unavailable)
        # The reason lives on one dedicated row rather than on every N/A cell.
        status = next(
            row for row in comparison if row["metric"] == "relevance_evaluation_status"
        )
        assert "not a fair pool" in status["note"]

    def test_comparison_keeps_the_measured_differences(self, client):
        comparison = client.get("/api/evaluation/comparison").json()["comparison"]
        terms = next(row for row in comparison if row["metric"] == "index_terms")
        assert terms["pipeline_a"] == "22552"
        assert terms["pipeline_b"] == "21305"
        assert terms["selected"] == "pipeline_b"
        assert terms["unit"] == "terms"
        assert "phase3" in terms["source"]

    def test_validation_rules_all_pass(self, client):
        body = client.get("/api/evaluation/validation").json()
        assert body["total"] == 16
        assert body["passed"] == 16
        assert body["all_passed"] is True

    def test_exposes_the_judgment_rubric(self, client):
        rubric = client.get("/api/evaluation/summary").json()["judgments"]["rubric"]
        assert "relevance = 1" in rubric
        assert "relevance = 0" in rubric


class TestJudgments:
    def test_lists_the_committed_dataset(self, client):
        body = client.get("/api/evaluation/judgments", params={"limit": 200}).json()
        assert body["total"] == 139

    def test_filters_by_query(self, client):
        body = client.get(
            "/api/evaluation/judgments", params={"query_id": "Q06", "limit": 50}
        ).json()
        assert body["total"] == 6
        assert all(row["query_id"] == "Q06" for row in body["items"])

    def test_filters_by_label(self, client):
        relevant = client.get(
            "/api/evaluation/judgments", params={"relevance": True, "limit": 200}
        ).json()
        not_relevant = client.get(
            "/api/evaluation/judgments", params={"relevance": False, "limit": 200}
        ).json()
        assert relevant["total"] == 130
        assert not_relevant["total"] == 9
        assert all(row["relevance"] == "1" for row in relevant["items"])
        assert all(row["relevance"] == "0" for row in not_relevant["items"])

    def test_every_judgment_is_within_the_pool_depth(self, client):
        body = client.get("/api/evaluation/judgments", params={"limit": 200}).json()
        per_query: dict = {}
        for row in body["items"]:
            per_query[row["query_id"]] = per_query.get(row["query_id"], 0) + 1
        assert len(per_query) == 15
        assert max(per_query.values()) <= 10


# ----------------------------------------------------------------------
# the single write path
# ----------------------------------------------------------------------
class TestSaveJudgment:
    @pytest.fixture()
    def stored(self, client, temp_judgment_file):
        """Save one judgment, then hand the caller the resulting CSV path."""
        response = client.post(
            "/api/evaluation/judgments",
            json={
                "query_id": "Q01",
                "unit_id": "D01_P999_SEC_9_9_PAR999",
                "relevance": False,
                "annotator": "api-test",
                "notes": "a footer that only matches lexically",
            },
        )
        assert response.status_code == 201, response.text
        return response.json(), temp_judgment_file

    def test_returns_201_and_echoes_the_judgment(self, stored):
        body, _ = stored
        assert body["saved"] is True
        assert body["judgment"]["query_id"] == "Q01"
        assert body["judgment"]["relevance"] == 0
        assert body["judgment"]["relevance_label"] == "not_relevant"
        assert body["judgment"]["annotator"] == "api-test"

    def test_the_label_reaches_the_csv(self, stored):
        _, path = stored
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        match = [r for r in rows if r["unit_id"] == "D01_P999_SEC_9_9_PAR999"]
        assert len(match) == 1
        assert match[0]["relevance"] == "0"
        assert match[0]["annotator"] == "api-test"
        assert list(rows[0]) == list(FIELDNAMES)

    def test_re_saving_the_same_pair_replaces_rather_than_appends(self, client, stored):
        client.post(
            "/api/evaluation/judgments",
            json={
                "query_id": "Q01",
                "unit_id": "D01_P999_SEC_9_9_PAR999",
                "relevance": True,
                "annotator": "api-test",
                "notes": "re-read: it does discuss inflation",
            },
        )
        _, path = stored
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        match = [r for r in rows if r["unit_id"] == "D01_P999_SEC_9_9_PAR999"]
        assert len(match) == 1
        assert match[0]["relevance"] == "1"

    def test_the_new_judgment_is_readable_through_the_api(self, client, stored):
        body = client.get(
            "/api/evaluation/judgments", params={"query_id": "Q01", "limit": 50}
        ).json()
        assert body["total"] == 11

    def test_tells_the_caller_how_to_recompute_the_metrics(self, stored):
        body, _ = stored
        assert "src.phase4.run" in body["message"]

    def test_refuses_a_retrieval_score_as_input(self, client):
        """A label derived from a score would make the evaluation circular."""
        response = client.post(
            "/api/evaluation/judgments",
            json={
                "query_id": "Q01", "unit_id": "D01_A", "relevance": True,
                "annotator": "api-test", "score": 0.99,
            },
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "payload",
        [
            {"query_id": "Q1", "unit_id": "D01_A", "relevance": True, "annotator": "a"},
            {"query_id": "Q01", "unit_id": "../escape", "relevance": True, "annotator": "a"},
            {"query_id": "Q01", "unit_id": "D01_A", "relevance": True, "annotator": ""},
            {"query_id": "Q01", "unit_id": "D01_A", "relevance": "yes", "annotator": "a"},
            {"query_id": "Q01", "unit_id": "D01_A", "relevance": True},
        ],
    )
    def test_rejects_malformed_judgments(self, client, payload):
        assert client.post("/api/evaluation/judgments", json=payload).status_code == 422


# ----------------------------------------------------------------------
# input rejection across the whole surface
# ----------------------------------------------------------------------
class TestInputRejection:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/documents/../../etc/passwd",
            "/api/documents/..%2F..%2Fetc%2Fpasswd",
            "/api/units/..%2F..%2Fconfig",
            "/api/pipelines/..%2F..",
        ],
    )
    def test_path_traversal_never_serves_a_file(self, client, path):
        response = client.get(path)
        assert response.status_code == 404
        assert "root:" not in response.text

    def test_every_error_shares_one_shape(self, client):
        for path in ("/api/documents/D99", "/api/does-not-exist"):
            body = client.get(path).json()
            assert set(body) <= {"error", "detail", "hint"}
            assert body["error"] and body["detail"]

    def test_a_validation_error_names_the_field(self, client):
        body = client.post("/api/search", json={"query": "x", "top_k": 0}).json()
        assert body["error"] == "unprocessable"
        assert "top_k" in body["detail"]
        assert body["hint"]

    def test_the_frontend_origin_is_allowed_by_cors(self, client):
        response = client.get(
            "/api/health", headers={"Origin": "http://localhost:3000"}
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_an_unknown_origin_is_not_allowed(self, client):
        response = client.get(
            "/api/health", headers={"Origin": "https://example.invalid"}
        )
        assert "access-control-allow-origin" not in response.headers

    def test_oversized_page_limits_are_rejected(self, client):
        assert client.get("/api/documents", params={"limit": 100000}).status_code == 422
        assert client.get("/api/experiments/ner", params={"limit": 100000}).status_code == 422
        assert client.get("/api/experiments/ngrams", params={"limit": 100000}).status_code == 422
