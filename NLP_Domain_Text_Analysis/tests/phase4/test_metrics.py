"""Metric definitions, including the zero-division cases that decide a score."""

from __future__ import annotations

import pytest

from src.phase4.metrics import (
    NO_DENOMINATOR,
    Counts,
    Metric,
    average_precision,
    evaluate_ranking,
    f1_of,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_divides_by_k_not_by_the_number_retrieved(self):
        # Three results, two relevant: P@3 is 2/3, not 2/2.
        assert precision_at_k(["a", "b", "c"], {"a": 1, "b": 1, "c": 0}, 3).value == pytest.approx(2 / 3)

    def test_unjudged_units_still_occupy_a_slot(self):
        # An unjudged unit in the top K is not evidence of relevance, so it
        # lowers precision rather than being quietly dropped.
        assert precision_at_k(["a", "z"], {"a": 1}, 2).value == pytest.approx(0.5)

    def test_unjudged_units_do_not_count_as_relevant(self):
        assert precision_at_k(["z", "a"], {"a": 1}, 2).value == pytest.approx(0.5)

    def test_k_larger_than_the_ranking_still_divides_by_k(self):
        # One relevant in the top 2 out of a K of 10: 0.1, not 0.5. A query that
        # returns fewer than K results is not credited for the empty slots.
        assert precision_at_k(["a", "b"], {"a": 1, "b": 0}, 10).value == pytest.approx(0.1)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_is_undefined(self, k):
        metric = precision_at_k(["a"], {"a": 1}, k)
        assert metric.value is None
        assert metric.reason

    def test_empty_ranking_is_undefined_not_zero(self):
        metric = precision_at_k([], {"a": 1}, 10)
        assert metric.value is None
        assert metric.reason == NO_DENOMINATOR


class TestRecallAtK:
    def test_recall_is_bounded_by_the_judged_pool(self):
        # Three relevant units exist in the pool; only one was retrieved.
        assert recall_at_k(["a"], {"a": 1, "b": 1, "c": 1}, 10).value == pytest.approx(1 / 3)

    def test_explicit_relevant_total_overrides_the_pool(self):
        assert recall_at_k(["a"], {"a": 1}, 10, relevant_total=4).value == pytest.approx(0.25)

    def test_no_relevant_units_is_undefined(self):
        # 0/0 is not 0. Reporting 0 would understate a system that returned
        # nothing wrong on a query with no relevant material.
        metric = recall_at_k(["a"], {"a": 0}, 10)
        assert metric.value is None
        assert "relevant" in metric.reason.lower()


class TestF1:
    def test_symmetric(self):
        assert f1_of(1.0, 1.0).value == pytest.approx(1.0)
        assert f1_of(0.5, 0.5).value == pytest.approx(0.5)

    def test_harmonic_mean(self):
        # P=1, R=0.5 -> 2PR/(P+R) = 1/1.5
        assert f1_of(1.0, 0.5).value == pytest.approx(2 / 3)

    def test_undefined_when_either_side_is_undefined(self):
        assert f1_of(None, 1.0).value is None
        assert f1_of(1.0, None).value is None

    def test_both_zero_is_zero(self):
        assert f1_of(0.0, 0.0).value == pytest.approx(0.0)


class TestAveragePrecision:
    def test_relevant_first_beats_relevant_last(self):
        early = average_precision(["a", "b", "c", "d"], {"a": 1, "c": 1})
        late = average_precision(["d", "c", "b", "a"], {"a": 1, "c": 1})
        assert early.value > late.value

    def test_perfect_ranking_is_one(self):
        assert average_precision(["a", "b"], {"a": 1, "b": 1}).value == pytest.approx(1.0)

    def test_no_relevant_retrieved_is_zero_not_undefined(self):
        assert average_precision(["a", "b"], {"a": 0, "b": 0}).value == pytest.approx(0.0)

    def test_empty_ranking_scores_zero_and_says_why(self):
        # Nothing was retrieved, so nothing relevant was retrieved either: AP is
        # 0 on the R_precision variant, and the reason is carried alongside so a
        # reader can tell it apart from a query that returned wrong units.
        metric = average_precision([], {"a": 1})
        assert metric.value == pytest.approx(0.0)
        assert metric.reason == NO_DENOMINATOR

    def test_average_precision_uses_total_relevant_items_denominator(self):
        # Case: relevant set = {A, B, C}, ranking = [A, X, B]
        # P@1 = 1/1 = 1.0, P@3 = 2/3 ≈ 0.6667
        # Expected AP = (1.0 + 2/3) / 3 = (5/3) / 3 = 5/9 ≈ 0.5556
        labels = {"A": 1, "B": 1, "C": 1, "X": 0}
        ap = average_precision(["A", "X", "B"], labels, relevant_total=3)
        assert ap.value == pytest.approx(5 / 9)


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        labels = {"a": 1, "b": 1, "c": 0}
        assert ndcg_at_k(["a", "b", "c"], labels, k=3).value == pytest.approx(1.0)

    def test_relevant_result_appearing_later_is_between_zero_and_one(self):
        labels = {"a": 1, "b": 1, "x": 0}
        metric = ndcg_at_k(["x", "a", "b"], labels, k=3)
        assert 0.0 < metric.value < 1.0

    def test_no_relevant_items_is_zero(self):
        labels = {"a": 0, "b": 0}
        assert ndcg_at_k(["a", "b"], labels, k=2).value == pytest.approx(0.0)

    def test_k_larger_than_ranking_length(self):
        labels = {"a": 1, "b": 0}
        metric = ndcg_at_k(["a", "b"], labels, k=10)
        assert metric.value == pytest.approx(1.0)

    def test_binary_labels_only(self):
        labels = {"a": 1, "b": 0, "c": 1}
        m1 = ndcg_at_k(["a", "b", "c"], labels, k=1)
        m3 = ndcg_at_k(["a", "b", "c"], labels, k=3)
        assert m1.value == pytest.approx(1.0)
        assert 0.0 <= m3.value <= 1.0


class TestEvaluateRanking:
    def test_counts_a_mixed_ranking(self):
        counts = evaluate_ranking(["a", "b", "c", "d"], {"a": 1, "b": 0, "c": 1})
        assert counts.retrieved == 4
        assert counts.judged == 3
        assert counts.unjudged == 1
        assert counts.relevant_retrieved == 2
        assert counts.relevant_total == 2
        assert counts.false_positives == 1      # b: retrieved and judged not relevant
        assert counts.false_negatives == 0      # both relevant units were retrieved
        assert counts.precision.value == pytest.approx(2 / 3)
        assert counts.recall.value == pytest.approx(1.0)

    def test_unjudged_is_excluded_from_precision_by_default(self):
        counts = evaluate_ranking(["a", "z"], {"a": 1})
        assert counts.judged == 1
        assert counts.precision.value == pytest.approx(1.0)

    def test_unjudged_can_be_counted_as_not_relevant(self):
        counts = evaluate_ranking(
            ["a", "z"], {"a": 1}, unjudged_policy="count_as_not_relevant"
        )
        assert counts.judged == 2
        assert counts.precision.value == pytest.approx(0.5)

    def test_relevant_total_may_exceed_what_was_retrieved(self):
        counts = evaluate_ranking(["a"], {"a": 1, "b": 1, "c": 1})
        assert counts.false_negatives == 2
        assert counts.recall.value == pytest.approx(1 / 3)

    def test_empty_ranking_leaves_metrics_undefined(self):
        counts = evaluate_ranking([], {"a": 1})
        assert counts.retrieved == 0
        assert counts.precision.value is None
        assert counts.recall.value == pytest.approx(0.0)


class TestMetric:
    def test_repr_carries_the_reason(self):
        assert "no denominator" in repr(Metric(None, "no denominator"))

    def test_defined_reports_no_reason(self):
        assert Metric(0.0, "").defined
        assert Metric(None, "x").defined is False

    def test_rounded_keeps_none_as_none(self):
        assert Metric(0.123456, "").rounded(2) == pytest.approx(0.12)
        assert Metric(None, "x").rounded() is None

    def test_counts_serialise_to_plain_types(self):
        payload = Counts(retrieved=1, relevant_total=2).as_json()
        assert set(payload) == {
            "retrieved", "judged", "unjudged", "relevant_retrieved",
            "relevant_total", "false_positives", "false_negatives",
        }
