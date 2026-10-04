"""Retrieval evaluation metrics.

All metrics are computed from real counts. Division by zero is never hidden: a
metric whose denominator is zero is reported as ``None`` and the reason is
carried alongside, so "no relevant units" can never masquerade as "perfect".

Definitions used here (``R`` is the ranking of retrieved content units)::

    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)
    F1          = 2 * Precision * Recall / (Precision + Recall)
    P@K         = |relevant retrieved in top K| / K
    R@K         = |relevant retrieved in top K| / |all judged relevant|
    MAP         = mean over queries of the average of the precisions at the
                  ranks of each judged relevant unit that the query retrieved

Unjudged units are not silently scored as wrong. A retrieved unit with no
judgment is excluded from the precision denominator and counted separately, and
recall is measured against the *judged* relevant set - which makes recall
pool-bounded by construction. The pool depth is reported with every result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

#: Reason strings attached to a metric that could not be computed.
NO_DENOMINATOR = "no retrieved units to score"
NO_RELEVANT = "no judged relevant unit for this query"
NO_PRECISION = "no retrieved unit was judged, so precision is undefined"
NO_RECALL = "no judged relevant unit, so recall is undefined"


@dataclass
class Metric:
    """One metric value, or the reason it is undefined."""

    value: Optional[float]
    reason: str = ""

    @property
    def defined(self) -> bool:
        return self.value is not None

    def rounded(self, digits: int = 4) -> Optional[float]:
        return None if self.value is None else round(self.value, digits)

    def as_json(self) -> Optional[float]:
        return self.rounded()


def _ratio(numerator: float, denominator: float, reason: str) -> Metric:
    if denominator == 0:
        return Metric(None, reason)
    return Metric(numerator / denominator)


def f1_of(precision: Optional[float], recall: Optional[float]) -> Metric:
    if precision is None or recall is None:
        return Metric(None, "F1 needs both precision and recall")
    total = precision + recall
    if total == 0:
        # precision == recall == 0: the ranking returned nothing relevant at all.
        # Defined as 0, not as undefined, and stated here so the choice is visible.
        return Metric(0.0, "")
    return Metric(2 * precision * recall / total)


# ----------------------------------------------------------------------
@dataclass
class Counts:
    """The confusion counts behind one query-level measurement."""

    retrieved: int = 0            # units considered (the ranking prefix)
    judged: int = 0               # of those, how many carry a judgment
    unjudged: int = 0             # retrieved but never judged
    relevant_retrieved: int = 0   # TP
    judged_non_relevant: int = 0  # FP
    relevant_total: int = 0       # every judged relevant unit for this query

    @property
    def true_positives(self) -> int:
        return self.relevant_retrieved

    @property
    def false_positives(self) -> int:
        return self.judged_non_relevant

    @property
    def false_negatives(self) -> int:
        return max(0, self.relevant_total - self.relevant_retrieved)

    @property
    def precision(self) -> Metric:
        return _ratio(self.true_positives, self.judged, NO_PRECISION)

    @property
    def recall(self) -> Metric:
        return _ratio(self.true_positives, self.relevant_total, NO_RECALL)

    @property
    def f1(self) -> Metric:
        return f1_of(self.precision.value, self.recall.value)

    def as_json(self) -> Dict[str, object]:
        return {
            "retrieved": self.retrieved,
            "judged": self.judged,
            "unjudged": self.unjudged,
            "relevant_retrieved": self.relevant_retrieved,
            "relevant_total": self.relevant_total,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def evaluate_ranking(
    ranking: Sequence[str],
    labels: Dict[str, int],
    relevant_total: Optional[int] = None,
    unjudged_policy: str = "exclude_from_precision_denominator",
) -> Counts:
    """Count TP/FP/FN/unjudged over a ranking of unit ids.

    ``labels`` maps unit id to ``1`` (relevant) or ``0`` (not relevant); a unit
    absent from the mapping is unjudged. ``relevant_total`` defaults to the
    number of units labelled relevant in the whole judgment pool for the query,
    which is what makes recall pool-bounded.
    """
    counts = Counts()
    if relevant_total is None:
        relevant_total = sum(1 for value in labels.values() if value == 1)
    counts.relevant_total = int(relevant_total)

    for unit_id in ranking:
        counts.retrieved += 1
        if unit_id in labels:
            counts.judged += 1
            if labels[unit_id] == 1:
                counts.relevant_retrieved += 1
            else:
                counts.judged_non_relevant += 1
        else:
            counts.unjudged += 1
            if unjudged_policy == "count_as_not_relevant":
                counts.judged += 1
                counts.judged_non_relevant += 1
    return counts


def precision_at_k(ranking: Sequence[str], labels: Dict[str, int], k: int) -> Metric:
    """``|relevant in top K| / K`` - unjudged units count in the denominator."""
    if k <= 0:
        return Metric(None, "K must be positive")
    head = list(ranking)[:k]
    if not head:
        return Metric(None, NO_DENOMINATOR)
    hits = sum(1 for unit_id in head if labels.get(unit_id) == 1)
    return Metric(hits / k, "")


def recall_at_k(
    ranking: Sequence[str], labels: Dict[str, int], k: int, relevant_total: Optional[int] = None
) -> Metric:
    """``|relevant in top K| / |all judged relevant|``."""
    if k <= 0:
        return Metric(None, "K must be positive")
    total = relevant_total if relevant_total is not None else sum(
        1 for value in labels.values() if value == 1
    )
    if total == 0:
        return Metric(None, NO_RELEVANT)
    head = list(ranking)[:k]
    hits = sum(1 for unit_id in head if labels.get(unit_id) == 1)
    return Metric(hits / total, "")


def average_precision(
    ranking: Sequence[str], labels: Dict[str, int], relevant_total: Optional[int] = None
) -> Metric:
    """Mean precision at the ranks of relevant units divided by total relevant items R_q.

    AP(q) = (1 / R_q) * sum_{k: rel(k)=1} P@k
    where R_q is the total number of judged relevant items for query q in the pool.
    """
    total = relevant_total if relevant_total is not None else sum(
        1 for value in labels.values() if value == 1
    )
    if total == 0:
        return Metric(0.0, "" if ranking else NO_RELEVANT)

    hits = 0
    precision_sum = 0.0
    for position, unit_id in enumerate(ranking, start=1):
        if labels.get(unit_id) == 1:
            hits += 1
            precision_sum += hits / position

    if precision_sum == 0.0:
        return Metric(0.0, "" if ranking else NO_DENOMINATOR)
    return Metric(precision_sum / total, "")


def dcg_at_k(ranking: Sequence[str], labels: Dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain at K for binary relevance labels."""
    if k <= 0:
        return 0.0
    dcg = 0.0
    for position, unit_id in enumerate(list(ranking)[:k], start=1):
        rel = 1 if labels.get(unit_id) == 1 else 0
        if rel > 0:
            dcg += rel / math.log2(position + 1)
    return dcg


def idcg_at_k(labels: Dict[str, int], k: int, relevant_total: Optional[int] = None) -> float:
    """Ideal DCG at K assuming ideal ranking with all relevant items first."""
    if k <= 0:
        return 0.0
    total_rel = relevant_total if relevant_total is not None else sum(1 for v in labels.values() if v == 1)
    ideal_count = min(k, total_rel)
    idcg = 0.0
    for position in range(1, ideal_count + 1):
        idcg += 1.0 / math.log2(position + 1)
    return idcg


def ndcg_at_k(
    ranking: Sequence[str], labels: Dict[str, int], k: int, relevant_total: Optional[int] = None
) -> Metric:
    """Normalized Discounted Cumulative Gain at K (nDCG@K)."""
    if k <= 0:
        return Metric(None, "K must be positive")
    idcg = idcg_at_k(labels, k, relevant_total=relevant_total)
    if idcg == 0.0:
        return Metric(0.0, "" if ranking else NO_RELEVANT)
    dcg = dcg_at_k(ranking, labels, k)
    return Metric(dcg / idcg, "")


def reciprocal_rank(ranking: Sequence[str], labels: Dict[str, int]) -> Metric:
    """Reciprocal Rank (RR): 1 / rank of first relevant unit retrieved."""
    for position, unit_id in enumerate(ranking, start=1):
        if labels.get(unit_id) == 1:
            return Metric(1.0 / position, "")
    return Metric(0.0, "" if ranking else NO_DENOMINATOR)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
@dataclass
class Aggregate:
    """Macro / micro / count summary over a set of query-level measurements."""

    queries: int = 0
    queries_scored: int = 0
    queries_undefined: int = 0
    undefined_reasons: Dict[str, int] = field(default_factory=dict)
    macro: Dict[str, Optional[float]] = field(default_factory=dict)
    micro: Dict[str, Optional[float]] = field(default_factory=dict)
    micro_counts: Dict[str, int] = field(default_factory=dict)

    def as_json(self) -> Dict[str, object]:
        return {
            "queries": self.queries,
            "queries_scored": self.queries_scored,
            "queries_undefined": self.queries_undefined,
            "undefined_reasons": dict(sorted(self.undefined_reasons.items())),
            "macro": {k: (None if v is None else round(v, 4)) for k, v in self.macro.items()},
            "micro": {k: (None if v is None else round(v, 4)) for k, v in self.micro.items()},
            "micro_counts": dict(self.micro_counts),
        }


def _macro(values: Sequence[Metric]) -> Optional[float]:
    """Unweighted mean over the queries where the metric is defined."""
    defined = [m.value for m in values if m.value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def aggregate(measurements: Sequence[Dict[str, object]]) -> Aggregate:
    """Aggregate query-level rows.

    Each row must carry ``precision``, ``recall``, ``f1`` as ``Metric`` objects
    and a ``counts`` :class:`Counts` for the micro totals.
    """
    result = Aggregate(queries=len(measurements))
    keys = ("precision", "recall", "f1")

    for row in measurements:
        if any(getattr(row[k], "value", None) is None for k in keys):
            result.queries_undefined += 1
            for k in keys:
                metric = row.get(k)
                reason = getattr(metric, "reason", "") or "undefined"
                result.undefined_reasons[reason] = result.undefined_reasons.get(reason, 0) + 1
        else:
            result.queries_scored += 1

    for k in keys:
        result.macro[k] = _macro([row[k] for row in measurements if isinstance(row.get(k), Metric)])
    for k in keys:
        result.macro[f"{k}_at_defined_queries"] = result.macro[k]

    # Micro: pool the confusion counts, then recompute the ratios. Micro
    # precision/recall are ratios of sums, never means of ratios.
    totals = Counts()
    for row in measurements:
        counts = row.get("counts")
        if not isinstance(counts, Counts):
            continue
        totals.retrieved += counts.retrieved
        totals.judged += counts.judged
        totals.unjudged += counts.unjudged
        totals.relevant_retrieved += counts.relevant_retrieved
        totals.judged_non_relevant += counts.judged_non_relevant
        totals.relevant_total += counts.relevant_total
    result.micro_counts = {
        "retrieved": totals.retrieved,
        "judged": totals.judged,
        "unjudged": totals.unjudged,
        "relevant_retrieved": totals.relevant_retrieved,
        "relevant_total": totals.relevant_total,
    }
    result.micro["precision"] = totals.precision.rounded()
    result.micro["recall"] = totals.recall.rounded()
    result.micro["f1"] = totals.f1.rounded()
    return result


def aggregate_at_k(
    measurements: Sequence[Dict[str, object]], k: int
) -> Dict[str, object]:
    """Macro P@K / R@K plus the pooled hit count behind them."""
    p_values: List[float] = []
    r_values: List[float] = []
    hits = 0
    relevant_total = 0
    for row in measurements:
        p = row.get(f"precision_at_{k}")
        r = row.get(f"recall_at_{k}")
        if isinstance(p, Metric) and p.value is not None:
            p_values.append(p.value)
        if isinstance(r, Metric) and r.value is not None:
            r_values.append(r.value)
        counts = row.get("counts")
        if isinstance(counts, Counts):
            relevant_total += counts.relevant_total
            # recompute the hit count at k from the stored ranking head
            head = row.get("ranking_head", [])[:k]
            labels = row.get("labels", {})
            hits += sum(1 for unit_id in head if labels.get(unit_id) == 1)
    return {
        "k": k,
        "macro_precision_at_k": None if not p_values else round(sum(p_values) / len(p_values), 4),
        "macro_recall_at_k": None if not r_values else round(sum(r_values) / len(r_values), 4),
        "queries_with_defined_precision_at_k": len(p_values),
        "queries_with_defined_recall_at_k": len(r_values),
        "relevant_retrieved_at_k": hits,
        "relevant_total": relevant_total,
        "pooled_precision_at_k": None if k == 0 else round(hits / k, 4),
    }
