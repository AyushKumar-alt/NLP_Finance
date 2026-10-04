"""Run the Phase 3 retrieval subsystem against the Phase 4 judgments.

The evaluator never re-implements ranking, tokenization or boolean logic. It
calls :class:`src.phase3.retrieval.RetrievalEngine`, takes the ranking it
returns, joins the human labels, and reports the metrics.

Two views are produced for every query:

``judged_depth``
    the ranking truncated to the depth at which judgments exist. Every item
    considered carries a label, so precision, recall and F1 are exact on the
    judged set.

``full_ranking``
    the whole ranking the engine returned. Items beyond the judged depth have
    no label; they are counted as ``unjudged`` and excluded from the precision
    denominator rather than being silently marked wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.phase3.retrieval import RetrievalEngine
from src.phase3.query_parser import QuerySyntaxError

from .config import Phase4Config, log_event
from .metrics import (
    Aggregate,
    Counts,
    Metric,
    aggregate,
    aggregate_at_k,
    average_precision,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .relevance import RelevanceStore


@dataclass
class QueryEvaluation:
    """Everything measured for one query on one pipeline."""

    query_id: str
    query: str
    query_type: str
    description: str = ""
    pipeline: str = ""
    ranking: List[str] = field(default_factory=list)
    labels: Dict[str, int] = field(default_factory=dict)
    counts: Counts = field(default_factory=Counts)
    full_counts: Counts = field(default_factory=Counts)
    metrics: Dict[str, Metric] = field(default_factory=dict)
    document_ranking: List[str] = field(default_factory=list)
    document_labels: Dict[str, int] = field(default_factory=dict)
    document_metrics: Dict[str, Metric] = field(default_factory=dict)
    document_counts: Counts = field(default_factory=Counts)
    execution_time_ms: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)
    retrieved_units: int = 0
    retrieved_documents: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class EvaluationRun:
    """The whole evaluation, at every depth and for every pipeline tried."""

    pipeline: str
    queries: List[QueryEvaluation] = field(default_factory=list)
    judged_depth: int = 0
    unjudged_policy: str = ""
    aggregates: Dict[str, Aggregate] = field(default_factory=dict)
    at_k: Dict[int, Dict[str, object]] = field(default_factory=dict)
    document_aggregate: Optional[Aggregate] = None
    problems: List[str] = field(default_factory=list)

    def evaluated_queries(self) -> List[QueryEvaluation]:
        return [q for q in self.queries if q.ok]

    def as_json(self) -> Dict[str, object]:
        return {
            "pipeline": self.pipeline,
            "judged_depth": self.judged_depth,
            "unjudged_policy": self.unjudged_policy,
            "queries_evaluated": len(self.evaluated_queries()),
            "queries_failed": len(self.queries) - len(self.evaluated_queries()),
            "aggregates": {name: agg.as_json() for name, agg in self.aggregates.items()},
            "at_k": {str(k): v for k, v in self.at_k.items()},
            "document_level": self.document_aggregate.as_json() if self.document_aggregate else None,
            "problems": list(self.problems),
        }


# ----------------------------------------------------------------------
def _labels_for(store: RelevanceStore, query_id: str, allowed: Optional[set]) -> Dict[str, int]:
    """Labels for one query, optionally restricted to a set of unit ids."""
    labels: Dict[str, int] = {}
    for (qid, unit_id), judgment in store.judgments.items():
        if qid != query_id:
            continue
        if allowed is not None and unit_id not in allowed:
            continue
        labels[unit_id] = judgment.relevance
    return labels


def evaluate_query(
    engine: RetrievalEngine,
    query_id: str,
    query_text: str,
    query_type: str,
    store: RelevanceStore,
    judged_depth: int,
    description: str = "",
    pipeline: str = "",
    unjudged_policy: str = "exclude_from_precision_denominator",
    k_values: Sequence[int] = (5, 10),
    include_document_granularity: bool = True,
) -> QueryEvaluation:
    """Run one query and measure it. Never raises: a failure is recorded."""
    result = QueryEvaluation(
        query_id=query_id,
        query=query_text,
        query_type=query_type,
        description=description,
        pipeline=pipeline or engine.spec.key,
    )

    try:
        outcome = engine.search(query_text, query_type or None)
    except QuerySyntaxError as error:
        result.error = f"invalid query: {error}"
        return result
    except Exception as error:                       # noqa: BLE001 - recorded, not raised
        result.error = f"{type(error).__name__}: {error}"
        return result

    result.ranking = [hit.unit_id for hit in outcome.unit_hits]
    result.retrieved_units = outcome.unit_count
    result.retrieved_documents = outcome.document_count
    result.execution_time_ms = outcome.execution_time_ms
    result.matched_terms = list(outcome.matched_terms)
    result.missing_terms = list(outcome.missing_terms)

    # --- unit level, at the depth that was actually judged ---------------
    judged_set = {
        unit_id for (qid, unit_id) in store.judgments if qid == query_id
    }
    depth = judged_depth if judged_depth > 0 else len(result.ranking)
    head = result.ranking[:depth]
    labels = _labels_for(store, query_id, judged_set)
    counts = evaluate_ranking(head, labels, unjudged_policy=unjudged_policy)
    result.labels = labels
    result.counts = counts
    metrics: Dict[str, Metric] = {
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "average_precision": average_precision(head, labels, counts.relevant_total),
        "reciprocal_rank": reciprocal_rank(head, labels),
    }
    all_k = sorted(set(list(k_values) + [1, 3, 5, 10]))
    for k in all_k:
        metrics[f"precision_at_{k}"] = precision_at_k(head, labels, k)
        metrics[f"recall_at_{k}"] = recall_at_k(head, labels, k, counts.relevant_total)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(head, labels, k, counts.relevant_total)
    result.metrics = metrics

    # --- full ranking view ------------------------------------------------
    full = evaluate_ranking(result.ranking, labels, unjudged_policy=unjudged_policy)
    result.full_counts = full

    # --- document level, a secondary view ---------------------------------
    if include_document_granularity:
        unit_to_doc = {hit.unit_id: hit.document_id for hit in outcome.unit_hits}
        doc_ranking: List[str] = []
        seen: set = set()
        for hit in outcome.unit_hits:
            if hit.document_id not in seen:
                seen.add(hit.document_id)
                doc_ranking.append(hit.document_id)
        result.document_ranking = doc_ranking
        # A document counts as relevant when it contains at least one judged
        # relevant unit, which is the standard document-level reduction of
        # unit-level judgments.
        result.document_labels = {
            document_id: (1 if any(labels.get(unit_id) == 1 for unit_id in doc_units) else 0)
            for document_id, doc_units in _group_by_document(unit_to_doc).items()
        }
        doc_counts = evaluate_ranking(
            doc_ranking[:judged_depth if judged_depth > 0 else len(doc_ranking)],
            result.document_labels,
            unjudged_policy=unjudged_policy,
        )
        result.document_metrics = {
            "precision": doc_counts.precision,
            "recall": doc_counts.recall,
            "f1": doc_counts.f1,
        }
        result.document_counts = doc_counts
    return result


def _group_by_document(unit_to_document: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for unit_id, document_id in unit_to_document.items():
        grouped.setdefault(document_id, []).append(unit_id)
    return grouped


# ----------------------------------------------------------------------
def evaluate_pipeline(
    engine: RetrievalEngine,
    queries: Sequence[Dict[str, str]],
    store: RelevanceStore,
    config: Phase4Config,
    logger: Optional[logging.Logger] = None,
) -> EvaluationRun:
    """Evaluate every configured query on one pipeline."""
    pipeline = engine.spec.key
    run = EvaluationRun(
        pipeline=pipeline,
        judged_depth=config.pool_depth,
        unjudged_policy=config.unjudged_policy,
    )
    k_values = config.k_values

    for query in queries:
        query_id = str(query.get("id", "")).strip()
        if not query_id:
            run.problems.append(f"skipping malformed query entry: {query!r}")
            continue
        evaluation = evaluate_query(
            engine=engine,
            query_id=query_id,
            query_text=str(query.get("query", "")).strip(),
            query_type=str(query.get("type", "")).strip(),
            store=store,
            judged_depth=config.pool_depth,
            description=str(query.get("description", "")).strip(),
            pipeline=pipeline,
            unjudged_policy=config.unjudged_policy,
            k_values=k_values,
            include_document_granularity=bool(
                config.get("evaluation.include_document_granularity", True)
            ),
        )
        run.queries.append(evaluation)
        if evaluation.ok and logger is not None:
            counts = evaluation.counts
            log_event(
                logger,
                "INFO",
                "evaluate",
                f"{query_id} {evaluation.query!r} [{evaluation.query_type}]: "
                f"{counts.relevant_retrieved}/{counts.judged} judged retrieved relevant, "
                f"{counts.relevant_total} judged relevant in total, "
                f"{evaluation.execution_time_ms}ms",
            )
        elif not evaluation.ok and logger is not None:
            log_event(logger, "ERROR", "evaluate", f"{query_id}: {evaluation.error}")

    rows = []
    for query in run.evaluated_queries():
        row: Dict[str, object] = {"counts": query.counts, "labels": query.labels}
        row.update(query.metrics)
        row["ranking_head"] = query.ranking[: max(k_values or [0])]
        rows.append(row)
    run.aggregates["judged_depth"] = aggregate(rows)
    for k in k_values:
        run.at_k[k] = aggregate_at_k(rows, k)
        run.at_k[k]["micro"] = _micro_at_k(run.evaluated_queries(), k)
    if any(q.document_metrics for q in run.evaluated_queries()):
        doc_rows = [{"counts": getattr(q, "document_counts"), "labels": q.document_labels}
                    for q in run.evaluated_queries() if q.document_metrics]
        for row in doc_rows:
            row.update({"precision": row["counts"].precision,
                        "recall": row["counts"].recall,
                        "f1": row["counts"].f1})
        run.document_aggregate = aggregate(doc_rows)
    return run


def _micro_at_k(queries: Sequence[QueryEvaluation], k: int) -> Dict[str, object]:
    """Pooled P@K and R@K over the queries, with the honest denominators."""
    hits = 0
    judged = 0
    relevant_total = 0
    for query in queries:
        head = query.ranking[:k]
        judged += len(head)
        relevant_total += query.counts.relevant_total
        hits += sum(1 for unit_id in head if query.labels.get(unit_id) == 1)
    return {
        "k": k,
        "pooled_hits": hits,
        "pooled_slots": judged,
        "pooled_precision_at_k": None if judged == 0 else round(hits / judged, 4),
        "pooled_recall_at_k": None if relevant_total == 0 else round(hits / relevant_total, 4),
        "relevant_total": relevant_total,
    }


def probe_pipeline(
    engine: RetrievalEngine,
    queries: Sequence[Dict[str, str]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Retrieval statistics for a pipeline, with **no** relevance metrics.

    Used for the runner-up. The judgment pool was built by pooling the *selected*
    pipeline's rankings, so those labels are not a fair pool for a different
    index: most of its top-N units carry no label. Scoring it anyway would
    produce a number that looks like a comparison and is not one. This returns
    only what *is* comparable - answerability, throughput and volume.
    """
    answered = 0
    units = 0
    documents = 0
    elapsed = 0.0
    missing_term_queries = 0
    failures: List[str] = []
    per_type: Dict[str, Dict[str, int]] = {}

    for query in queries:
        query_id = str(query.get("id", "")).strip()
        text = str(query.get("query", "")).strip()
        query_type = str(query.get("type", "")).strip()
        try:
            outcome = engine.search(text, query_type or None)
        except Exception as error:                   # noqa: BLE001 - recorded, not raised
            failures.append(f"{query_id}: {type(error).__name__}: {error}")
            continue
        answered += 1
        units += outcome.unit_count
        documents += outcome.document_count
        elapsed += outcome.execution_time_ms
        if outcome.missing_terms:
            missing_term_queries += 1
        bucket = per_type.setdefault(outcome.query_type, {"queries": 0, "answered": 0, "units": 0})
        bucket["queries"] += 1
        bucket["answered"] += 1
        bucket["units"] += outcome.unit_count

    total = len(queries)
    return {
        "pipeline": engine.spec.key,
        "pipeline_name": engine.spec.name,
        "queries": total,
        "answered": answered,
        "answerability": round(answered / total, 4) if total else None,
        "total_units_returned": units,
        "total_documents_returned": documents,
        "mean_units_per_query": round(units / answered, 4) if answered else None,
        "mean_execution_time_ms": round(elapsed / answered, 3) if answered else None,
        "queries_with_missing_terms": missing_term_queries,
        "by_query_type": per_type,
        "failures": failures,
        "metrics_available": False,
        "note": (
            "retrieval statistics only. Relevance metrics are N/A for this pipeline "
            "because the judgment pool was built by pooling the selected pipeline's "
            "rankings, which is not a fair pool for a different index."
        ),
    }


def evaluate(
    config: Phase4Config,
    queries: Sequence[Dict[str, str]],
    store: RelevanceStore,
    engines: Dict[str, RetrievalEngine],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, EvaluationRun]:
    """Evaluate every requested pipeline."""
    runs: Dict[str, EvaluationRun] = {}
    for key, engine in engines.items():
        runs[key] = evaluate_pipeline(engine, queries, store, config, logger=logger)
    return runs

