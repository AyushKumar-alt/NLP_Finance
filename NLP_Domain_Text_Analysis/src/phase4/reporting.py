"""Write the Phase 4 artefact set.

Every file written here is derived from real Phase 1-3 artefacts, the human
judgments in ``relevance_judgments.csv`` and metrics computed by
:mod:`src.phase4.metrics`. Nothing is invented, and a metric that could not be
computed is written as an empty cell with the reason in an adjacent column
rather than as a zero.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Phase4Config, log_event
from .evaluator import EvaluationRun, QueryEvaluation
from .metrics import Metric
from .relevance import RelevanceStore

EVALUATION_FIELDS = (
    "query_id", "query", "query_type", "description", "pipeline", "evaluable",
    "retrieved_units", "retrieved_documents", "judged_depth", "considered_units",
    "judged_units", "unjudged_units", "relevant_retrieved", "relevant_total",
    "false_positives", "false_negatives",
    "precision", "recall", "f1", "average_precision",
    "precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10",
    "document_precision", "document_recall", "document_f1",
    "matched_terms", "missing_terms", "execution_time_ms",
    "recall_is_pool_bounded", "undefined_reason", "error",
)

SUMMARY_FIELDS = (
    "scope", "pipeline", "queries", "queries_scored", "queries_undefined",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_micro", "recall_micro", "f1_micro",
    "precision_at_5_macro", "recall_at_5_macro",
    "precision_at_10_macro", "recall_at_10_macro",
    "precision_at_10_micro", "recall_at_10_micro",
    "judged_pairs", "judged_relevant", "judged_not_relevant", "judged_depth",
    "notes",
)

PIPELINE_FIELDS = (
    "metric", "pipeline_a", "pipeline_b", "selected", "unit", "source",
    "note",
)

FINAL_FIELDS = ("section", "metric", "value", "unit", "source", "note")


def _cell(metric: Optional[Metric]) -> str:
    if metric is None:
        return ""
    if metric.value is None:
        return ""
    return str(metric.rounded())


def _reason(metrics: Dict[str, Metric]) -> str:
    parts = []
    for key in ("precision", "recall", "f1", "average_precision"):
        metric = metrics.get(key)
        if metric is not None and metric.value is None and metric.reason:
            parts.append(f"{key}: {metric.reason}")
    return "; ".join(parts)


# ----------------------------------------------------------------------
def evaluation_rows(run: EvaluationRun) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for query in run.queries:
        counts = query.counts if query.ok else None
        full = query.full_counts
        document = query.document_metrics if query.document_metrics else {}
        row: Dict[str, Any] = {
            "query_id": query.query_id,
            "query": query.query,
            "query_type": query.query_type,
            "description": query.description,
            "pipeline": query.pipeline,
            "evaluable": "true" if query.ok else "false",
            "retrieved_units": query.retrieved_units,
            "retrieved_documents": query.retrieved_documents,
            "judged_depth": run.judged_depth,
            "considered_units": counts.retrieved if counts else 0,
            "judged_units": counts.judged if counts else 0,
            "unjudged_units": counts.unjudged if counts else 0,
            "relevant_retrieved": counts.relevant_retrieved if counts else 0,
            "relevant_total": counts.relevant_total if counts else 0,
            "false_positives": counts.false_positives if counts else 0,
            "false_negatives": counts.false_negatives if counts else 0,
            "precision": _cell(query.metrics.get("precision")),
            "recall": _cell(query.metrics.get("recall")),
            "f1": _cell(query.metrics.get("f1")),
            "average_precision": _cell(query.metrics.get("average_precision")),
            "precision_at_5": _cell(query.metrics.get("precision_at_5")),
            "recall_at_5": _cell(query.metrics.get("recall_at_5")),
            "precision_at_10": _cell(query.metrics.get("precision_at_10")),
            "recall_at_10": _cell(query.metrics.get("recall_at_10")),
            "document_precision": _cell(document.get("precision")),
            "document_recall": _cell(document.get("recall")),
            "document_f1": _cell(document.get("f1")),
            "matched_terms": ";".join(query.matched_terms),
            "missing_terms": ";".join(query.missing_terms),
            "execution_time_ms": query.execution_time_ms,
            "recall_is_pool_bounded": "true" if query.ok else "",
            "undefined_reason": _reason(query.metrics),
            "error": query.error,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


# ----------------------------------------------------------------------
def summary_rows(
    runs: Dict[str, EvaluationRun],
    store: RelevanceStore,
    config: Phase4Config,
) -> List[Dict[str, Any]]:
    coverage = store.coverage()
    judged_pairs = len(store)
    judged_relevant = sum(1 for j in store.judgments.values() if j.is_relevant)
    judged_not = judged_pairs - judged_relevant
    rows: List[Dict[str, Any]] = []

    for pipeline, run in sorted(runs.items()):
        aggregate = run.aggregates.get("judged_depth")
        at10 = run.at_k.get(10, {})
        at5 = run.at_k.get(5, {})
        macro = aggregate.macro if aggregate else {}
        micro = aggregate.micro if aggregate else {}
        rows.append(
            {
                "scope": "content_unit",
                "pipeline": pipeline,
                "queries": aggregate.queries if aggregate else 0,
                "queries_scored": aggregate.queries_scored if aggregate else 0,
                "queries_undefined": aggregate.queries_undefined if aggregate else 0,
                "precision_macro": macro.get("precision"),
                "recall_macro": macro.get("recall"),
                "f1_macro": macro.get("f1"),
                "precision_micro": micro.get("precision"),
                "recall_micro": micro.get("recall"),
                "f1_micro": micro.get("f1"),
                "precision_at_5_macro": at5.get("macro_precision_at_k"),
                "recall_at_5_macro": at5.get("macro_recall_at_k"),
                "precision_at_10_macro": at10.get("macro_precision_at_k"),
                "recall_at_10_macro": at10.get("macro_recall_at_k"),
                "precision_at_10_micro": (at10.get("micro") or {}).get("pooled_precision_at_k"),
                "recall_at_10_micro": (at10.get("micro") or {}).get("pooled_recall_at_k"),
                "judged_pairs": judged_pairs,
                "judged_relevant": judged_relevant,
                "judged_not_relevant": judged_not,
                "judged_depth": run.judged_depth,
                "notes": (
                    "precision/recall/F1 measured at the judged depth, where every "
                    "considered unit carries a label; recall is bounded by the judgment "
                    "pool and is therefore a pool-bounded recall, not an absolute one"
                ),
            }
        )

        document = run.document_aggregate
        if document is not None:
            rows.append(
                {
                    "scope": "document",
                    "pipeline": pipeline,
                    "queries": document.queries,
                    "queries_scored": document.queries_scored,
                    "queries_undefined": document.queries_undefined,
                    "precision_macro": document.macro.get("precision"),
                    "recall_macro": document.macro.get("recall"),
                    "f1_macro": document.macro.get("f1"),
                    "precision_micro": document.micro.get("precision"),
                    "recall_micro": document.micro.get("recall"),
                    "f1_micro": document.micro.get("f1"),
                    "judged_pairs": judged_pairs,
                    "judged_relevant": judged_relevant,
                    "judged_not_relevant": judged_not,
                    "judged_depth": run.judged_depth,
                    "notes": (
                        "secondary view: a document is relevant when it contains at "
                        "least one judged relevant content unit"
                    ),
                }
            )
    return rows


# ----------------------------------------------------------------------
PIPELINE_METRIC_SOURCES = {
    "token_count": ("total_tokens", "tokens", "results/phase3/pipeline_comparison.csv"),
    "unique_tokens": ("unique_tokens", "tokens", "results/phase3/pipeline_comparison.csv"),
    "vocabulary_size": ("vocabulary_size", "terms", "results/phase3/pipeline_comparison.csv"),
    "index_terms": ("index_terms", "terms", "results/phase3/pipeline_comparison.csv"),
    "meaningful_ngrams": ("phrase_terms", "phrases", "results/phase3/pipeline_comparison.csv"),
    "index_size_postings": ("total_postings", "postings", "results/phase3/pipeline_comparison.csv"),
    "processing_time_seconds": (
        "processing_time_seconds", "seconds", "results/phase3/pipeline_comparison.csv",
    ),
    "domain_term_preservation": (
        "domain_term_recall", "rate", "results/phase3/pipeline_comparison.csv",
    ),
    "financial_expression_preservation": (
        "financial_expression_preservation", "rate", "results/phase3/pipeline_comparison.csv",
    ),
    "variant_collapse_rate": (
        "variant_collapse_rate", "rate", "results/phase3/pipeline_comparison.csv",
    ),
    "query_answerability": (
        "query_answerability", "rate", "results/phase3/pipeline_comparison.csv",
    ),
    "selection_score": ("total_score", "weighted score",
                        "results/phase3/pipeline_comparison.csv"),
}


def pipeline_comparison_rows(
    phase3_rows: Dict[str, Dict[str, str]],
    runs: Dict[str, EvaluationRun],
    selected: str,
    probes: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, Any]]:
    """Table J: every metric both pipelines can supply, in one place."""
    probes = probes or {}
    keys = sorted(set(phase3_rows) | set(runs) | set(probes)) or ["pipeline_a", "pipeline_b"]
    rows: List[Dict[str, Any]] = []
    for metric, (column, unit, source) in PIPELINE_METRIC_SOURCES.items():
        row: Dict[str, Any] = {
            "metric": metric,
            "unit": unit,
            "source": source,
            "selected": selected,
            "note": "",
        }
        for key in keys:
            row[key] = phase3_rows.get(key, {}).get(column, "")
        rows.append(row)

    for name, getter in (
        ("precision_macro", lambda r: (r.aggregates.get("judged_depth").macro.get("precision")
                                       if r.aggregates.get("judged_depth") else None)),
        ("recall_macro", lambda r: (r.aggregates.get("judged_depth").macro.get("recall")
                                    if r.aggregates.get("judged_depth") else None)),
        ("f1_macro", lambda r: (r.aggregates.get("judged_depth").macro.get("f1")
                                if r.aggregates.get("judged_depth") else None)),
        ("precision_at_10_macro", lambda r: r.at_k.get(10, {}).get("macro_precision_at_k")),
        ("recall_at_10_macro", lambda r: r.at_k.get(10, {}).get("macro_recall_at_k")),
        ("precision_at_10_micro", lambda r: (r.at_k.get(10, {}).get("micro") or {})
         .get("pooled_precision_at_k")),
        ("recall_at_10_micro", lambda r: (r.at_k.get(10, {}).get("micro") or {})
         .get("pooled_recall_at_k")),
    ):
        row = {"metric": name, "unit": "rate", "source": "results/phase4/evaluation_summary.csv",
               "selected": selected, "note": ""}
        for key in keys:
            run = runs.get(key)
            value = getter(run) if run is not None else None
            if run is None:
                row[key] = "N/A"
            else:
                row[key] = "" if value is None else str(round(value, 4))
        rows.append(row)

    # Retrieval statistics are comparable across pipelines and are reported for
    # both, including the pipeline that could not be scored for relevance.
    for name, key_name, unit in (
        ("retrieval_answerability", "answerability", "rate"),
        ("retrieval_units_returned", "total_units_returned", "units"),
        ("retrieval_documents_returned", "total_documents_returned", "documents"),
        ("retrieval_mean_units_per_query", "mean_units_per_query", "units/query"),
        ("retrieval_mean_execution_ms", "mean_execution_time_ms", "ms/query"),
        ("retrieval_queries_with_missing_terms", "queries_with_missing_terms", "queries"),
    ):
        row = {"metric": name, "unit": unit, "source": "python -m src.phase4.run",
               "selected": selected, "note": ""}
        for key in keys:
            run = runs.get(key)
            if run is not None:
                values = [q.retrieved_units for q in run.evaluated_queries()]
                answered = len(values)
                row[key] = {
                    "retrieval_answerability": round(answered / max(1, len(run.queries)), 4),
                    "retrieval_units_returned": sum(values),
                    "retrieval_documents_returned": sum(
                        q.retrieved_documents for q in run.evaluated_queries()
                    ),
                    "retrieval_mean_units_per_query": round(
                        sum(values) / answered, 4) if answered else "",
                    "retrieval_mean_execution_ms": round(
                        sum(q.execution_time_ms for q in run.evaluated_queries()) / answered, 3
                    ) if answered else "",
                    "retrieval_queries_with_missing_terms": sum(
                        1 for q in run.evaluated_queries() if q.missing_terms),
                }[name]
            else:
                probe = probes.get(key, {})
                row[key] = probe.get(key_name, "N/A")
        rows.append(row)

    for key in keys:
        if key in runs:
            continue
        rows.append(
            {
                "metric": "relevance_evaluation_status",
                "unit": "",
                "source": "results/phase4",
                "selected": selected,
                "note": str(probes.get(key, {}).get("note", "not evaluated")),
                key: "N/A - not evaluated",
            }
        )
    return rows


# ----------------------------------------------------------------------
def final_summary_rows(
    phase1_validation: Dict[str, Any],
    phase2_summary: Dict[str, Any],
    phase3_summary: Dict[str, Any],
    runs: Dict[str, EvaluationRun],
    store: RelevanceStore,
    selected: str,
) -> List[Dict[str, Any]]:
    corpus = phase2_summary.get("corpus", {}) or {}
    rows: List[Dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, unit: str, source: str, note: str = "") -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": "" if value is None else str(value),
                "unit": unit,
                "source": source,
                "note": note,
            }
        )

    add("phase1", "documents_found", phase1_validation.get("documents_found"), "documents",
        "results/phase1/validation_report.json")
    add("phase1", "documents_processed", phase1_validation.get("documents_processed"), "documents",
        "results/phase1/validation_report.json")
    add("phase1", "pages", phase1_validation.get("pages_validated"), "pages",
        "results/phase1/validation_report.json")
    add("phase1", "content_units", phase1_validation.get("units_validated"), "units",
        "results/phase1/validation_report.json")
    add("phase1", "validation_status", phase1_validation.get("status"), "",
        "results/phase1/validation_report.json")

    add("phase2", "selected_units", corpus.get("selected_units"), "units",
        "results/phase2/phase2_summary.json",
        f"policy '{corpus.get('text_selection_policy')}'")
    add("phase2", "selected_characters", corpus.get("selected_characters"), "characters",
        "results/phase2/phase2_summary.json")
    add("phase2", "bpe_vocabulary", phase2_summary.get("bpe", {}).get("vocabulary_size"), "tokens",
        "results/phase2/phase2_summary.json")
    ner = phase2_summary.get("ner", {}) or {}
    add("phase2", "ner_entities", ner.get("total_entities"), "entities",
        "results/phase2/phase2_summary.json")
    add("phase2", "ner_domain_mentions", ner.get("domain_mentions"), "mentions",
        "results/phase2/phase2_summary.json")
    for row in phase2_summary.get("tokenization", []) or []:
        add("phase2", f"tokenization_{row.get('tokenizer')}_tokens", row.get("total_tokens"),
            "tokens", "results/phase2/tokenization/tokenization_comparison.csv")

    index = phase3_summary.get("index_statistics", {}) or {}
    add("phase3", "final_pipeline", selected, "", "results/phase3/final_pipeline.json")
    add("phase3", "index_terms", index.get("index_terms"), "terms",
        "results/phase3/index_statistics.csv")
    add("phase3", "total_postings", index.get("total_postings"), "postings",
        "results/phase3/index_statistics.csv")
    add("phase3", "postings_per_term", index.get("postings_per_term"), "postings/term",
        "results/phase3/index_statistics.csv")
    add("phase3", "phrase_terms", index.get("phrase_terms"), "phrases",
        "results/phase3/index_statistics.csv")
    add("phase3", "queries_answered",
        (phase3_summary.get("retrieval_statistics", {}) or {}).get("answered"), "queries",
        "results/phase3/phase3_summary.json")

    add("phase4", "judged_pairs", len(store), "query-unit pairs",
        "results/phase4/relevance_judgments.csv")
    add("phase4", "judged_relevant", sum(1 for j in store.judgments.values() if j.is_relevant),
        "pairs", "results/phase4/relevance_judgments.csv")
    add("phase4", "judged_not_relevant",
        sum(1 for j in store.judgments.values() if not j.is_relevant), "pairs",
        "results/phase4/relevance_judgments.csv")
    add("phase4", "judgment_pool_depth",
        next(iter(runs.values())).judged_depth if runs else None, "units per query",
        "config/phase4_config.yaml")

    primary = runs.get(selected) or (next(iter(runs.values())) if runs else None)
    if primary is not None:
        aggregate = primary.aggregates.get("judged_depth")
        if aggregate is not None:
            for name in ("precision", "recall", "f1"):
                add("phase4", f"{name}_macro", aggregate.macro.get(name), "rate",
                    "results/phase4/evaluation_summary.csv", "macro average over the query set")
                add("phase4", f"{name}_micro", aggregate.micro.get(name), "rate",
                    "results/phase4/evaluation_summary.csv",
                    "ratio of summed counts, not a mean of ratios")
        for k in sorted(primary.at_k):
            block = primary.at_k[k]
            add("phase4", f"precision_at_{k}_macro", block.get("macro_precision_at_k"), "rate",
                "results/phase4/evaluation_summary.csv")
            add("phase4", f"recall_at_{k}_macro", block.get("macro_recall_at_k"), "rate",
                "results/phase4/evaluation_summary.csv")
            micro = block.get("micro") or {}
            add("phase4", f"recall_at_{k}_micro", micro.get("pooled_recall_at_k"), "rate",
                "results/phase4/evaluation_summary.csv")
        add("phase4", "mean_query_execution_ms",
            round(sum(q.execution_time_ms for q in primary.evaluated_queries())
                  / max(1, len(primary.evaluated_queries())), 3), "ms",
            "results/phase4/evaluation_results.csv")
    return rows


# ----------------------------------------------------------------------
def gui_summary(
    phase1_validation: Dict[str, Any],
    phase2_summary: Dict[str, Any],
    phase3_summary: Dict[str, Any],
    runs: Dict[str, EvaluationRun],
    store: RelevanceStore,
    selected: str,
    config: Phase4Config,
) -> Dict[str, Any]:
    """Exactly the payload the dashboard renders, and nothing else."""
    corpus = phase2_summary.get("corpus", {}) or {}
    index = phase3_summary.get("index_statistics", {}) or {}
    retrieval = phase3_summary.get("retrieval_statistics", {}) or {}
    primary = runs.get(selected) or (next(iter(runs.values())) if runs else None)
    aggregate = primary.aggregates.get("judged_depth") if primary else None

    per_query = []
    if primary is not None:
        for query in primary.evaluated_queries():
            per_query.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "query_type": query.query_type,
                    "precision": query.metrics["precision"].as_json(),
                    "recall": query.metrics["recall"].as_json(),
                    "f1": query.metrics["f1"].as_json(),
                    "precision_at_10": query.metrics.get("precision_at_10", Metric(None)).as_json(),
                    "recall_at_10": query.metrics.get("recall_at_10", Metric(None)).as_json(),
                    "relevant_retrieved": query.counts.relevant_retrieved,
                    "relevant_total": query.counts.relevant_total,
                    "considered_units": query.counts.retrieved,
                }
            )

    per_type: Dict[str, Dict[str, object]] = {}
    if primary is not None:
        buckets: Dict[str, Dict[str, Any]] = {}
        for query in primary.evaluated_queries():
            bucket = buckets.setdefault(
                query.query_type,
                {"queries": 0, "precision": [], "recall": [], "f1": []},
            )
            bucket["queries"] += 1
            for name in ("precision", "recall", "f1"):
                value = query.metrics[name].value
                if value is not None:
                    bucket[name].append(value)
        for bucket in buckets.values():
            for name in ("precision", "recall", "f1"):
                values = bucket[name]
                bucket[name] = round(sum(values) / len(values), 4) if values else None
        per_type = buckets

    return {
        "generated_for": "gui",
        "selected_pipeline": selected,
        "judgments": {
            "pairs": len(store),
            "relevant": sum(1 for j in store.judgments.values() if j.is_relevant),
            "not_relevant": sum(1 for j in store.judgments.values() if not j.is_relevant),
            "pool_depth": config.pool_depth,
            "annotator": config.get("relevance.annotator"),
            "judgment_method": config.get("relevance.judgment_method"),
            "rubric": config.get("relevance.rubric"),
        },
        "corpus": {
            "documents": corpus.get("documents"),
            "pages": corpus.get("pages"),
            "units_total": corpus.get("units_total"),
            "selected_units": corpus.get("selected_units"),
            "selected_characters": corpus.get("selected_characters"),
            "text_selection_policy": corpus.get("text_selection_policy"),
            "units_by_type": corpus.get("units_by_type", {}),
        },
        "phase1": {
            "status": phase1_validation.get("status"),
            "documents_found": phase1_validation.get("documents_found"),
            "documents_processed": phase1_validation.get("documents_processed"),
            "pages_validated": phase1_validation.get("pages_validated"),
            "units_validated": phase1_validation.get("units_validated"),
        },
        "phase2": {
            "validation_passed": corpus.get("validation_passed"),
            "validation_total": corpus.get("validation_total"),
            "bpe_vocabulary_size": phase2_summary.get("bpe", {}).get("vocabulary_size"),
            "ner_total_entities": (phase2_summary.get("ner", {}) or {}).get("total_entities"),
            "ner_domain_mentions": (phase2_summary.get("ner", {}) or {}).get("domain_mentions"),
            "ner_labels": (phase2_summary.get("ner", {}) or {}).get("labels", []),
            "ngrams": phase2_summary.get("ngrams", []),
            "pos_gold_set": (phase2_summary.get("pos", {}) or {}).get("gold_set", {}),
        },
        "phase3": {
            "final_pipeline": phase3_summary.get("final_pipeline"),
            "final_pipeline_name": phase3_summary.get("final_pipeline_name"),
            "index": {
                key: index.get(key)
                for key in (
                    "index_terms", "unigram_terms", "phrase_terms", "total_postings",
                    "postings_per_term", "indexed_units", "indexed_documents",
                    "total_positions", "singleton_terms", "longest_posting_list",
                    "longest_posting_term", "mean_document_frequency", "hapax_ratio",
                )
            },
            "retrieval": {
                key: retrieval.get(key)
                for key in (
                    "queries", "answered", "answerability", "total_units_returned",
                    "mean_units_per_query", "mean_execution_time_ms", "by_query_type",
                )
            },
        },
        "phase4": {
            "evaluation_available": aggregate is not None,
            "queries_evaluated": len(primary.evaluated_queries()) if primary else 0,
            "macro": {
                "precision": aggregate.macro.get("precision") if aggregate else None,
                "recall": aggregate.macro.get("recall") if aggregate else None,
                "f1": aggregate.macro.get("f1") if aggregate else None,
            },
            "micro": {
                "precision": aggregate.micro.get("precision") if aggregate else None,
                "recall": aggregate.micro.get("recall") if aggregate else None,
                "f1": aggregate.micro.get("f1") if aggregate else None,
            },
            "at_k": {
                str(k): primary.at_k[k] for k in sorted(primary.at_k)
            } if primary else {},
            "per_query": per_query,
            "per_query_type": per_type,
            "recall_is_pool_bounded": True,
            "pool_depth": config.pool_depth,
            "note": (
                "Relevance labels are human judgments over a pooled candidate set of "
                f"depth {config.pool_depth}. Recall is measured against the judged "
                "relevant set and is therefore bounded by the pool; precision at the "
                "judged depth is exact because every considered unit carries a label."
            ),
        },
    }
