"""Phase 3 and Phase 4 services.

Phase 3 facts (index, pipelines, query registry, executed retrieval) are read
from the artefacts Phase 3 wrote. Phase 4 adds the evaluation metrics and the
one write path in the whole API: a pooled relevance judgment.

The judgment writer is delegated to ``src.phase4.relevance.RelevanceStore``, so
the CSV on disk keeps exactly the schema Phase 4 defined and the validator can
still read it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config.settings import Settings
from backend.services import artifacts
from backend.services.corpus import validate_id

# ----------------------------------------------------------------------
# Phase 3
# ----------------------------------------------------------------------


def _p3(settings: Settings, relative: str, required: bool = True):
    return artifacts.read_csv_rows(settings.path(f"{settings.phase3_results}/{relative}"), required)


def _p3j(settings: Settings, relative: str, required: bool = True):
    return artifacts.read_json(settings.path(f"{settings.phase3_results}/{relative}"), required)


def phase3_summary(settings: Settings) -> Dict[str, Any]:
    return _p3j(settings, "phase3_summary.json")


def phase3_validation(settings: Settings) -> List[Dict[str, str]]:
    return _p3(settings, "validation_report.csv")


def index_manifest(settings: Settings) -> Dict[str, Any]:
    return _p3j(settings, "index_manifest.json")


def final_pipeline(settings: Settings) -> Dict[str, Any]:
    """The Phase 3 selection decision, including the criteria breakdown."""
    return _p3j(settings, "final_pipeline.json")


def index_statistics(settings: Settings) -> Dict[str, Any]:
    """Phase 3's metric/value table, turned into a mapping."""
    return {
        row.get("metric", ""): artifacts.as_float(row.get("value"))
        for row in _p3(settings, "index_statistics.csv")
    }


def pipelines(settings: Settings) -> List[Dict[str, Any]]:
    """Both pipelines with every Phase 3 measurement, criterion breakdown included."""
    selection = final_pipeline(settings)
    comparison = _p3(settings, "pipeline_comparison.csv")
    chosen = selection.get("final_pipeline", "")
    scores = selection.get("scores", {}) or {}

    rows: List[Dict[str, Any]] = []
    for row in comparison:
        key = row.get("pipeline", "")
        detail = scores.get(key, {}) or {}
        criteria = {
            name: {
                "rate": artifacts.as_float(value.get("rate")),
                "weight": artifacts.as_float(value.get("weight")),
                "weighted": artifacts.as_float(value.get("weighted")),
                "numerator": artifacts.as_int(value.get("numerator")),
                "denominator": artifacts.as_int(value.get("denominator")),
            }
            for name, value in (detail.get("criteria") or {}).items()
        }
        rows.append(
            {
                "pipeline": key,
                "name": row.get("pipeline_name") or detail.get("pipeline_name"),
                "is_final": key == chosen,
                "tokenizer": row.get("tokenizer"),
                "stopword_strategy": row.get("stopword_strategy"),
                "morphology": row.get("morphology"),
                "term_representation": row.get("term_representation"),
                "order": row.get("order"),
                "total_score": artifacts.as_float(row.get("total_score")),
                "criteria": criteria,
                "criterion_weights": selection.get("criteria_weights", {}),
                "index_terms": artifacts.as_int(row.get("index_terms")),
                "vocabulary_size": artifacts.as_int(row.get("vocabulary_size")),
                "total_tokens": artifacts.as_int(row.get("total_tokens")),
                "unique_tokens": artifacts.as_int(row.get("unique_tokens")),
                "indexed_units": artifacts.as_int(row.get("indexed_units")),
                "total_postings": artifacts.as_int(row.get("total_postings")),
                "phrase_terms": artifacts.as_int(row.get("phrase_terms")),
                "missing_domain_terms": artifacts.as_int(row.get("missing_domain_terms")),
                "collapsed_variant_groups": artifacts.as_int(row.get("collapsed_variant_groups")),
                "unanswered_queries": artifacts.as_int(row.get("unanswered_queries")),
                "processing_time_seconds": artifacts.as_float(row.get("processing_time_seconds")),
                "index_build_seconds": artifacts.as_float(row.get("index_build_seconds")),
                "preserved_examples": artifacts.split_list(row.get("preserved_examples")),
                "lost_examples": artifacts.split_list(row.get("lost_examples")),
                "variant_collapse_detail": row.get("variant_collapse_rate_detail"),
                "domain_term_recall_detail": row.get("domain_term_recall_detail"),
                "answerability_detail": row.get("query_answerability_detail"),
            }
        )
    rows.sort(key=lambda r: (not r["is_final"], -(r["total_score"] or 0)))
    return rows


def pipeline_detail(settings: Settings, pipeline: str) -> Dict[str, Any]:
    validate_id(pipeline, "pipeline")
    row = next((p for p in pipelines(settings) if p["pipeline"] == pipeline), None)
    if row is None:
        return {}
    return {
        **row,
        "selection_reason": final_pipeline(settings).get("reason"),
        "margin_over_runner_up": artifacts.as_float(
            final_pipeline(settings).get("margin_over_runner_up")
        ),
        "tie_breakers": final_pipeline(settings).get("tie_breakers", []),
        "excluded_from_score": final_pipeline(settings).get("excluded_from_score", []),
    }


def pipeline_selection(settings: Settings) -> Dict[str, Any]:
    return phase3_summary(settings).get("selection", {}) or {}


def query_registry(settings: Settings) -> List[Dict[str, Any]]:
    """The 15 registered queries with how each one actually behaved."""
    rows = _p3(settings, "query_registry.csv")
    out = []
    for row in rows:
        out.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query"),
                "query_text": row.get("query"),
                "query_type": row.get("query_type"),
                "description": row.get("description"),
                "expected_domain": row.get("expected_domain"),
                "configured_terms": artifacts.split_list(row.get("configured_terms")),
                "excluded_terms": artifacts.split_list(row.get("excluded_terms")),
                "normalized_terms": artifacts.split_list(row.get("normalized_terms")),
                "matched_terms": artifacts.split_list(row.get("matched_terms")),
                "missing_terms": artifacts.split_list(row.get("missing_terms")),
                "syntax_valid": artifacts.as_bool(row.get("syntax_valid")),
                "answered": artifacts.as_bool(row.get("answered")),
                "result_units": artifacts.as_int(row.get("result_units")),
                "result_documents": artifacts.as_int(row.get("result_documents")),
                "execution_time_ms": artifacts.as_float(row.get("execution_time_ms")),
                "top_unit_ids": artifacts.split_list(row.get("top_unit_ids")),
                "top_documents": artifacts.split_list(row.get("top_documents")),
                "error": row.get("error") or None,
            }
        )
    return out


def retrieval_summary(settings: Settings) -> List[Dict[str, Any]]:
    """Per-query execution summary used by the query-history panel."""
    rows = _p3(settings, "retrieval_summary.csv")
    out = []
    for row in rows:
        out.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query"),
                "query_type": row.get("query_type"),
                "description": row.get("description"),
                "expected_domain": row.get("expected_domain"),
                "normalized_terms": artifacts.split_list(row.get("normalized_terms")),
                "matched_terms": artifacts.split_list(row.get("matched_terms")),
                "missing_terms": artifacts.split_list(row.get("missing_terms")),
                "syntax_valid": artifacts.as_bool(row.get("syntax_valid")),
                "answered": artifacts.as_bool(row.get("answered")),
                "result_units": artifacts.as_int(row.get("result_units")),
                "result_documents": artifacts.as_int(row.get("result_documents")),
                "top10_score_sum": artifacts.as_float(row.get("top10_score_sum")),
                "execution_time_ms": artifacts.as_float(row.get("execution_time_ms")),
            }
        )
    return out


def retrieval_results(
    settings: Settings, query_id: Optional[str] = None, limit: int = 100, offset: int = 0
) -> Dict[str, Any]:
    """The stored Phase 3 rankings, i.e. what the CLI already retrieved."""
    rows = _p3(settings, "retrieval_results.csv")
    if query_id:
        validate_id(query_id, "query_id")
        rows = [r for r in rows if r.get("query_id") == query_id]
    rows.sort(key=lambda r: (r.get("query_id", ""), artifacts.as_int(r.get("rank"), 0) or 0))
    return artifacts.page(rows, limit, offset)


def document_results(
    settings: Settings, query_id: Optional[str] = None, limit: int = 50, offset: int = 0
) -> Dict[str, Any]:
    rows = _p3(settings, "document_results.csv")
    if query_id:
        validate_id(query_id, "query_id")
        rows = [r for r in rows if r.get("query_id") == query_id]
    return artifacts.page(rows, limit, offset)


# ----------------------------------------------------------------------
# Phase 4
# ----------------------------------------------------------------------


def _p4(settings: Settings, relative: str, required: bool = True):
    return artifacts.read_csv_rows(settings.path(f"{settings.phase4_results}/{relative}"), required)


def _p4j(settings: Settings, relative: str, required: bool = True):
    return artifacts.read_json(settings.path(f"{settings.phase4_results}/{relative}"), required)


def phase4_summary(settings: Settings) -> Dict[str, Any]:
    return _p4j(settings, "final_summary.json")


def phase4_validation(settings: Settings) -> List[Dict[str, str]]:
    return _p4(settings, "validation_report.csv", required=False)


def evaluation_results(
    settings: Settings, query_id: Optional[str] = None, limit: int = 50, offset: int = 0
) -> Dict[str, Any]:
    """One row per query, with numbers coerced out of the CSV strings."""
    rows = _p4(settings, "evaluation_results.csv")
    if query_id:
        validate_id(query_id, "query_id")
        rows = [r for r in rows if r.get("query_id") == query_id]
    numeric = (
        "retrieved_units", "retrieved_documents", "judged_depth", "considered_units",
        "judged_units", "unjudged_units", "relevant_retrieved", "relevant_total",
        "false_positives", "false_negatives", "precision", "recall", "f1",
        "average_precision", "precision_at_5", "recall_at_5", "precision_at_10",
        "recall_at_10", "document_precision", "document_recall", "document_f1",
        "execution_time_ms",
    )
    shaped: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {k: v for k, v in row.items()}
        item["evaluable"] = artifacts.as_bool(row.get("evaluable"))
        item["recall_is_pool_bounded"] = artifacts.as_bool(row.get("recall_is_pool_bounded"))
        item["matched_terms_list"] = artifacts.split_list(row.get("matched_terms"))
        item["missing_terms_list"] = artifacts.split_list(row.get("missing_terms"))
        for name in numeric:
            item[name] = artifacts.as_float(row.get(name))
        for name in ("judged_depth", "retrieved_units", "retrieved_documents"):
            item[name] = artifacts.as_int(row.get(name))
        shaped.append(item)
    return artifacts.page(shaped, limit, offset)


def evaluation_aggregates(settings: Settings) -> List[Dict[str, Any]]:
    """Unit-level and document-level aggregates, as Phase 4 wrote them."""
    out = []
    for row in _p4(settings, "evaluation_summary.csv"):
        out.append(
            {
                "scope": row.get("scope"),
                "pipeline": row.get("pipeline"),
                "queries": artifacts.as_int(row.get("queries")),
                "queries_scored": artifacts.as_int(row.get("queries_scored")),
                "queries_undefined": artifacts.as_int(row.get("queries_undefined")),
                "precision_macro": artifacts.as_float(row.get("precision_macro")),
                "recall_macro": artifacts.as_float(row.get("recall_macro")),
                "f1_macro": artifacts.as_float(row.get("f1_macro")),
                "precision_micro": artifacts.as_float(row.get("precision_micro")),
                "recall_micro": artifacts.as_float(row.get("recall_micro")),
                "f1_micro": artifacts.as_float(row.get("f1_micro")),
                "precision_at_5_macro": artifacts.as_float(row.get("precision_at_5_macro")),
                "recall_at_5_macro": artifacts.as_float(row.get("recall_at_5_macro")),
                "precision_at_10_macro": artifacts.as_float(row.get("precision_at_10_macro")),
                "recall_at_10_macro": artifacts.as_float(row.get("recall_at_10_macro")),
                "precision_at_10_micro": artifacts.as_float(row.get("precision_at_10_micro")),
                "recall_at_10_micro": artifacts.as_float(row.get("recall_at_10_micro")),
                "judged_pairs": artifacts.as_int(row.get("judged_pairs")),
                "judged_relevant": artifacts.as_int(row.get("judged_relevant")),
                "judged_not_relevant": artifacts.as_int(row.get("judged_not_relevant")),
                "judged_depth": artifacts.as_int(row.get("judged_depth")),
                "notes": row.get("notes"),
            }
        )
    return out


def pipeline_comparison(settings: Settings) -> List[Dict[str, Any]]:
    """Pipeline A against Pipeline B. Relevance rows may be N/A by design."""
    return [
        {
            "metric": row.get("metric"),
            "pipeline_a": row.get("pipeline_a"),
            "pipeline_b": row.get("pipeline_b"),
            "selected": row.get("selected"),
            "unit": row.get("unit"),
            "source": row.get("source"),
            "note": row.get("note"),
            "available": str(row.get("pipeline_a", "")).strip().upper() != "N/A",
        }
        for row in _p4(settings, "pipeline_final_comparison.csv")
    ]


def judgments(
    settings: Settings,
    query_id: Optional[str] = None,
    relevance: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    rows = _p4(settings, "relevance_judgments.csv")
    if query_id:
        validate_id(query_id, "query_id")
        rows = [r for r in rows if r.get("query_id") == query_id]
    if relevance is not None:
        # The column is the label 1/0, not the word "relevant".
        wanted = "1" if relevance else "0"
        rows = [r for r in rows if str(r.get("relevance", "")).strip() == wanted]
    rows.sort(key=lambda r: (r.get("query_id", ""), artifacts.as_int(r.get("rank"), 0) or 0))
    return artifacts.page(rows, limit, offset)


def evaluation_report(settings: Settings) -> Dict[str, Any]:
    """The single payload the /evaluation page renders, with honest metadata."""
    summary = phase4_summary(settings)
    judged = (summary.get("evaluation") or {}).get("pipeline_b") or {}
    aggregates = {row["scope"]: row for row in evaluation_aggregates(settings)}
    return {
        "phase": summary.get("phase"),
        "selected_pipeline": summary.get("selected_pipeline"),
        "judged_pipeline": "pipeline_b",
        "inputs": summary.get("inputs", {}),
        "queries": summary.get("queries", []),
        "judgment_summary": summary.get("judgments", {}),
        "judged_depth": judged.get("judged_depth"),
        "unjudged_policy": judged.get("unjudged_policy"),
        "queries_evaluated": judged.get("queries_evaluated"),
        "queries_failed": judged.get("queries_failed"),
        "problems": judged.get("problems", []),
        "runner_up": summary.get("runner_up_retrieval_statistics", {}),
        "unit_level": aggregates.get("content_unit", {}),
        "document_level": aggregates.get("document", {}),
        "at_k": judged.get("at_k", {}),
        "per_query": evaluation_results(settings, limit=100),
        "comparison": pipeline_comparison(settings),
        "validation": phase4_validation(settings),
        "notes": _evaluation_notes(summary),
    }


def _evaluation_notes(summary: Dict[str, Any]) -> List[str]:
    """Every caveat a reader needs, derived from the artefacts themselves."""
    notes: List[str] = []
    judgment = summary.get("judgments", {}) or {}
    depth = judgment.get("pool_depth")
    if depth:
        notes.append(
            f"Relevance was judged on a pooled top-{depth} list per query, so recall is "
            "pool-bounded: it measures how much of the judged pool a pipeline returns, "
            "not how much of the whole corpus."
        )
    if judgment.get("annotator"):
        notes.append(f"Annotator: {judgment['annotator']}.")
    if judgment.get("judgment_method"):
        notes.append(str(judgment["judgment_method"]))
    runner_up = summary.get("runner_up_retrieval_statistics", {}) or {}
    for key, stats in runner_up.items():
        if stats.get("note"):
            notes.append(f"{key}: {stats['note']}")
    policy = ((summary.get("evaluation") or {}).get("pipeline_b") or {}).get("unjudged_policy")
    if policy:
        notes.append(f"Unjudged units inside the pool: {policy}.")
    return notes


# ----------------------------------------------------------------------
# The only write path in the API.
# ----------------------------------------------------------------------
def save_judgment(
    settings: Settings,
    query_id: str,
    unit_id: str,
    relevance: bool,
    annotator: str,
    notes: str = "",
    query: str = "",
    rank: Optional[int] = None,
    document_id: str = "",
) -> Dict[str, Any]:
    """Record one pooled judgment through the Phase 4 store, then revalidate.

    ``RelevanceStore.upsert`` is the same call the Phase 4 CLI uses, so the CSV
    keeps the exact schema the validator and the metrics expect.
    """
    from src.phase4.relevance import RelevanceStore

    path = settings.path(f"{settings.phase4_results}/relevance_judgments.csv")
    store = RelevanceStore.load(path, strict=False)
    judgment = store.upsert(
        query_id=query_id,
        unit_id=unit_id,
        relevance=1 if relevance else 0,
        notes=notes,
        annotator=annotator,
        judgment_method=(
            "pooled manual judgment recorded through the web GUI, against the "
            "Phase 4 rubric"
        ),
        query=query or "",
        rank=str(rank) if rank is not None else "",
        document_id=document_id or "",
    )
    store.save(annotator=annotator)
    artifacts.clear_cache()

    rows = _p4(settings, "relevance_judgments.csv", required=False)
    validation = phase4_validation(settings)
    passed = sum(1 for r in validation if str(r.get("status", "")).upper() == "PASS")
    return {
        "saved": True,
        "judgment": {
            "query_id": judgment.query_id,
            "unit_id": judgment.unit_id,
            "relevance": judgment.relevance,
            "relevance_label": "relevant" if judgment.relevance else "not_relevant",
            "rank": judgment.rank,
            "notes": judgment.notes,
            "annotator": judgment.annotator,
            "judgment_method": judgment.judgment_method,
            "judged_at": judgment.judged_at,
        },
        "judgment_count": len(rows),
        "relevant_count": sum(
            1 for r in rows if str(r.get("relevance", "")).strip() == "1"
        ),
        "validation": {
            "rules": len(validation),
            "passed": passed,
            "all_passed": passed == len(validation) and bool(validation),
        },
        "message": (
            "Judgment saved to relevance_judgments.csv. The metric CSVs are not "
            "regenerated on request; run `python -m src.phase4.run` to recompute "
            "the evaluation from the updated judgments."
        ),
    }
