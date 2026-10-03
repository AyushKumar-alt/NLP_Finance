"""Aggregate statistics for the home page.

Assembled by reading the four phase summaries and the Phase 1 registry. Every
number is taken from an artefact; nothing is recomputed, and a phase whose
artefacts are missing is reported as missing rather than as zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config.settings import Settings
from backend.services import artifacts, corpus, evaluation, experiments

MISSING: Dict[str, Any] = {
    "available": False,
    "reason": "phase artefacts not found; run the phase CLI first",
}


def _or_missing(value: Optional[Dict[str, Any]], key: str = "reason") -> Dict[str, Any]:
    if not value:
        return dict(MISSING)
    return value


def project(settings: Settings) -> Dict[str, Any]:
    p1 = {}
    try:
        p1 = corpus.phase1_validation(settings)
    except artifacts.ArtifactMissing:
        p1 = {}
    p3 = {}
    try:
        p3 = evaluation.phase3_summary(settings)
    except artifacts.ArtifactMissing:
        p3 = {}
    p4 = {}
    try:
        p4 = evaluation.phase4_summary(settings)
    except artifacts.ArtifactMissing:
        p4 = {}
    p2 = {}
    try:
        p2 = experiments.summary(settings)
    except artifacts.ArtifactMissing:
        p2 = {}

    return {
        "phases": [1, 2, 3, 4],
        "domain": "Indian financial and economic document analysis",
        "phases_available": {
            "phase1": bool(p1),
            "phase2": bool(p2),
            "phase3": bool(p3),
            "phase4": bool(p4),
        },
        "generated_by": {
            "phase1": p1.get("generated_by"),
            "phase3": p3.get("generated_by"),
            "phase4": p4.get("generated_by"),
        },
        "elapsed_seconds": {
            "phase3": artifacts.as_float(p3.get("elapsed_seconds")),
            "phase4": artifacts.as_float(p4.get("elapsed_seconds")),
        },
    }


def corpus_overview(settings: Settings) -> Dict[str, Any]:
    documents = corpus.document_summary(settings)
    sources = corpus.source_summary(settings)
    totals = {
        "documents": len(documents),
        "sources": len(sources),
        "pages": sum(d.get("page_count") or 0 for d in documents),
        "words": sum(d.get("words") or 0 for d in documents),
        "characters": sum(d.get("characters") or 0 for d in documents),
        "baseline_tokens": sum(d.get("baseline_token_count") or 0 for d in documents),
        "sections": sum(d.get("sections") or 0 for d in documents),
        "units": sum(d.get("units") or 0 for d in documents),
        "paragraphs": sum(d.get("paragraphs") or 0 for d in documents),
        "tables": sum(d.get("tables") or 0 for d in documents),
        "figures": sum(d.get("figures") or 0 for d in documents),
        "footnotes": sum(d.get("footnotes") or 0 for d in documents),
    }

    years: Dict[str, int] = {}
    for document in documents:
        year = document.get("year") or "unknown"
        years[year] = years.get(year, 0) + 1

    try:
        manifest = evaluation.index_manifest(settings)
        unit_types = (manifest.get("corpus") or {}).get("units_by_type", {})
        selection_policy = manifest.get("text_selection_policy")
        selected_units = manifest.get("indexed_units")
    except artifacts.ArtifactMissing:
        unit_types, selection_policy, selected_units = {}, None, None

    return {
        **totals,
        "units_by_type": dict(sorted(unit_types.items())),
        "documents_by_year": dict(sorted(years.items())),
        "text_selection_policy": selection_policy,
        "selected_units": selected_units,
        "per_document": [
            {
                "document_id": d["document_id"],
                "title": d.get("title"),
                "words": d.get("words"),
                "units": d.get("units"),
                "pages": d.get("page_count"),
            }
            for d in documents
        ],
        "per_source": [
            {
                "source_id": s["source_id"],
                "source_title": s.get("source_title"),
                "documents": s.get("documents"),
                "pages": s.get("observed_pages"),
                "units": s.get("observed_units"),
            }
            for s in sources
        ],
    }


def index_overview(settings: Settings) -> Dict[str, Any]:
    try:
        manifest = evaluation.index_manifest(settings)
        statistics = manifest.get("statistics", {}) or {}
        final = manifest.get("final_pipeline")
    except artifacts.ArtifactMissing:
        return dict(MISSING)
    flat = evaluation.index_statistics(settings)
    return {
        "available": True,
        "final_pipeline": final,
        "pipeline_name": manifest.get("pipeline_name"),
        "terms": artifacts.as_int(statistics.get("index_terms")),
        "postings": artifacts.as_int(statistics.get("total_postings")),
        "units": manifest.get("indexed_units"),
        "documents": manifest.get("documents"),
        "index_bytes": artifacts.as_int((manifest.get("index") or {}).get("bytes")),
        "postings_per_term": artifacts.as_float(statistics.get("postings_per_term")),
        "singleton_terms": artifacts.as_int(statistics.get("singleton_terms")),
        "hapax_ratio": artifacts.as_float(statistics.get("hapax_ratio")),
        "mean_term_frequency": artifacts.as_float(statistics.get("mean_term_frequency")),
        "mean_document_frequency": artifacts.as_float(statistics.get("mean_document_frequency")),
        "longest_posting_list": artifacts.as_int(statistics.get("longest_posting_list")),
        "longest_posting_term": statistics.get("longest_posting_term"),
        "indexed_statistics": flat,
    }


def pipeline_overview(settings: Settings) -> Dict[str, Any]:
    try:
        rows = evaluation.pipelines(settings)
    except artifacts.ArtifactMissing:
        return dict(MISSING)
    selection = evaluation.final_pipeline(settings)
    return {
        "available": True,
        "final_pipeline": selection.get("final_pipeline"),
        "final_pipeline_name": selection.get("final_pipeline_name"),
        "score": artifacts.as_float(selection.get("score")),
        "margin_over_runner_up": artifacts.as_float(selection.get("margin_over_runner_up")),
        "reason": selection.get("reason"),
        "criteria_weights": selection.get("criteria_weights", {}),
        "pipelines": rows,
    }


def phase2_overview(settings: Settings) -> Dict[str, Any]:
    try:
        summary = experiments.summary(settings)
    except artifacts.ArtifactMissing:
        return dict(MISSING)
    tokenizers = summary.get("tokenization") or []
    best = None
    for row in tokenizers:
        if best is None or (row.get("total_tokens") or 0) < (best.get("total_tokens") or 0):
            best = row
    ngrams = summary.get("ngrams") or []
    return {
        "available": True,
        "tokenizer_count": len(tokenizers),
        "tokenizers": [
            {
                "tokenizer": row.get("tokenizer") or row.get("method"),
                "total_tokens": artifacts.as_int(row.get("total_tokens")),
                "vocabulary_size": artifacts.as_int(row.get("vocabulary_size")),
                "execution_time_seconds": artifacts.as_float(row.get("execution_time_seconds")),
            }
            for row in tokenizers
        ],
        "most_compact_tokenizer": (best.get("tokenizer") or best.get("method")) if best else None,
        "stopwords": summary.get("stopwords", {}),
        "stemming": summary.get("stemming", {}),
        "lemmatization": summary.get("lemmatization", {}),
        "preprocessing": summary.get("preprocessing", {}),
        "pos": summary.get("pos", {}),
        "ner": summary.get("ner", {}),
        "ngrams": ngrams,
        "bpe": summary.get("bpe", {}),
        "date_number": summary.get("date_number", {}),
        "corpus": summary.get("corpus", {}),
    }


def phase3_overview(settings: Settings) -> Dict[str, Any]:
    try:
        summary = evaluation.phase3_summary(settings)
    except artifacts.ArtifactMissing:
        return dict(MISSING)
    retrieval = summary.get("retrieval_statistics", {}) or {}
    coverage = summary.get("query_coverage", {}) or {}
    return {
        "available": True,
        "final_pipeline": summary.get("final_pipeline"),
        "final_pipeline_name": summary.get("final_pipeline_name"),
        "queries_defined": artifacts.as_int(coverage.get("queries_defined")),
        "queries_answered": artifacts.as_int(coverage.get("queries_answered")),
        "answerability": artifacts.as_float(coverage.get("answerability")),
        "total_units_returned": artifacts.as_int(retrieval.get("total_units_returned")),
        "total_documents_returned": artifacts.as_int(retrieval.get("total_documents_returned")),
        "mean_execution_time_ms": artifacts.as_float(retrieval.get("mean_execution_time_ms")),
        "query_types": coverage.get("by_query_type", {}),
    }


def phase4_overview(settings: Settings) -> Dict[str, Any]:
    try:
        summary = evaluation.phase4_summary(settings)
    except artifacts.ArtifactMissing:
        return dict(MISSING)
    judgment = summary.get("judgments", {}) or {}
    judged = (summary.get("evaluation") or {}).get("pipeline_b") or {}
    aggregates = {row["scope"]: row for row in evaluation.evaluation_aggregates(settings)}
    unit = aggregates.get("content_unit", {})
    document = aggregates.get("document", {})
    validation = evaluation.phase4_validation(settings)
    passed = sum(1 for row in validation if str(row.get("status", "")).upper() == "PASS")
    return {
        "available": True,
        "judgments": {
            "pairs": artifacts.as_int(judgment.get("pairs")),
            "relevant": artifacts.as_int(judgment.get("relevant")),
            "not_relevant": artifacts.as_int(judgment.get("not_relevant")),
            "pool_depth": artifacts.as_int(judgment.get("pool_depth")),
            "annotator": judgment.get("annotator"),
            "rubric": judgment.get("rubric"),
            "per_query": judgment.get("queries", {}),
        },
        "queries_evaluated": artifacts.as_int(judged.get("queries_evaluated")),
        "judged_depth": artifacts.as_int(judged.get("judged_depth")),
        "unjudged_policy": judged.get("unjudged_policy"),
        "unit_level": {
            "precision_macro": unit.get("precision_macro"),
            "recall_macro": unit.get("recall_macro"),
            "f1_macro": unit.get("f1_macro"),
            "precision_micro": unit.get("precision_micro"),
            "recall_micro": unit.get("recall_micro"),
            "f1_micro": unit.get("f1_micro"),
            "precision_at_5_macro": unit.get("precision_at_5_macro"),
            "recall_at_5_macro": unit.get("recall_at_5_macro"),
            "precision_at_10_macro": unit.get("precision_at_10_macro"),
            "recall_at_10_macro": unit.get("recall_at_10_macro"),
        },
        "document_level": {
            "precision_macro": document.get("precision_macro"),
            "recall_macro": document.get("recall_macro"),
            "f1_macro": document.get("f1_macro"),
            "precision_micro": document.get("precision_micro"),
            "recall_micro": document.get("recall_micro"),
            "f1_micro": document.get("f1_micro"),
        },
        "runner_up": summary.get("runner_up_retrieval_statistics", {}),
        "validation": {"rules": len(validation), "passed": passed},
    }


def validation_overview(settings: Settings) -> Dict[str, Any]:
    blocks: Dict[str, Any] = {}
    for name, reader in (
        ("phase1", lambda: corpus.phase1_rules(settings)),
        ("phase2", lambda: experiments.validation(settings)),
        ("phase3", lambda: evaluation.phase3_validation(settings)),
        ("phase4", lambda: evaluation.phase4_validation(settings)),
    ):
        try:
            rows = reader()
        except artifacts.ArtifactMissing:
            blocks[name] = {"available": False, "rules": [], "passed": 0, "total": 0}
            continue
        passed = sum(1 for row in rows if str(row.get("status", "")).upper() == "PASS")
        blocks[name] = {
            "available": True,
            "total": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
            "all_passed": passed == len(rows),
            "rules": rows,
        }
    total = sum(block.get("total", 0) for block in blocks.values())
    passed = sum(block.get("passed", 0) for block in blocks.values())
    return {
        "phases": blocks,
        "total": total,
        "passed": passed,
        "all_passed": total > 0 and passed == total,
    }


def statistics(settings: Settings) -> Dict[str, Any]:
    """The whole /statistics payload."""
    corpus_block = corpus_overview(settings)
    return {
        "project": project(settings),
        "corpus": corpus_block,
        "sources": corpus.source_summary(settings),
        "index": index_overview(settings),
        "pipelines": pipeline_overview(settings),
        "phase2": phase2_overview(settings),
        "phase3": phase3_overview(settings),
        "phase4": phase4_overview(settings),
        "validation": validation_overview(settings),
        "sources_note": (
            "Every figure is read from the artefact that produced it. Phases whose "
            "artefacts are missing are reported as unavailable, never as zero."
        ),
    }


def figures(settings: Settings) -> Dict[str, Any]:
    """The Phase 2 figures and Phase 3 markdown reports, for the docs page."""
    out: Dict[str, Any] = {"phase2_figures": experiments.figures(settings), "reports": {}}
    for phase, name in ((1, "README.md"), (2, "README.md"), (3, "README.md"), (4, "README.md")):
        directory = {
            1: settings.phase1_results,
            2: settings.phase2_results,
            3: settings.phase3_results,
            4: settings.phase4_results,
        }[phase]
        path = settings.path(f"{directory}/{name}")
        if path.is_file():
            out["reports"][f"phase{phase}"] = artifacts.read_text(path, required=False)
    return out
