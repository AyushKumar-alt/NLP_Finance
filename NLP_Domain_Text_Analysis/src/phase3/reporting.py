"""Narrative Phase 3 outputs: the comparison, the decision and the README.

Markdown tables here are generated from the same objects that produce the CSV
files, so a narrative number can never drift from a measured one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Phase3Config
from .index_builder import BuildResult
from .load_phase2 import Phase2Evidence
from .pipeline_comparison import PipelineComparison, PipelineMeasurement
from .query_registry import QueryRecord, coverage_summary
from .retrieval import RetrievalEngine
from .statistics import IndexStatistics, RetrievalStatistics, selection_payload

#: Phase 2 facts reused as Phase 3 inputs, quoted with their source file.
EVIDENCE_NOTE = (
    "Phase 3 consumes Phase 1 (structured corpus) and Phase 2 (NLP experiments) as "
    "inputs and never rewrites either phase's artefacts."
)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(header) for header in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _rate(value: float) -> str:
    return f"{value:.4f}"


def comparison_markdown(
    comparison: PipelineComparison,
    evidence: Phase2Evidence,
    builds: Dict[str, BuildResult],
) -> str:
    """``pipeline_comparison.md`` - the A vs B write-up."""
    parts: List[str] = [
        "# Phase 3 - Pipeline A vs Pipeline B",
        "",
        EVIDENCE_NOTE,
        "",
        "## 1. The two pipelines",
        "",
    ]
    pipeline_rows = []
    for key in sorted(comparison.measurements):
        spec = comparison.measurements[key].spec
        pipeline_rows.append(
            [
                spec.key,
                spec.tokenizer,
                spec.stopword_strategy,
                spec.morphology_label,
                spec.term_kind,
                " -> ".join(spec.order),
            ]
        )
    parts.append(
        _table(
            ["pipeline", "tokenizer", "stopwords", "morphology", "index terms are", "order"],
            pipeline_rows,
        )
    )
    for key in sorted(comparison.measurements):
        spec = comparison.measurements[key].spec
        if spec.notes:
            parts += ["", f"**{spec.key}**: {spec.notes}"]

    parts += ["", "## 2. Phase 2 evidence used as input", ""]
    for citation in evidence.citations():
        parts.append(f"- {citation}")
    parts += [
        "",
        f"- {evidence.summary_line()}",
        "",
        "## 3. Measured criteria",
        "",
        "Every value below is computed from the artifacts each pipeline produced; "
        "nothing is copied from a previous phase except the inputs listed above.",
        "",
    ]
    criteria = comparison.measurements[sorted(comparison.measurements)[0]].criteria
    pipeline_keys = sorted(comparison.measurements)
    rows = []
    for criterion in criteria:
        rows.append(
            [f"{criterion.name}<br>(weight {criterion.weight} x = "
             f"{criterion.weighted:.4f})"]
            + [
                f"{_rate(comparison.measurements[key].criterion(criterion.name).rate)}<br>"
                f"{comparison.measurements[key].criterion(criterion.name).as_detail()}"
                for key in pipeline_keys
            ]
        )
    rows.append(
        ["**total score**"]
        + [f"**{comparison.measurements[key].total_score:.4f}**" for key in pipeline_keys]
    )
    parts.append(_table(["criterion"] + pipeline_keys, rows))

    parts += ["", "## 4. What each criterion measures", ""]
    date_number_total = evidence.date_number_row("TOTAL") or {}
    intact_rate = date_number_total.get("intact_rate_percent", "12.85")
    preservation = [
        comparison.measurements[key].criterion("financial_expression_preservation")
        for key in sorted(comparison.measurements)
    ]
    if len({(c.numerator, c.denominator) for c in preservation}) == 1:
        parts += [
            "> Both pipelines preserve exactly "
            f"{preservation[0].numerator} of {preservation[0].denominator} financial "
            "expressions. That is expected, not a copy-paste error: expression protection "
            "runs *before* morphology and the Phase 2 tokenizer already emits "
            "`7.4 per cent` as one token, and stemming leaves digits and currency "
            "expressions untouched. This criterion therefore confirms the protection step "
            "works, but it does not separate the two pipelines on this corpus.",
            "",
        ]
    parts += [
        "- **financial_expression_preservation** - share of the typed financial expressions "
        "found in the raw text (`7.4 per cent`, `FY2025-26`, `Rs 1.25 lakh crore`) that are "
        "still a single index term. Phase 2 measured that standard tokenization keeps only "
        f"{intact_rate}% of them intact, which is why this criterion carries the largest weight.",
        "- **domain_term_recall** - share of the domain vocabulary (the configured "
        "reference terms plus the `domain_terms_in_phrase` values Phase 2 exported) that "
        "resolves to an index term.",
        "- **variant_collapse_rate** - share of the configured variant groups whose surface "
        "forms all map onto one indexed term, so a query for `investment` also finds "
        "`investments` and `investors`.",
        "- **query_answerability** - share of the 15 configured domain queries that return at "
        "least one result on that pipeline's own index.",
        "",
        f"Timing is measured and reported but excluded from the score "
        f"(`exclude_from_score: {comparison.excluded_from_score}`): a slower pipeline that "
        "answers more queries is still the better index.",
        "",
        "## 5. Decision",
        "",
        f"**Selected: {comparison.winner.name}** ({comparison.winner_key})",
        "",
        comparison.reason,
        "",
    ]
    if comparison.margin:
        runner_up_key = next(k for k in sorted(comparison.measurements) if k != comparison.winner_key)
        runner_up = comparison.measurements[runner_up_key]
        parts += [
            f"Runner-up: {runner_up.name} ({runner_up_key}) at {runner_up.total_score:.4f}.",
            "",
        ]
    parts += ["Tie-breakers, in order: " + ", ".join(comparison.tie_breakers), ""]

    parts += ["## 6. Index size consequence", ""]
    size_rows = []
    for key in sorted(comparison.measurements):
        measurement = comparison.measurements[key]
        stats = measurement.index_stats
        size_rows.append(
            [
                key,
                stats.get("index_terms", 0),
                stats.get("unigram_terms", 0),
                stats.get("phrase_terms", 0),
                stats.get("total_postings", 0),
                stats.get("indexed_units", 0),
                measurement.pipeline_stats.get("total_tokens", 0),
                measurement.pipeline_stats.get("processing_time_seconds", 0.0),
            ]
        )
    parts.append(
        _table(
            ["pipeline", "index terms", "unigrams", "phrases", "postings", "units",
             "tokens", "processing s"],
            size_rows,
        )
    )

    parts += ["", "## 7. What survived, and what did not", ""]
    for key in sorted(comparison.measurements):
        measurement = comparison.measurements[key]
        parts += [
            f"**{key}**",
            "",
            f"- preserved: {'; '.join(measurement.preserved_examples) or 'none recorded'}",
            f"- not preserved: {'; '.join(measurement.lost_examples) or 'none recorded'}",
            f"- collapsed variant groups: "
            f"{'; '.join(measurement.collapsed_groups) or 'none'}",
            f"- domain terms missing from the index: "
            f"{'; '.join(measurement.missing_domain_terms) or 'none'}",
            f"- unanswered queries: {'; '.join(measurement.unanswered_queries) or 'none'}",
            "",
        ]
    parts += [
        "The expressions that do *not* survive are the bare currency amounts "
        "(`₹ 2.5`, `$25`): the Phase 2 tokenizer splits the symbol from the number, so the "
        "expression-protection step has nothing to re-join. This is a tokenizer-level "
        "limitation inherited from Phase 2, reported here rather than patched, because "
        "Phase 3 must not change the tokenizer the earlier phase measured.",
        "",
    ]
    return "\n".join(parts) + "\n"


def final_pipeline_markdown(
    comparison: PipelineComparison,
    winner: BuildResult,
    index_stats: IndexStatistics,
    retrieval_stats: RetrievalStatistics,
    records: Sequence[QueryRecord],
    engine: RetrievalEngine,
) -> str:
    """``final_pipeline.md`` - the decision and the index it produced."""
    spec = winner.spec
    coverage = coverage_summary(records)
    parts: List[str] = [
        f"# Phase 3 - Final pipeline: {spec.name}",
        "",
        f"Pipeline key: `{spec.key}`",
        "",
        "## Component order",
        "",
    ]
    parts.append(_table(["#", "step", "component"],
                        [[s["order"], s["step"], s["component"]] for s in spec.step_labels()]))
    parts += [
        "",
        f"Index representation: `{spec.index_representation}` (case-folded, positional, "
        "provenance preserved per posting).",
        "",
        "## Why this pipeline",
        "",
        comparison.reason,
        "",
        "## Selected index",
        "",
    ]
    parts.append(
        _table(
            ["metric", "value"],
            [
                ["index terms", index_stats.index_terms],
                ["unigram terms", index_stats.unigram_terms],
                ["phrase terms (2-3 grams, min frequency "
                 f"{engine.config.get('ngrams.min_frequency')})", index_stats.phrase_terms],
                ["postings", index_stats.total_postings],
                ["postings per term", index_stats.postings_per_term],
                ["positions stored", index_stats.total_positions],
                ["content units indexed", index_stats.indexed_units],
                ["documents indexed", index_stats.indexed_documents],
                ["units per document", index_stats.units_per_document],
                ["singleton terms", index_stats.singleton_terms],
                ["hapax ratio", index_stats.hapax_ratio],
                ["terms holding 90% of postings", index_stats.terms_in_90_percent_of_postings],
                ["longest posting list", f"{index_stats.longest_posting_term} "
                                        f"({index_stats.longest_posting_list})"],
            ],
        )
    )
    parts += [
        "",
        "## Retrieval behaviour",
        "",
        _table(
            ["metric", "value"],
            [
                ["queries executed", retrieval_stats.queries],
                ["queries answered", retrieval_stats.answered],
                ["answerability", retrieval_stats.answerability],
                ["units returned in total", retrieval_stats.total_units_returned],
                ["mean units per query", retrieval_stats.mean_units_per_query],
                ["median units per query", retrieval_stats.median_units_per_query],
                ["mean documents per query", retrieval_stats.mean_documents_per_query],
                ["mean execution time (ms)", retrieval_stats.mean_execution_time_ms],
                ["max execution time (ms)", retrieval_stats.max_execution_time_ms],
            ],
        ),
        "",
        "## Query coverage",
        "",
        _table(
            ["query", "type", "units", "documents", "missing terms", "ms"],
            [
                [
                    f"{r.query_id} {r.query}",
                    r.query_type,
                    r.unit_count,
                    r.document_count,
                    ", ".join(r.missing_terms) or "-",
                    f"{r.execution_time_ms:.3f}",
                ]
                for r in records
            ],
        ),
        "",
        f"Answerability: {coverage['answered']}/{coverage['queries']} "
        f"({coverage['answerability'] * 100:.1f}%). "
        f"Expected domains covered: {', '.join(coverage['expected_domains'])}.",
        "",
        "Retrieval quality (precision/recall of these results) is **not** measured here; "
        "that is the job of Phase 4.",
        "",
        "## Traceability",
        "",
        "Every result row keeps the full Phase 1 chain: `source_id -> document_id -> "
        "page_number -> section_id -> unit_id`, plus a snippet of the matched text.",
        "",
    ]
    return "\n".join(parts) + "\n"


def readme_markdown(
    config: Phase3Config,
    comparison: PipelineComparison,
    index_stats: IndexStatistics,
    retrieval_stats: RetrievalStatistics,
    coverage: Dict[str, Any],
    validation_rows: Sequence[Dict[str, Any]],
    outputs: Sequence[Dict[str, str]],
    query_rows: Sequence[Dict[str, Any]],
    evidence: Phase2Evidence,
) -> str:
    """``README.md`` - what Phase 3 produced and how to read it."""
    passed = sum(1 for row in validation_rows if row.get("status") == "PASS")
    parts: List[str] = [
        "# Phase 3 results - pipeline comparison and retrieval",
        "",
        EVIDENCE_NOTE,
        "",
        "## Outcome",
        "",
        f"- Selected pipeline: **{comparison.winner.name}** (`{comparison.winner_key}`), "
        f"score {comparison.winner.total_score:.4f}",
        f"- Runner-up margin: {comparison.margin:+.4f}",
        f"- Index: {index_stats.index_terms} terms ({index_stats.unigram_terms} unigrams, "
        f"{index_stats.phrase_terms} phrases), {index_stats.total_postings} postings, "
        f"{index_stats.indexed_units} content units, {index_stats.indexed_documents} documents",
        f"- Queries: {coverage['answered']}/{coverage['queries']} answered "
        f"({coverage['answerability'] * 100:.1f}%), "
        f"{retrieval_stats.total_units_returned} unit results in total",
        f"- Validation: {passed}/{len(validation_rows)} rules PASS",
        "",
        "## Files",
        "",
        _table(
            ["file", "what it contains"],
            [
                [row["path"], row["description"]]
                for row in outputs
            ],
        ),
        "",
        "## Query set",
        "",
        _table(
            ["id", "query", "type", "units", "documents", "answered"],
            [
                [
                    row["query_id"],
                    row["query"],
                    row["query_type"],
                    row["result_units"],
                    row["result_documents"],
                    "yes" if str(row["answered"]) == "True" else "no",
                ]
                for row in query_rows
            ],
        ),
        "",
        "## Inputs used",
        "",
        "Phase 2 evidence cited by the comparison:",
        "",
    ]
    for citation in evidence.citations():
        parts.append(f"- {citation}")
    parts += [
        "",
        "## How to reproduce",
        "",
        "```",
        "python -m src.phase3.run",
        "```",
        "",
        f"Random seed: {config.random_seed}. All ordering is deterministic (posting lists "
        "sorted by `unit_id`, ties in ranking broken by `unit_id`), so a second run "
        "produces byte-identical result tables.",
        "",
    ]
    return "\n".join(parts) + "\n"
