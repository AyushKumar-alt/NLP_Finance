"""Phase 3 entry point.

    python -m src.phase3.run

Steps
-----
 1. load the configuration and clean only the Phase 3 output paths
 2. load the Phase 1 corpus and the Phase 2 evidence
 3. execute both pipelines and build both indexes
 4. run the 15 domain queries against both indexes
 5. score the pipelines on the declared criteria and select the final one
 6. re-run the query set against the winner and write the result tables
 7. write the index, statistics, narrative and validation artefacts

Phase 1 and Phase 2 artefacts are read-only inputs: nothing outside the paths
declared in ``config/phase3_config.yaml`` is written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.core.utils import sha256_file, write_csv, write_json, write_jsonl
from src.phase2.load_corpus import Corpus, load_corpus

from . import reporting, statistics as stats_module, validation
from .config import (
    Phase3Config,
    clean_managed_output,
    load_config,
    log_event,
    seed_everything,
    setup_logging,
)
from .index_builder import BuildResult, IndexBuilder
from .load_phase2 import load_phase2_evidence
from .pipeline_comparison import PipelineComparison, compare_pipelines
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec, load_pipeline_specs
from .query_registry import (
    QueryRecord,
    coverage_summary,
    execute_query_set,
    validate_query_vocabulary,
)
from .result_formatter import (
    DOCUMENT_FIELDS,
    RETRIEVAL_FIELDS,
    SUMMARY_FIELDS,
    ResultFormatter,
)
from .retrieval import RetrievalEngine, build_engine
from .statistics import compute_index_statistics, compute_retrieval_statistics, selection_payload

BANNER = "=" * 74
SUB = "-" * 74

#: Descriptions used in the README file table.
OUTPUT_DESCRIPTIONS = {
    "pipeline_comparison.csv": "One row per pipeline with every criterion, raw rate and weighted score",
    "pipeline_comparison.md": "The A vs B write-up, with the Phase 2 evidence it relies on",
    "final_pipeline.json": "The selection decision as data (scores, weights, reason, tie-breakers)",
    "final_pipeline.md": "The selected pipeline, its index statistics and its query coverage",
    "inverted_index.json": "The serialized inverted index: terms, postings, positions and provenance",
    "index_statistics.csv": "Index size, distribution and concentration measurements",
    "index_manifest.json": "What was indexed, from which inputs, with what settings and hashes",
    "query_registry.csv": "The 15 domain queries with their normalization, results and timing",
    "retrieval_results.csv": "One row per retrieved content unit, ranked, with provenance and snippet",
    "retrieval_summary.csv": "One row per query: answerability, coverage and execution time",
    "document_results.csv": "The same queries aggregated to document level",
    "query_examples/": "A readable text example per query",
    "validation_report.csv": "Every Phase 3 validation rule with its status and evidence",
    "phase3_summary.json": "The machine-readable summary of the whole Phase 3 run",
    "README.md": "How to read this directory",
}


def run(config: Phase3Config, verbose: bool = False) -> Dict[str, Any]:
    started = time.time()
    logger = setup_logging(config, verbose=verbose)
    seed_everything(config.random_seed)
    print(BANNER)
    print("PHASE 3 - pipeline comparison, inverted index and retrieval")
    print(BANNER)

    # ------------------------------------------------------------------ 1. setup
    cleaned = clean_managed_output(config, logger)
    specs = load_pipeline_specs(config)
    log_event(logger, "INFO", "setup",
              f"config {config.config_path.name}, seed {config.random_seed}, "
              f"cleaned {len(cleaned)} output path(s)")

    # ------------------------------------------------------------------ 2. inputs
    print("Loading Phase 1 corpus and Phase 2 evidence...", flush=True)
    corpus = load_corpus(config.phase2_config(), logger)
    units = corpus.selected_units(config.active_policy)
    log_event(logger, "INFO", "inputs",
              f"Phase 1: {corpus.document_count} documents, {corpus.page_count} pages, "
              f"{len(corpus.units)} units; policy '{corpus.policy}' selects {len(units)} units")
    evidence = load_phase2_evidence(config, logger)
    log_event(logger, "INFO", "inputs",
              f"Phase 2: {len(evidence.files_read)} input file(s) read; "
              f"{len(evidence.citations())} citation(s) available")

    runner = PipelineRunner(config, logger)
    builder = IndexBuilder(config, runner, logger)

    # ------------------------------------------------------------------ 3. pipelines + indexes
    builds: Dict[str, BuildResult] = {}
    pipeline_stats: Dict[str, Dict[str, Any]] = {}
    engines: Dict[str, RetrievalEngine] = {}
    for key in sorted(specs):
        spec = specs[key]
        print(f"Running {spec.key}: {spec.name}", flush=True)
        result = runner.run(spec, units)
        build = builder.build(
            result,
            corpus,
            entities_by_unit=evidence.ner_entities_by_unit,
            domain_entities_by_unit=evidence.domain_entities_by_unit,
            pipeline_metadata={
                "policy": corpus.policy,
                "components": result.stats.get("order"),
                "index_representation": spec.index_representation,
                "text_selection_policy": corpus.policy,
            },
        )
        builds[key] = build
        pipeline_stats[key] = result.stats
        engines[key] = build_engine(build.index, spec, runner, config, units=units, logger=logger)

    # ------------------------------------------------------------------ 4. queries on both
    query_outcomes: Dict[str, Sequence[Any]] = {}
    for key in sorted(engines):
        records = execute_query_set(config, engines[key], logger=logger)
        query_outcomes[key] = [record.outcome for record in records if record.outcome is not None]

    # ------------------------------------------------------------------ 5. selection
    print("Comparing pipelines...", flush=True)
    comparison = compare_pipelines(
        config=config,
        runner=runner,
        builds=builds,
        pipeline_stats=pipeline_stats,
        query_outcomes=query_outcomes,
        units=units,
        evidence=evidence,
        logger=logger,
    )
    write_csv(
        config.out_path("pipeline_comparison_csv"),
        comparison.rows(),
        list(comparison.rows()[0].keys()) if comparison.rows() else [],
    )
    config.out_path("pipeline_comparison_md").write_text(
        reporting.comparison_markdown(comparison, evidence, builds), encoding="utf-8"
    )
    write_json(config.out_path("final_pipeline_json"), selection_payload(comparison))
    log_event(logger, "INFO", "selection",
              f"final pipeline: {comparison.winner_key} "
              f"(score {comparison.winner.total_score:.4f})")

    # ------------------------------------------------------------------ 6. final pipeline retrieval
    final_key = comparison.winner_key
    engine = engines[final_key]
    records: List[QueryRecord] = execute_query_set(config, engine, logger=logger)
    coverage = coverage_summary(records)
    missing_vocabulary = validate_query_vocabulary(engine, records)
    if missing_vocabulary:
        log_event(logger, "WARNING", "queries",
                  f"configured query terms absent from the index: {missing_vocabulary}")

    formatter = ResultFormatter(config)
    rows = formatter.all_rows(records)
    write_csv(config.out_path("retrieval_results_csv"), rows["units"], list(RETRIEVAL_FIELDS))
    write_csv(config.out_path("document_results_csv"), rows["documents"], list(DOCUMENT_FIELDS))
    write_csv(config.out_path("retrieval_summary_csv"), rows["summary"], list(SUMMARY_FIELDS))
    write_csv(
        config.out_path("query_registry_csv"),
        [record.as_row() for record in records],
        list(records[0].as_row().keys()) if records else [],
    )
    examples_dir = config.out_dir("query_examples_dir")
    for record in records:
        (examples_dir / f"{record.query_id}.txt").write_text(
            formatter.query_example(record), encoding="utf-8"
        )

    # ------------------------------------------------------------------ 7. index + statistics
    print(f"Serializing the {final_key} index...", flush=True)
    winner_build = builds[final_key]
    winner_index = winner_build.index
    index_payload = winner_index.save(config.out_path("inverted_index_json"))
    index_stats = compute_index_statistics(
        winner_index, top_n=int(config.get("retrieval.top_terms_to_report", 25))
    )
    write_csv(config.out_path("index_statistics_csv"), index_stats.as_rows(),
              ["metric", "value"])
    retrieval_stats = compute_retrieval_statistics(records)

    manifest = {
        "phase": 3,
        "final_pipeline": final_key,
        "pipeline_name": comparison.winner.name,
        "text_selection_policy": corpus.policy,
        "corpus": corpus.summary(),
        "indexed_units": index_stats.indexed_units,
        "documents": index_stats.indexed_documents,
        "inputs": {
            key: {
                "path": str(config.project_path(str(value))),
                "sha256": _hash_if_present(config.project_path(str(value))),
            }
            for key, value in config.section("input").items()
        },
        "settings": {
            "ngrams": config.section("ngrams"),
            "retrieval": config.section("retrieval"),
            "selection": config.section("selection"),
            "financial_expression_patterns": {
                key: value for key, value in (config.get("financial_expressions") or {}).items()
                if key.endswith("pattern")
            },
        },
        "index": index_payload,
        "statistics": {row["metric"]: row["value"] for row in index_stats.as_rows()},
        "reproducibility": {
            "random_seed": config.random_seed,
            "deterministic_ordering": bool(config.get("reproducibility.deterministic_ordering", True)),
            "ranking": config.get("retrieval.ranking"),
            "tie_break": config.get("retrieval.ranking.tie_break"),
        },
    }
    write_json(config.out_path("index_manifest_json"), manifest)
    config.out_path("final_pipeline_md").write_text(
        reporting.final_pipeline_markdown(
            comparison, winner_build, index_stats, retrieval_stats, records, engine
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ 8. validation
    print("Validating Phase 3 outputs...", flush=True)
    placeholder_summary = _summary_payload(config, comparison, evidence, index_stats,
                                          retrieval_stats, coverage, records, [], outputs=None)
    write_json(config.out_path("summary_json"), placeholder_summary)
    # placeholders, so that the "required outputs exist" rule checks the complete
    # deliverable set; both are rewritten below with the real content
    write_csv(
        config.out_path("validation_report_csv"),
        [{"rule_id": "P00", "rule": "validation report pending", "status": "PENDING",
          "detail": "written before validation so the deliverable set is complete",
          "evidence": ""}],
        ["rule_id", "rule", "status", "detail", "evidence"],
    )
    config.out_path("readme").write_text(
        "# Phase 3 results\n\nValidation in progress; rewritten when the run finishes.\n",
        encoding="utf-8",
    )

    validation_results = validation.run_validation(
        config=config,
        index=winner_index,
        comparison=comparison,
        records=records,
        engine=engine,
        unit_rows=rows["units"],
        logger=logger,
    )
    # second pass: now every required artefact exists on disk, so the artefact
    # rules see the same state a reader of results/phase3 will see
    validation_results = validation.run_validation(
        config=config,
        index=winner_index,
        comparison=comparison,
        records=records,
        engine=engine,
        unit_rows=rows["units"],
        logger=logger,
    )
    write_csv(
        config.out_path("validation_report_csv"),
        [result.as_row() for result in validation_results],
        ["rule_id", "rule", "status", "detail", "evidence"],
    )

    # ------------------------------------------------------------------ 9. summary + README
    # The inventory is taken after every artefact exists, so the README and the
    # summary JSON list the same complete set of files.
    config.out_path("readme").write_text(
        reporting.readme_markdown(
            config, comparison, index_stats, retrieval_stats, coverage,
            [result.as_row() for result in validation_results], [],
            [record.as_row() for record in records], evidence,
        ),
        encoding="utf-8",
    )
    outputs = _output_inventory(config, corpus)
    summary = _summary_payload(config, comparison, evidence, index_stats, retrieval_stats,
                               coverage, records, validation_results, outputs=outputs)
    write_json(config.out_path("summary_json"), summary)
    config.out_path("readme").write_text(
        reporting.readme_markdown(
            config, comparison, index_stats, retrieval_stats, coverage,
            [result.as_row() for result in validation_results], outputs,
            [record.as_row() for record in records], evidence,
        ),
        encoding="utf-8",
    )

    elapsed = time.time() - started
    log_event(logger, "INFO", "run",
              f"Phase 3 finished in {elapsed:.1f}s; {len(outputs)} artefact(s) written")
    _final_report(logger, comparison, index_stats, retrieval_stats, coverage,
                  validation_results, records, outputs, elapsed,
                  str(config.out_path("results_dir")))
    return {
        "validation": [result.as_row() for result in validation_results],
        "outputs": outputs,
        "final_pipeline": final_key,
        "score": comparison.winner.total_score,
        "coverage": coverage,
        "index_statistics": {row["metric"]: row["value"] for row in index_stats.as_rows()},
        "retrieval_statistics": retrieval_stats.as_dict(),
        "runtime_seconds": round(elapsed, 3),
    }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _hash_if_present(path: Path) -> str:
    try:
        if path.is_file():
            return sha256_file(path)
    except OSError:  # pragma: no cover - defensive
        pass
    return ""


def _output_inventory(config: Phase3Config, corpus: Corpus) -> List[Dict[str, str]]:
    """Every file Phase 3 wrote, with its description."""
    results_dir = config.out_path("results_dir")
    inventory: List[Dict[str, str]] = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(results_dir.parent).as_posix()
        name = path.relative_to(results_dir).as_posix()
        if path.parent == results_dir:
            description = OUTPUT_DESCRIPTIONS.get(name, "Phase 3 output")
        elif name.startswith("query_examples/"):
            description = "Readable result example for one query"
        else:
            description = "Phase 3 output"
        inventory.append({"path": relative, "description": description})
    return inventory


def _summary_payload(
    config: Phase3Config,
    comparison: PipelineComparison,
    evidence: Any,
    index_stats: Any,
    retrieval_stats: Any,
    coverage: Dict[str, Any],
    records: Sequence[QueryRecord],
    validation_results: Sequence[Any],
    outputs: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    passed = sum(1 for result in validation_results if result.status == "PASS")
    return {
        "phase": 3,
        "final_pipeline": comparison.winner_key,
        "final_pipeline_name": comparison.winner.name,
        "selection": selection_payload(comparison),
        "inputs": {
            "phase1_corpus": str(config.corpus_jsonl),
            "phase2_config": str(config.phase2_config().config_path),
            "phase2_files_read": evidence.files_read,
            "phase2_evidence": evidence.citations(),
        },
        "index_statistics": {row["metric"]: row["value"] for row in index_stats.as_rows()},
        "retrieval_statistics": retrieval_stats.as_dict(),
        "query_coverage": coverage,
        "queries": [record.as_row() for record in records],
        "validation": {
            "passed": passed,
            "total": len(validation_results),
            "results": [result.as_row() for result in validation_results],
        },
        "outputs": list(outputs or []),
    }


def _final_report(
    logger: Any,
    comparison: PipelineComparison,
    index_stats: Any,
    retrieval_stats: Any,
    coverage: Dict[str, Any],
    validation_results: Sequence[Any],
    records: Sequence[QueryRecord],
    outputs: Sequence[Dict[str, str]],
    elapsed: float,
    results_dir: str,
) -> None:
    print(f"\n{SUB}\nFINAL PIPELINE\n{SUB}")
    print(f"  selected : {comparison.winner.name} ({comparison.winner_key})")
    print(f"  score    : {comparison.winner.total_score:.4f} "
          f"(margin {comparison.margin:+.4f})")
    for criterion in comparison.winner.criteria:
        print(f"    - {criterion.name:38s} {criterion.as_detail():>16s} "
              f"x{criterion.weight} = {criterion.weighted:.4f}")
    print(f"  {comparison.reason}")
    print(f"\n{SUB}\nINDEX\n{SUB}")
    print(f"  terms {index_stats.index_terms} "
          f"(unigrams {index_stats.unigram_terms}, phrases {index_stats.phrase_terms}), "
          f"postings {index_stats.total_postings}, positions {index_stats.total_positions}")
    print(f"  units {index_stats.indexed_units} in {index_stats.indexed_documents} documents, "
          f"postings/term {index_stats.postings_per_term}, hapax ratio {index_stats.hapax_ratio}")
    print(f"\n{SUB}\nRETRIEVAL\n{SUB}")
    print(f"  {coverage['answered']}/{coverage['queries']} queries answered "
          f"({coverage['answerability'] * 100:.1f}%), "
          f"{retrieval_stats.total_units_returned} unit results, "
          f"mean {retrieval_stats.mean_execution_time_ms:.3f} ms/query")
    for record in records:
        flag = "ok " if record.answered else "MISS"
        print(f"  [{flag}] {record.query_id} {record.query:34s} "
              f"{record.unit_count:>4d} units  {record.document_count:>3d} docs  "
              f"{record.execution_time_ms:7.3f} ms")
    passed = sum(1 for result in validation_results if result.status == "PASS")
    print(f"\n{SUB}\nVALIDATION\n{SUB}")
    print(f"  {passed}/{len(validation_results)} rules PASS")
    for result in validation_results:
        if result.status != "PASS":
            print(f"  {result.status} {result.rule_id}: {result.detail}")
    print(f"\n{SUB}\nOUTPUTS ({len(outputs)} files in {results_dir})\n{SUB}")
    for entry in outputs:
        print(f"  {entry['path']}")
    print(f"\n{BANNER}\nPhase 3 complete in {elapsed:.1f}s")
    print(BANNER)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 (pipelines, index, retrieval).")
    parser.add_argument("--config", default=None, help="path to phase3_config.yaml")
    parser.add_argument("--verbose", action="store_true", help="verbose console logging")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    outcome = run(config, verbose=args.verbose)
    failures = [row for row in outcome["validation"] if row["status"] == "FAIL"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
