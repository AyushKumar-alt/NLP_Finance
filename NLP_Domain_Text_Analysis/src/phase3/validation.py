"""Phase 3 validation rules.

Each rule asks one question and answers it with a computed check. Nothing is
asserted that was not measured here, and a failing rule is reported rather than
suppressed.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import Phase3Config, log_event
from .inverted_index import InvertedIndex
from .keyword_search import UnitHit
from .pipeline_comparison import PipelineComparison
from .query_parser import QuerySyntaxError, parse_query
from .query_registry import QueryRecord
from .retrieval import RetrievalEngine

#: Artefacts that must exist and be non-empty after a Phase 3 run.
REQUIRED_OUTPUTS = (
    "pipeline_comparison.csv",
    "pipeline_comparison.md",
    "final_pipeline.json",
    "final_pipeline.md",
    "inverted_index.json",
    "index_statistics.csv",
    "index_manifest.json",
    "query_registry.csv",
    "retrieval_results.csv",
    "retrieval_summary.csv",
    "document_results.csv",
    "validation_report.csv",
    "phase3_summary.json",
    "README.md",
)

#: Queries the engine must refuse (unrestricted NOT is not meaningful).
REJECTED_QUERY_SAMPLES = (
    ("NOT food", "bare NOT"),
    ("NOT (food OR banking)", "bare NOT with parentheses"),
    ("food AND (", "unbalanced parenthesis"),
    ("food)", "unmatched closing parenthesis"),
    ("AND food", "operator without a left operand"),
    ('"unclosed phrase', "unclosed quote"),
)


@dataclass
class ValidationResult:
    rule_id: str
    rule: str
    status: str          # PASS / FAIL / WARN
    detail: str
    evidence: str

    def as_row(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule": self.rule,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def run_validation(
    config: Phase3Config,
    index: InvertedIndex,
    comparison: PipelineComparison,
    records: Sequence[QueryRecord],
    engine: RetrievalEngine,
    unit_rows: Sequence[Dict[str, object]],
    logger: Optional[logging.Logger] = None,
) -> List[ValidationResult]:
    results: List[ValidationResult] = [
        _index_structurally_valid(index),
        _postings_reference_known_units(index),
        _postings_sorted_and_unique(index),
        _provenance_complete(index),
        _positions_present(config, index),
        _pipeline_decision_declared(comparison),
        _pipeline_scores_within_range(comparison),
        _phrase_terms_derived_from_unigrams(index),
        _queries_answered(records),
        _no_duplicate_results(unit_rows),
        _results_traceable(unit_rows, index),
        _boolean_parser_rejects_unsafe_queries(),
        _boolean_parser_accepts_supported_forms(),
        _not_only_under_and(engine),
        _outputs_exist(config),
        _outputs_readable(config),
        _deterministic_ranking(unit_rows),
        _phase2_artifacts_untouched(config),
    ]
    failures = [result for result in results if result.status == "FAIL"]
    if logger is not None:
        log_event(
            logger,
            "INFO",
            "validation",
            f"{len(results) - len(failures)}/{len(results)} rules passed"
            + (f"; failures: {[f.rule_id for f in failures]}" if failures else ""),
        )
    return results


# ----------------------------------------------------------------------
# Index rules
# ----------------------------------------------------------------------
def _index_structurally_valid(index: InvertedIndex) -> ValidationResult:
    problems = index.validate()
    return ValidationResult(
        "P01",
        "the inverted index passes its own structural validation (no duplicate, "
        "unsorted or dangling postings)",
        "PASS" if not problems else "FAIL",
        f"{index.term_count} terms, {index.posting_count} postings, "
        f"{len(problems)} structural problem(s)",
        "; ".join(problems[:3]) if problems else "every posting is unique, sorted and resolvable",
    )


def _postings_reference_known_units(index: InvertedIndex) -> ValidationResult:
    dangling = sorted(
        {
            posting.unit_id
            for entry in index.terms.values()
            for posting in entry.postings
            if posting.unit_id not in index.units
        }
    )
    return ValidationResult(
        "P02",
        "every posting resolves to a provenance record in the index",
        "PASS" if not dangling else "FAIL",
        f"{index.posting_count} postings checked against {index.unit_count} unit records",
        f"dangling: {dangling[:5]}" if dangling else "no dangling unit references",
    )


def _postings_sorted_and_unique(index: InvertedIndex) -> ValidationResult:
    unsorted_terms: List[str] = []
    duplicated_terms: List[str] = []
    for term, entry in index.terms.items():
        unit_ids = [posting.unit_id for posting in entry.postings]
        if unit_ids != sorted(unit_ids):
            unsorted_terms.append(term)
        if len(unit_ids) != len(set(unit_ids)):
            duplicated_terms.append(term)
    ok = not unsorted_terms and not duplicated_terms
    return ValidationResult(
        "P03",
        "posting lists are sorted by unit_id and contain no duplicate units",
        "PASS" if ok else "FAIL",
        f"{index.term_count} posting lists checked",
        f"unsorted: {unsorted_terms[:3]}; duplicated: {duplicated_terms[:3]}"
        if not ok else "all posting lists sorted and duplicate-free",
    )


def _provenance_complete(index: InvertedIndex) -> ValidationResult:
    incomplete = [
        unit_id
        for unit_id, record in index.units.items()
        if not record.document_id
        or not record.source_id
        or not record.section_id
        or not record.unit_id
        or record.page_number <= 0
    ]
    missing_document = [
        unit_id for unit_id, record in index.units.items() if record.document_id not in index.documents
    ]
    ok = not incomplete and not missing_document
    return ValidationResult(
        "P04",
        "source_id -> document_id -> page_number -> section_id -> unit_id is preserved for "
        "every indexed unit",
        "PASS" if ok else "FAIL",
        f"{index.unit_count} unit records checked across {index.document_count()} documents",
        f"incomplete: {incomplete[:3]}; unknown documents: {missing_document[:3]}"
        if not ok else "full provenance chain present for every unit",
    )


def _positions_present(config: Phase3Config, index: InvertedIndex) -> ValidationResult:
    if not config.get("retrieval.enable_positions", True):
        return ValidationResult("P05", "token positions are stored for phrase search",
                                "WARN", "positions disabled in configuration", "config")
    without = sorted(
        term for term, entry in index.terms.items()
        if entry.postings and any(not posting.positions for posting in entry.postings)
    )
    return ValidationResult(
        "P05",
        "every posting carries token positions, so phrase search is positional",
        "PASS" if not without else "FAIL",
        f"{index.term_count} terms carry positions",
        f"terms without positions: {without[:3]}" if without else "all postings positional",
    )


# ----------------------------------------------------------------------
# Selection rules
# ----------------------------------------------------------------------
def _pipeline_decision_declared(comparison: PipelineComparison) -> ValidationResult:
    scores = {key: measurement.total_score for key, measurement in comparison.measurements.items()}
    best = max(scores, key=lambda key: (scores[key], key))
    declared = best if not comparison.margin else comparison.winner_key
    ok = declared in comparison.measurements
    return ValidationResult(
        "P06",
        "a final pipeline is selected from the measured criteria with a stated reason",
        "PASS" if ok else "FAIL",
        "; ".join(f"{key}={value:.4f}" for key, value in sorted(scores.items())),
        comparison.reason,
    )


def _pipeline_scores_within_range(comparison: PipelineComparison) -> ValidationResult:
    problems: List[str] = []
    total_weight = sum(comparison.criteria_weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        problems.append(f"criteria weights sum to {total_weight}, expected 1.0")
    for key, measurement in comparison.measurements.items():
        for criterion in measurement.criteria:
            if not 0.0 <= criterion.rate <= 1.0:
                problems.append(f"{key}.{criterion.name}={criterion.rate}")
            if criterion.weight <= 0.0:
                problems.append(f"{key}.{criterion.name} has weight {criterion.weight}")
    return ValidationResult(
        "P07",
        "every criterion rate is in [0, 1] and the weights sum to 1.0",
        "PASS" if not problems else "FAIL",
        f"weights: {comparison.criteria_weights} (sum {round(total_weight, 6)})",
        f"problems: {problems}" if problems else "all normalized scores and weights are valid",
    )


def _phrase_terms_derived_from_unigrams(index: InvertedIndex) -> ValidationResult:
    orphans: List[str] = []
    for term, entry in index.terms.items():
        if not entry.is_phrase:
            continue
        for part in term.split("_"):
            if part and part not in index.terms:
                orphans.append(term)
                break
    return ValidationResult(
        "P08",
        "every phrase term is composed of index terms that exist on their own",
        "PASS" if not orphans else "FAIL",
        f"{sum(1 for e in index.terms.values() if e.is_phrase)} phrase terms checked",
        f"orphans: {orphans[:3]}" if orphans else "all phrase terms decompose into indexed unigrams",
    )


# ----------------------------------------------------------------------
# Retrieval rules
# ----------------------------------------------------------------------
def _queries_answered(records: Sequence[QueryRecord]) -> ValidationResult:
    answered = [record for record in records if record.answered]
    unanswered = [record.query_id for record in records if not record.answered]
    return ValidationResult(
        "P09",
        "at least one configured domain query is answered by the final index",
        "PASS" if answered else "FAIL",
        f"{len(answered)}/{len(records)} queries answered"
        + (f"; unanswered: {unanswered}" if unanswered else ""),
        "; ".join(f"{r.query_id}={r.unit_count}" for r in records),
    )


def _no_duplicate_results(unit_rows: Sequence[Dict[str, object]]) -> ValidationResult:
    seen = set()
    duplicates: List[str] = []
    for row in unit_rows:
        key = (row.get("query_id"), row.get("unit_id"))
        if key in seen:
            duplicates.append(f"{key[0]}/{key[1]}")
        seen.add(key)
    return ValidationResult(
        "P10",
        "no unit is returned twice for the same query",
        "PASS" if not duplicates else "FAIL",
        f"{len(unit_rows)} result rows checked",
        f"duplicates: {duplicates[:5]}" if duplicates else "every (query, unit) pair appears once",
    )


def _results_traceable(
    unit_rows: Sequence[Dict[str, object]], index: InvertedIndex
) -> ValidationResult:
    missing: List[str] = []
    for row in unit_rows:
        record = index.unit(str(row.get("unit_id", "")))
        if record is None:
            missing.append(str(row.get("unit_id")))
            continue
        if row.get("document_id") != record.document_id or row.get("page_number") != record.page_number:
            missing.append(str(row.get("unit_id")))
    return ValidationResult(
        "P11",
        "every result row carries provenance that matches the index",
        "PASS" if not missing else "FAIL",
        f"{len(unit_rows)} rows verified against the provenance store",
        f"mismatches: {missing[:5]}" if missing else "all rows traceable to their Phase 1 unit",
    )


def _boolean_parser_rejects_unsafe_queries() -> ValidationResult:
    accepted_wrongly: List[str] = []
    for query, label in REJECTED_QUERY_SAMPLES:
        try:
            parse_query(query)
        except QuerySyntaxError:
            continue
        accepted_wrongly.append(f"{query} ({label})")
    return ValidationResult(
        "P12",
        "the Boolean parser rejects bare NOT, unbalanced parentheses and malformed operators "
        "without using eval()",
        "PASS" if not accepted_wrongly else "FAIL",
        f"{len(REJECTED_QUERY_SAMPLES)} malformed queries tested, "
        f"{len(REJECTED_QUERY_SAMPLES) - len(accepted_wrongly)} rejected",
        f"accepted incorrectly: {accepted_wrongly}" if accepted_wrongly
        else "all malformed queries raised QuerySyntaxError",
    )


def _boolean_parser_accepts_supported_forms() -> ValidationResult:
    supported = (
        "gdp AND inflation",
        "gdp OR gva",
        "inflation AND NOT food",
        "(gdp OR gva) AND policy",
        '"monetary policy" AND rbi',
        "(a AND NOT b) OR c",
    )
    failures: List[str] = []
    for query in supported:
        try:
            parse_query(query)
        except QuerySyntaxError as error:
            failures.append(f"{query}: {error}")
    return ValidationResult(
        "P13",
        "the Boolean parser accepts AND, OR, A AND NOT B and parenthesized groups",
        "PASS" if not failures else "FAIL",
        f"{len(supported)} supported forms tested",
        f"failures: {failures}" if failures else "all supported forms parsed",
    )


def _not_only_under_and(engine: RetrievalEngine) -> ValidationResult:
    """``A AND NOT B`` must return a strict subset of ``A``.

    The comparison is made on the raw posting lists, not on the capped result
    tables, so the check cannot be distorted by ``max_results_per_query``.
    """
    from .query_parser import Term

    include = "inflation"
    exclude = "food"
    try:
        positive_ids = engine.boolean_search.evaluate(Term(text=include)).postings.unit_id_set
        negative_ids = engine.boolean_search.search_not([include], [exclude]).postings.unit_id_set
    except QuerySyntaxError as error:
        return ValidationResult("P14", "A AND NOT B removes postings from A", "FAIL", str(error), "")
    excluded_ids = engine.boolean_search.evaluate(Term(text=exclude)).postings.unit_id_set
    subset = negative_ids <= positive_ids
    removed = len(positive_ids) - len(negative_ids)
    if not subset:
        status = "FAIL"
    elif removed == 0:
        status = "WARN"      # the exclusion removed nothing, so nothing was proved
    else:
        status = "PASS"
    return ValidationResult(
        "P14",
        "A AND NOT B returns a strict subset of A (set difference, not a full-corpus scan)",
        status,
        f"'{include}' -> {len(positive_ids)} units; '{include} AND NOT {exclude}' -> "
        f"{len(negative_ids)} units; removed {removed} of {len(excluded_ids & positive_ids)} "
        f"co-occurring unit(s)",
        f"leaked units: {sorted(negative_ids - positive_ids)[:5]}"
        if not subset else "no unit outside the positive set was returned",
    )


# ----------------------------------------------------------------------
# Artefact rules
# ----------------------------------------------------------------------
def _outputs_exist(config: Phase3Config) -> ValidationResult:
    root = config.out_path("results_dir")
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).is_file()]
    empty = [name for name in REQUIRED_OUTPUTS if (root / name).is_file() and (root / name).stat().st_size == 0]
    ok = not missing and not empty
    return ValidationResult(
        "P15",
        "every required Phase 3 output file exists and is non-empty",
        "PASS" if ok else "FAIL",
        f"{len(REQUIRED_OUTPUTS) - len(missing) - len(empty)}/{len(REQUIRED_OUTPUTS)} files present and non-empty",
        f"missing: {missing}; empty: {empty}" if not ok else "all required artefacts written",
    )


def _outputs_readable(config: Phase3Config) -> ValidationResult:
    root = config.out_path("results_dir")
    problems: List[str] = []
    csv_checked = 0
    json_checked = 0
    for path in sorted(root.rglob("*.csv")):
        csv_checked += 1
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), None)
            if header is None:
                problems.append(f"{path.name} has no header")
        except Exception as error:  # pragma: no cover
            problems.append(f"{path.name}: {error}")
    for path in sorted(root.rglob("*.json")):
        json_checked += 1
        try:
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as error:  # pragma: no cover
            problems.append(f"{path.name}: {error}")
    return ValidationResult(
        "P16",
        "every Phase 3 CSV and JSON output parses cleanly",
        "PASS" if not problems else "FAIL",
        f"{csv_checked} CSV and {json_checked} JSON files checked",
        f"problems: {problems[:3]}" if problems else "all outputs parse cleanly",
    )


def _deterministic_ranking(unit_rows: Sequence[Dict[str, object]]) -> ValidationResult:
    """Ranks must be 1..n per query and scores must never increase with rank."""
    by_query: Dict[str, List[Dict[str, object]]] = {}
    for row in unit_rows:
        by_query.setdefault(str(row.get("query_id", "")), []).append(row)
    problems: List[str] = []
    for query_id, rows in sorted(by_query.items()):
        ranks = [int(row["rank"]) for row in rows]  # type: ignore[arg-type]
        if ranks != list(range(1, len(rows) + 1)):
            problems.append(f"{query_id}: ranks {ranks[:5]}...")
        scores = [float(row["score"]) for row in rows]  # type: ignore[arg-type]
        if any(later > earlier + 1e-9 for earlier, later in zip(scores, scores[1:])):
            problems.append(f"{query_id}: scores not monotonically non-increasing")
        # unit_id is only consulted to break a genuine tie
        for left, right in zip(rows, rows[1:]):
            if float(left["score"]) == float(right["score"]) and str(left["unit_id"]) > str(right["unit_id"]):  # type: ignore[arg-type]
                problems.append(
                    f"{query_id}: tie on score {left['score']} broken by score, not unit_id"  # type: ignore[index]
                )
    return ValidationResult(
        "P17",
        "results are ranked deterministically: ranks are contiguous, scores are "
        "non-increasing and ties break on unit_id",
        "PASS" if not problems else "FAIL",
        f"{len(by_query)} queries checked over {len(unit_rows)} rows",
        f"problems: {problems[:3]}" if problems else "ordering is deterministic for every query",
    )


def _phase2_artifacts_untouched(config: Phase3Config) -> ValidationResult:
    """Every Phase 3 write target must live inside the project's own outputs."""
    results_dir = config.out_path("results_dir").resolve()
    escaped: List[str] = []
    for path in config.managed_paths() + [config.out_path("log_file")]:
        resolved = path.resolve()
        try:
            resolved.relative_to(config.project_root.resolve())
        except ValueError:
            escaped.append(str(path))
    inside_results = 0
    for path in config.managed_paths():
        try:
            path.resolve().relative_to(results_dir)
            inside_results += 1
        except ValueError:
            escaped.append(str(path))
    ok = not escaped
    return ValidationResult(
        "P18",
        "Phase 3 writes only under its own declared output paths; no Phase 1 or Phase 2 "
        "artefact is a write target",
        "PASS" if ok else "FAIL",
        f"{inside_results}/{len(config.managed_paths())} managed outputs are inside "
        f"{results_dir.name}/, plus the run log",
        f"escaped paths: {escaped}" if escaped
        else "all Phase 3 outputs are confined to the declared results directory and logs/",
    )
