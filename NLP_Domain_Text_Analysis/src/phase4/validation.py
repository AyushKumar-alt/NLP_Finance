"""Phase 4 validation rules.

Each rule is a check that either passes or fails with evidence. A rule is never
given the benefit of the doubt: if the evidence cannot be produced, the rule
fails and says why.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.core.utils import read_csv_rows

from .config import Phase4Config
from .evaluator import EvaluationRun
from .relevance import RelevanceStore

RuleResult = Dict[str, str]
Check = Callable[[], RuleResult]

#: Phase 1-3 artefacts Phase 4 depends on. Hashing them proves the run left
#: them alone; a changed hash would mean a previous phase was rerun or edited.
GUARDED_INPUTS = (
    ("data/corpus/structured/corpus.jsonl", "Phase 1 structured corpus"),
    ("data/corpus/metadata/document_registry.csv", "Phase 1 document registry"),
    ("results/phase1/validation_report.json", "Phase 1 validation report"),
    ("results/phase2/phase2_summary.json", "Phase 2 experiment summary"),
    ("results/phase2/phase2_validation_report.csv", "Phase 2 validation report"),
    ("results/phase3/phase3_summary.json", "Phase 3 run summary"),
    ("results/phase3/index_statistics.csv", "Phase 3 index statistics"),
    ("results/phase3/pipeline_comparison.csv", "Phase 3 pipeline comparison"),
    ("results/phase3/inverted_index.json", "Phase 3 inverted index"),
)


def _rule(rule_id: str, rule: str, ok: bool, detail: str, evidence: str) -> RuleResult:
    return {
        "rule_id": rule_id,
        "rule": rule,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
        "evidence": evidence,
    }


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_checks(
    config: Phase4Config,
    store: RelevanceStore,
    runs: Dict[str, EvaluationRun],
    selected: str,
    baselines: Dict[str, str],
    output_files: Sequence[Path],
) -> List[Check]:
    """Assemble the rule set. Each check runs independently and records why."""

    checks: List[Check] = []

    def phase1_3_readable() -> RuleResult:
        missing = []
        present = 0
        for relative, label in GUARDED_INPUTS:
            path = config.project_root / relative
            if path.is_file() and path.stat().st_size > 0:
                present += 1
            else:
                missing.append(f"{label} ({relative})")
        return _rule(
            "V01",
            "every Phase 1, Phase 2 and Phase 3 input Phase 4 depends on is present and readable",
            not missing,
            f"{present}/{len(GUARDED_INPUTS)} input artefact(s) readable"
            + ("" if not missing else f"; missing: {', '.join(missing)}"),
            "data/corpus/**, results/phase1/**, results/phase2/**, results/phase3/**",
        )

    checks.append(phase1_3_readable)

    def inputs_unmodified() -> RuleResult:
        changed = []
        for relative, label in GUARDED_INPUTS:
            expected = baselines.get(relative)
            path = config.project_root / relative
            if not expected or not path.is_file():
                continue
            if _digest(path) != expected:
                changed.append(label)
        return _rule(
            "V02",
            "Phase 4 did not modify any Phase 1, Phase 2 or Phase 3 artefact",
            not changed,
            "all guarded input hashes unchanged since the start of this run"
            if not changed else f"changed: {', '.join(changed)}",
            "sha256 of 9 guarded inputs compared before and after the run",
        )

    checks.append(inputs_unmodified)

    def judgments_valid() -> RuleResult:
        ok = not store.problems
        return _rule(
            "V03",
            "every relevance label is 0 or 1 and every pair has a query id and a unit id",
            ok,
            f"{len(store)} judgment(s) loaded, {len(store.problems)} problem(s)"
            + ("" if not store.problems else f": {'; '.join(store.problems[:3])}"),
            "results/phase4/relevance_judgments.csv",
        )

    checks.append(judgments_valid)

    def judgments_not_derived() -> RuleResult:
        """A label must carry a decision note, i.e. it is not a computed score."""
        empty = [j for j in store.judgments.values() if not j.notes]
        return _rule(
            "V04",
            "every relevance label records a human decision note (labels are judgments, not scores)",
            not empty,
            f"{len(store) - len(empty)}/{len(store)} label(s) carry a decision note"
            + ("" if not empty else f"; {len(empty)} without a note"),
            "relevance_judgments.csv notes column",
        )

    checks.append(judgments_not_derived)

    def pool_covers_queries() -> RuleResult:
        judged = set(store.query_ids())
        expected = {
            query.query_id for run in runs.values() for query in run.queries
        }
        missing = sorted(expected - judged)
        return _rule(
            "V05",
            "every evaluated query has at least one relevance judgment",
            bool(expected) and not missing,
            f"{len(judged & expected)}/{len(expected)} evaluated query/queries judged"
            + ("" if not missing else f"; unjudged: {', '.join(missing)}"),
            "relevance_judgments.csv query_id column",
        )

    checks.append(pool_covers_queries)

    def metrics_defined() -> RuleResult:
        run = runs.get(selected)
        if run is None:
            return _rule(
                "V06", "precision, recall and F1 are computed for the selected pipeline",
                False, "the selected pipeline was not evaluated", "config/phase4_config.yaml",
            )
        evaluated = run.evaluated_queries()
        defined = [q for q in evaluated if q.metrics["f1"].value is not None]
        return _rule(
            "V06",
            "precision, recall and F1 are computed for the selected pipeline",
            len(evaluated) > 0 and len(defined) == len(evaluated),
            f"{len(defined)}/{len(evaluated)} evaluated query/queries have all three metrics defined",
            "results/phase4/evaluation_results.csv",
        )

    checks.append(metrics_defined)

    def at_k_defined() -> RuleResult:
        run = runs.get(selected)
        if run is None:
            return _rule("V07", "P@K and R@K are computed at every configured cut-off",
                         False, "the selected pipeline was not evaluated", "config/phase4_config.yaml")
        problems = []
        for k, block in sorted(run.at_k.items()):
            if block.get("macro_precision_at_k") is None and block.get("macro_recall_at_k") is None:
                problems.append(f"K={k}")
        return _rule(
            "V07", "P@K and R@K are computed at every configured cut-off",
            not problems,
            f"cut-offs evaluated: {sorted(run.at_k)}"
            + ("" if not problems else f"; undefined at: {', '.join(problems)}"),
            "results/phase4/evaluation_summary.csv",
        )

    checks.append(at_k_defined)

    def zero_division_explicit() -> RuleResult:
        """A metric with an empty denominator must say so, not report 0."""
        run = runs.get(selected)
        if run is None:
            return _rule("V08", "an undefined metric is reported as undefined, never as zero",
                         False, "the selected pipeline was not evaluated", "results/phase4")
        undocumented = []
        for query in run.evaluated_queries():
            for name, metric in query.metrics.items():
                if metric.value is None and not metric.reason:
                    undocumented.append(f"{query.query_id}:{name}")
        return _rule(
            "V08", "an undefined metric is reported as undefined, never as zero",
            not undocumented,
            "every undefined metric carries a reason"
            if not undocumented else f"{len(undocumented)} undocumented: {', '.join(undocumented[:5])}",
            "src/phase4/metrics.py Metric.reason",
        )

    checks.append(zero_division_explicit)

    def search_returns_real_units() -> RuleResult:
        run = runs.get(selected)
        if run is None:
            return _rule("V09", "every evaluated query returned real, traceable content units",
                         False, "the selected pipeline was not evaluated", "results/phase4")
        with_units = [q for q in run.evaluated_queries() if q.ranking]
        # Phase 1 numbers content units D<nn>_<kind><nnn>... where <kind> is the
        # unit type: P for a page, T for a table, BOX for a pulled-out box, etc.
        pattern = re.compile(r"^D\d{2}_(P\d{3}|T\d{3}|BOX\d{3}|FIG\d{3}|FN\d{3})")
        untraceable = [
            unit_id
            for q in with_units
            for unit_id in q.ranking
            if not pattern.match(unit_id)
        ]
        return _rule(
            "V09", "every evaluated query returned real, traceable content units",
            bool(with_units) and not untraceable,
            f"{len(with_units)}/{len(run.evaluated_queries())} query/queries returned units; "
            f"{sum(len(q.ranking) for q in with_units)} returned unit id(s) all carry a "
            f"document and page/table/box locator"
            if not untraceable
            else f"{len(untraceable)} unit id(s) are not page-qualified, e.g. {untraceable[:3]}",
            "results/phase4/evaluation_results.csv retrieved_units",
        )

    checks.append(search_returns_real_units)

    def all_query_types_covered() -> RuleResult:
        run = runs.get(selected)
        if run is None:
            return _rule("V10", "keyword, phrase, AND, OR, NOT and grouped queries are all evaluated",
                         False, "the selected pipeline was not evaluated", "results/phase4")
        found = {q.query_type for q in run.evaluated_queries()}
        required = {"keyword", "phrase", "boolean_and", "boolean_or", "boolean_not", "boolean_group"}
        missing = sorted(required - found)
        return _rule(
            "V10", "keyword, phrase, AND, OR, NOT and grouped queries are all evaluated",
            not missing,
            f"query types evaluated: {sorted(found)}"
            + ("" if not missing else f"; missing: {', '.join(missing)}"),
            "results/phase3/query_registry.csv",
        )

    checks.append(all_query_types_covered)

    def outputs_exist() -> RuleResult:
        missing = [p.name for p in output_files if not p.is_file()]
        empty = [p.name for p in output_files if p.is_file() and p.stat().st_size == 0]
        return _rule(
            "V11", "every declared Phase 4 output artefact was written and is non-empty",
            not missing and not empty,
            f"{len(output_files)} artefact(s) declared"
            + ("" if not missing else f"; missing: {', '.join(missing)}")
            + ("" if not empty else f"; empty: {', '.join(empty)}"),
            "results/phase4/**",
        )

    checks.append(outputs_exist)

    def no_fabricated_values() -> RuleResult:
        """A metric cell must be empty, or equal to a value Python computed."""
        run = runs.get(selected)
        if run is None:
            return _rule("V12", "every metric in the report equals a computed value",
                         False, "the selected pipeline was not evaluated", "results/phase4")
        problems = []
        for query in run.evaluated_queries():
            for name, metric in query.metrics.items():
                value = metric.value
                if value is None:
                    continue
                if not (0.0 <= value <= 1.0) and not name.startswith("average_precision"):
                    problems.append(f"{query.query_id}:{name}={value}")
        return _rule(
            "V12", "every metric in the report equals a computed value in [0, 1]",
            not problems,
            "all rate metrics are within [0, 1]"
            if not problems else f"out of range: {', '.join(problems[:5])}",
            "results/phase4/evaluation_results.csv",
        )

    checks.append(no_fabricated_values)

    def judgments_persisted() -> RuleResult:
        path = config.relevance_judgments_path
        exists = path.is_file()
        rows: List[Dict[str, str]] = read_csv_rows(path) if exists else []
        return _rule(
            "V13", "relevance judgments persist to results/phase4/relevance_judgments.csv",
            exists and bool(rows) and len(rows) == len(store),
            f"{len(rows)} row(s) on disk, {len(store)} in memory",
            "results/phase4/relevance_judgments.csv",
        )

    checks.append(judgments_persisted)

    def phase3_selection_respected() -> RuleResult:
        run = runs.get(selected)
        return _rule(
            "V14", "the evaluation uses the pipeline Phase 3 actually selected",
            run is not None,
            f"selected pipeline '{selected}' evaluated"
            if run is not None else f"selected pipeline '{selected}' was not evaluated",
            "results/phase3/final_pipeline.json",
        )

    checks.append(phase3_selection_respected)

    def csv_schema_stable() -> RuleResult:
        from .reporting import EVALUATION_FIELDS

        path = config.out_path("evaluation_results_csv")
        if not path.is_file():
            return _rule("V15", "evaluation_results.csv keeps a stable column schema",
                         False, "evaluation_results.csv was not written", str(path))
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next((line for line in handle if line.strip()), "").strip()
        expected = ",".join(EVALUATION_FIELDS)
        return _rule(
            "V15", "evaluation_results.csv keeps a stable column schema",
            header == expected,
            f"{len(header.split(','))} columns" if header else "empty header",
            ",".join(EVALUATION_FIELDS[:6]) + ",...",
        )

    checks.append(csv_schema_stable)

    def phase4_readonly_outside_results() -> RuleResult:
        results_dir = config.results_dir.resolve()
        strays = []
        for path in output_files:
            try:
                path.resolve().relative_to(results_dir)
            except ValueError:
                strays.append(path.name)
        return _rule(
            "V16", "Phase 4 writes only inside results/phase4",
            not strays,
            f"{len(output_files)} declared output(s) all inside {results_dir.name}/"
            if not strays else f"outside: {', '.join(strays)}",
            "config/phase4_config.yaml output section",
        )

    checks.append(phase4_readonly_outside_results)

    return checks


def run_checks(checks: Sequence[Check]) -> List[RuleResult]:
    return [check() for check in checks]


def summarise(results: Sequence[RuleResult]) -> Dict[str, int]:
    passed = sum(1 for r in results if r["status"] == "PASS")
    return {"passed": passed, "failed": len(results) - passed, "total": len(results)}


def snapshot_inputs(config: Phase4Config) -> Dict[str, str]:
    """sha256 of the guarded inputs, taken before the evaluation runs."""
    digests: Dict[str, str] = {}
    for relative, _ in GUARDED_INPUTS:
        path = config.project_root / relative
        if path.is_file():
            digests[relative] = _digest(path)
    return digests
