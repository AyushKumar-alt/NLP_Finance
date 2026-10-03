"""Measure Pipeline A against Pipeline B and select the final pipeline.

Four criteria, declared in ``config/phase3_config.yaml`` *before* the
measurements were taken, each computed from the artifacts a pipeline actually
produced:

``financial_expression_preservation`` (0.35)
    Share of the typed financial expressions found in the raw text that are
    still a single index term after the pipeline (``7.4 per cent``,
    ``FY2025-26``, ``Rs 1.25 lakh crore``). Phase 2 measured that standard
    tokenization keeps only 12.85% of them intact, so this is the criterion
    that decides whether numbers stay searchable.

``domain_term_recall`` (0.25)
    Share of the domain vocabulary (the ``reference_terms`` list plus the
    ``domain_terms_in_phrase`` values exported by Phase 2) that resolves to an
    index term.

``variant_collapse_rate`` (0.25)
    Share of the configured variant groups whose surface forms all map onto one
    index term. A query for ``investment`` then also finds ``investments`` and
    ``investors``.

``query_answerability`` (0.15)
    Share of the configured domain queries that return at least one result on
    that pipeline's own index.

Processing time is measured and reported but excluded from the score: a slower
pipeline that answers more queries is still the better index.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.phase2.load_corpus import Unit

from .config import Phase3Config, log_event
from .index_builder import BuildResult
from .inverted_index import InvertedIndex
from .load_phase2 import Phase2Evidence
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec

#: Criterion -> configuration key in ``selection.criteria``.
CRITERION_KEYS = (
    ("financial_expression_preservation", "financial_expression_preservation"),
    ("domain_term_recall", "domain_term_recall"),
    ("variant_collapse_rate", "variant_collapse_rate"),
    ("query_answerability", "query_answerability"),
)

_EXPRESSION_PATTERNS = (
    ("percentage", re.compile(r"\d+(?:\.\d+)?\s*(?:%|per\s*cent|percent)", re.IGNORECASE)),
    ("fiscal_year", re.compile(r"FY\d{2,4}(?:-\d{2,4})?|fiscal\s+\d{4}", re.IGNORECASE)),
    ("currency", re.compile(r"(?:Rs\.?|INR|₹|US\$|\$)\s*\d[\d,.]*", re.IGNORECASE)),
)


@dataclass
class CriterionScore:
    """One criterion's raw measurement, normalized score and weighted score."""

    name: str
    weight: float
    numerator: int
    denominator: int
    normalized: float

    @property
    def weighted(self) -> float:
        return round(self.normalized * self.weight, 6)

    @property
    def rate(self) -> float:
        return round(self.normalized, 6)

    def as_detail(self) -> str:
        return f"{self.numerator}/{self.denominator} = {self.rate:.4f}"


@dataclass
class PipelineMeasurement:
    """Everything measured about one pipeline, including the score."""

    spec: PipelineSpec
    criteria: List[CriterionScore]
    index_stats: Dict[str, Any] = field(default_factory=dict)
    pipeline_stats: Dict[str, Any] = field(default_factory=dict)
    preserved_examples: List[str] = field(default_factory=list)
    lost_examples: List[str] = field(default_factory=list)
    missing_domain_terms: List[str] = field(default_factory=list)
    collapsed_groups: List[str] = field(default_factory=list)
    unanswered_queries: List[str] = field(default_factory=list)
    per_criterion_detail: Dict[str, str] = field(default_factory=dict)

    # ---------------- derived ----------------
    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def total_score(self) -> float:
        return round(sum(criterion.weighted for criterion in self.criteria), 6)

    def criterion(self, name: str) -> CriterionScore:
        return next(c for c in self.criteria if c.name == name)

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "pipeline": self.spec.key,
            "pipeline_name": self.spec.name,
            "tokenizer": self.spec.tokenizer,
            "stopword_strategy": self.spec.stopword_strategy,
            "morphology": self.spec.morphology_label,
            "term_representation": self.spec.term_kind,
            "order": " -> ".join(self.spec.order),
        }
        for criterion in self.criteria:
            row[criterion.name] = criterion.rate
            row[f"{criterion.name}_detail"] = criterion.as_detail()
            row[f"{criterion.name}_weight"] = criterion.weight
            row[f"{criterion.name}_weighted"] = criterion.weighted
        row.update(
            {
                "total_score": self.total_score,
                "index_terms": self.index_stats.get("index_terms", 0),
                "vocabulary_size": self.pipeline_stats.get("vocabulary_size", 0),
                "total_tokens": self.pipeline_stats.get("total_tokens", 0),
                "unique_tokens": self.pipeline_stats.get("unique_tokens", 0),
                "indexed_units": self.index_stats.get("indexed_units", 0),
                "total_postings": self.index_stats.get("total_postings", 0),
                "phrase_terms": self.index_stats.get("phrase_terms", 0),
                "preserved_examples": ";".join(self.preserved_examples),
                "lost_examples": ";".join(self.lost_examples),
                "missing_domain_terms": ";".join(self.missing_domain_terms),
                "collapsed_variant_groups": ";".join(self.collapsed_groups),
                "unanswered_queries": ";".join(self.unanswered_queries),
                "processing_time_seconds": self.pipeline_stats.get("processing_time_seconds", 0.0),
                "index_build_seconds": self.index_stats.get("index_build_seconds", 0.0),
            }
        )
        return row


@dataclass
class PipelineComparison:
    """The full A-vs-B comparison and the decision."""

    measurements: Dict[str, PipelineMeasurement]
    winner_key: str
    winner: PipelineMeasurement
    margin: float
    reason: str
    criteria_weights: Dict[str, float]
    tie_breakers: List[str]
    excluded_from_score: List[str]

    def loser(self) -> PipelineMeasurement:
        for key, measurement in self.measurements.items():
            if key != self.winner_key:
                return measurement
        raise KeyError("no runner-up pipeline")

    def rows(self) -> List[Dict[str, Any]]:
        return [self.measurements[key].as_row() for key in sorted(self.measurements)]


# ----------------------------------------------------------------------
# Individual measurements
# ----------------------------------------------------------------------
def find_financial_expressions(units: Sequence[Unit]) -> List[Tuple[str, str]]:
    """Every typed financial expression occurrence in the raw text.

    Returns ``(category, surface)`` pairs in corpus order, so both pipelines are
    scored against exactly the same occurrences.
    """
    found: List[Tuple[str, str]] = []
    for unit in units:
        text = unit.text
        for category, pattern in _EXPRESSION_PATTERNS:
            for match in pattern.finditer(text):
                surface = " ".join(match.group().split())
                if surface:
                    found.append((category, surface))
    return found


def measure_financial_expressions(
    expressions: Sequence[Tuple[str, str]],
    spec: PipelineSpec,
    runner: PipelineRunner,
    index: InvertedIndex,
) -> Tuple[int, int, List[str], List[str], Dict[str, Tuple[int, int]]]:
    """Count how many financial expressions survive as a single index term."""
    preserved = 0
    examples: List[str] = []
    lost: List[str] = []
    per_category: Dict[str, Tuple[int, int]] = {}

    for category, surface in expressions:
        tokens = runner.tokenize_text(spec, surface)
        merged = runner.protect_expressions(spec, surface, tokens)
        indexable = [runner.term_for(spec, token) for token in merged]
        term = indexable[0] if len(indexable) == 1 else ""
        kept = bool(term) and index.has_term(term)
        total, hits = per_category.get(category, (0, 0))
        per_category[category] = (total + 1, hits + (1 if kept else 0))
        if kept:
            preserved += 1
            if len(examples) < 8:
                examples.append(f"{surface} -> '{term}'")
        else:
            if len(lost) < 8:
                lost.append(surface)

    return preserved, len(expressions), examples, lost, {
        category: counts for category, counts in sorted(per_category.items())
    }


def load_domain_terms(config: Phase3Config, evidence: Optional[Phase2Evidence] = None) -> List[str]:
    """Domain vocabulary: the configured reference terms plus Phase 2's own.

    Phase 2 exported ``domain_terms_in_phrase`` for every phrase it considered
    domain relevant; those terms come from the corpus rather than from this
    file, which is what makes the recall figure meaningful.
    """
    terms: List[str] = []
    seen: set = set()

    def add(value: str) -> None:
        cleaned = " ".join(str(value).split()).strip()
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            terms.append(cleaned)

    for term in config.get("financial_expressions.reference_terms", []) or []:
        add(str(term))
    for term in config.get("domain_preservation.config_terms", []) or []:
        add(str(term))
    if evidence is not None:
        for row in evidence.domain_phrases:
            for value in str(row.get("domain_terms_in_phrase", "")).split(";"):
                add(value)
    return terms


def measure_domain_terms(
    domain_terms: Sequence[str],
    spec: PipelineSpec,
    runner: PipelineRunner,
    index: InvertedIndex,
) -> Tuple[int, int, List[str]]:
    """How many domain terms resolve to an index term."""
    found = 0
    missing: List[str] = []
    for term in domain_terms:
        normalized = runner.normalize_query(spec, term)
        hit = bool(normalized) and all(index.has_term(item) for item in normalized)
        if hit:
            found += 1
        elif len(missing) < 15:
            missing.append(term)
    return found, len(domain_terms), missing


def measure_variant_collapse(
    spec: PipelineSpec,
    runner: PipelineRunner,
    index: InvertedIndex,
    config: Phase3Config,
) -> Tuple[int, int, List[str], List[Dict[str, Any]]]:
    """How many variant groups collapse onto a single indexed term."""
    groups = config.get("variant_groups", []) or []
    collapsed = 0
    collapsed_names: List[str] = []
    detail: List[Dict[str, Any]] = []
    for group in groups:
        surfaces = [str(item) for item in group if str(item).strip()]
        if len(surfaces) < 2:
            continue
        mapping: Dict[str, List[str]] = {}
        for surface in surfaces:
            normalized = runner.normalize_query(spec, surface)
            term = normalized[0] if len(normalized) == 1 else (normalized[0] if normalized else "")
            mapping.setdefault(term, []).append(surface)
        target = max(
            (term for term in mapping if term and index.has_term(term)),
            key=lambda term: len(mapping[term]),
            default="",
        )
        group_terms = {term for term in mapping if term and index.has_term(term)}
        is_collapsed = bool(target) and len(mapping[target]) == len(surfaces) and len(group_terms) == 1
        if is_collapsed:
            collapsed += 1
            collapsed_names.append("~".join(surfaces) + f" -> '{target}'")
        detail.append(
            {
                "group": " ~ ".join(surfaces),
                "resolved_term": target,
                "in_index": bool(target) and index.has_term(target),
                "collapsed": is_collapsed,
                "surface_to_term": "; ".join(
                    f"{surface}->{term or '<empty>'}"
                    for term, surface in sorted(mapping.items())
                ),
            }
        )
    return collapsed, len(groups), collapsed_names, detail


def measure_query_answerability(
    outcomes: Sequence[Any],
) -> Tuple[int, int, List[str]]:
    """Share of queries returning at least one result."""
    answered = 0
    unanswered: List[str] = []
    for outcome in outcomes:
        if outcome.unit_count > 0:
            answered += 1
        else:
            unanswered.append(outcome.query)
    return answered, len(outcomes), unanswered


# ----------------------------------------------------------------------
# Comparison driver
# ----------------------------------------------------------------------
def compare_pipelines(
    config: Phase3Config,
    runner: PipelineRunner,
    builds: Dict[str, BuildResult],
    pipeline_stats: Dict[str, Dict[str, Any]],
    query_outcomes: Dict[str, Sequence[Any]],
    units: Sequence[Unit],
    evidence: Optional[Phase2Evidence] = None,
    logger: Optional[logging.Logger] = None,
) -> PipelineComparison:
    """Score every pipeline on the four criteria and pick the final one."""
    weights = {
        key: float(value) for key, value in (config.section("selection").get("criteria", {}) or {}).items()
    }
    if not weights:
        raise ValueError("selection.criteria is empty in the Phase 3 configuration")
    tie_breakers = [str(item) for item in (config.get("selection.tie_breakers", []) or [])]
    excluded = [str(item) for item in (config.get("selection.exclude_from_score", []) or [])]

    expressions = find_financial_expressions(units)
    domain_terms = load_domain_terms(config, evidence)
    measurements: Dict[str, PipelineMeasurement] = {}

    for key in sorted(builds):
        build = builds[key]
        spec = build.spec
        index = build.index

        preserved, expression_total, examples, lost, _by_category = measure_financial_expressions(
            expressions, spec, runner, index
        )
        domain_found, domain_total, missing_terms = measure_domain_terms(
            domain_terms, spec, runner, index
        )
        collapsed, group_total, collapsed_names, _detail = measure_variant_collapse(
            spec, runner, index, config
        )
        answered, query_total, unanswered = measure_query_answerability(
            query_outcomes.get(key, [])
        )

        criteria = [
            CriterionScore(
                name=label,
                weight=float(weights.get(key_name, 0.0)),
                numerator=numerator,
                denominator=denominator,
                normalized=(numerator / denominator) if denominator else 0.0,
            )
            for (label, key_name), (numerator, denominator) in (
                (("financial_expression_preservation", "financial_expression_preservation"),
                 (preserved, expression_total)),
                (("domain_term_recall", "domain_term_recall"), (domain_found, domain_total)),
                (("variant_collapse_rate", "variant_collapse_rate"), (collapsed, group_total)),
                (("query_answerability", "query_answerability"), (answered, query_total)),
            )
        ]

        measurement = PipelineMeasurement(
            spec=spec,
            criteria=criteria,
            index_stats=dict(build.stats),
            pipeline_stats=dict(pipeline_stats.get(key, {})),
            preserved_examples=examples,
            lost_examples=lost,
            missing_domain_terms=missing_terms,
            collapsed_groups=collapsed_names,
            unanswered_queries=unanswered,
        )
        measurements[key] = measurement
        if logger is not None:
            log_event(
                logger, "INFO", "comparison",
                f"{spec.key}: score={measurement.total_score} "
                + " | ".join(f"{c.name}={c.as_detail()} (w={c.weight})" for c in criteria),
            )

    winner_key, margin, reason = _decide(measurements, tie_breakers, logger)
    return PipelineComparison(
        measurements=measurements,
        winner_key=winner_key,
        winner=measurements[winner_key],
        margin=margin,
        reason=reason,
        criteria_weights=weights,
        tie_breakers=tie_breakers,
        excluded_from_score=excluded,
    )


def _decide(
    measurements: Dict[str, PipelineMeasurement],
    tie_breakers: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, float, str]:
    """Highest weighted score wins; declared tie-breakers resolve exact ties."""
    ordered = sorted(
        measurements.items(),
        key=lambda item: (-item[1].total_score, item[1].name),
    )
    best_key, best = ordered[0]
    runner_key, runner_up = ordered[1] if len(ordered) > 1 else (best_key, best)
    margin = round(best.total_score - runner_up.total_score, 6)

    if margin != 0.0:
        reason = (
            f"{best.name} wins on the weighted score "
            f"({best.total_score:.4f} vs {runner_up.total_score:.4f}, margin {margin:+.4f}). "
            f"Decomposition: "
            + "; ".join(
                f"{c.name} {c.as_detail()} x{c.weight} = {c.weighted:.4f}" for c in best.criteria
            )
            + "."
        )
        contributors = [
            (
                criterion,
                best.criterion(criterion.name).rate - runner_up.criterion(criterion.name).rate,
            )
            for criterion in best.criteria
            if best.criterion(criterion.name).rate != runner_up.criterion(criterion.name).rate
        ]
        if contributors:
            deciding = ", ".join(
                f"{criterion.name} ({best.criterion(criterion.name).as_detail()} vs "
                f"{runner_up.criterion(criterion.name).as_detail()}, "
                f"{delta:+.4f} x{criterion.weight} = {criterion.weight * delta:+.4f})"
                for criterion, delta in contributors
            )
            reason += (
                f" The two pipelines are equal on "
                f"{len(best.criteria) - len(contributors)} of {len(best.criteria)} criteria; "
                f"the margin comes from {deciding}."
            )
        else:  # pragma: no cover - a zero margin cannot reach this branch
            reason += " All criteria are equal, so the margin is zero."
    else:
        parts: List[str] = []
        for rule in tie_breakers:
            if "vocabulary" in rule:
                key = ("index_terms", lambda m: int(m.index_stats.get("index_terms", 0)))
            elif "per posting" in rule:
                key = ("postings", lambda m: int(m.index_stats.get("total_postings", 0)))
            else:
                key = (rule, lambda m: m.name)
            left = key[1](best)
            right = key[1](runner_up)
            if left != right:
                parts.append(f"tie broken by '{rule}': {left} vs {right}")
                winner = best if left < right else runner_up
                return (
                    winner.spec.key,
                    0.0,
                    f"Scores are equal ({best.total_score:.4f}); " + "; ".join(parts),
                )
        reason = (
            f"Scores are equal ({best.total_score:.4f}) and no tie-breaker separated the "
            f"pipelines; the lexicographically first pipeline name is kept."
        )
    if logger is not None:
        log_event(logger, "INFO", "selection", reason)
    return best_key, margin, reason
