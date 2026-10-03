"""The domain query set and its coverage report.

``config/phase3_config.yaml`` declares 15 domain queries. This module validates
that every query can actually be answered by the final pipeline's index and
records, per query, how the pipeline normalized it and which configured terms
were found. A query whose terms are all absent is reported as such rather than
being silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import Phase3Config, log_event
from .inverted_index import InvertedIndex
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec
from .query_parser import QuerySyntaxError, parse_query, query_type_of
from .retrieval import RetrievalEngine, SearchOutcome

#: Query types the engine supports, matching the configuration vocabulary.
QUERY_TYPES = ("keyword", "phrase", "boolean_and", "boolean_or", "boolean_not", "boolean_group")


@dataclass
class QueryRecord:
    """One configured query and the outcome of running it."""

    query_id: str
    query: str
    query_type: str
    description: str
    expected_domain: str
    configured_terms: List[str]
    excluded_terms: List[str]
    normalized_terms: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)
    unit_count: int = 0
    document_count: int = 0
    answered: bool = False
    syntax_ok: bool = True
    error: str = ""
    execution_time_ms: float = 0.0
    top_unit_ids: List[str] = field(default_factory=list)
    top_documents: List[str] = field(default_factory=list)
    outcome: Optional[SearchOutcome] = None

    def as_row(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type,
            "description": self.description,
            "expected_domain": self.expected_domain,
            "configured_terms": ";".join(self.configured_terms),
            "excluded_terms": ";".join(self.excluded_terms),
            "normalized_terms": ";".join(self.normalized_terms),
            "matched_terms": ";".join(self.matched_terms),
            "missing_terms": ";".join(self.missing_terms),
            "syntax_valid": self.syntax_ok,
            "answered": self.answered,
            "result_units": self.unit_count,
            "result_documents": self.document_count,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "top_unit_ids": ";".join(self.top_unit_ids),
            "top_documents": ";".join(self.top_documents),
            "error": self.error,
        }


def load_queries(config: Phase3Config) -> List[Dict[str, Any]]:
    """Read the query list from the configuration, keeping declaration order."""
    queries = config.get("queries", []) or []
    if not queries:
        raise ValueError("no queries configured; add entries to config/phase3_config.yaml")
    return [dict(entry) for entry in queries]


def _split(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def execute_query_set(
    config: Phase3Config,
    engine: RetrievalEngine,
    queries: Optional[Sequence[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[QueryRecord]:
    """Run every configured query against ``engine`` and collect the records."""
    spec: PipelineSpec = engine.spec
    runner: PipelineRunner = engine.runner
    entries = list(queries if queries is not None else load_queries(config))
    records: List[QueryRecord] = []

    for entry in entries:
        query = str(entry.get("query", "")).strip()
        declared_type = str(entry.get("type", "") or query_type_of(query))
        if declared_type not in QUERY_TYPES:
            raise ValueError(
                f"query {entry.get('id')} declares unsupported type '{declared_type}'; "
                f"allowed: {QUERY_TYPES}"
            )
        record = QueryRecord(
            query_id=str(entry.get("id", "")),
            query=query,
            query_type=declared_type,
            description=str(entry.get("description", "")),
            expected_domain=str(entry.get("expected_domain", "")),
            configured_terms=_split(entry.get("terms")),
            excluded_terms=_split(entry.get("excluded")),
        )
        record.normalized_terms = _normalized_terms(spec, runner, query, declared_type)

        try:
            outcome = engine.search(query, declared_type)
            record.outcome = outcome
            record.matched_terms = list(outcome.matched_terms)
            record.missing_terms = list(outcome.missing_terms)
            record.unit_count = outcome.unit_count
            record.document_count = outcome.document_count
            record.execution_time_ms = outcome.execution_time_ms
            record.answered = outcome.unit_count > 0
            record.top_unit_ids = [hit.unit_id for hit in outcome.unit_hits[:5]]
            record.top_documents = sorted({hit.document_id for hit in outcome.unit_hits})
        except QuerySyntaxError as error:
            record.syntax_ok = False
            record.error = str(error)
            if logger is not None:
                log_event(logger, "WARNING", "queries",
                          f"{record.query_id} rejected: {error}")
        records.append(record)

    if logger is not None:
        answered = sum(1 for r in records if r.answered)
        log_event(
            logger, "INFO", "queries",
            f"{answered}/{len(records)} domain queries answered on {spec.key}",
        )
    return records


def _normalized_terms(
    spec: PipelineSpec, runner: PipelineRunner, query: str, query_type: str
) -> List[str]:
    """The index terms the query reduces to under the final pipeline."""
    if query_type in ("keyword", "phrase"):
        return runner.normalize_query(spec, query)
    try:
        node = parse_query(query)
    except QuerySyntaxError:
        return []
    return _collect_terms(node)


def _collect_terms(node: Any) -> List[str]:
    """Depth-first collection of every leaf term in an AST, in query order."""
    from .query_parser import And, Not, Or, Term

    if isinstance(node, Term):
        return [node.text]
    if isinstance(node, Not):
        return _collect_terms(node.child)
    if isinstance(node, (And, Or)):
        return _collect_terms(node.left) + _collect_terms(node.right)
    return []


def coverage_summary(records: Sequence[QueryRecord]) -> Dict[str, Any]:
    """Aggregate answerability and coverage of the query set."""
    total = len(records)
    answered = sum(1 for record in records if record.answered)
    by_type: Dict[str, Dict[str, int]] = {}
    for record in records:
        bucket = by_type.setdefault(record.query_type, {"queries": 0, "answered": 0, "units": 0})
        bucket["queries"] += 1
        bucket["answered"] += 1 if record.answered else 0
        bucket["units"] += record.unit_count
    domains = sorted({record.expected_domain for record in records if record.expected_domain})
    return {
        "queries": total,
        "answered": answered,
        "answerability": round(answered / total, 4) if total else 0.0,
        "syntax_valid": sum(1 for record in records if record.syntax_ok),
        "by_type": dict(sorted(by_type.items())),
        "expected_domains": domains,
        "queries_without_results": [r.query for r in records if not r.answered],
        "queries_with_missing_terms": [r.query_id for r in records if r.missing_terms],
    }


def validate_query_vocabulary(
    engine: RetrievalEngine, records: Sequence[QueryRecord]
) -> List[str]:
    """Configured terms that do not resolve to any index term.

    Terms are normalized with the final pipeline first, so a query word that
    only survives as a lemma is reported as present, not missing.
    """
    index = engine.index
    runner = engine.runner
    missing: List[str] = []
    for record in records:
        for term in record.configured_terms + record.excluded_terms:
            if not term:
                continue
            normalized = runner.normalize_query(engine.spec, term)
            resolved = bool(normalized) and all(index.has_term(item) for item in normalized)
            if not resolved and term not in missing:
                missing.append(term)
    return missing
