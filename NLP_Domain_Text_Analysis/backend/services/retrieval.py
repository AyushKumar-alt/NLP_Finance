"""Retrieval service: a thin, cached bridge to the Phase 3 engine.

There is no search logic here. Every call goes through
``src.phase3.retrieval.RetrievalEngine``, which is the Phase 3 implementation
the CLI already uses, so the GUI and ``python -m src.phase3.run`` cannot
disagree about what a query returns.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any, Dict, List, Optional

from src.phase3.query_parser import (
    And,
    Not,
    Or,
    QuerySyntaxError,
    Term,
    describe_expression,
    query_type_of,
)
from src.phase3.query_parser import parse_query as parse_expression
from src.phase3.retrieval import RetrievalEngine

from backend.config.settings import Settings
from backend.services import artifacts

_LOCK = threading.Lock()
_RUNTIME = None
_RUNTIME_ERROR: Optional[str] = None
_ENGINE: Optional[RetrievalEngine] = None


class RetrievalUnavailable(RuntimeError):
    """Phase 1 or Phase 3 has not been run, so the API cannot answer a search."""


def runtime(settings: Settings, refresh: bool = False):
    """Load the Phase 3 runtime once and reuse it.

    Loading costs about 1.2 s (corpus + the 9.9 MB index), so it is deferred
    until the first request that actually needs it.
    """
    global _RUNTIME, _RUNTIME_ERROR, _ENGINE
    if refresh:
        with _LOCK:
            _RUNTIME, _RUNTIME_ERROR, _ENGINE = None, None, None
    if _RUNTIME is not None:
        return _RUNTIME
    with _LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        from src.phase4.runtime import load_phase3_runtime

        try:
            loaded = load_phase3_runtime(str(settings.path(settings.phase3_config)))
        except Exception as error:                    # noqa: BLE001 - reported to the client
            _RUNTIME_ERROR = f"{type(error).__name__}: {error}"
            raise RetrievalUnavailable(_RUNTIME_ERROR) from error
        _RUNTIME = loaded
        _RUNTIME_ERROR = None
        return _RUNTIME


def engine(settings: Settings, pipeline: Optional[str] = None) -> RetrievalEngine:
    """The retrieval engine bound to the Phase 3 selected pipeline."""
    global _ENGINE
    selected = pipeline or ""
    with _LOCK:
        if _ENGINE is not None and not selected:
            return _ENGINE
    built = runtime(settings).engine(selected or None)
    if not selected:
        with _LOCK:
            _ENGINE = built
    return built


def available(settings: Settings) -> Dict[str, Any]:
    """What /api/health reports; never raises."""
    try:
        loaded = runtime(settings)
    except RetrievalUnavailable as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "final_pipeline": loaded.final_pipeline,
        "final_pipeline_name": loaded.spec.name,
        "index_terms": loaded.index.term_count,
        "postings": loaded.index.posting_count,
        "documents": len(loaded.corpus.documents),
        "units": len(loaded.corpus.units),
        "indexed_units": len(loaded.corpus.selected_units()),
        "load_seconds": loaded.load_seconds,
        "pipelines": sorted(loaded.specs),
    }


def has_unquoted_ampersand(query: str) -> bool:
    """Check for standalone '&' outside quotes."""
    without_quotes = re.sub(r'"[^"]*"|\'[^\']*\'', '', query or '')
    return bool(re.search(r'(?:^|\s)&(?:\s|$)', without_quotes))


# ----------------------------------------------------------------------
CITATION_TEMPLATE = "{document_id} | p{page_number} | {section_number} | {unit_id}"


def _hit_payload(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Add the citation string and the human labels the UI shows."""
    unit_id = hit.get("unit_id", "")
    payload = dict(hit)
    payload["citation"] = CITATION_TEMPLATE.format(
        document_id=hit.get("document_id", ""),
        page_number=hit.get("page_number", ""),
        section_number=hit.get("section_number", ""),
        unit_id=unit_id,
    )
    payload["matched_terms_list"] = artifacts.split_list(hit.get("matched_terms"))
    payload["matched_term_frequency_map"] = {
        item.split(":", 1)[0]: artifacts.as_int(item.split(":", 1)[1], 0)
        for item in artifacts.split_list(hit.get("matched_term_frequency"))
        if ":" in item
    }
    payload["has_section_title"] = bool(hit.get("section_title")) and hit.get("section_title") != "UNKNOWN"

    # Exact Ranking Score breakdown based on verified Phase 3 formula:
    # score = 1.0 * matched_term_count + 2.0 * phrase_match + 0.25 * log10(1 + total_tf)
    matched_count = int(hit.get("matched_term_count", len(payload["matched_terms_list"])))
    term_component = round(1.0 * matched_count, 4)
    phrase_hit = bool(hit.get("phrase_match", False))
    phrase_component = 2.0 if phrase_hit else 0.0
    total_tf = sum(payload["matched_term_frequency_map"].values())
    tf_component = round(0.25 * math.log10(1 + max(0, total_tf)), 4)
    total_score = round(float(hit.get("score", term_component + phrase_component + tf_component)), 4)

    payload["score_breakdown"] = {
        "matched_term_count": matched_count,
        "term_component": term_component,
        "phrase_hit": phrase_hit,
        "phrase_component": phrase_component,
        "total_tf": total_tf,
        "tf_component": tf_component,
        "total_score": total_score,
    }
    return payload


def search(
    settings: Settings,
    query: str,
    query_type: Optional[str] = None,
    top_k: int = 10,
    pipeline: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one query through the Phase 3 engine and shape the response."""
    text = (query or "").strip()
    if not text:
        raise ValueError("query must not be empty")
    if has_unquoted_ampersand(text):
        raise ValueError("& is not a supported Boolean operator. Use AND instead.")
    resolved = query_type or query_type_of(text)
    try:
        active = engine(settings, pipeline)
    except KeyError as error:
        raise ValueError(f"unknown pipeline: {pipeline}") from error

    outcome = active.search(text, resolved)
    limit = max(1, min(int(top_k or 10), settings.max_top_k))
    hits = outcome.unit_hits[:limit]
    return {
        "query": text,
        "requested_query_type": query_type,
        "query_type": outcome.query_type,
        "pipeline": outcome.pipeline,
        "pipeline_name": active.spec.name,
        "retrieval_method": outcome.retrieval_method,
        "total_results": outcome.unit_count,
        "returned": len(hits),
        "top_k": limit,
        "result_documents": outcome.document_count,
        "matched_terms": outcome.matched_terms,
        "matched_terms_list": list(outcome.matched_terms),
        "missing_terms": outcome.missing_terms,
        "missing_terms_list": list(outcome.missing_terms),
        # A NOT term is not "missing": it is present in the index and was
        # deliberately removed. Reporting it separately lets the UI say which
        # term was excluded instead of leaving the user to guess.
        "excluded_terms": excluded_terms(text, outcome.query_type),
        "execution_time_ms": outcome.execution_time_ms,
        "max_results_available": int(
            active.config.get("retrieval.max_results_per_query", 50)
        ),
        "results": [_hit_payload(hit.to_row()) for hit in hits],
        "documents": outcome.document_hits()[:limit],
    }


def excluded_terms(query: str, query_type: str) -> List[str]:
    """The terms a ``NOT`` removed, read from the Phase 3 parse tree.

    Returns ``[]`` for any query without a ``NOT``. The walk reuses the Phase 3
    parser's own node types, so a term is never extracted by a second, divergent
    rule.
    """
    if query_type not in ("boolean_not", "boolean_group"):
        return []
    try:
        tree = parse_expression(query)
    except QuerySyntaxError:
        return []

    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Not):
            if isinstance(node.child, Term):
                found.append(node.child.text)
            else:
                walk(node.child)
            return
        if isinstance(node, (And, Or)):
            walk(node.left)
            walk(node.right)

    walk(tree)
    return found


def parse_query(query: str, query_type: Optional[str] = None) -> Dict[str, Any]:
    """Validate and normalize a query without running it. Powers the live builder.

    Boolean queries go through the Phase 3 parser, so the GUI reports the same
    error the engine would raise instead of inventing its own rules. The type is
    taken from :func:`query_type_of` rather than sniffed from the text: a
    truncated ``"GDP AND"`` still classifies as boolean, and a keyword path would
    have waved it through as a valid two-word search.
    """
    text = (query or "").strip()
    if not text:
        raise ValueError("query must not be empty")
    if has_unquoted_ampersand(text):
        return {
            "query": text,
            "query_type": query_type or query_type_of(text),
            "normalized": None,
            "valid": False,
            "error": "& is not a supported Boolean operator. Use AND instead.",
        }
    kind = query_type or query_type_of(text)
    if kind in ("keyword", "phrase"):
        return {
            "query": text,
            "query_type": kind,
            "normalized": text,
            "valid": True,
            "error": None,
        }
    try:
        tree = parse_expression(text)
    except QuerySyntaxError as error:
        return {
            "query": text,
            "query_type": kind,
            "normalized": None,
            "valid": False,
            "error": str(error),
        }
    return {
        "query": text,
        "query_type": kind,
        "normalized": describe_expression(tree),
        "valid": True,
        "error": None,
    }


def term_lookup(settings: Settings, term: str, limit: int = 50) -> Dict[str, Any]:
    """Resolve a surface term to index postings, through the Phase 3 index."""
    text = (term or "").strip()
    if not text:
        raise ValueError("term must not be empty")
    loaded = runtime(settings)
    active = engine(settings)
    normalized = loaded.runner.normalize_query(active.spec, text)
    matches = [t for t in normalized if loaded.index.has_term(t)]
    missing = [t for t in normalized if not loaded.index.has_term(t)]

    payload: Dict[str, Any] = {
        "term": text,
        "normalized_terms": normalized,
        "matched_terms": matches,
        "missing_terms": missing,
        "pipeline": active.spec.key,
        "searchable": bool(matches),
        "max_results_available": int(
            active.config.get("retrieval.max_results_per_query", 50)
        ),
    }
    if not matches:
        return payload

    postings = loaded.index.posting_list(matches[0])
    unit_ids = postings.unit_ids
    shown = unit_ids[: max(1, min(int(limit), settings.max_index_postings))]
    entries: List[Dict[str, Any]] = []
    for unit_id in shown:
        record = loaded.index.unit(unit_id)
        posting = postings.posting_for(unit_id)
        if record is None:
            continue
        unit = loaded.unit(unit_id)
        entries.append(
            {
                "unit_id": unit_id,
                "document_id": record.document_id,
                "source_id": record.source_id,
                "page_number": record.page_number,
                "section_id": record.section_id,
                "section_number": record.section_number,
                "section_title": record.section_title,
                "unit_type": record.unit_type,
                "term_frequency": posting.tf if posting else 0,
                "positions": list(posting.positions) if posting else [],
                "snippet": " ".join(unit.text.split())[:220] if unit is not None else "",
            }
        )
    document_frequency = loaded.index.document_frequency(matches[0])
    payload.update(
        {
            "term_key": matches[0],
            "posting_frequency": postings.posting_frequency,
            "total_frequency": postings.total_frequency,
            "document_frequency": document_frequency,
            "postings_total": len(unit_ids),
            "postings_returned": len(entries),
            "document_ids": sorted({e["document_id"] for e in entries}),
            "postings": entries,
        }
    )
    return payload


def term_statistics(settings: Settings, limit: int = 30, order: str = "posting_frequency") -> Dict[str, Any]:
    """The longest posting lists, read from the index's own statistics."""
    loaded = runtime(settings)
    rows = loaded.index.term_statistics()
    if order == "document_frequency":
        rows.sort(key=lambda r: -artifacts.as_int(r.get("document_frequency"), 0))
    else:
        rows.sort(key=lambda r: -artifacts.as_int(r.get("posting_frequency"), 0))
    return {
        "order": order,
        "available_orders": ["posting_frequency", "document_frequency"],
        "total": len(rows),
        "items": rows[: max(1, min(int(limit), 200))],
    }
