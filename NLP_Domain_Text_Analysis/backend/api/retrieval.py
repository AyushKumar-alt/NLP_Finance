"""Retrieval routes.

Search is a thin pass-through to ``RetrievalEngine``. No scoring, tokenization
or ranking happens here; if the API and the Phase 3 CLI ever disagreed, it would
be a bug in this layer, which is why it does as little as possible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from backend.config.settings import ID_PATTERN, get_settings
from backend.schemas.models import IndexTermResponse, ParseQueryRequest, SearchRequest
from backend.services import evaluation, retrieval

router = APIRouter(tags=["phase3"])


@router.get("/queries", summary="The 15 registered queries and how each behaved")
def queries() -> List[Dict[str, Any]]:
    return evaluation.query_registry(get_settings())


@router.get("/queries/history", summary="Per-query retrieval summary")
def query_history() -> List[Dict[str, Any]]:
    return evaluation.retrieval_summary(get_settings())


@router.get("/queries/{query_id}/results", summary="Stored Phase 3 ranking for a query")
def stored_results(
    query_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    settings = get_settings()
    if not _is_query_id(query_id):
        raise HTTPException(status_code=400, detail=f"invalid query_id: {query_id!r}")
    return evaluation.retrieval_results(settings, query_id=query_id, limit=limit, offset=offset)


@router.post("/search", summary="Run a query through the Phase 3 retrieval engine")
def search(request: SearchRequest) -> Dict[str, Any]:
    settings = get_settings()
    try:
        return retrieval.search(
            settings,
            query=request.query,
            query_type=request.query_type,
            top_k=request.top_k,
            pipeline=request.pipeline,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except retrieval.RetrievalUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


@router.post("/search/parse", summary="Validate and normalize a query without running it")
def parse(request: ParseQueryRequest) -> Dict[str, Any]:
    try:
        return retrieval.parse_query(request.query, request.query_type)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/index/term", response_model=IndexTermResponse, summary="Postings for one term")
def term(
    term: str = Query(..., min_length=1, max_length=120, description="Surface term to look up"),
    limit: int = Query(default=25, ge=1, le=200),
) -> Dict[str, Any]:
    settings = get_settings()
    limit = min(limit, settings.max_index_postings)
    try:
        return retrieval.term_lookup(settings, term, limit=limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except retrieval.RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/index/terms", summary="Term statistics ordered by posting or document frequency")
def terms(
    limit: int = Query(default=30, ge=1, le=200),
    order: str = Query(default="posting_frequency", pattern=r"^(posting_frequency|document_frequency)$"),
) -> Dict[str, Any]:
    try:
        return retrieval.term_statistics(get_settings(), limit=limit, order=order)
    except retrieval.RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/index/statistics", summary="Inverted index statistics from Phase 3")
def index_statistics() -> Dict[str, Any]:
    return {
        "manifest": evaluation.index_manifest(get_settings()),
        "statistics": evaluation.index_statistics(get_settings()),
        "runtime": retrieval.available(get_settings()),
    }


@router.get("/pipelines", summary="Pipeline A and Pipeline B with Phase 3 measurements")
def pipelines() -> List[Dict[str, Any]]:
    return evaluation.pipelines(get_settings())


@router.get("/pipelines/selection", summary="The Phase 3 selection decision")
def selection() -> Dict[str, Any]:
    return evaluation.final_pipeline(get_settings())


@router.get("/pipelines/{pipeline}", summary="One pipeline in full")
def pipeline(pipeline: str) -> Dict[str, Any]:
    settings = get_settings()
    if not _matches(pipeline):
        raise HTTPException(status_code=400, detail=f"invalid pipeline id: {pipeline!r}")
    detail = evaluation.pipeline_detail(settings, pipeline)
    if not detail:
        raise HTTPException(status_code=404, detail=f"unknown pipeline {pipeline!r}")
    return detail


@router.get("/validation", summary="Phase 3 validation rules")
def phase3_validation() -> Dict[str, Any]:
    return {
        "rules": evaluation.phase3_validation(get_settings()),
        "summary": evaluation.phase3_summary(get_settings()).get("validation", {}),
    }


def _is_query_id(value: str) -> bool:
    return value.startswith("Q") and value[1:].isdigit() and 3 <= len(value) <= 5


def _matches(value: str) -> bool:
    import re

    return bool(re.match(ID_PATTERN, value))
