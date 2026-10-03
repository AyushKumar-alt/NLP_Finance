"""System and corpus routes."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.config.settings import get_settings
from backend.schemas.models import DocumentResponse, HealthResponse
from backend.services import artifacts, corpus, retrieval, statistics

router = APIRouter(tags=["system"])

_STARTED = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service and artefact health")
def health() -> Dict[str, Any]:
    """Reports whether the project has actually been run.

    The retrieval check loads the Phase 3 runtime, which is what tells a client
    whether /api/search will work. It never raises: an unavailable phase is
    reported as ``retrieval_available: false`` with the reason.
    """
    settings = get_settings()
    detail = retrieval.available(settings)
    return {
        "status": "ok" if detail.get("available") else "degraded",
        "project": "NLP_Domain_Text_Analysis",
        "phase": 4,
        "generated_by": "FastAPI backend over Phases 1-3 artefacts",
        "retrieval_available": bool(detail.get("available")),
        "retrieval_detail": detail,
        "phases": statistics.project(settings)["phases_available"],
        "uptime_seconds": round(time.time() - _STARTED, 3),
    }


@router.get("/statistics", summary="Aggregated statistics across all four phases")
def all_statistics() -> Dict[str, Any]:
    return statistics.statistics(get_settings())


@router.get("/project", summary="Project identity and which phases have run")
def project() -> Dict[str, Any]:
    return statistics.project(get_settings())


@router.get("/validation", summary="Validation rules from every phase")
def validation() -> Dict[str, Any]:
    return statistics.validation_overview(get_settings())


@router.get("/reports", summary="Generated phase documentation and figures")
def reports() -> Dict[str, Any]:
    return statistics.figures(get_settings())


# ----------------------------------------------------------------------
@router.get("/sources", summary="Source registry with observed counts")
def sources() -> List[Dict[str, Any]]:
    return corpus.source_summary(get_settings())


@router.get("/documents", summary="Document registry joined with Phase 1 statistics")
def documents(
    source_id: Optional[str] = Query(default=None, pattern=r"^SRC\d{1,4}$"),
    document_type: Optional[str] = Query(default=None, max_length=60),
    ingestion_status: Optional[str] = Query(default=None, max_length=60),
    search: Optional[str] = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    settings = get_settings()
    limit = min(limit, settings.max_page_size)
    rows = corpus.document_summary(settings)
    if source_id:
        rows = [r for r in rows if r.get("source_id") == source_id]
    if document_type:
        rows = [r for r in rows if r.get("document_type") == document_type]
    if ingestion_status:
        rows = [r for r in rows if r.get("ingestion_status") == ingestion_status]
    if search:
        needle = search.strip().lower()
        rows = [
            r
            for r in rows
            if needle in str(r.get("title", "")).lower()
            or needle in str(r.get("filename", "")).lower()
            or needle in str(r.get("document_id", "")).lower()
            or needle in str(r.get("publisher", "")).lower()
        ]
    window = artifacts.page(rows, limit, offset)
    window["filters"] = {
        "source_id": source_id,
        "document_type": document_type,
        "ingestion_status": ingestion_status,
        "search": search,
    }
    return window


@router.get("/documents/{document_id}", response_model=DocumentResponse, summary="One document")
def document(document_id: str) -> Dict[str, Any]:
    settings = get_settings()
    try:
        detail = corpus.document_detail(settings, document_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not detail:
        raise HTTPException(
            status_code=404, detail=f"document {document_id!r} is not in the registry"
        )
    return {"document_id": document_id, "document": detail}


@router.get("/units/{unit_id}", summary="One content unit with its provenance")
def unit(unit_id: str) -> Dict[str, Any]:
    settings = get_settings()
    try:
        record = corpus.unit(settings, unit_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if record is None:
        raise HTTPException(status_code=404, detail=f"unit {unit_id!r} does not exist")
    return {"unit_id": unit_id, "unit": record}
