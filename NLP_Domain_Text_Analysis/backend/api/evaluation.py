"""Phase 4 evaluation routes, including the only write path in the API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.config.settings import get_settings
from backend.schemas.models import JudgmentRequest, JudgmentResponse
from backend.services import evaluation

router = APIRouter(prefix="/evaluation", tags=["phase4"])


@router.get("", summary="Full evaluation report: metrics, comparison and caveats")
def report() -> Dict[str, Any]:
    return evaluation.evaluation_report(get_settings())


@router.get("/summary", summary="final_summary.json as the GUI consumes it")
def final_summary() -> Dict[str, Any]:
    return evaluation.phase4_summary(get_settings())


@router.get("/results", summary="Per-query evaluation rows")
def results(
    query_id: Optional[str] = Query(default=None, pattern=r"^Q\d{2,4}$"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    return evaluation.evaluation_results(get_settings(), query_id=query_id, limit=limit, offset=offset)


@router.get("/aggregates", summary="Unit-level and document-level aggregates")
def aggregates() -> Dict[str, Any]:
    return {"aggregates": evaluation.evaluation_aggregates(get_settings())}


@router.get("/comparison", summary="Pipeline A against Pipeline B")
def comparison() -> Dict[str, Any]:
    return {
        "comparison": evaluation.pipeline_comparison(get_settings()),
        "selection": evaluation.final_pipeline(get_settings()),
    }


@router.get("/judgments", summary="Pooled relevance judgments")
def list_judgments(
    query_id: Optional[str] = Query(default=None, pattern=r"^Q\d{2,4}$"),
    relevance: Optional[bool] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    return evaluation.judgments(
        get_settings(), query_id=query_id, relevance=relevance, limit=limit, offset=offset
    )


@router.post(
    "/judgments",
    response_model=JudgmentResponse,
    status_code=201,
    summary="Record or replace one pooled relevance judgment",
)
def create_judgment(request: JudgmentRequest) -> Dict[str, Any]:
    """Writes to ``results/phase4/relevance_judgments.csv``.

    The judgment is a label about the unit's own text, supplied by a human. The
    endpoint deliberately accepts no score: an automatic label derived from a
    retrieval score would make the evaluation circular.
    """
    settings = get_settings()
    try:
        from src.phase4.relevance import RelevanceValidationError

        return evaluation.save_judgment(
            settings,
            query_id=request.query_id,
            unit_id=request.unit_id,
            relevance=request.relevance,
            annotator=request.annotator,
            notes=request.notes,
            query=request.query,
            rank=request.rank,
            document_id=request.document_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:                        # noqa: BLE001 - mapped below
        name = type(error).__name__
        if name == "RelevanceValidationError":
            raise HTTPException(status_code=422, detail=str(error)) from error
        raise HTTPException(
            status_code=500, detail=f"could not save the judgment: {name}: {error}"
        ) from error


@router.get("/validation", summary="Phase 4 validation rules")
def phase4_validation() -> Dict[str, Any]:
    rows = evaluation.phase4_validation(get_settings())
    passed = sum(1 for row in rows if str(row.get("status", "")).upper() == "PASS")
    return {
        "rules": rows,
        "total": len(rows),
        "passed": passed,
        "all_passed": bool(rows) and passed == len(rows),
    }
