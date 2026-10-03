"""FastAPI application for the Phase 4 GUI backend.

The API is a read-mostly view over the artefacts Phases 1-3 already produced.
It loads those artefacts, runs queries through the existing Phase 3
``RetrievalEngine``, and exposes the Phase 4 evaluation. It does not reimplement
any NLP step and does not write anything except a relevance judgment.

Run it with::

    python -m uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import evaluation as evaluation_api
from backend.api import experiments as experiments_api
from backend.api import retrieval as retrieval_api
from backend.api import system as system_api
from backend.config.settings import get_settings
from backend.services.artifacts import ArtifactMissing
from backend.services.retrieval import RetrievalUnavailable

logger = logging.getLogger("backend")

DESCRIPTION = """
Read-mostly REST API over the artefacts of a four-phase NLP pipeline for Indian
financial and economic documents.

* **Phases 1-3 remain the source of truth.** Every number this API returns is read
  from a CSV or JSON those phases wrote. No NLP step is repeated here.
* **Search runs the real engine.** `POST /api/search` calls
  `src.phase3.retrieval.RetrievalEngine`, the same object the Phase 3 CLI uses.
* **One write path.** `POST /api/evaluation/judgments` appends a human relevance
  label to `results/phase4/relevance_judgments.csv` through
  `src.phase4.relevance.RelevanceStore`. It accepts no score, because a label
  derived from a retrieval score would make the evaluation circular.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if settings.is_wildcard_cors and not settings.allow_credentialed_cors:
        logger.warning(
            "CORS is set to '*'. That lets any page read this API from a browser. "
            "Set NLP_API_CORS_ORIGINS to the frontend origin instead."
        )
    logger.info("project root: %s", settings.project_root)
    logger.info(
        "serving %s / %s / %s / %s",
        settings.phase1_results, settings.phase2_results,
        settings.phase3_results, settings.phase4_results,
    )
    yield
    logger.info("shutting down")


settings = get_settings()

app = FastAPI(
    title="NLP Domain Text Analysis API",
    version="4.0.0",
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.allow_credentialed_cors,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    expose_headers=["X-Total-Count"],
)

app.include_router(system_api.router, prefix="/api")
app.include_router(experiments_api.router, prefix="/api")
app.include_router(retrieval_api.router, prefix="/api")
app.include_router(evaluation_api.router, prefix="/api")


# ----------------------------------------------------------------------
# Errors. Every failure leaves the API in the same shape, so the frontend has
# exactly one error path to implement.
# ----------------------------------------------------------------------
def _payload(error: str, detail: str, hint: str | None = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"error": error, "detail": detail}
    if hint:
        body["hint"] = hint
    return body


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    names = {400: "bad_request", 404: "not_found", 405: "method_not_allowed",
             422: "unprocessable", 503: "phase_unavailable"}
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(names.get(exc.status_code, "http_error"), detail),
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn Pydantic errors into a readable message plus the offending fields."""
    problems = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
        problems.append(f"{location or 'body'}: {item.get('msg')}")
    return JSONResponse(
        status_code=422,
        content=_payload(
            "unprocessable",
            "; ".join(problems) or "the request body did not validate",
            "identifiers accept letters, digits and _ . : + - only",
        ),
    )


@app.exception_handler(ArtifactMissing)
async def artifact_missing(request: Request, exc: ArtifactMissing) -> JSONResponse:
    """A declared artefact is absent: tell the client which phase to run."""
    missing = str(exc)
    phase = "phase 1"
    for name in ("phase4", "phase3", "phase2", "phase1"):
        if f"{name}" in missing.replace("\\", "/"):
            phase = name
            break
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_payload(
            "artefact_missing",
            f"required artefact is missing: {missing}",
            f"run the {phase} CLI to generate it",
        ),
    )


@app.exception_handler(RetrievalUnavailable)
async def retrieval_unavailable(request: Request, exc: RetrievalUnavailable) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_payload(
            "phase_unavailable",
            str(exc),
            "run `python -m src.phase3.run` so the inverted index exists",
        ),
    )


@app.exception_handler(ValueError)
async def value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content=_payload("bad_request", str(exc)))


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_payload(
            "internal_error",
            f"{type(exc).__name__}: {exc}",
            "this is a bug in the backend; the traceback is in the server log",
        ),
    )


@app.get("/", include_in_schema=False)
def root() -> Dict[str, Any]:
    return {
        "service": "NLP Domain Text Analysis API",
        "version": app.version,
        "docs": "/docs",
        "health": "/api/health",
        "frontend": "set NEXT_PUBLIC_API_BASE_URL to this server's origin",
    }
