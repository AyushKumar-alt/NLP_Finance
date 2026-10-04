"""Pydantic request/response models.

Models exist for two reasons: a client gets a documented, typed contract, and a
malformed request is rejected with a 422 before it reaches a service. Anything
that names a document, unit, term or query is constrained to an identifier
pattern, so a request can never name a path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from backend.config.settings import ID_PATTERN

Identifier = Field(
    ...,
    pattern=ID_PATTERN,
    description="Project identifier: letters, digits and _ . : + - only.",
)

QueryType = Literal[
    "keyword", "phrase", "boolean_and", "boolean_or", "boolean_not", "boolean_group"
]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ----------------------------------------------------------------------
# envelope
# ----------------------------------------------------------------------
class HealthResponse(ApiModel):
    status: str = Field(..., description="'ok' when the retrieval runtime loads.")
    project: str
    phase: int
    generated_by: str
    retrieval_available: bool
    retrieval_detail: Dict[str, Any]
    phases: Dict[str, Any]
    uptime_seconds: float


class PageMeta(BaseModel):
    total: int
    offset: int
    limit: int
    returned: int
    has_more: bool


class Paged(ApiModel):
    items: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    returned: int
    has_more: bool


class ErrorResponse(ApiModel):
    """The single error shape every failure returns."""

    error: str
    detail: str
    hint: Optional[str] = None


# ----------------------------------------------------------------------
# system / corpus
# ----------------------------------------------------------------------
class StatisticsResponse(ApiModel):
    project: Dict[str, Any]
    corpus: Dict[str, Any]
    sources: List[Dict[str, Any]]
    index: Dict[str, Any]
    pipelines: Dict[str, Any]
    phase2: Dict[str, Any]
    phase3: Dict[str, Any]
    phase4: Dict[str, Any]
    validation: Dict[str, Any]
    sources_note: str


class DocumentListResponse(ApiModel):
    items: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    returned: int
    has_more: bool
    filters: Dict[str, Any]


class DocumentResponse(ApiModel):
    document_id: str
    document: Dict[str, Any]


class UnitResponse(ApiModel):
    unit_id: str
    unit: Dict[str, Any]


# ----------------------------------------------------------------------
# retrieval
# ----------------------------------------------------------------------
class SearchRequest(ApiModel):
    query: str = Field(..., min_length=1, max_length=500)
    query_type: Optional[QueryType] = Field(
        default=None,
        description="Inferred from the query text when omitted, using the Phase 3 rule.",
    )
    top_k: int = Field(default=10, ge=1, le=200)
    pipeline: Optional[str] = Field(default=None, pattern=r"^pipeline_[ab](_lemma|_stem)?$")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


class ParseQueryRequest(ApiModel):
    query: str = Field(..., min_length=1, max_length=500)
    query_type: Optional[QueryType] = None


class IndexTermResponse(ApiModel):
    term: str
    normalized_terms: List[str]
    matched_terms: List[str]
    missing_terms: List[str]
    pipeline: str
    searchable: bool
    max_results_available: int
    term_key: Optional[str] = None
    posting_frequency: Optional[int] = None
    total_frequency: Optional[int] = None
    document_frequency: Optional[int] = None
    postings_total: Optional[int] = None
    postings_returned: Optional[int] = None
    document_ids: List[str] = Field(default_factory=list)
    postings: List[Dict[str, Any]] = Field(default_factory=list)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
class JudgmentRequest(ApiModel):
    """One pooled relevance judgment.

    ``relevance`` is a label about the unit's own text. It is never derived from
    a retrieval score, and the API does not accept a score as input.
    """

    query_id: str = Field(..., pattern=r"^Q\d{2,4}$")
    unit_id: str = Identifier
    relevance: StrictBool
    annotator: str = Field(..., min_length=1, max_length=120)
    notes: str = Field(default="", max_length=2000)
    query: str = Field(default="", max_length=500)
    rank: Optional[int] = Field(default=None, ge=1)
    document_id: str = Field(default="", pattern=r"^$|^D\d{1,4}$")


class JudgmentResponse(ApiModel):
    saved: bool
    judgment: Dict[str, Any]
    judgment_count: int
    relevant_count: int
    validation: Dict[str, Any]
    message: str
