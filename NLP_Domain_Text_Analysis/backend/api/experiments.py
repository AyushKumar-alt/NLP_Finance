"""Phase 2 experiment routes: tokenization through BPE."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.config.settings import get_settings
from backend.services import artifacts, experiments

router = APIRouter(prefix="/experiments", tags=["phase2"])


@router.get("/summary", summary="Phase 2 master summary")
def summary() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "summary": experiments.summary(settings),
        "master_comparison": experiments.master_comparison(settings),
        "summary_table": experiments.summary_table(settings),
        "validation": experiments.validation(settings),
    }


@router.get("/tokenization", summary="Tokenizer comparison, custom rules, date and number handling")
def tokenization() -> Dict[str, Any]:
    return experiments.tokenization(get_settings())


@router.get("/preprocessing", summary="Cleaning, stopwords and stopword/morphology order")
def preprocessing() -> Dict[str, Any]:
    return experiments.preprocessing(get_settings())


@router.get("/stemming", summary="Stemmer comparison and families")
def stemming() -> Dict[str, Any]:
    return experiments.stemming(get_settings())


@router.get("/lemmatization", summary="Lemmatizer comparison")
def lemmatization() -> Dict[str, Any]:
    return experiments.lemmatization(get_settings())


@router.get("/pos", summary="POS tagging, custom dictionary and ML comparison")
def pos() -> Dict[str, Any]:
    return experiments.pos(get_settings())


@router.get("/ner", summary="Named entity rows, filterable and paged")
def ner(
    kind: str = Query(default="general", pattern=r"^(general|domain)$"),
    label: Optional[str] = Query(default=None, max_length=60),
    document_id: Optional[str] = Query(default=None, pattern=r"^D\d{1,4}$"),
    search: Optional[str] = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    settings = get_settings()
    limit = min(limit, settings.max_ner_rows)
    return experiments.ner(
        settings, kind=kind, label=label, document_id=document_id, search=search,
        limit=limit, offset=offset,
    )


@router.get("/ner/sidebar", summary="Entity label counts and error analysis")
def ner_sidebar() -> Dict[str, Any]:
    return experiments.ner_sidebar(get_settings())


@router.get("/ngrams", summary="N-gram frequencies for n = 1..5")
def ngrams(
    n: int = Query(default=2, ge=1, le=5),
    search: Optional[str] = Query(default=None, max_length=120),
    domain_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=300),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    settings = get_settings()
    limit = min(limit, settings.max_ngram_rows)
    return experiments.ngrams(
        settings, n=n, search=search, domain_only=domain_only, limit=limit, offset=offset
    )


@router.get("/bpe", summary="BPE statistics, examples, manifest and vocabulary")
def bpe(
    search: Optional[str] = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    settings = get_settings()
    return experiments.bpe(settings, search=search, limit=limit, offset=offset)
