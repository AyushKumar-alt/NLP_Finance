"""Focused regression tests for Search GUI Correctness and Explainability Fixes.

Covers:
- Query type auto-detection vs explicit selection
- Exact phrase positional retrieval (quotes stripped, adjacent match)
- Boolean AND semantics (every unit satisfies all positive terms)
- Boolean OR semantics (units satisfy either or both terms)
- Boolean NOT semantics (excluded term removed, positive terms present)
- Grouped Boolean semantics ((GDP OR GVA) AND policy)
- Mixed phrase + Boolean (RBI AND "repo rate")
- Standalone '&' validation (rejected with exact error message)
- Pipeline A vs Pipeline B isolation
- Score formula exact breakdown
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict

import pytest

from backend.services.retrieval import has_unquoted_ampersand, parse_query as backend_parse_query
from src.phase3.query_parser import QuerySyntaxError, query_type_of, tokenize_query

BASE_URL = "http://127.0.0.1:8000"


def post_api(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status_code": resp.status, "data": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
        return {"status_code": err.code, "error": parsed}


# ----------------------------------------------------------------------
# 1. Query Type Auto-Detection & Ampersand Validation
# ----------------------------------------------------------------------
def test_query_type_detection():
    assert query_type_of("inflation") == "keyword"
    assert query_type_of('"monetary policy"') == "phrase"
    assert query_type_of("GDP AND inflation") == "boolean_and"
    assert query_type_of("GDP OR GVA") == "boolean_or"
    assert query_type_of("inflation AND NOT food") == "boolean_not"
    assert query_type_of("(GDP OR GVA) AND policy") == "boolean_group"
    assert query_type_of('RBI AND "repo rate"') == "boolean_and"


def test_ampersand_rejected_in_tokenizer():
    with pytest.raises(QuerySyntaxError) as exc_info:
        tokenize_query("monetary policy & repo rate")
    assert "& is not a supported Boolean operator. Use AND instead." in str(exc_info.value)


def test_has_unquoted_ampersand_helper():
    assert has_unquoted_ampersand("monetary policy & repo rate") is True
    assert has_unquoted_ampersand("& repo rate") is True
    assert has_unquoted_ampersand("repo rate &") is True
    assert has_unquoted_ampersand('"monetary & policy"') is False
    assert has_unquoted_ampersand("monetary AND policy") is False


def test_backend_parse_query_ampersand_warning():
    parsed = backend_parse_query("monetary policy & repo rate")
    assert parsed["valid"] is False
    assert parsed["error"] == "& is not a supported Boolean operator. Use AND instead."


# ----------------------------------------------------------------------
# 2. Mandatory Functional Retrieval Matrix (Queries 1-8)
# ----------------------------------------------------------------------
def test_test1_keyword_inflation():
    res = post_api("/api/search", {"query": "inflation", "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert data["query_type"] == "keyword"
    assert data["retrieval_method"] == "keyword"
    assert len(data["results"]) > 0
    top = data["results"][0]
    assert any(t in ("inflation", "inflat") for t in top["matched_terms_list"])


def test_test2_phrase_monetary_policy():
    res = post_api("/api/search", {"query": '"monetary policy"', "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert data["query_type"] == "phrase"
    assert data["retrieval_method"] == "phrase"
    assert len(data["results"]) > 0
    for hit in data["results"][:5]:
        assert hit["phrase_match"] is True


def test_test3_boolean_and_gdp_inflation():
    res = post_api("/api/search", {"query": "GDP AND inflation", "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert data["query_type"] == "boolean_and"
    assert len(data["results"]) > 0
    for hit in data["results"]:
        m = set(hit["matched_terms_list"])
        assert "gdp" in m
        assert ("inflation" in m or "inflat" in m)


def test_test4_boolean_or_gdp_gva():
    res = post_api("/api/search", {"query": "GDP OR GVA", "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert data["query_type"] == "boolean_or"
    assert len(data["results"]) > 0
    has_both = False
    for hit in data["results"]:
        m = set(hit["matched_terms_list"])
        assert ("gdp" in m or "gva" in m)
        if "gdp" in m and "gva" in m:
            has_both = True
    assert has_both is True


def test_test5_boolean_not_inflation_food():
    res = post_api("/api/search", {"query": "inflation AND NOT food", "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert data["query_type"] == "boolean_not"
    assert "food" in data["excluded_terms"]
    assert len(data["results"]) > 0
    for hit in data["results"]:
        m = set(hit["matched_terms_list"])
        assert ("inflation" in m or "inflat" in m)
        assert "food" not in m


def test_test6_boolean_group():
    res = post_api("/api/search", {"query": "(GDP OR GVA) AND policy", "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert data["query_type"] == "boolean_group"
    assert len(data["results"]) > 0
    for hit in data["results"]:
        m = set(hit["matched_terms_list"])
        assert ("gdp" in m or "gva" in m)
        assert ("policy" in m or "polici" in m)


def test_test7_mixed_boolean_and_phrase():
    res = post_api("/api/search", {"query": 'RBI AND "repo rate"', "top_k": 10})
    assert res["status_code"] == 200
    data = res["data"]
    assert len(data["results"]) > 0
    for hit in data["results"]:
        m = set(hit["matched_terms_list"])
        assert "rbi" in m
        assert hit["phrase_match"] is True


def test_test8_ampersand_rejected():
    # Parse rejection
    res_parse = post_api("/api/search/parse", {"query": "monetary policy & repo rate"})
    assert res_parse["status_code"] == 200
    assert res_parse["data"]["valid"] is False
    assert "& is not a supported Boolean operator. Use AND instead." in res_parse["data"]["error"]

    # Search rejection
    res_search = post_api("/api/search", {"query": "monetary policy & repo rate"})
    assert res_search["status_code"] == 400
    assert "& is not a supported Boolean operator. Use AND instead." in str(res_search["error"])


# ----------------------------------------------------------------------
# 3. Pipeline A/B Isolation
# ----------------------------------------------------------------------
def test_pipeline_ab_isolation():
    res_a = post_api("/api/search", {"query": "banking AND credit", "pipeline": "pipeline_a_lemma"})
    res_b = post_api("/api/search", {"query": "banking AND credit", "pipeline": "pipeline_b_stem"})
    assert res_a["status_code"] == 200
    assert res_b["status_code"] == 200
    data_a = res_a["data"]
    data_b = res_b["data"]
    assert data_a["pipeline"] == "pipeline_a_lemma"
    assert data_b["pipeline"] == "pipeline_b_stem"
    # Lemmatization produces 'banking' in A, while Snowball stems to 'bank' in B
    assert data_a["matched_terms"] == ["banking", "credit"]
    assert data_b["matched_terms"] == ["bank", "credit"]
    assert data_a["results"][0]["unit_id"] != data_b["results"][0]["unit_id"]


# ----------------------------------------------------------------------
# 4. Score Breakdown Formula
# ----------------------------------------------------------------------
def test_score_breakdown_formula():
    res = post_api("/api/search", {"query": "GDP AND inflation", "top_k": 5})
    assert res["status_code"] == 200
    data = res["data"]
    hit = data["results"][0]
    sb = hit.get("score_breakdown")
    assert sb is not None
    assert sb["matched_term_count"] == 2
    assert sb["term_component"] == 2.0
    assert sb["phrase_component"] == 0.0
    assert sb["total_score"] == hit["score"]
