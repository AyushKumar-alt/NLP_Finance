"""Smoke-check every endpoint once, reporting status and payload size.

Run from the project root::

    python -m scripts.check_api

This is a diagnostic, not the test suite: it prints what each route returned so
a wiring mistake is visible immediately. ``tests/test_api.py`` is the suite.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Tuple

from fastapi.testclient import TestClient

from backend.main import app

GETS: List[Tuple[str, Dict[str, Any]]] = [
    ("/api/health", {}),
    ("/api/statistics", {}),
    ("/api/project", {}),
    ("/api/validation", {}),
    ("/api/sources", {}),
    ("/api/documents", {"params": {"limit": 3}}),
    ("/api/documents/D01", {}),
    ("/api/experiments/summary", {}),
    ("/api/experiments/tokenization", {}),
    ("/api/experiments/preprocessing", {}),
    ("/api/experiments/stemming", {}),
    ("/api/experiments/lemmatization", {}),
    ("/api/experiments/pos", {}),
    ("/api/experiments/ner", {"params": {"limit": 2}}),
    ("/api/experiments/ner/sidebar", {}),
    ("/api/experiments/ngrams", {"params": {"n": 2, "limit": 2}}),
    ("/api/experiments/bpe", {"params": {"limit": 2}}),
    ("/api/queries", {}),
    ("/api/queries/history", {}),
    ("/api/queries/Q01/results", {"params": {"limit": 3}}),
    ("/api/index/term", {"params": {"term": "inflation", "limit": 3}}),
    ("/api/index/terms", {"params": {"limit": 3}}),
    ("/api/index/statistics", {}),
    ("/api/pipelines", {}),
    ("/api/pipelines/selection", {}),
    ("/api/pipelines/pipeline_a", {}),
    ("/api/validation", {}),
    ("/api/evaluation", {}),
    ("/api/evaluation/summary", {}),
    ("/api/evaluation/results", {"params": {"limit": 2}}),
    ("/api/evaluation/aggregates", {}),
    ("/api/evaluation/comparison", {}),
    ("/api/evaluation/judgments", {"params": {"query_id": "Q04"}}),
    ("/api/evaluation/validation", {}),
]

POSTS: List[Tuple[str, Dict[str, Any]]] = [
    ("/api/search", {"json": {"query": "monetary policy", "query_type": "phrase", "top_k": 3}}),
    ("/api/search", {"json": {"query": "GDP AND NOT inflation", "top_k": 3}}),
    ("/api/search/parse", {"json": {"query": 'RBI AND "repo rate" AND NOT (GDP OR GVA)'}}),
]

#: Requests that must be rejected or come back empty, with the status they must
#: produce. A well-formed identifier that does not exist is a 404, not a 400:
#: 400 is reserved for a value that could never name a project object.
NEGATIVE: List[Tuple[str, Dict[str, Any], int]] = [
    ("/api/search", {"json": {"query": "   "}}, 422),
    ("/api/search", {"json": {"query": "x", "top_k": 9999}}, 422),
    ("/api/search", {"json": {"query": "x", "query_type": "nonsense"}}, 422),
    ("/api/search", {"json": {"query": "x", "unexpected": 1}}, 422),
    ("/api/documents/D99", {}, 404),
    ("/api/documents/..%2F..%2Fetc%2Fpasswd", {}, 404),
    ("/api/documents/bad%2Fid", {}, 404),
    ("/api/units/nope", {}, 404),
    ("/api/pipelines/pipeline_c", {}, 404),
    ("/api/queries/Q1/results", {}, 400),
    ("/api/evaluation/judgments", {"json": {"query_id": "bad", "unit_id": "x",
                                            "relevance": True, "annotator": "a"}}, 422),
    ("/api/evaluation/judgments", {"json": {"query_id": "Q01", "unit_id": "../x",
                                            "relevance": True, "annotator": "a"}}, 422),
    ("/api/experiments/ngrams", {"params": {"n": 9}}, 422),
    ("/api/experiments/ner", {"params": {"kind": "other"}}, 422),
    ("/api/does-not-exist", {}, 404),
]


def main() -> int:
    client = TestClient(app, raise_server_exceptions=False)
    failures = 0

    print("== GET endpoints")
    for path, kwargs in GETS:
        started = time.perf_counter()
        response = client.get(path, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000
        body = response.json() if response.headers.get("content-type", "").startswith(
            "application/json"
        ) else {}
        size = len(response.content)
        mark = "ok " if response.status_code == 200 else "FAIL"
        if response.status_code != 200:
            failures += 1
        print(f"  {mark} {response.status_code} {path} ({size:,}B, {elapsed:.0f}ms)")
        if response.status_code != 200:
            print(f"       {str(body)[:300]}")

    print("== POST endpoints")
    for path, kwargs in POSTS:
        started = time.perf_counter()
        response = client.post(path, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000
        body = response.json() if response.headers.get("content-type", "").startswith(
            "application/json"
        ) else {}
        mark = "ok " if response.status_code == 200 else "FAIL"
        if response.status_code != 200:
            failures += 1
        detail = ""
        if isinstance(body, dict):
            detail = f"type={body.get('query_type')} returned={body.get('returned')}"
        print(f"  {mark} {response.status_code} {path} {detail} ({elapsed:.0f}ms)")
        if response.status_code != 200:
            print(f"       {str(body)[:300]}")

    print("== negative cases")
    for path, kwargs, expected in NEGATIVE:
        method = client.post if "json" in kwargs else client.get
        response = method(path, **kwargs)
        mark = "ok " if response.status_code == expected else "FAIL"
        if response.status_code != expected:
            failures += 1
        print(f"  {mark} {response.status_code} (want {expected}) {path}")
        if response.status_code != expected:
            print(f"       {response.text[:300]}")

    print()
    print(f"{len(GETS) + len(POSTS) + len(NEGATIVE) - failures}"
          f"/{len(GETS) + len(POSTS) + len(NEGATIVE)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
