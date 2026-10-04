"""Automated verification of the Mandatory Functional Test Matrix (Queries 1-8)
and Pipeline A/B isolation check.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict

BASE_URL = "http://127.0.0.1:8000"


def post_json(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8")
        return {"_status_code": err.code, "_error": body}


def main():
    print("============================================================")
    print("MANDATORY FUNCTIONAL TEST MATRIX (API LEVEL)")
    print("============================================================")

    # TEST 1: inflation
    print("\n--- TEST 1: inflation ---")
    t1 = post_json("/api/search", {"query": "inflation", "top_k": 10})
    print(f"Query: {t1.get('query')}")
    print(f"Query type: {t1.get('query_type')}, method: {t1.get('retrieval_method')}")
    print(f"Total results: {t1.get('total_results')}, returned: {t1.get('returned')}")
    assert t1.get("query_type") == "keyword", f"Expected keyword, got {t1.get('query_type')}"
    assert t1.get("returned") > 0
    for hit in t1["results"][:3]:
        print(f"  Hit {hit['unit_id']}: score={hit['score']}, matched={hit['matched_terms_list']}")
        assert "inflation" in hit["matched_terms_list"] or "inflat" in hit["matched_terms_list"]

    # TEST 2: "monetary policy"
    print("\n--- TEST 2: \"monetary policy\" ---")
    t2 = post_json("/api/search", {"query": '"monetary policy"', "top_k": 10})
    print(f"Query: {t2.get('query')}")
    print(f"Query type: {t2.get('query_type')}, method: {t2.get('retrieval_method')}")
    print(f"Total results: {t2.get('total_results')}, returned: {t2.get('returned')}")
    assert t2.get("query_type") == "phrase", f"Expected phrase, got {t2.get('query_type')}"
    assert t2.get("retrieval_method") == "phrase"
    assert t2.get("returned") > 0
    for hit in t2["results"][:3]:
        print(f"  Hit {hit['unit_id']}: score={hit['score']}, phrase_match={hit['phrase_match']}, snippet={hit['snippet'][:60]}...")
        assert hit["phrase_match"] is True

    # TEST 3: GDP AND inflation
    print("\n--- TEST 3: GDP AND inflation ---")
    t3 = post_json("/api/search", {"query": "GDP AND inflation", "top_k": 10})
    print(f"Query: {t3.get('query')}")
    print(f"Query type: {t3.get('query_type')}, method: {t3.get('retrieval_method')}")
    print(f"Total results: {t3.get('total_results')}, returned: {t3.get('returned')}")
    assert t3.get("query_type") == "boolean_and"
    for hit in t3["results"]:
        matched = set(hit["matched_terms_list"])
        print(f"  Hit {hit['unit_id']}: matched={hit['matched_terms_list']}")
        assert "gdp" in matched and ("inflation" in matched or "inflat" in matched)

    # TEST 4: GDP OR GVA
    print("\n--- TEST 4: GDP OR GVA ---")
    t4 = post_json("/api/search", {"query": "GDP OR GVA", "top_k": 10})
    print(f"Query: {t4.get('query')}")
    print(f"Query type: {t4.get('query_type')}, method: {t4.get('retrieval_method')}")
    print(f"Total results: {t4.get('total_results')}, returned: {t4.get('returned')}")
    assert t4.get("query_type") == "boolean_or"
    both_count = 0
    for hit in t4["results"]:
        matched = set(hit["matched_terms_list"])
        assert "gdp" in matched or "gva" in matched
        if "gdp" in matched and "gva" in matched:
            both_count += 1
    print(f"  Results matching both GDP and GVA in top 10: {both_count}")

    # TEST 5: inflation AND NOT food
    print("\n--- TEST 5: inflation AND NOT food ---")
    t5 = post_json("/api/search", {"query": "inflation AND NOT food", "top_k": 10})
    print(f"Query: {t5.get('query')}")
    print(f"Query type: {t5.get('query_type')}, method: {t5.get('retrieval_method')}")
    print(f"Excluded terms: {t5.get('excluded_terms')}")
    print(f"Total results: {t5.get('total_results')}, returned: {t5.get('returned')}")
    assert t5.get("query_type") == "boolean_not"
    assert "food" in t5.get("excluded_terms", [])
    for hit in t5["results"]:
        matched = set(hit["matched_terms_list"])
        assert "inflation" in matched or "inflat" in matched
        assert "food" not in matched

    # TEST 6: (GDP OR GVA) AND policy
    print("\n--- TEST 6: (GDP OR GVA) AND policy ---")
    t6 = post_json("/api/search", {"query": "(GDP OR GVA) AND policy", "top_k": 10})
    print(f"Query: {t6.get('query')}")
    print(f"Query type: {t6.get('query_type')}, method: {t6.get('retrieval_method')}")
    print(f"Total results: {t6.get('total_results')}, returned: {t6.get('returned')}")
    assert t6.get("query_type") == "boolean_group"
    for hit in t6["results"]:
        matched = set(hit["matched_terms_list"])
        assert ("gdp" in matched or "gva" in matched) and ("policy" in matched or "polici" in matched)

    # TEST 7: RBI AND "repo rate"
    print("\n--- TEST 7: RBI AND \"repo rate\" ---")
    t7 = post_json("/api/search", {"query": 'RBI AND "repo rate"', "top_k": 10})
    print(f"Query: {t7.get('query')}")
    print(f"Query type: {t7.get('query_type')}, method: {t7.get('retrieval_method')}")
    print(f"Total results: {t7.get('total_results')}, returned: {t7.get('returned')}")
    assert t7.get("returned") > 0
    for hit in t7["results"]:
        matched = set(hit["matched_terms_list"])
        print(f"  Hit {hit['unit_id']}: phrase_match={hit['phrase_match']}, matched={hit['matched_terms_list']}")
        assert "rbi" in matched
        assert hit["phrase_match"] is True

    # TEST 8: monetary policy & repo rate
    print("\n--- TEST 8: monetary policy & repo rate ---")
    # First check parse endpoint
    p8 = post_json("/api/search/parse", {"query": "monetary policy & repo rate"})
    print(f"Parse result: valid={p8.get('valid')}, error={p8.get('error')}")
    assert p8.get("valid") is False
    assert "& is not a supported Boolean operator. Use AND instead." in p8.get("error", "")

    # Now check search endpoint rejects with 400
    s8 = post_json("/api/search", {"query": "monetary policy & repo rate", "top_k": 10})
    print(f"Search endpoint rejection: status={s8.get('_status_code')}, error={s8.get('_error')}")
    assert s8.get("_status_code") == 400
    assert "& is not a supported Boolean operator. Use AND instead." in s8.get("_error", "")

    # PIPELINE B REGRESSION CHECK
    print("\n============================================================")
    print("PIPELINE B REGRESSION CHECK (banking AND credit)")
    print("============================================================")
    res_a = post_json("/api/search", {"query": "banking AND credit", "pipeline": "pipeline_a_lemma", "top_k": 10})
    res_b = post_json("/api/search", {"query": "banking AND credit", "pipeline": "pipeline_b_stem", "top_k": 10})
    print(f"Pipeline A ({res_a.get('pipeline')}): {res_a.get('total_results')} results, Top hit: {res_a['results'][0]['unit_id'] if res_a['results'] else None}, score={res_a['results'][0]['score'] if res_a['results'] else None}")
    print(f"Pipeline B ({res_b.get('pipeline')}): {res_b.get('total_results')} results, Top hit: {res_b['results'][0]['unit_id'] if res_b['results'] else None}, score={res_b['results'][0]['score'] if res_b['results'] else None}")
    print(f"Pipeline A matched terms: {res_a['matched_terms']}")
    print(f"Pipeline B matched terms: {res_b['matched_terms']}")
    assert res_a.get("pipeline") == "pipeline_a_lemma"
    assert res_b.get("pipeline") == "pipeline_b_stem"

    # Score breakdown verification
    print("\n============================================================")
    print("SCORE BREAKDOWN VERIFICATION")
    print("============================================================")
    top_hit = t3["results"][0]
    print(f"Hit: {top_hit['unit_id']}")
    print(f"Score breakdown: {top_hit.get('score_breakdown')}")
    sb = top_hit["score_breakdown"]
    assert sb["matched_term_count"] == 2
    assert sb["term_component"] == 2.0
    assert sb["phrase_component"] == 0.0
    assert sb["total_score"] == top_hit["score"]
    print("Score breakdown exactly matches formula!")

    print("\n>>> ALL MATRIX CHECKS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
