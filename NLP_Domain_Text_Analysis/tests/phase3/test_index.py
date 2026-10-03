"""Inverted index structure, provenance and persistence."""

from __future__ import annotations

import json

from src.phase3.index_builder import is_indexable
from src.phase3.inverted_index import InvertedIndex


def test_index_contains_the_expected_terms(builds):
    index = builds["pipeline_a"].index
    # 'banking' and 'lending' are stored as their WordNet lemmas ('bank', 'lend');
    # that normalization is exactly what Pipeline A is being measured on.
    for term in ("gdp", "inflation", "disinflation", "cpi", "repo", "rate", "bank", "lend"):
        assert term in index, f"expected '{term}' in the Pipeline A index"


def test_pipeline_b_uses_stems_while_pipeline_a_uses_lemmas(builds):
    a_terms = set(builds["pipeline_a"].index.terms)
    b_terms = set(builds["pipeline_b"].index.terms)
    assert "inflation" in a_terms
    assert "inflation" not in b_terms, "Pipeline B stems, so the surface form should be gone"
    assert "inflat" in b_terms
    assert builds["pipeline_a"].spec.term_kind == "lemma"
    assert builds["pipeline_b"].spec.term_kind == "stem"


def test_index_has_no_structural_problems(builds):
    for key, build in builds.items():
        assert build.index.validate() == [], f"{key} index failed self-validation"


def test_every_posting_resolves_to_a_unit_record(builds):
    for build in builds.values():
        index = build.index
        for entry in index.terms.values():
            for posting in entry.postings:
                assert posting.unit_id in index.units


def test_posting_lists_are_sorted_and_duplicate_free(builds):
    for build in builds.values():
        for term, entry in build.index.terms.items():
            unit_ids = [posting.unit_id for posting in entry.postings]
            assert unit_ids == sorted(unit_ids), f"{term} postings are unsorted"
            assert len(unit_ids) == len(set(unit_ids)), f"{term} has duplicate postings"


def test_provenance_chain_is_complete(builds, synthetic_units):
    expected = {unit.unit_id: unit for unit in synthetic_units}
    index = builds["pipeline_a"].index
    assert index.unit_count == len(expected)
    for unit_id, record in index.units.items():
        source = expected[unit_id]
        assert record.source_id == source.source_id
        assert record.document_id == source.document_id
        assert record.page_number == source.page_number
        assert record.section_id == source.section_id
        assert record.section_number == source.section_number
        assert record.section_title == source.section_title
        assert record.document_id in index.documents


def test_document_frequency_counts_distinct_documents(builds):
    index = builds["pipeline_a"].index
    postings = index.posting_list("inflation")
    documents = {
        index.unit(posting.unit_id).document_id for posting in postings
    }
    assert index.document_frequency("inflation") == len(documents)
    assert documents <= {"D01", "D02", "D03"}
    assert index.document_frequency("no_such_term_exists_here") == 0


def test_positions_are_stored_for_every_posting(builds):
    for build in builds.values():
        for term, entry in build.index.terms.items():
            for posting in entry.postings:
                assert posting.positions, f"{term}/{posting.unit_id} has no positions"
                assert all(position >= 0 for position in posting.positions)


def test_positions_are_monotonic_within_a_posting(builds):
    for build in builds.values():
        for entry in build.index.terms.values():
            for posting in entry.postings:
                assert list(posting.positions) == sorted(posting.positions)


def test_phrase_terms_are_flagged_and_decompose(builds):
    join = "_"
    for build in builds.values():
        index = build.index
        phrases = [term for term, entry in index.terms.items() if entry.is_phrase]
        assert phrases, "no phrase terms were indexed"
        for phrase in phrases:
            for part in phrase.split(join):
                assert part in index, f"phrase '{phrase}' has unknown part '{part}'"


def test_financial_expressions_survive_as_single_terms(builds):
    for key, build in builds.items():
        index = build.index
        assert "fy26" in index, f"{key} lost the fiscal year"
        assert any("per cent" in term for term in index.terms), (
            f"{key} lost the percentage expression"
        )


def test_index_round_trips_through_json(builds, tmp_path):
    original = builds["pipeline_a"].index
    path = tmp_path / "index.json"
    payload = original.save(path)
    assert payload["terms"] == original.term_count
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["pipeline"] == "pipeline_a"

    reloaded = InvertedIndex.load(path)
    assert reloaded.term_count == original.term_count
    assert reloaded.posting_count == original.posting_count
    assert reloaded.unit_count == original.unit_count
    assert reloaded.document_count() == original.document_count()
    assert set(reloaded.terms) == set(original.terms)
    for term, entry in original.terms.items():
        assert reloaded.terms[term].postings == entry.postings
    assert reloaded.validate() == []


def test_saved_index_is_byte_identical_for_the_same_input(builds, tmp_path):
    original = builds["pipeline_a"].index
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    original.save(first)
    original.save(second)
    assert first.read_bytes() == second.read_bytes()


def test_punctuation_only_tokens_are_not_indexed():
    assert is_indexable("gdp") is True
    assert is_indexable("7.4 per cent") is True
    assert is_indexable("fy26") is True
    assert is_indexable(".") is False
    assert is_indexable("2026") is True          # bare year still carries signal
    assert is_indexable("--") is False
    assert is_indexable("") is False


def test_unknown_term_returns_an_empty_posting_list(builds):
    index = builds["pipeline_a"].index
    postings = index.posting_list("definitely_not_a_term")
    assert len(postings) == 0
    assert postings.unit_ids == []
    assert "definitely_not_a_term" not in index
