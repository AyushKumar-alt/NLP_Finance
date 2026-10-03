"""Posting-list set operations used by Boolean retrieval."""

from __future__ import annotations

from src.phase3.posting_list import (
    Posting,
    PostingList,
    difference,
    document_frequency,
    empty_list,
    intersect,
    positions_are_adjacent,
    union,
)


def make(term: str, pairs: dict, is_phrase: bool = False) -> PostingList:
    return PostingList(
        term=term,
        postings=[Posting(unit_id=unit_id, tf=tf, positions=tuple(range(tf)))
                  for unit_id, tf in sorted(pairs.items())],
        is_phrase=is_phrase,
    )


def test_intersection_keeps_only_shared_units():
    left = make("gdp", {"u1": 2, "u2": 1, "u3": 1})
    right = make("inflation", {"u2": 3, "u3": 1, "u4": 1})
    merged = intersect([left, right])
    assert merged.unit_ids == ["u2", "u3"]
    assert merged.posting_for("u2").tf == 4          # tf is summed over both lists
    assert merged.posting_for("u3").positions == (0,)


def test_intersection_of_disjoint_lists_is_empty():
    merged = intersect([make("a", {"u1": 1}), make("b", {"u2": 1})])
    assert len(merged) == 0


def test_intersection_of_a_single_list_is_a_copy():
    left = make("a", {"u1": 1, "u2": 2})
    merged = intersect([left])
    assert merged.unit_ids == left.unit_ids
    merged.postings.clear()
    assert len(left) == 2, "the original list must not be aliased"


def test_union_removes_duplicates_and_sums_frequencies():
    merged = union([make("gdp", {"u1": 1, "u2": 2}), make("gva", {"u2": 1, "u3": 1})])
    assert merged.unit_ids == ["u1", "u2", "u3"]
    assert merged.posting_for("u2").tf == 3
    assert len(merged) == 3


def test_union_of_a_single_list_is_a_copy():
    left = make("a", {"u1": 1})
    merged = union([left])
    merged.postings.clear()
    assert len(left) == 1


def test_difference_removes_excluded_units():
    left = make("inflation", {"u1": 1, "u2": 1, "u3": 1})
    right = make("food", {"u2": 4})
    merged = difference(left, right)
    assert merged.unit_ids == ["u1", "u3"]
    assert merged.term == "inflation AND NOT food"


def test_difference_with_an_empty_exclusion_returns_everything():
    merged = difference(make("a", {"u1": 1}), empty_list("b"))
    assert merged.unit_ids == ["u1"]


def test_empty_list_is_well_formed():
    empty = empty_list("nothing")
    assert len(empty) == 0
    assert empty.unit_id_set == set()
    assert empty.total_frequency == 0
    assert empty.positions_for("u1") == ()


def test_posting_list_helpers():
    postings = make("gdp", {"u1": 2, "u2": 1})
    assert postings.posting_frequency == 2
    assert postings.total_frequency == 3
    assert "u1" in postings
    assert "u9" not in postings
    assert postings.posting_for("u9") is None
    assert postings.positions_for("u1") == (0, 1)


def test_document_frequency_uses_the_unit_to_document_map():
    postings = make("gdp", {"u1": 1, "u2": 1, "u3": 1})
    mapping = {"u1": "D1", "u2": "D1", "u3": "D2"}
    assert document_frequency(postings, mapping) == 2


def test_positions_are_adjacent_detects_order():
    assert positions_are_adjacent([3], [4]) is True
    assert positions_are_adjacent([3], [5]) is False
    assert positions_are_adjacent([], [1]) is False


def test_posting_list_serialization_round_trip():
    original = make("gdp", {"u1": 2, "u2": 1}, is_phrase=False)
    restored = PostingList.from_json(original.to_json())
    assert restored.postings == original.postings
    assert restored.term == original.term
    assert restored.is_phrase == original.is_phrase
