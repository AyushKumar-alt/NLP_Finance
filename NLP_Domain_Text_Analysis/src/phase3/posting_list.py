"""Posting-list data structure with Boolean set operations.

A posting list is the ordered list of content units in which a term occurs,
each posting carrying its term frequency and the token positions that produced
it. Positions make phrase search a positional intersection rather than a text
guess, and the unit ids make every operation a set operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Posting:
    """One unit's occurrence of one term."""

    unit_id: str
    tf: int
    positions: Tuple[int, ...] = ()

    def to_json(self) -> List[object]:
        return [self.unit_id, self.tf, list(self.positions)]

    @classmethod
    def from_json(cls, payload: Sequence[object]) -> "Posting":
        unit_id = str(payload[0])
        tf = int(payload[1]) if len(payload) > 1 else 1
        positions = tuple(int(p) for p in (payload[2] if len(payload) > 2 else []))
        return cls(unit_id=unit_id, tf=tf, positions=positions)


@dataclass
class PostingList:
    """All postings of a single term, sorted by unit id (deterministic)."""

    term: str
    postings: List[Posting] = field(default_factory=list)
    is_phrase: bool = False

    def __post_init__(self) -> None:
        self.postings.sort(key=lambda posting: posting.unit_id)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self):
        return iter(self.postings)

    def __contains__(self, unit_id: str) -> bool:
        return any(posting.unit_id == unit_id for posting in self.postings)

    @property
    def unit_ids(self) -> List[str]:
        return [posting.unit_id for posting in self.postings]

    @property
    def unit_id_set(self) -> Set[str]:
        return {posting.unit_id for posting in self.postings}

    @property
    def posting_frequency(self) -> int:
        return len(self.postings)

    @property
    def total_frequency(self) -> int:
        return sum(posting.tf for posting in self.postings)

    def posting_for(self, unit_id: str) -> Optional[Posting]:
        return next((posting for posting in self.postings if posting.unit_id == unit_id), None)

    def positions_for(self, unit_id: str) -> Tuple[int, ...]:
        posting = self.posting_for(unit_id)
        return posting.positions if posting else ()

    def to_json(self) -> Dict[str, object]:
        return {
            "term": self.term,
            "is_phrase": self.is_phrase,
            "posting_frequency": len(self.postings),
            "term_frequency": self.total_frequency,
            "postings": [posting.to_json() for posting in self.postings],
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "PostingList":
        return cls(
            term=str(payload.get("term", "")),
            postings=[Posting.from_json(item) for item in payload.get("postings", [])],  # type: ignore[arg-type]
            is_phrase=bool(payload.get("is_phrase", False)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PostingList({self.term!r}, postings={len(self.postings)})"


# ----------------------------------------------------------------------
# Boolean set operations
# ----------------------------------------------------------------------
def intersect(lists: Sequence[PostingList]) -> PostingList:
    """Units present in **every** list (set intersection)."""
    if not lists:
        return PostingList(term="<empty>")
    terms = " AND ".join(plist.term for plist in lists)
    if len(lists) == 1:
        only = lists[0]
        return PostingList(term=terms, postings=list(only.postings), is_phrase=only.is_phrase)

    common = set(lists[0].unit_id_set)
    for plist in lists[1:]:
        common &= plist.unit_id_set
    if not common:
        return PostingList(term=terms)

    merged: List[Posting] = []
    for unit_id in sorted(common):
        positions: Set[int] = set()
        total_tf = 0
        phrase = True
        for plist in lists:
            posting = plist.posting_for(unit_id)
            if posting is None:  # pragma: no cover - defensive
                phrase = False
                break
            total_tf += posting.tf
            positions.update(posting.positions)
            phrase = phrase and plist.is_phrase
        merged.append(Posting(unit_id=unit_id, tf=total_tf, positions=tuple(sorted(positions))))
    return PostingList(term=terms, postings=merged, is_phrase=phrase)


def union(lists: Sequence[PostingList]) -> PostingList:
    """Units present in **at least one** list (set union, duplicates removed)."""
    terms = " OR ".join(plist.term for plist in lists)
    if not lists:
        return PostingList(term="<empty>")
    if len(lists) == 1:
        only = lists[0]
        return PostingList(term=terms, postings=list(only.postings), is_phrase=only.is_phrase)

    merged_positions: Dict[str, Set[int]] = {}
    totals: Dict[str, int] = {}
    is_phrase = False
    for plist in lists:
        is_phrase = is_phrase or plist.is_phrase
        for posting in plist.postings:
            merged_positions.setdefault(posting.unit_id, set()).update(posting.positions)
            totals[posting.unit_id] = totals.get(posting.unit_id, 0) + posting.tf

    merged = [
        Posting(unit_id=unit_id, tf=totals[unit_id], positions=tuple(sorted(merged_positions[unit_id])))
        for unit_id in sorted(merged_positions)
    ]
    return PostingList(term=terms, postings=merged, is_phrase=is_phrase)


def difference(left: PostingList, right: PostingList) -> PostingList:
    """``left AND NOT right`` (set difference)."""
    excluded = right.unit_id_set
    kept = [posting for posting in left.postings if posting.unit_id not in excluded]
    return PostingList(
        term=f"{left.term} AND NOT {right.term}",
        postings=kept,
        is_phrase=left.is_phrase,
    )


def empty_list(term: str) -> PostingList:
    """A well formed, empty posting list (missing terms never raise)."""
    return PostingList(term=term, postings=[], is_phrase=False)


def document_frequency(plist: PostingList, unit_to_document: Dict[str, str]) -> int:
    """Number of distinct documents the term appears in."""
    return len({unit_to_document[posting.unit_id] for posting in plist.postings if posting.unit_id in unit_to_document})


def positions_are_adjacent(left: Iterable[int], right: Iterable[int], offset: int = 1) -> bool:
    """True when any ``right`` position follows a ``left`` position by ``offset``."""
    left_set = set(left)
    return any(position - offset in left_set for position in right)
