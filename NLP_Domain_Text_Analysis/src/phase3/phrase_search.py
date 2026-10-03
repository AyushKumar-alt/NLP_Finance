"""Phrase search using token positions.

Phrase search is **not** a disjunction of its words: ``"monetary policy"`` must
match only units where ``monetary`` is immediately followed by ``policy``.

Two details decide whether a phrase search is correct or merely plausible:

*Stopwords.*  A phrase often contains one - ``current account deficit``,
``repo rate``. The index vocabulary deliberately excludes stopwords, so the
positional stream of the *index* cannot answer those phrases: after stopword
removal ``current`` is gone and the remaining tokens are not the ones the reader
wrote. Phrase matching therefore runs on a **stopword-preserving stream**
(:meth:`~src.phase3.pipeline_runner.PipelineRunner.phrase_stream`) recomputed
from the unit text, while the *candidate* units still come from the index, so
the index is never scanned end to end.

*Order.*  The two words must be adjacent, and in the written order. The n-gram
phrase terms stored in the index (``monetary_policy``) provide a second,
independent check: a unit carrying the phrase term is a match by construction.
Both signals are recorded so the method can be audited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .config import log_event
from .inverted_index import InvertedIndex
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec
from .posting_list import Posting, PostingList


@dataclass
class PhraseMatch:
    unit_id: str
    positions: Tuple[int, ...]
    via: str          # "positions" | "phrase_term" | "both"


class PhraseSearch:
    """Positional phrase matching over the unit token stream."""

    def __init__(self, index: InvertedIndex, spec: PipelineSpec, runner: PipelineRunner,
                 join_with: str = "_", stream_provider: Optional[Callable[[str], List[str]]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.index = index
        self.spec = spec
        self.runner = runner
        self.join_with = join_with
        #: returns the stopword-preserving term stream of a unit (may be None,
        #: in which case the index positions are used)
        self.stream_provider = stream_provider
        self.logger = logger
        self._cache: Dict[str, List[str]] = {}
        self._stream_cache: Dict[str, List[str]] = {}
        self.stream_hits = 0
        self.index_position_hits = 0

    # ------------------------------------------------------------------
    def normalize(self, phrase: str) -> List[str]:
        """Phrase terms, with stopwords kept so the phrase can be located."""
        return self.runner.phrase_stream_terms(self.spec, phrase)

    def phrase_term(self, terms: Sequence[str]) -> str:
        return self.join_with.join(terms)

    # ------------------------------------------------------------------
    def unit_stream(self, unit_id: str) -> Optional[List[str]]:
        """The stopword-preserving term stream of one unit (cached)."""
        if self.stream_provider is None:
            return None
        if unit_id not in self._stream_cache:
            self._stream_cache[unit_id] = self.stream_provider(unit_id)
        return self._stream_cache[unit_id]

    def _positions_in_stream(self, unit_id: str, terms: Sequence[str]) -> Set[int]:
        """Starting positions where ``terms`` occur adjacently, in order."""
        stream = self.unit_stream(unit_id)
        if stream is None:
            return set()
        starts: Set[int] = set()
        first = terms[0]
        width = len(terms)
        for position, term in enumerate(stream):
            if term != first:
                continue
            if all(
                position + offset < len(stream) and stream[position + offset] == terms[offset]
                for offset in range(width)
            ):
                starts.add(position)
        return starts

    def _positions_in_index(self, unit_id: str, terms: Sequence[str]) -> Set[int]:
        """Starting positions according to the index's own positional stream."""
        current = set(self.index.posting_list(terms[0]).positions_for(unit_id))
        for offset, term in enumerate(terms[1:], start=1):
            positions = set(self.index.posting_list(term).positions_for(unit_id))
            if not positions:
                return set()
            current = {p + offset for p in current if (p + offset) in positions}
            if not current:
                return set()
        return current

    def match_unit(self, unit_id: str, terms: Sequence[str]) -> Optional[PhraseMatch]:
        """Positional match of ``terms`` inside one unit."""
        if not terms:
            return None

        via_stream = self._positions_in_stream(unit_id, terms)
        if via_stream:
            self.stream_hits += 1
            positions = via_stream
            via = "stream"
        else:
            positions = self._positions_in_index(unit_id, terms)
            if not positions:
                return None
            self.index_position_hits += 1
            via = "index_positions"

        phrase_term = self.phrase_term(terms)
        if self.index.has_term(phrase_term) and unit_id in self.index.posting_list(
            phrase_term
        ).unit_id_set:
            via = f"{via}+phrase_term"
        return PhraseMatch(unit_id=unit_id, positions=tuple(sorted(positions)), via=via)

    # ------------------------------------------------------------------
    def search(self, phrase: str, candidate_limit: int = 20000) -> Dict[str, Any]:
        """Return ``{'terms', 'missing', 'matches'}`` for a quoted phrase."""
        terms = self.normalize(phrase)
        if not terms:
            return {
                "terms": [],
                "missing": [],
                "matches": [],
                "unit_ids": [],
                "phrase_term": "",
                "note": "phrase normalized to an empty term sequence (no indexable token)",
            }

        phrase_term = self.phrase_term(terms)
        candidates: Set[str] = set()
        if self.index.has_term(phrase_term):
            candidates = set(self.index.posting_list(phrase_term).unit_ids)

        # The anchor is the first phrase term that is actually in the index; a
        # leading stopword ("current account deficit") is skipped, never assumed.
        stopwords = self.runner.stopword_words(self.spec)
        indexed = [term for term in terms if self.index.has_term(term)]
        missing = [
            term for term in terms
            if not self.index.has_term(term) and term not in stopwords
        ]
        stopword_terms = [term for term in terms if term in stopwords]
        if indexed and len(candidates) < candidate_limit:
            candidates |= set(self.index.posting_list(indexed[0]).unit_ids[:candidate_limit])

        matches: List[PhraseMatch] = []
        for unit_id in sorted(candidates):
            found = self.match_unit(unit_id, terms)
            if found is not None:
                matches.append(found)

        if self.logger is not None:
            log_event(
                self.logger,
                "INFO",
                "phrase_search",
                f"'{phrase}' -> {len(matches)} unit(s) "
                f"via {sorted({m.via for m in matches}) or ['none']}",
            )
        return {
            "terms": terms,
            "indexed_terms": indexed,
            "stopword_terms": stopword_terms,
            "missing": missing,
            "matches": matches,
            "unit_ids": [match.unit_id for match in matches],
            "phrase_term": phrase_term,
            "note": "positional intersection on the stopword-preserving token stream",
        }
