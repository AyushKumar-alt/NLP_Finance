"""Boolean retrieval over posting lists.

Every operator is a set operation on posting lists:

``A AND B``      intersection
``A OR B``       union (duplicates removed)
``A AND NOT B``  difference

The AST produced by :mod:`src.phase3.query_parser` is interpreted here; no part
of the user query is ever evaluated as Python code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set

from .config import log_event
from .inverted_index import InvertedIndex
from .phrase_search import PhraseSearch
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec
from .posting_list import Posting, PostingList, difference, empty_list, intersect, union
from .query_parser import And, Expression, Not, Or, QuerySyntaxError, Term


@dataclass
class BooleanResult:
    """Posting-list outcome of evaluating an AST."""

    postings: PostingList
    matched_terms: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)
    phrase_units: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)


class BooleanSearch:
    """Evaluates a Boolean AST against the index."""

    def __init__(self, index: InvertedIndex, spec: PipelineSpec, runner: PipelineRunner,
                 join_with: str = "_", enable_phrase: bool = True,
                 phrase_search: Optional[PhraseSearch] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.index = index
        self.spec = spec
        self.runner = runner
        self.logger = logger
        if phrase_search is not None:
            self.phrase_search = phrase_search
        elif enable_phrase:
            self.phrase_search = PhraseSearch(
                index, spec, runner, join_with=join_with, logger=logger
            )
        else:
            self.phrase_search = None

    # ------------------------------------------------------------------
    def evaluate(self, node: Expression) -> BooleanResult:
        if isinstance(node, Term):
            return self._evaluate_term(node)
        if isinstance(node, Not):
            raise QuerySyntaxError(
                "NOT is only supported in the form 'A AND NOT B'; a bare NOT has no "
                "meaningful result set"
            )
        if isinstance(node, And):
            return self._evaluate_and(node)
        if isinstance(node, Or):
            return self._evaluate_or(node)
        raise TypeError(f"unknown AST node {node!r}")

    # ------------------------------------------------------------------
    def _evaluate_term(self, node: Term) -> BooleanResult:
        normalized = self.runner.normalize_query(self.spec, node.text)
        if node.is_phrase or len(normalized) > 1:
            if self.phrase_search is None:
                raise QuerySyntaxError("phrase search is disabled in this configuration")
            found = self.phrase_search.search(node.text)
            missing = list(found["missing"])
            unit_ids: List[str] = list(found["unit_ids"])
            postings = self._postings_for_units(node.text, unit_ids, is_phrase=True)
            return BooleanResult(
                postings=postings,
                matched_terms=list(found["indexed_terms"]),
                missing_terms=missing,
                phrase_units=set(unit_ids),
                notes=[str(found.get("note", ""))],
            )
        if not normalized:
            return BooleanResult(
                postings=empty_list(node.text),
                missing_terms=[node.text],
                notes=["term normalized to an empty sequence (stopword or punctuation only)"],
            )
        term = normalized[0]
        if not self.index.has_term(term):
            return BooleanResult(postings=empty_list(term), missing_terms=[term],
                                 notes=[f"term '{term}' is not in the index"])
        return BooleanResult(postings=self.index.posting_list(term), matched_terms=[term])

    def _postings_for_units(self, term: str, unit_ids: Sequence[str], is_phrase: bool) -> PostingList:
        """Aggregate tf/positions for the units a phrase matched."""
        if not unit_ids:
            return PostingList(term=term, postings=[], is_phrase=is_phrase)
        totals: Dict[str, int] = {}
        for unit_id in unit_ids:
            entry = self.index.terms.get(self._term_of(term))
            if entry is not None:
                posting = next((p for p in entry.postings if p.unit_id == unit_id), None)
                totals[unit_id] = posting.tf if posting else 1
        postings = [
            Posting(unit_id=unit_id, tf=totals.get(unit_id, 1), positions=())
            for unit_id in sorted(set(unit_ids))
        ]
        return PostingList(term=term, postings=postings, is_phrase=is_phrase)

    def _term_of(self, term: str) -> str:
        normalized = self.runner.normalize_query(self.spec, term)
        return normalized[0] if normalized else term

    # ------------------------------------------------------------------
    def _evaluate_and(self, node: And) -> BooleanResult:
        left = self.evaluate(node.left)
        notes = list(left.notes)
        if isinstance(node.right, Not):
            # 'A AND NOT B' is a set difference; the NOT operand is evaluated on
            # its own, because a bare NOT is refused rather than evaluated.
            right = self.evaluate(node.right.child)
            merged = difference(left.postings, right.postings)
            notes += list(right.notes)
            notes.append(
                f"excluded {len(right.postings)} unit(s) containing "
                f"{', '.join(right.matched_terms) or node.right.child.text}"
            )
            # the excluded terms are deliberately *not* reported as matched:
            # no returned unit contains them
            matched = sorted(set(left.matched_terms))
        else:
            right = self.evaluate(node.right)
            merged = intersect([left.postings, right.postings])
            notes += list(right.notes)
            matched = sorted(set(left.matched_terms) | set(right.matched_terms))
        return BooleanResult(
            postings=merged,
            matched_terms=matched,
            missing_terms=sorted(set(left.missing_terms) | set(right.missing_terms)),
            phrase_units=left.phrase_units | right.phrase_units,
            notes=notes,
        )

    def _evaluate_or(self, node: Or) -> BooleanResult:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        return BooleanResult(
            postings=union([left.postings, right.postings]),
            matched_terms=sorted(set(left.matched_terms) | set(right.matched_terms)),
            missing_terms=sorted(set(left.missing_terms) | set(right.missing_terms)),
            phrase_units=left.phrase_units | right.phrase_units,
            notes=list(left.notes) + list(right.notes),
        )

    # ------------------------------------------------------------------
    # explicit, testable entry points (Section 20-22 of the assignment)
    # ------------------------------------------------------------------
    def search_and(self, terms: Sequence[str]) -> BooleanResult:
        if len(terms) < 2:
            raise QuerySyntaxError("AND needs at least two terms")
        lists: List[PostingList] = []
        matched: List[str] = []
        missing: List[str] = []
        for term in terms:
            single = self.evaluate(Term(text=term))
            matched.extend(single.matched_terms)
            missing.extend(single.missing_terms)
            lists.append(single.postings)
        return BooleanResult(
            postings=intersect(lists),
            matched_terms=sorted(set(matched)),
            missing_terms=sorted(set(missing)),
        )

    def search_or(self, terms: Sequence[str]) -> BooleanResult:
        if not terms:
            raise QuerySyntaxError("OR needs at least one term")
        lists: List[PostingList] = []
        matched: List[str] = []
        missing: List[str] = []
        for term in terms:
            single = self.evaluate(Term(text=term))
            matched.extend(single.matched_terms)
            missing.extend(single.missing_terms)
            lists.append(single.postings)
        return BooleanResult(
            postings=union(lists),
            matched_terms=sorted(set(matched)),
            missing_terms=sorted(set(missing)),
        )

    def search_not(self, include_terms: Sequence[str], exclude_terms: Sequence[str]) -> BooleanResult:
        """``A AND NOT B``: include terms must all be present, exclude terms removed."""
        if not include_terms:
            raise QuerySyntaxError("A AND NOT B needs at least one include term")
        if not exclude_terms:
            raise QuerySyntaxError("A AND NOT B needs at least one exclude term")
        return self.evaluate(
            And(
                left=self._as_expression(include_terms),
                right=Not(child=self._as_expression(exclude_terms)),
            )
        )

    def _as_expression(self, terms: Sequence[str]) -> Expression:
        node: Expression = Term(text=terms[0])
        for term in terms[1:]:
            node = And(left=node, right=Term(text=term))
        return node
