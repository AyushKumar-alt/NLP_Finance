"""The retrieval engine: one API over keyword, phrase and Boolean search.

Ranking is deterministic and documented in ``config/phase3_config.yaml``::

    score = term_weight * (number of distinct query terms matched)
          + phrase_bonus (1 when the query phrase occurs adjacently in the unit)
          + tf_log_weight * log10(1 + total matched term frequency)

Ties are broken by ``unit_id`` ascending, so repeated runs produce identical
result tables. No claim is made that this ranking is better than another one -
Phase 4 evaluates the retrieval quality.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from src.phase2.load_corpus import Corpus, Unit

from .boolean_search import BooleanResult, BooleanSearch
from .config import Phase3Config, log_event
from .inverted_index import InvertedIndex
from .keyword_search import KeywordSearch, SearchOutcome, UnitHit
from .phrase_search import PhraseSearch
from .pipeline_runner import PipelineRunner
from .pipelines import PipelineSpec
from .posting_list import PostingList
from .query_parser import (
    Expression,
    QuerySyntaxError,
    describe_expression,
    parse_query,
    query_type_of,
)


@dataclass
class RetrievalEngine:
    """Search API bound to one pipeline's index."""

    index: InvertedIndex
    spec: PipelineSpec
    runner: PipelineRunner
    config: Phase3Config
    units: Dict[str, Unit] = field(default_factory=dict)
    logger: Optional[logging.Logger] = None

    def __post_init__(self) -> None:
        join_with = str(self.config.get("ngrams.join_with", "_"))
        self._stream_cache: Dict[str, List[str]] = {}
        self.keyword_search = KeywordSearch(self.index, self.spec, self.runner, self.logger)
        self.phrase_search = PhraseSearch(
            self.index, self.spec, self.runner, join_with=join_with,
            stream_provider=self._stream_for_unit, logger=self.logger,
        )
        self.boolean_search = BooleanSearch(
            self.index, self.spec, self.runner, join_with=join_with,
            enable_phrase=bool(self.config.get("retrieval.enable_phrase_search", True)),
            phrase_search=self.phrase_search, logger=self.logger,
        )

    # ------------------------------------------------------------------
    def _stream_for_unit(self, unit_id: str) -> List[str]:
        """Stopword-preserving term stream of one unit, cached per run."""
        stream = self._stream_cache.get(unit_id)
        if stream is None:
            unit = self.units.get(unit_id)
            stream = (
                self.runner.phrase_stream(self.spec, unit.text) if unit is not None else []
            )
            self._stream_cache[unit_id] = stream
        return stream

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _score(self, matched_count: int, phrase_match: bool, total_tf: int) -> float:
        ranking = self.config.section("retrieval").get("ranking", {}) or {}
        term_weight = float(ranking.get("term_weight", 1.0))
        phrase_bonus = float(ranking.get("phrase_bonus", 2.0))
        tf_weight = float(ranking.get("tf_log_weight", 0.25))
        score = term_weight * matched_count
        if phrase_match:
            score += phrase_bonus
        score += tf_weight * math.log10(1 + max(0, total_tf))
        # rounded *before* ranking so that the stored score and the sort key are
        # the same number: equal displayed scores must be real ties
        return round(score, 4)

    # ------------------------------------------------------------------
    # snippet construction
    # ------------------------------------------------------------------
    def snippet_for(self, unit: Unit, matched_terms: Sequence[str], phrase_terms: Sequence[str]) -> str:
        """Human readable excerpt around the matched terms.

        The surface forms are taken from the pipeline token stream of the unit,
        so a lemma or stem match (``inflate`` -> ``inflation``) still produces a
        snippet that contains the word the reader sees in the document.
        """
        snippet_chars = int(self.config.get("retrieval.snippet_chars", 220))
        context = int(self.config.get("retrieval.snippet_context_chars", 90))
        text = " ".join(unit.text.split())
        if not text:
            return ""
        wanted = {term.casefold() for term in matched_terms} | {t.casefold() for t in phrase_terms}
        if not wanted:
            return text[:snippet_chars].strip()

        surfaces: List[str] = []
        for token in self.runner.tokenize_text(self.spec, unit.text):
            term = self.runner.term_for(self.spec, token)
            if term in wanted and token not in surfaces:
                surfaces.append(token)

        lowered = text.casefold()
        positions = [lowered.find(surface.casefold()) for surface in surfaces]
        positions = [p for p in positions if p >= 0]
        if not positions:
            return text[:snippet_chars].strip()

        centre = min(positions, key=lambda p: abs(p - min(positions)))
        start = max(0, centre - context)
        end = min(len(text), centre + snippet_chars)
        excerpt = text[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
        return excerpt

    # ------------------------------------------------------------------
    # hit construction
    # ------------------------------------------------------------------
    def _hits_from_postings(
        self,
        postings: PostingList,
        matched_terms: Sequence[str],
        missing_terms: Sequence[str],
        phrase_units: Optional[Set[str]] = None,
        query_terms: Optional[Sequence[str]] = None,
    ) -> List[UnitHit]:
        phrase_units = phrase_units or set()
        max_results = int(self.config.get("retrieval.max_results_per_query", 50))
        query_terms = list(query_terms or matched_terms)

        hits: List[UnitHit] = []
        for posting in postings:
            record = self.index.unit(posting.unit_id)
            unit = self.units.get(posting.unit_id)
            if record is None:
                continue
            per_term: Dict[str, int] = {}
            positions: Dict[str, List[int]] = {}
            for term in matched_terms:
                term_posting = self.index.posting_list(term).posting_for(posting.unit_id)
                if term_posting is not None:
                    per_term[term] = term_posting.tf
                    positions[term] = list(term_posting.positions)
            matched_here = sorted(per_term)
            if not matched_here:
                continue
            phrase_hit = posting.unit_id in phrase_units
            snippet = self.snapshot(unit, matched_here, query_terms) if unit else ""
            hits.append(
                UnitHit(
                    unit_id=posting.unit_id,
                    document_id=record.document_id,
                    source_id=record.source_id,
                    page_number=record.page_number,
                    section_id=record.section_id,
                    section_number=record.section_number,
                    section_title=record.section_title,
                    unit_type=record.unit_type,
                    score=self._score(len(matched_here), phrase_hit, sum(per_term.values())),
                    matched_terms=matched_here,
                    matched_positions=positions,
                    matched_term_frequency=per_term,
                    snippet=snippet,
                    phrase_match=phrase_hit,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.unit_id))
        if max_results and len(hits) > max_results:
            hits = hits[:max_results]
        for position, hit in enumerate(hits, start=1):
            hit.rank = position
        return hits

    def snapshot(self, unit: Unit, matched_terms: Sequence[str], query_terms: Sequence[str]) -> str:
        return self.snippet_for(unit, matched_terms, query_terms)

    # ------------------------------------------------------------------
    # public API (Section 31 of the assignment)
    # ------------------------------------------------------------------
    def search_keyword(self, query: str) -> SearchOutcome:
        started = time.perf_counter()
        normalized = self.runner.normalize_query(self.spec, query)
        missing = [term for term in normalized if not self.index.has_term(term)]
        matched = [term for term in normalized if self.index.has_term(term)]
        if not matched:
            return SearchOutcome(
                query=query, query_type="keyword", matched_terms=[], missing_terms=missing,
                unit_hits=[], execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
                pipeline=self.spec.key, retrieval_method="keyword",
            )
        postings = self.index.posting_list(matched[0])
        hits = self._hits_from_postings(postings, matched, missing, set(), normalized)
        return SearchOutcome(
            query=query, query_type="keyword", matched_terms=matched, missing_terms=missing,
            unit_hits=hits,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            pipeline=self.spec.key, retrieval_method="keyword",
        )

    def search_phrase(self, phrase: str) -> SearchOutcome:
        started = time.perf_counter()
        found = self.phrase_search.search(phrase)
        matched = list(found["indexed_terms"])
        missing = list(found["missing"])
        unit_ids = list(found["unit_ids"])
        postings = self.boolean_search._postings_for_units(phrase, unit_ids, is_phrase=True)
        hits = self._hits_from_postings(postings, matched, missing, set(unit_ids), matched)
        return SearchOutcome(
            query=phrase, query_type="phrase", matched_terms=matched, missing_terms=missing,
            unit_hits=hits,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            pipeline=self.spec.key, retrieval_method="phrase",
        )

    def search_boolean_and(self, terms: Sequence[str]) -> SearchOutcome:
        started = time.perf_counter()
        result = self.boolean_search.search_and(terms)
        hits = self._hits_from_postings(
            result.postings, result.matched_terms, result.missing_terms, result.phrase_units
        )
        return SearchOutcome(
            query=" AND ".join(terms), query_type="boolean_and",
            matched_terms=result.matched_terms, missing_terms=result.missing_terms,
            unit_hits=hits,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            pipeline=self.spec.key, retrieval_method="boolean_and",
        )

    def search_boolean_or(self, terms: Sequence[str]) -> SearchOutcome:
        started = time.perf_counter()
        result = self.boolean_search.search_or(terms)
        hits = self._hits_from_postings(
            result.postings, result.matched_terms, result.missing_terms, result.phrase_units
        )
        return SearchOutcome(
            query=" OR ".join(terms), query_type="boolean_or",
            matched_terms=result.matched_terms, missing_terms=result.missing_terms,
            unit_hits=hits,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            pipeline=self.spec.key, retrieval_method="boolean_or",
        )

    def search_boolean_not(self, include_terms: Sequence[str], exclude_terms: Sequence[str]) -> SearchOutcome:
        started = time.perf_counter()
        result = self.boolean_search.search_not(include_terms, exclude_terms)
        hits = self._hits_from_postings(
            result.postings, result.matched_terms, result.missing_terms, result.phrase_units
        )
        return SearchOutcome(
            query=" AND ".join(include_terms) + " AND NOT " + " AND ".join(exclude_terms),
            query_type="boolean_not",
            matched_terms=result.matched_terms, missing_terms=result.missing_terms,
            unit_hits=hits,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            pipeline=self.spec.key, retrieval_method="boolean_not",
        )

    def search(self, query: str, query_type: Optional[str] = None) -> SearchOutcome:
        """Route to the right method; ``query_type`` defaults to auto-detection."""
        resolved = query_type or query_type_of(query)
        text = (query or "").strip()
        try:
            if resolved == "keyword":
                return self.search_keyword(text)
            if resolved == "phrase":
                return self.search_phrase(text)
            if resolved == "boolean_and":
                return self._search_expression(text, "boolean_and")
            if resolved == "boolean_or":
                return self._search_expression(text, "boolean_or")
            if resolved == "boolean_not":
                return self._search_expression(text, "boolean_not")
            if resolved == "boolean_group":
                return self._search_expression(text, "boolean_group")
        except QuerySyntaxError as error:
            if self.logger is not None:
                log_event(self.logger, "ERROR", "search", f"malformed query {query!r}: {error}")
            raise
        raise ValueError(f"unknown query type '{resolved}' for query {query!r}")

    def _search_expression(self, query: str, query_type: str) -> SearchOutcome:
        started = time.perf_counter()
        expression: Expression = parse_query(
            query,
            allow_parentheses=bool(self.config.get("retrieval.enable_parentheses", True)),
            allow_not=bool(self.config.get("retrieval.enable_boolean", True)),
        )
        result: BooleanResult = self.boolean_search.evaluate(expression)
        hits = self._hits_from_postings(
            result.postings, result.matched_terms, result.missing_terms, result.phrase_units
        )
        return SearchOutcome(
            query=describe_expression(expression), query_type=query_type,
            matched_terms=result.matched_terms, missing_terms=result.missing_terms,
            unit_hits=hits,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            pipeline=self.spec.key, retrieval_method=query_type,
        )


def build_engine(
    index: InvertedIndex,
    spec: PipelineSpec,
    runner: PipelineRunner,
    config: Phase3Config,
    corpus: Optional[Corpus] = None,
    units: Optional[Sequence[Unit]] = None,
    logger: Optional[logging.Logger] = None,
) -> RetrievalEngine:
    """Create a retrieval engine, caching unit texts for snippet building."""
    if units is None and corpus is not None:
        units = corpus.selected_units()
    unit_map = {unit.unit_id: unit for unit in (units or [])}
    return RetrievalEngine(
        index=index, spec=spec, runner=runner, config=config, units=unit_map, logger=logger
    )
