"""Shared measurement helpers for every Phase 2 experiment.

The point of this module is comparability: vocabulary size, token count,
document frequency and "percentage reduction" must be computed *identically*
for every method, otherwise the comparison tables are meaningless.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .load_corpus import Unit

RE_HAS_LETTER = re.compile(r"[A-Za-z]")


@dataclass
class TokenizedCorpus:
    """A tokenized view of the experiment sample, with provenance kept intact."""

    name: str
    units: List[Unit]
    tokens: List[List[str]]

    def __post_init__(self) -> None:
        if len(self.units) != len(self.tokens):
            raise ValueError("units and tokens must be the same length")

    # ---------------- flattened views ----------------
    @property
    def flat(self) -> List[str]:
        return [token for unit_tokens in self.tokens for token in unit_tokens]

    @property
    def lower_flat(self) -> List[str]:
        return [token.casefold() for token in self.flat]

    def lower_unit_tokens(self) -> List[List[str]]:
        return [[token.casefold() for token in unit_tokens] for unit_tokens in self.tokens]

    # ---------------- counts ----------------
    @property
    def total_tokens(self) -> int:
        return sum(len(unit_tokens) for unit_tokens in self.tokens)

    @property
    def unique_tokens(self) -> int:
        return len(set(self.flat))

    @property
    def vocabulary_size(self) -> int:
        """Distinct case-folded tokens that contain at least one letter.

        Pure numbers and punctuation are excluded from the *vocabulary* because
        they are unbounded in a numeric document and would make vocabulary
        comparison meaningless. They are still counted as tokens.
        """
        return len(word_like_vocabulary(self.lower_flat))

    @property
    def document_count(self) -> int:
        return len({unit.document_id for unit in self.units})

    @property
    def page_count(self) -> int:
        return len({unit.page_id for unit in self.units})

    def document_tokens(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for unit, unit_tokens in zip(self.units, self.tokens):
            grouped.setdefault(unit.document_id, []).extend(unit_tokens)
        return grouped

    def type_token_ratio(self) -> float:
        return round(self.unique_tokens / self.total_tokens, 6) if self.total_tokens else 0.0

    def avg_tokens_per_document(self) -> float:
        docs = self.document_count
        return round(self.total_tokens / docs, 2) if docs else 0.0

    def avg_tokens_per_unit(self) -> float:
        units = len(self.units)
        return round(self.total_tokens / units, 2) if units else 0.0

    def freq(self, lower: bool = True) -> Counter:
        return Counter(self.lower_flat if lower else self.flat)

    def doc_freq(self, lower: bool = True) -> Counter:
        """Document frequency: in how many documents does a token occur."""
        counter: Counter = Counter()
        for unit, unit_tokens in zip(self.units, self.lower_unit_tokens() if lower else self.tokens):
            counter.update(set(unit_tokens))
        return counter

    def first_occurrence(self, lower: bool = True) -> Dict[str, Tuple[str, int, str, str]]:
        """token -> (document_id, page_number, section_title, unit_id) of first hit."""
        seen: Dict[str, Tuple[str, int, str, str]] = {}
        for unit, unit_tokens in zip(self.units, self.lower_unit_tokens() if lower else self.tokens):
            for token in unit_tokens:
                if token not in seen:
                    seen[token] = (
                        unit.document_id,
                        unit.page_number,
                        unit.section_title,
                        unit.unit_id,
                    )
        return seen


def word_like_vocabulary(tokens: Iterable[str]) -> set:
    return {token for token in tokens if RE_HAS_LETTER.search(token)}


def build_tokenized(name: str, units: Sequence[Unit], tokens: Sequence[Sequence[str]]) -> TokenizedCorpus:
    return TokenizedCorpus(name=name, units=list(units), tokens=[list(t) for t in tokens])


def percent_reduction(before: int, after: int) -> float:
    if not before:
        return 0.0
    return round(100.0 * (before - after) / before, 2)


def top_terms(counter: Counter, limit: int = 25, exclude: Optional[set] = None) -> List[Tuple[str, int]]:
    exclude = exclude or set()
    rows = [(term, count) for term, count in counter.most_common() if term not in exclude]
    return rows[:limit]


@dataclass
class VocabularyMetrics:
    """The metric block shared by every preprocessing comparison table."""

    stage: str
    method: str
    total_tokens: int
    unique_tokens: int
    vocabulary_size: int
    documents: int
    units: int
    avg_tokens_per_document: float
    extra: Dict[str, object] = field(default_factory=dict)

    def as_row(self, baseline_total: Optional[int] = None, baseline_vocab: Optional[int] = None) -> Dict[str, object]:
        row: Dict[str, object] = {
            "stage": self.stage,
            "method": self.method,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "vocabulary_size": self.vocabulary_size,
            "type_token_ratio": round(self.unique_tokens / self.total_tokens, 6) if self.total_tokens else 0.0,
            "documents": self.documents,
            "units": self.units,
            "avg_tokens_per_document": self.avg_tokens_per_document,
            "avg_tokens_per_unit": round(self.total_tokens / self.units, 2) if self.units else 0.0,
        }
        if baseline_total:
            row["token_change_vs_baseline_percent"] = percent_reduction(baseline_total, self.total_tokens)
        if baseline_vocab:
            row["vocabulary_change_vs_baseline_percent"] = percent_reduction(baseline_vocab, self.vocabulary_size)
        row.update(self.extra)
        return row


def vocabulary_metrics(stage: str, method: str, tokenized: TokenizedCorpus, **extra) -> VocabularyMetrics:
    return VocabularyMetrics(
        stage=stage,
        method=method,
        total_tokens=tokenized.total_tokens,
        unique_tokens=tokenized.unique_tokens,
        vocabulary_size=tokenized.vocabulary_size,
        documents=tokenized.document_count,
        units=len(tokenized.units),
        avg_tokens_per_document=tokenized.avg_tokens_per_document(),
        extra=extra,
    )
