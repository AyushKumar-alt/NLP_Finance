"""Date and number aware tokenization for financial text.

Why this module exists
----------------------
In economic writing, temporal and quantitative expressions carry the meaning::

    "Real GDP growth is projected at 7.4% in FY2025-26, against 6.5% in 2024-25."
    "The outstanding amount is Rs 1.25 lakh crore as on 31 March 2026."
    "Disinflation of 10.5 per cent y-o-y in Q1 FY26."

A generic tokenizer scatters such expressions into fragments that no later stage
can reliably recombine, and a retrieval query written by a domain user will
never match ``['FY2025', '-', '26']`` against a document that says ``FY2025-26``.

This module therefore provides a *deliberately split* representation, the
complement of ``custom_tokenizer.py``:

* ``custom_tokenizer`` keeps the expression **whole** (good for exact-match
  indexing of ``FY2025-26``).
* ``date_number_tokenizer`` splits it into typed components (good for numeric
  aggregation, trend analysis and Phase 3 numeric filtering):

      "7.4%"            -> ['7.4', '%',  type=percentage]
      "Rs 1.25 lakh crore" -> ['Rs', '1.25', 'lakh', 'crore', type=currency_amount]
      "FY2025-26"       -> ['FY', '2025-26', type=fiscal_year]
      "31 March 2026"   -> ['31', 'March', '2026', type=date]
      "Q1 FY26"         -> ['Q1', 'FY26', type=quarter]

Both representations are measured against the same corpus in
``date_number_comparison.csv`` so the trade-off is quantified rather than
asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .custom_tokenizer import preprocess

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
MONTH_RE = f"(?:{MONTHS})[a-z]*"

CURRENCY = r"(?:\u20b9|US\s?\$|US\$|Rs\.?|INR|USD|\$|\u20ac|EUR|GBP|\u00a3)"
SCALE = r"(?:lakh\s+crore|trillion|billion|million|crore|lakh|thousand|bn|mn)"

#: Typed expression patterns, ordered by specificity.
EXPRESSION_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
    (
        "fiscal_year_range",
        r"\bFY\s*\d{2,4}\s*[-/\u2013\u2014]\s*\d{2,4}\b",
        "fiscal year range with FY prefix (FY2025-26)",
    ),
    (
        "fiscal_year",
        r"\bFY\s*\d{2,4}\b",
        "fiscal year with FY prefix (FY26, FY 2025)",
    ),
    (
        "year_range",
        r"\b(?:19|20)\d{2}\s*[-/\u2013\u2014]\s*\d{2,4}\b",
        "bare year range (2024-25)",
    ),
    (
        "quarter",
        r"\bQ[1-4]\s*(?:FY\s*\d{2,4}|'?\d{2,4})?|\b[1-4](?:st|nd|rd|th)\s+quarter\b",
        "fiscal quarter (Q1 FY26)",
    ),
    (
        "day_month_year",
        rf"\b\d{{1,2}}\s*(?:of\s+)?{MONTH_RE}\.?\s*\d{{4}}\b",
        "numeric day-month-year date (31 March 2026)",
    ),
    (
        "month_year",
        rf"\b{MONTH_RE}\.?\s*\d{{4}}\b",
        "month and year (January 2026)",
    ),
    (
        "day_month",
        rf"\b\d{{1,2}}\s+{MONTH_RE}\b",
        "day and month (31 March)",
    ),
    (
        "slash_date",
        r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b|\b(?:19|20)\d{2}\s*/\s*\d{2,4}\b",
        "numerically written date (31/03/2026, 2025/26)",
    ),
    (
        "year",
        r"\b(?:19|20)\d{2}\b",
        "bare calendar year (2025)",
    ),
    (
        "percentage",
        r"[+-]?\d[\d,]*(?:\.\d+)?\s*(?:per\s+cent|percent|pct\.?|%)",
        "percentage, symbol or written out (7.4%, 7.4 per cent)",
    ),
    (
        "currency_amount",
        rf"{CURRENCY}\s*\d[\d,]*(?:\.\d+)?\s*(?:{SCALE})?",
        "currency-marked amount (Rs 1.25 lakh crore, US$ 2.5 billion)",
    ),
    (
        "scaled_magnitude",
        rf"\b\d[\d,]*(?:\.\d+)?\s+{SCALE}\b",
        "magnitude with a scale word (5 lakh crore, 3.2 million)",
    ),
    (
        "grouped_number",
        r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b",
        "digit-grouped number (1,25,000)",
    ),
    (
        "decimal_number",
        r"[+-]?\d+\.\d+",
        "decimal number (7.4)",
    ),
    (
        "integer",
        r"[+-]?\d+",
        "integer (250)",
    ),
)

_COMPILED: List[Tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE)) for name, pattern, _ in EXPRESSION_PATTERNS
]

#: The ordered master used when splitting an expression into typed components.
_SPLIT_RULES: Tuple[Tuple[str, str, str], ...] = (
    (
        "percentage",
        r"([+-]?\d[\d,]*(?:\.\d+)?)\s*+((?:per\s+cent|percent|pct\.?|%))",
        "value and unit separated",
    ),
    (
        "currency_amount",
        rf"({CURRENCY})\s*+(\d[\d,]*(?:\.\d+)?)\s*+({SCALE})?",
        "currency symbol, value and scale separated",
    ),
    (
        "scaled_magnitude",
        rf"(\d[\d,]*(?:\.\d+)?)\s+({SCALE})",
        "value and scale word separated",
    ),
    (
        "fiscal_year_range",
        r"(FY)\s*+(\d{2,4})\s*+[-/\u2013\u2014]\s*+(\d{2,4})",
        "prefix and year range separated",
    ),
    (
        "fiscal_year",
        r"(FY)\s*+(\d{2,4})",
        "prefix and year separated",
    ),
    (
        "quarter",
        r"(Q[1-4])\s*+((?:FY\s*+\d{2,4}|'?\d{2,4}))?",
        "quarter and fiscal year separated",
    ),
    (
        "day_month_year",
        rf"(\d{{1,2}})\s*+(?:of\s+)?({MONTH_RE})\.?\s*+(\d{{4}})",
        "day, month and year separated",
    ),
    (
        "month_year",
        rf"({MONTH_RE})\.?\s+(\d{{4}})",
        "month and year separated",
    ),
    (
        "year_range",
        r"((?:19|20)\d{2})\s*+[-/\u2013\u2014]\s*+(\d{2,4})",
        "year range split into both years",
    ),
    (
        "number",
        r"[+-]?\d[\d,]*(?:\.\d+)?",
        "value emitted as one token",
    ),
    (
        "word",
        r"[A-Za-z][A-Za-z'\-/]*",
        "alphabetic token",
    ),
    (
        "punct",
        r"[^\sA-Za-z0-9]",
        "punctuation",
    ),
)

@dataclass
class TypedToken:
    """A token plus the expression category it came from."""

    token: str
    category: str
    expression: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.token, self.category)


@dataclass
class DateNumberResult:
    typed_tokens: List[TypedToken] = field(default_factory=list)
    expressions: List[Dict[str, str]] = field(default_factory=list)

    @property
    def tokens(self) -> List[str]:
        return [t.token for t in self.typed_tokens]

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for typed in self.typed_tokens:
            if typed.category not in ("word", "punct"):
                counts[typed.category] = counts.get(typed.category, 0) + 1
        return counts

    def expression_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for expression in self.expressions:
            key = f"{expression['category']}_expressions"
            counts[key] = counts.get(key, 0) + 1
        return counts


_SPLIT_COMPILED: List[Tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE)) for name, pattern, _ in _SPLIT_RULES
]
_SPLIT_MASTER = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern, _ in _SPLIT_RULES), re.UNICODE
)

#: Group number of each rule's own named group inside ``_SPLIT_MASTER``. Needed
#: because ``match.groups()`` also returns the named group, which would
#: otherwise duplicate the whole expression next to its components.
_SPLIT_NAMED_GROUP: Dict[str, int] = {}
_group_counter = 1
for _name, _pattern, _ in _SPLIT_RULES:
    _SPLIT_NAMED_GROUP[_name] = _group_counter
    _group_counter += 1 + re.compile(_pattern).groups


def detect_expressions(text: str) -> List[Dict[str, str]]:
    """Detect typed date/number expressions without dropping the spans."""
    surface = preprocess(text)
    found: List[Dict[str, str]] = []
    taken: List[Tuple[int, int]] = []
    for name, pattern in _COMPILED:
        for match in pattern.finditer(surface):
            start, end = match.span()
            if any(start < e and end > s for s, e in taken):
                continue
            taken.append((start, end))
            found.append(
                {
                    "category": name,
                    "text": match.group().strip(),
                    "start": str(start),
                    "end": str(end),
                }
            )
    found.sort(key=lambda row: (int(row["start"]), -int(row["end"])))
    return found


def date_number_tokenize(text: str) -> DateNumberResult:
    """Tokenize with typed date/number handling, keeping expression context."""
    surface = preprocess(text)
    result = DateNumberResult()
    for name, pattern in _COMPILED:
        for match in pattern.finditer(surface):
            result.expressions.append(
                {
                    "category": name,
                    "text": match.group().strip(),
                    "start": str(match.start()),
                    "end": str(match.end()),
                }
            )

    for match in _SPLIT_MASTER.finditer(surface):
        rule = match.lastgroup or "word"
        expression = match.group()
        if not expression.strip():
            continue
        # Rules with capture groups emit their components ("Rs 1.25 lakh crore"
        # -> ['Rs', '1.25', 'lakh', 'crore']); rules without emit the whole match.
        named = _SPLIT_NAMED_GROUP.get(rule)
        parts = [
            g
            for index, g in enumerate(match.groups(), start=1)
            if index != named and g and g.strip()
        ]
        if not parts:
            parts = [expression]
        category = "word" if rule in ("word", "punct") else rule
        for part in parts:
            value = part.strip()
            if not value:
                continue
            result.typed_tokens.append(
                TypedToken(token=value, category=category, expression=expression.strip())
            )
    return result


def date_number_tokens(text: str) -> List[str]:
    return date_number_tokenize(text).tokens


# ----------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------
def expression_category_totals(texts: Sequence[str]) -> Dict[str, int]:
    """Aggregate typed expression counts across a corpus slice."""
    totals: Dict[str, int] = {}
    for text in texts:
        result = date_number_tokenize(text)
        for key, value in result.expression_counts().items():
            totals[key] = totals.get(key, 0) + value
        for key, value in result.category_counts().items():
            totals[f"tokens_{key}"] = totals.get(f"tokens_{key}", 0) + value
    return dict(sorted(totals.items()))


def pattern_rows() -> List[Dict[str, str]]:
    """Rows describing every expression pattern (for the report)."""
    rows: List[Dict[str, str]] = []
    for name, pattern, description in EXPRESSION_PATTERNS:
        rows.append(
            {
                "category": name,
                "pattern": pattern,
                "description": description,
            }
        )
    return rows
