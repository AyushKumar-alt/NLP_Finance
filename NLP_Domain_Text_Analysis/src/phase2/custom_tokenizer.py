"""Custom financial / economic domain tokenizer.

Why a custom tokenizer is needed
--------------------------------
A generic word tokenizer treats punctuation as an unconditional delimiter, which
destroys the expressions that carry most of the meaning in economic writing:

    "The repo rate was 6.50% in FY2025-26."
    NLTK   -> ['The','repo','rate','was','6.50','%','in','FY2025-26','.']   ('%' orphaned)
    "Rs 1.25 lakh crore"
    NLTK   -> ['Rs','1.25','lakh','crore']   (unit detached from magnitude)
    "year-on-year"
    NLTK   -> ['year-on-year']  (ok) but spaCy/NLTK differ on whether the hyphen
             survives, so retrieval cannot rely on it.

This module therefore implements an *ordered, inspectable rule set*. The rules
are applied left-to-right, longest-pattern-first, and every rule is exported to
``results/phase2/tokenization/custom_tokenizer_rules.csv`` so the decisions are
auditable.

Explicit design decisions (justified against the Phase 1 corpus):

* ``7.4%``  -> ``['7.4%']`` (one token). Rationale: a percentage is a single
  measurement; splitting it orphans ``%`` and makes ``7.4`` indistinguishable
  from a plain growth rate figure. A *separate* experiment
  (``date_number_tokenizer.py``) deliberately splits percentages to quantify
  the trade-off, so both representations are available to Phase 3.
* ``FY2025-26`` -> ``['FY2025-26']``. Rationale: the fiscal-year token is the
  primary temporal index in Indian economic documents; fragmenting it into
  ``FY / 2025 / - / 26`` makes exact-match fiscal-year queries impossible.
* ``₹50,000`` / ``Rs. 50,000`` / ``US$ 2.5`` -> single token, because the
  currency symbol changes the meaning of the magnitude.
* Hyphenated economic terms (``year-on-year``, ``repo-rate``, ``poverty-line``)
  are kept whole, but ``-`` used as a *range* separator inside numbers
  (``2024-25``, ``2019-20``) is likewise kept.
* Dotted capitalised abbreviations (``B.Tech``, ``M.Sc.``, ``Jan.``) are kept
  whole; a trailing full stop that is really sentence punctuation is not
  absorbed.
* All-caps acronyms of 2-6 letters (``GDP``, ``RBI``, ``IMF``) are kept as one
  token, which is what a domain reader expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

# ----------------------------------------------------------------------
# Rule table
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One transparent tokenizer rule.

    ``pattern`` is a Python regular expression. ``group`` names the capture
    group holding the token when the rule fires.
    """

    rule_id: str
    pattern: str
    description: str
    example: str
    expected_behavior: str
    group: int = 0
    priority: float = 0.0

    @property
    def compiled(self) -> re.Pattern:
        return _compile(self.rule_id, self.pattern)


_COMPILE_CACHE: Dict[str, re.Pattern] = {}


def _compile(rule_id: str, pattern: str) -> re.Pattern:
    cached = _COMPILE_CACHE.get(rule_id)
    if cached is None:
        cached = re.compile(pattern, re.UNICODE)
        _COMPILE_CACHE[rule_id] = cached
    return cached


# Ordered by priority: higher priority = matched earlier.
RULES: Tuple[Rule, ...] = (
    # ---------------- magnitudes with scale words ----------------
    Rule(
        rule_id="R01",
        pattern=(
            r"\b\d{1,3}(?:,\d{2,3})++(?:\.\d++)?\s*+"
            r"(?:lakh\s+crore|trillion|billion|million|crore|lakh|thousand|bn|mn)\b"
        ),
        description="Indian-style grouped magnitude with a scale word (e.g. '1,25,000 crore', '5 lakh crore')",
        example="1,25,000 crore",
        expected_behavior="kept as a single magnitude token",
        priority=100.0,
    ),
    Rule(
        rule_id="R02",
        pattern=(
            r"(?:\u20b9|US\s?\$|US\$|Rs\.?|rs\.?|INR|inr|USD|usd|\$|\u20ac|EUR|eur|GBP|gbp|\u00a3)\s*+"
            r"\d++[\d,]*+(?:\.\d++)?\s*+"
            r"(?:lakh\s+crore|trillion|billion|million|crore|lakh|bn|mn)?"
        ),
        description="currency-marked amount (symbol + number + optional scale word)",
        example="\u20b951,000 crore",
        expected_behavior="kept whole - the currency symbol changes the meaning of the magnitude",
        priority=99.0,
    ),
    Rule(
        rule_id="R03",
        pattern=(
            r"\b\d++[\d,]*+(?:\.\d++)?\s*+"
            r"(?:lakh\s+crore|trillion|billion|million|crore|lakh|thousand|bn|mn)\b"
        ),
        description="bare magnitude with an English/Indian scale word",
        example="2.5 billion",
        expected_behavior="kept whole so the unit stays attached to the number",
        priority=98.0,
    ),
    # ---------------- percentages ----------------
    Rule(
        rule_id="R04",
        pattern=r"\b\d++[\d,]*+(?:\.\d++)?\s*+(?:per\s+cent|percent|pct\.?|basis\s+points?|bps)\b",
        description="percentage written out as words",
        example="7.4 per cent",
        expected_behavior="kept whole",
        priority=97.0,
    ),
    Rule(
        rule_id="R05",
        pattern=r"[+-]?\d++[\d,]*+(?:\.\d++)?\s*+%",
        description="symbol percentage (sign, grouped digits, decimals)",
        example="7.4%",
        expected_behavior="kept whole - see module docstring for the justification",
        priority=96.0,
    ),
    # ---------------- fiscal years ----------------
    Rule(
        rule_id="R06",
        pattern=(
            r"\bFY\s*+\d{2,4}\s*+[-/\u2013\u2014]\s*+\d{2,4}+\b"
            r"|\bFY\s*+\d{2,4}+\b"
            r"|\b\d{4}\s*+[-/\u2013\u2014]\s*+\d{2,4}+\b"
        ),
        description="fiscal year / financial-year range with or without the 'FY' prefix",
        example="FY2025-26",
        expected_behavior="kept whole - the fiscal year is the primary temporal index",
        priority=95.0,
    ),
    Rule(
        rule_id="R07",
        pattern=(
            r"\bQ[1-4]\s*+(?:FY\s*+\d{2,4}(?:\s*+[-/\u2013\u2014]\s*+\d{2,4}+)?|'?\d{2,4}(?:\s*+[-/\u2013]\s*+\d{2,4}+)?)?\b"
            r"|\b[1-4](?:st|nd|rd|th)\s+quarter\b"
        ),
        description="fiscal quarter reference",
        example="Q1 FY26",
        expected_behavior="kept whole",
        priority=94.0,
    ),
    # ---------------- dates ----------------
    Rule(
        rule_id="R08",
        pattern=(
            r"\b\d{1,2}\s*+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[a-z]*\.?\s*+\d{4}+\b"
        ),
        description="day-month-year date (numeric day first)",
        example="31 March 2026",
        expected_behavior="kept whole",
        priority=93.0,
    ),
    Rule(
        rule_id="R09",
        pattern=(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}+\b"
            r"|\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}+\b"
            r"|\b\d{4}\s*/\s*\d{2,4}+\b"
        ),
        description="month-year, day/month/year and year/slash-year forms",
        example="January 2026",
        expected_behavior="kept whole",
        priority=92.0,
    ),
    # ---------------- dotted abbreviations ----------------
    Rule(
        rule_id="R10",
        pattern=(
            r"\b(?:[A-Za-z]\.){2,}+[A-Za-z]{1,6}+\b"
            r"|\b[A-Za-z]{1,6}+\.(?:[A-Za-z]\.){1,}+[A-Za-z]{0,6}\b"
            r"|\b[A-Za-z]{1,4}+\.(?:[A-Za-z]{1,4}+\.){1,3}+(?![\w])"
            r"|\b[A-Za-z]\.[A-Za-z]{1,6}+\b"
        ),
        description="dotted abbreviation / qualification (e.g. 'B.Tech', 'M.Sc.', 'M. finance')",
        example="B.Tech",
        expected_behavior="kept whole - the dots are part of the abbreviation, not sentence punctuation",
        priority=91.0,
    ),
    # ---------------- acronyms ----------------
    Rule(
        rule_id="R11",
        pattern=r"\b[A-Z]{2,6}+\b",
        description="all-caps domain acronym (GDP, GVA, CPI, WPI, FDI, RBI, SEBI, IMF, NPA, GST)",
        example="GDP",
        expected_behavior="kept as one acronym token",
        priority=90.0,
    ),
    # ---------------- alphanumerics and coded terms ----------------
    Rule(
        rule_id="R12",
        pattern=(
            r"\b[A-Za-z]{2,}+\s*+[-/\u2013]\s*+\d++[A-Za-z0-9\-]*+\b"   # COVID-19
            r"|\b[A-Z]{2,}+\d++[A-Za-z0-9\-]*+\b"                        # FY26, COVID19
            r"|\b[A-Za-z]+\d++[A-Za-z0-9]*+\b"                           # GSTIN, M4, Covid19
        ),
        description="alphanumeric code / coined term containing digits",
        example="COVID-19",
        expected_behavior="kept whole",
        priority=90.5,
    ),
    # ---------------- hyphenated financial terms ----------------
    Rule(
        rule_id="R13",
        pattern=r"\b[A-Za-z]{2,}+(?:[-/][A-Za-z]{2,}+)+\b",
        description="hyphenated or slashed economic term (year-on-year, repo-rate, mark-to-market, FDI/FPI)",
        example="year-on-year",
        expected_behavior="kept whole - splitting it makes a domain query unmatchable",
        priority=88.0,
    ),
    # ---------------- plain numbers ----------------
    Rule(
        rule_id="R14",
        pattern=r"[+-]?\d++[\d,]*+(?:\.\d++)?",
        description="plain number, including Indian digit grouping and decimals",
        example="1,25,000",
        expected_behavior="kept whole - digit grouping is not a token boundary",
        priority=87.0,
    ),
    # ---------------- contractions ----------------
    Rule(
        rule_id="R15",
        pattern=r"\b[A-Za-z]+['\u2019](?:s|t|re|ve|ll|d|m)\b",
        description="English contraction",
        example="bank's",
        expected_behavior="kept whole",
        priority=86.0,
    ),
    # ---------------- words ----------------
    Rule(
        rule_id="R16",
        pattern=r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*+",
        description="ordinary alphabetic word",
        example="growth",
        expected_behavior="emitted as a word token",
        priority=85.0,
    ),
    # ---------------- punctuation ----------------
    Rule(
        rule_id="R17",
        pattern=r"[^\sA-Za-z0-9]",
        description="standalone punctuation or symbol (kept because the POS/NER experiments need it)",
        example=".",
        expected_behavior="emitted as a single punctuation token",
        priority=84.0,
    ),
)

RULE_BY_ID: Dict[str, Rule] = {rule.rule_id: rule for rule in RULES}


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
#: The master pattern is the union of all rules, longest-first by construction.
#: Sorting the alternatives by descending source length guarantees that
#: '1,25,000 crore' is preferred over the bare-number rule.
_MASTER_RE = re.compile(
    "|".join(
        f"(?P<{rule.rule_id}>{rule.pattern})"
        for rule in sorted(RULES, key=lambda r: (-r.priority, -len(r.pattern)))
    ),
    re.UNICODE,
)


def preprocess(text: str) -> str:
    """Normalise the surface form before rule matching.

    * U+FFFD (unmapped publisher glyph) and other non-printable characters --
      including the newlines Phase 1 uses to separate table cells -- become
      spaces, so a token can never span a cell boundary.
    * Typographic quotes and dashes are folded to ASCII so one pattern set is
      enough for both spellings.
    """
    text = text.replace("\ufffd", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\u00a0", " ")
    return "".join(c if c.isprintable() else " " for c in text)


def custom_tokenize(text: str, keep_punctuation: bool = True) -> List[str]:
    """Tokenize ``text`` with the financial-domain rule set.

    Parameters
    ----------
    keep_punctuation:
        When ``False`` standalone punctuation tokens are dropped. The default is
        ``True`` because Phase 2 POS / NER experiments need the punctuation.
    """
    if not text:
        return []
    surface = preprocess(text)
    tokens: List[str] = []
    for match in _MASTER_RE.finditer(surface):
        rule_id = match.lastgroup
        rule = RULE_BY_ID.get(rule_id or "")
        if rule is None:  # pragma: no cover - defensive
            continue
        value = match.group(rule.group).strip()
        if not value:
            continue
        if not keep_punctuation and rule.rule_id == "R17":
            continue
        tokens.append(value)
    return tokens


def custom_tokenize_batch(texts: Sequence[str], keep_punctuation: bool = True) -> List[List[str]]:
    return [custom_tokenize(t, keep_punctuation=keep_punctuation) for t in texts]


@lru_cache(maxsize=200_000)
def _cached_tokenize(text: str, keep_punctuation: bool) -> str:
    return "\t".join(custom_tokenize(text, keep_punctuation=keep_punctuation))


def custom_tokenize_cached(text: str, keep_punctuation: bool = True) -> List[str]:
    """Cached variant: Phase 1 emits many identical short strings (table headers)."""
    return _cached_tokenize(text, keep_punctuation).split("\t") if text else []


# ----------------------------------------------------------------------
# Rule introspection helpers
# ----------------------------------------------------------------------
def rule_rows() -> List[Dict[str, str]]:
    """Rows for ``custom_tokenizer_rules.csv``."""
    rows = []
    for rule in sorted(RULES, key=lambda r: -r.priority):
        rows.append(
            {
                "rule_id": rule.rule_id,
                "priority": rule.priority,
                "pattern": rule.pattern,
                "description": rule.description,
                "example": rule.example,
                "expected_behavior": rule.expected_behavior,
                "actual_behavior": " -> ".join(custom_tokenize(rule.example)),
            }
        )
    return rows


def explain(text: str) -> List[Dict[str, str]]:
    """Return, for every token in ``text``, which rule produced it.

    Used by tests and by the rule-verification block of the report.
    """
    surface = preprocess(text)
    out: List[Dict[str, str]] = []
    for match in _MASTER_RE.finditer(surface):
        rule_id = match.lastgroup or ""
        out.append(
            {
                "token": match.group().strip(),
                "rule_id": rule_id,
                "description": RULE_BY_ID[rule_id].description if rule_id in RULE_BY_ID else "",
            }
        )
    return out
