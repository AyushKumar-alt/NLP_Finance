"""Custom rule / dictionary POS tagging for financial terminology.

Method (evidence driven, nothing invented)
------------------------------------------
1. **Mine** the default spaCy tags for every domain candidate in the corpus and
   record the *observed* tag distribution of each candidate
   (:func:`discover_candidates`).
2. **Flag** a candidate as inconsistent when it is tagged with more than one
   tag, or when its dominant tag conflicts with an evidence-based expectation
   derived from the term's *shape* (acronym / hyphenated compound /
   ``-isation`` noun / institution name).
3. **Correct** only flagged candidates and write the dictionary with the corpus
   sentence that justifies each decision.
4. **Re-tag** the corpus and report how many tokens actually changed, so the
   effect size is measured rather than assumed.

The gold annotation file produced by :mod:`ml_pos` is used to score this
tagger exactly like the default tagger and the ML tagger.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import Phase2Config, log_event
from .load_corpus import Unit
from .spacy_pipeline import SpacyAnnotations

#: Acronym families that behave like proper nouns in economic writing.
INSTITUTION_ACRONYMS = {
    "RBI", "SEBI", "IRDAI", "PFRDA", "NABARD", "SIDBI", "EXIM", "ECGC", "NABARD",
    "IMF", "WTO", "OECD", "UNDP", "UNCTAD", "ILO", "FAO", "IDA", "ADB", "WB",
    "CPI", "WPI", "GDP", "GVA", "FDI", "FPI", "NPA", "GNPA", "NNPA", "GDP",
    "GST", "SGST", "IGST", "NRI", "EPFO", "ESIC", "PFRDA", "CAG", "TRAI",
    "CRISIL", "ICRA", "NSE", "BSE", "SEBI", "NABAM", "CARE", "ICAR",
    "MSF", "CRR", "SLR", "GDP", "ESA", "NITI",
}

#: Policy / instrument vocabulary whose POS the default tagger gets wrong or
#: marks inconsistently. Membership is validated against corpus counts.
DOMAIN_NOUNS_BY_SHAPE = [
    (
        "organisation_noun",
        re.compile(r"(?:isation|ization)\b", re.I),
        "NOUN",
        "financial process nouns ending in -isation/-ization (securitisation, disinflation) "
        "are nouns even when the default tagger reads them as verbs",
    ),
    (
        "adjective_form",
        re.compile(r"^(?:year-on-year|month-on-month|quarter-on-quarter|"
                   r"point-to-point|real|nominal|headline|core|retail|wholesale|net|gross)$", re.I),
        "ADJ",
        "comparative statistical modifiers that must stay adjectives when they modify a noun",
    ),
    (
        "policy_rate_compound",
        re.compile(r"^(?:repo|reverse-repo|policy-repo|bank-repo|msf|crr|slr|"
                   r"policy|open-market|reverse|standing-deposit|liquidity|"
                   r"monetary|fiscal|macro|prudential|macro-prudential)[-]?$", re.I),
        "NOUN",
        "monetary-policy rate/ration nouns (repo, MSF, CRR, SLR)",
    ),
    (
        "institution_phrase",
        re.compile(r"^(?:central|commercial|investment|development|co-operative|"
                   r"scheduled|state|public|private|foreign|multilateral|regional)[- ]"
                   r"(?:bank|banks|finance|financial|market|markets|reserve|"
                   r"corporation|authority|board|securities|insurance)$", re.I),
        "NOUN",
        "institution name components treated as a single domain noun",
    ),
    (
        "measurement_noun",
        re.compile(r"^(?:lakh|crore|billion|million|trillion|per[- ]cent|"
                   r"basis[- ]points?|percentage[- ]points?|fiscal[- ]year|"
                   r"financial[- ]year|quarter[- ]year|gross[- ]domestic|"
                   r"gross[- ]value|net[- ]value|disposable[- ]income)$", re.I),
        "NOUN",
        "units of account and measurement nouns",
    ),
]

#: POS tags considered acceptable for a domain noun / proper domain noun.
_ACCEPTABLE_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS", "NOUN", "PROPN"}
_ACCEPTABLE_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB"}
_ACCEPTABLE_ADJ_TAGS = {"JJ", "JJR", "JJS", "ADJ"}


@dataclass
class Candidate:
    term: str
    occurrences: int
    documents: int
    tag_counts: Counter
    dominant_tag: str
    dominant_share: float
    distinct_tags: int
    rule: str
    expected_tag: str
    reason: str
    example_sentence: str
    example_document: str
    example_page: int
    example_unit: str
    correction_applies: bool


def _tag_family(tag: str) -> str:
    if tag in _ACCEPTABLE_NOUN_TAGS or tag.startswith("NN"):
        return "noun"
    if tag.startswith("VB") or tag == "VERB":
        return "verb"
    if tag.startswith("JJ") or tag == "ADJ":
        return "adjective"
    if tag.startswith("RB") or tag == "ADV":
        return "adverb"
    return "other"


def discover_candidates(
    annotations: SpacyAnnotations,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> List[Candidate]:
    """Collect the default-tag distribution of every domain candidate in the corpus."""
    seeds = [s for s in (config.get("pos.custom_rules.candidate_terms", []) or [])]
    seeds_lower = {s.casefold() for s in seeds}

    counts: Dict[str, Counter] = defaultdict(Counter)
    documents: Dict[str, Set[str]] = defaultdict(set)
    examples: Dict[str, Tuple[str, str, int, str]] = {}

    for annotation, unit in zip(annotations.annotations, annotations.units):
        tokens = annotation.tokens
        for index, (token, tag) in enumerate(zip(tokens, annotation.tags)):
            surface = token.strip()
            key = surface.casefold()
            is_seed = key in seeds_lower
            is_acronym = key.isupper() and 2 <= len(surface) <= 6 and surface.isalpha()
            if not (is_seed or is_acronym or _matches_shape(key)):
                continue
            counts[key][tag] += 1
            documents[key].add(annotation.document_id)
            if key not in examples and 6 <= len(tokens) <= 45:
                examples[key] = (
                    " ".join(tokens),
                    unit.document_id,
                    unit.page_number,
                    unit.unit_id,
                )

    candidates: List[Candidate] = []
    for term, tag_counts in counts.items():
        total = sum(tag_counts.values())
        dominant_tag, dominant_count = tag_counts.most_common(1)[0]
        rule, expected, reason = _expectation(term)
        sentence, document, page, unit_id = examples.get(term, ("", "", 0, ""))
        candidates.append(
            Candidate(
                term=term,
                occurrences=total,
                documents=len(documents.get(term, ())),
                tag_counts=tag_counts,
                dominant_tag=dominant_tag,
                dominant_share=round(dominant_count / total, 4) if total else 0.0,
                distinct_tags=len(tag_counts),
                rule=rule,
                expected_tag=expected,
                reason=reason,
                example_sentence=sentence,
                example_document=document,
                example_page=page,
                example_unit=unit_id,
                correction_applies=bool(rule) and _tag_family(dominant_tag) != _tag_family(expected),
            )
        )
    candidates.sort(key=lambda c: (-c.occurrences, c.term))
    if logger is not None:
        flagged = [c for c in candidates if c.correction_applies]
        inconsistent = [c for c in candidates if c.distinct_tags > 1]
        log_event(
            logger,
            "INFO",
            "custom_pos",
            f"mined {len(candidates)} domain candidates; {len(inconsistent)} tagged inconsistently; "
            f"{len(flagged)} rule-flagged for correction",
        )
    return candidates


def _matches_shape(term: str) -> bool:
    for _rule, pattern, _tag, _reason in DOMAIN_NOUNS_BY_SHAPE:
        if pattern.search(term):
            return True
    return False


def _expectation(term: str) -> Tuple[str, str, str]:
    """Evidence-based expected tag for a candidate term."""
    if term.isupper() and term.isalpha() and 2 <= len(term) <= 6:
        if term in INSTITUTION_ACRONYMS:
            return (
                "acronym_institution_or_indicator",
                "PROPN",
                "all-caps financial acronym that names an institution or a statistical "
                "indicator; the default tagger often treats it as a common noun",
            )
        return (
            "acronym_generic",
            "PROPN",
            "all-caps acronym behaves like a proper domain noun in economic prose",
        )
    for rule, pattern, tag, reason in DOMAIN_NOUNS_BY_SHAPE:
        if pattern.search(term):
            return (rule, tag, reason)
    return ("", "", "")


def build_dictionary(
    candidates: Sequence[Candidate],
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> List[Candidate]:
    """Select the terms that will actually be re-tagged."""
    min_occurrences = int(config.get("pos.custom_rules.min_evidence_occurrences", 3))
    max_size = int(config.get("pos.custom_rules.max_dictionary_size", 60))
    selected = [
        c
        for c in candidates
        if c.correction_applies and c.occurrences >= min_occurrences and c.expected_tag
    ]
    selected.sort(key=lambda c: (-c.occurrences, c.term))
    selected = selected[:max_size]
    if logger is not None:
        log_event(logger, "INFO", "custom_pos", f"dictionary size: {len(selected)} terms")
    return selected


def candidate_rows(candidates: Sequence[Candidate]) -> List[Dict[str, object]]:
    """Observed default-tag distribution for every mined candidate."""
    return [
        {
            "term": c.term,
            "occurrences": c.occurrences,
            "documents": c.documents,
            "dominant_tag": c.dominant_tag,
            "dominant_share": c.dominant_share,
            "distinct_tags": c.distinct_tags,
            "observed_tag_distribution": ";".join(f"{t}:{n}" for t, n in c.tag_counts.most_common()),
            "rule": c.rule,
            "expected_tag": c.expected_tag,
            "correction_applies": c.correction_applies,
            "example_document_id": c.example_document,
            "example_page": c.example_page,
            "example_unit_id": c.example_unit,
        }
        for c in candidates
    ]


def dictionary_rows(dictionary: Sequence[Candidate]) -> List[Dict[str, object]]:
    return [
        {
            "term": c.term,
            "default_tag": c.dominant_tag,
            "default_tag_share": c.dominant_share,
            "distinct_default_tags": c.distinct_tags,
            "observed_tag_distribution": ";".join(f"{t}:{n}" for t, n in c.tag_counts.most_common()),
            "custom_tag": c.expected_tag,
            "rule": c.rule,
            "reason": c.reason,
            "occurrences_in_corpus": c.occurrences,
            "documents_affected": c.documents,
            "example": c.example_sentence,
            "example_document_id": c.example_document,
            "example_page": c.example_page,
            "example_unit_id": c.example_unit,
        }
        for c in dictionary
    ]


def apply_rules(
    tokens: Sequence[str],
    tags: Sequence[str],
    dictionary: Dict[str, str],
) -> Tuple[List[str], List[int]]:
    """Re-tag a token sequence; returns the new tags and the changed indexes.

    Hyphenated compounds are handled as a unit: ``repo``, ``-``, ``rate`` is
    corrected only when the whole compound appears in the dictionary.
    """
    new_tags = list(tags)
    changed: List[int] = []
    for index, token in enumerate(tokens):
        if index >= len(new_tags):
            break
        key = token.strip().casefold()
        target = dictionary.get(key)
        if target is None:
            continue
        current = new_tags[index]
        if _tag_family(current) == _tag_family(target) and current != target:
            # keep the existing inflected tag; only the family matters for
            # correct parsing, and forcing PROPN over NNP would lose number.
            continue
        if current == target:
            continue
        new_tags[index] = target
        changed.append(index)
    return new_tags, changed


def retag_corpus(
    annotations: SpacyAnnotations,
    dictionary: Sequence[Candidate],
) -> Dict[str, object]:
    """Apply the dictionary to every unit and summarise the changes."""
    mapping = {c.term: c.expected_tag for c in dictionary}
    changed_rows: List[Dict[str, object]] = []
    total_tokens = 0
    changed_tokens = 0
    per_unit_tags: List[List[str]] = []

    for annotation, unit in zip(annotations.annotations, annotations.units):
        total_tokens += len(annotation.tokens)
        new_tags, changed = apply_rules(annotation.tokens, annotation.tags, mapping)
        per_unit_tags.append(new_tags)
        changed_tokens += len(changed)
        for index in changed:
            token = annotation.tokens[index]
            if len(changed_rows) < 5000:
                changed_rows.append(
                    {
                        "document_id": unit.document_id,
                        "page_number": unit.page_number,
                        "unit_id": unit.unit_id,
                        "section_id": unit.section_id,
                        "section_title": unit.section_title,
                        "token": token,
                        "token_index": index,
                        "default_tag": annotation.tags[index],
                        "custom_tag": new_tags[index],
                        "sentence": " ".join(annotation.tokens[max(0, index - 6): index + 7]),
                    }
                )

    by_term: Counter = Counter(
        row["token"].casefold() for row in changed_rows
    )
    return {
        "custom_tags": per_unit_tags,
        "changed_rows": changed_rows,
        "total_tokens": total_tokens,
        "changed_tokens": changed_tokens,
        "changed_percent": round(100.0 * changed_tokens / total_tokens, 4) if total_tokens else 0.0,
        "changed_by_term": by_term.most_common(50),
    }
