"""Named Entity Recognition: general NER, domain dictionary, error analysis.

Three artefacts, all corpus-derived:

1. ``ner_entities.csv`` - every entity the general NER model finds, with full
   provenance (document, page, section, unit) and a context window.
2. ``ner_error_analysis.csv`` - errors found by a **documented manual review**
   procedure over a stratified sample, plus systematic checks that do not need
   annotation (e.g. an economic indicator such as ``GDP`` tagged PERSON, or a
   ministry tagged as a person).
3. ``domain_entity_dictionary.csv`` + ``custom_ner_results.csv`` - a
   complementary financial-domain entity layer that runs *in addition to* the
   general model, because a general OntoNotes model has no notion of
   ``FINANCIAL_INSTITUTION``, ``ECONOMIC_INDICATOR`` or ``POLICY_TERM``.

Accuracy for the general model is computed only on the manually reviewed
sample; everything else is reported as counts, never as a score.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import Phase2Config, log_event
from .spacy_pipeline import SpacyAnnotations

# ----------------------------------------------------------------------
# Financial / economic domain entity dictionary
# ----------------------------------------------------------------------
#: Category -> terms. Every term is a *domain category assignment*, and each
#: category is separately checked against the corpus before it is reported.
DOMAIN_ENTITY_TERMS: Dict[str, Tuple[str, ...]] = {
    "FINANCIAL_INSTITUTION": (
        "RBI", "Reserve Bank of India", "SBI", "State Bank of India", "SEBI",
        "NABARD", "SIDBI", "EXIM Bank", "IRDAI", "PFRDA", "EPFO", "ESIC",
        "NABAM", "credit cooperative bank", "commercial bank", "co-operative bank",
        "development finance institution", "NBFC", "non-banking financial company",
        "small finance bank", "payments bank", "microfinance institution",
    ),
    "ECONOMIC_INDICATOR": (
        "GDP", "GVA", "GDP growth", "CPI", "WPI", "inflation", "disinflation",
        "deflation", "fiscal deficit", "current account deficit", "current account balance",
        "balance of payments", "FDI", "FPI", "NPA", "GNPA", "NNPA", "per capita income",
        "poverty ratio", "unemployment rate", "employment rate", "money supply",
        "MSF", "CRR", "SLR", "repo rate", "reverse repo rate", "policy repo rate",
        "bank rate", "WACC", "NIM", "ROA", "ROE", "CAD", "PPI", "PMI", "IIP",
        "exim", "triple defic", "potential GDP", "output gap", "inflation target",
    ),
    "FINANCIAL_INSTRUMENT": (
        "repo", "reverse repo", "treasury bill", "T-bill", "government security",
        "G-sec", "corporate bond", "sovereign bond", "debenture", "commercial paper",
        "certificate of deposit", "equity", "index fund", "ETF", "mutual fund",
        "SIP", "NCD", "FD", "recurring deposit", "savings deposit", "loan",
        "microfinance loan", "priority sector lending", "MUDRA loan", "bill of exchange",
        "promissory note", "derivatives", "swap", "futures", "options", "gold bond",
    ),
    "POLICY_TERM": (
        "monetary policy", "fiscal policy", "macro-prudential policy", "liquidity management",
        "open market operations", "standing deposit facility", "cash reserve ratio",
        "statutory liquidity ratio", "repo-rate", "policy-rate", "repo window",
        "earnings season", "distress avenue", "resolution framework", "insolvency and bankruptcy",
        "budget", "fiscal consolidation", "fiscal stimulus", "capital expenditure",
        "disinvestment", "dividend tax", "corporate tax", "GST", "customs duty",
        "financial inclusion", "priority sector", "targeted refinancing",
    ),
    "REGULATORY_BODY": (
        "SEBI", "RBI", "IRDAI", "PFRDA", "Nabard", "TRAI", "CCI", "NCLT", "CAG",
        "Supreme Court", "High Court", "Central Board of Direct Taxes", "RBI Monetary Policy Committee",
    ),
    "GOVERNMENT_BODY": (
        "Ministry of Finance", "Ministry of Commerce and Industry", "Ministry of Statistics and Programme Implementation",
        "Ministry of Corporate Affairs", "Reserve Bank of India", "Central Board of Revenue and Customs",
        "NITI Aayog", "Planning Commission", "Finance Commission", "Central Statistical Organisation",
        "Election Commission of India", "Cabinet Secretariat", "Financial Stability Board",
    ),
    "COUNTRY_OR_REGION": (
        "India", "United States", "China", "European Union", "Euro area", "Japan",
        "United Kingdom", "Germany", "Russia", "Brazil", "Emerging markets", "EMDE",
        "advanced economies", "South Asia", "Southeast Asia", "West Asia", "G7", "G20",
        "BRICS", "SAARC", "ASEAN", "OPEC", "World", "India's States",
    ),
    "ECONOMIC_EVENT": (
        "Global Financial Crisis", "COVID-19 pandemic", "pandemic", "Global Recession",
        "financial crisis", "banking crisis", "currency crisis", "sovereign debt crisis",
        "taper tantrum", "liquidity crunch", "budget announcement", "monetary policy statement",
        "recession", "stagflation", "inflation spiral", "assembly elections", "election",
    ),
}

#: A multiword term is matched with flexible whitespace; a single term exactly.
_LONGEST_FIRST: Dict[str, str] = {}
for _category, _terms in DOMAIN_ENTITY_TERMS.items():
    for _term in _terms:
        if _term not in _LONGEST_FIRST:
            _LONGEST_FIRST[_term] = _category
_SORTED_TERMS = sorted(_LONGEST_FIRST, key=lambda t: (-len(t), t))
_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in _SORTED_TERMS) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

#: Label groups that are acceptable for a domain category, used by the review.
GENERAL_LABELS: Set[str] = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT"}
GENERAL_LABELS_FULL: Set[str] = {
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
    "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME",
    "WORK_OF_ART",
}

CATEGORY_TO_GENERAL = {
    "FINANCIAL_INSTITUTION": {"ORG"},
    "REGULATORY_BODY": {"ORG", "GPE"},
    "GOVERNMENT_BODY": {"ORG", "GPE"},
    "ECONOMIC_INDICATOR": set(),
    "FINANCIAL_INSTRUMENT": {"PRODUCT", "ORG"},
    "POLICY_TERM": set(),
    "COUNTRY_OR_REGION": {"GPE", "LOC", "NORP"},
    "ECONOMIC_EVENT": {"EVENT"},
}


# ----------------------------------------------------------------------
# General NER run
# ----------------------------------------------------------------------
def run_general_ner(
    annotations: SpacyAnnotations,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    max_entities = int(config.get("ner.max_entities_written", 20000))
    context_chars = int(config.get("ner.max_context_chars", 160))
    wanted = set(config.get("ner.entity_types_investigated", []) or [])

    rows: List[Dict[str, object]] = []
    label_counter: Counter = Counter()
    type_counter: Counter = Counter()
    entity_id = 0
    seen_texts: Counter = Counter()

    for annotation, unit in zip(annotations.annotations, annotations.units):
        tokens = annotation.tokens
        for text, label, start, end in annotation.entities:
            if wanted and label not in wanted and label not in GENERAL_LABELS_FULL:
                # keep everything the model produces, but flag the focus types
                pass
            label_counter[label] += 1
            seen_texts[text] += 1
            left = max(0, start - 12)
            right = min(len(tokens), end + 12)
            context = " ".join(tokens[left:right])
            type_counter[label if label in wanted else "other"] += 1
            if len(rows) >= max_entities:
                continue
            entity_id += 1
            rows.append(
                {
                    "entity_id": f"NER{entity_id:06d}",
                    "document_id": unit.document_id,
                    "source_id": unit.source_id,
                    "page_number": unit.page_number,
                    "unit_id": unit.unit_id,
                    "section_id": unit.section_id,
                    "section_number": unit.section_number,
                    "section_title": unit.section_title,
                    "unit_type": unit.unit_type,
                    "entity_text": text,
                    "entity_label": label,
                    "token_start": start,
                    "token_end": end,
                    "context": context[:context_chars],
                }
            )

    distribution = [
        {"entity_label": label, "count": count, "distinct_surface_forms": len(
            {t for t, c in seen_texts.items() if c}
        ) if label == sorted(label_counter)[0] else "", "percent_of_entities": round(
            100.0 * count / max(1, sum(label_counter.values())), 4
        )}
        for label, count in label_counter.most_common()
    ]

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "ner",
            f"general NER: {sum(label_counter.values())} entities, {len(label_counter)} labels, "
            f"{len(rows)} written",
        )

    return {
        "entities": rows,
        "label_distribution": distribution,
        "type_counts": dict(type_counter),
        "surface_forms": seen_texts,
        "total_entities": sum(label_counter.values()),
    }


# ----------------------------------------------------------------------
# Domain dictionary layer
# ----------------------------------------------------------------------
def run_domain_ner(
    annotations: SpacyAnnotations,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Complementary financial-domain entity layer over the same units."""
    max_entities = int(config.get("ner.max_entities_written", 20000))
    context_chars = int(config.get("ner.max_context_chars", 160))
    max_per_unit = 60

    rows: List[Dict[str, object]] = []
    category_counts: Counter = Counter()
    document_counts: Counter = Counter()
    entity_id = 0

    for annotation, unit in zip(annotations.annotations, annotations.units):
        text = " ".join(annotation.tokens)
        seen_in_unit: Counter = Counter()
        for match in _DOMAIN_RE.finditer(text):
            surface = match.group(1)
            category = _LONGEST_FIRST.get(surface) or _LONGEST_FIRST.get(
                next((t for t in _SORTED_TERMS if t.casefold() == surface.casefold()), surface)
            )
            seen_in_unit[category] += 1
            category_counts[category] += 1
            document_counts[f"{unit.document_id}|{category}"] += 1
            if len(rows) < max_entities and seen_in_unit[category] <= 3:
                entity_id += 1
                left = max(0, match.start() - 90)
                right = min(len(text), match.end() + 90)
                rows.append(
                    {
                        "entity_id": f"DOMNER{entity_id:06d}",
                        "document_id": unit.document_id,
                        "source_id": unit.source_id,
                        "page_number": unit.page_number,
                        "unit_id": unit.unit_id,
                        "section_id": unit.section_id,
                        "section_number": unit.section_number,
                        "section_title": unit.section_title,
                        "entity_text": surface,
                        "domain_category": category,
                        "also_tagged_by_general_ner": _overlap_labels(annotation.entities, surface),
                        "context": text[left:right][:context_chars],
                    }
                )

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "ner",
            f"domain NER: {sum(category_counts.values())} domain mentions across "
            f"{len(category_counts)} categories, {len(rows)} written",
        )
    return {
        "entities": rows,
        "category_counts": category_counts,
        "documents_per_category": document_counts,
        "total_mentions": sum(category_counts.values()),
    }


def _overlap_labels(entities: Sequence[Tuple[str, str, int, int]], surface: str) -> str:
    hits = sorted({label for text, label, _s, _e in entities if text == surface})
    return ";".join(hits)


def domain_dictionary_rows(
    corpus_counts: Counter,
    document_counts: Counter,
) -> List[Dict[str, object]]:
    """One row per dictionary term, with the corpus evidence that justifies it."""
    rows: List[Dict[str, object]] = []
    for term in _SORTED_TERMS:
        category = _LONGEST_FIRST[term]
        occurrences = corpus_counts.get(term, 0)
        if occurrences == 0:
            occurrences = corpus_counts.get(term.capitalize(), 0)
        rows.append(
            {
                "term": term,
                "domain_category": category,
                "token_length": len(term.split()),
                "corpus_mentions": occurrences,
                "documents_mentioned": document_counts.get(f"__{category}__{term}", 0),
                "general_ner_equivalent_labels": ";".join(sorted(CATEGORY_TO_GENERAL.get(category, set()))) or "none",
                "justification": _category_rationale(category),
            }
        )
    rows.sort(key=lambda row: (-int(row["corpus_mentions"]), str(row["term"])))
    return rows


def _category_rationale(category: str) -> str:
    return {
        "FINANCIAL_INSTITUTION": "entity that lends, regulates or supervises money; general NER tags many of these as ORG, but has no class for the sector",
        "ECONOMIC_INDICATOR": "a measured quantity, not a named entity; no general NER label exists for it",
        "FINANCIAL_INSTRUMENT": "a tradable/lendable instrument; general NER only covers a few as PRODUCT",
        "POLICY_TERM": "a policy concept or facility, not a named entity; invisible to general NER",
        "REGULATORY_BODY": "the body that issues the rules; general NER usually ORG but often missed when abbreviated",
        "GOVERNMENT_BODY": "a ministry or public institution; general NER is inconsistent on Indian ministries",
        "COUNTRY_OR_REGION": "country or region; general NER covers only a subset of the world and of regional groupings",
        "ECONOMIC_EVENT": "a macro-financial event with economic consequences; general NER EVENT is trained on news and rarely fires on economic writing",
    }[category]


# ----------------------------------------------------------------------
# Error analysis
# ----------------------------------------------------------------------
def build_error_analysis(
    annotations: SpacyAnnotations,
    general: Dict[str, object],
    domain: Dict[str, object],
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, object]]:
    """Systematic + manually reviewed error rows.

    Systematic checks (no annotation needed, they are type-level impossibilities
    of the general tagger):

    * an all-caps domain acronym labelled ``PERSON``;
    * a domain institution/indicator surface form labelled with a general label
      that contradicts its domain category;
    * a domain entity the general model missed entirely.

    Manually reviewed rows come from the annotation file written by
    :func:`write_reviewed_annotations`, and are tagged ``manually_reviewed``.
    """
    rows: List[Dict[str, object]] = []
    general_entities = general["entities"]  # type: ignore[index]
    domain_entities = domain["entities"]  # type: ignore[index]

    # ---- type-level impossibilities ----
    for entity in general_entities:  # type: ignore[union-attr]
        text = str(entity["entity_text"])
        label = str(entity["entity_label"])
        expected = _LONGEST_FIRST.get(text) or _LONGEST_FIRST.get(
            next((t for t in _SORTED_TERMS if t.casefold() == text.casefold()), "")
        )
        if expected and label == "PERSON" and text.isupper():
            rows.append(
                _error_row(
                    text=text,
                    default_label=label,
                    expected=expected,
                    error_type="wrong_general_label",
                    context=str(entity["context"]),
                    document_id=str(entity["document_id"]),
                    page_number=int(entity["page_number"]),
                    unit_id=str(entity["unit_id"]),
                    notes=f"'{text}' is a domain {expected} but the general model labelled it PERSON",
                )
            )
        elif expected and CATEGORY_TO_GENERAL.get(expected) and label not in CATEGORY_TO_GENERAL[expected]:
            rows.append(
                _error_row(
                    text=text,
                    default_label=label,
                    expected=expected,
                    error_type="domain_category_mismatch",
                    context=str(entity["context"]),
                    document_id=str(entity["document_id"]),
                    page_number=int(entity["page_number"]),
                    unit_id=str(entity["unit_id"]),
                    notes=(
                        f"domain category {expected} is not a general NER class; "
                        f"model label {label} is not an equivalent"
                    ),
                )
            )

    # ---- domain mentions the general model never found ----
    missed: Counter = Counter()
    missed_example: Dict[str, Dict[str, object]] = {}
    for entity in domain_entities:  # type: ignore[union-attr]
        if not str(entity["also_tagged_by_general_ner"]):
            key = str(entity["entity_text"])
            missed[key] += 1
            missed_example.setdefault(key, entity)
    for text, count in missed.most_common(400):
        example = missed_example[text]
        rows.append(
            _error_row(
                text=text,
                default_label="NOT_RECOGNISED",
                expected=_LONGEST_FIRST.get(text, "domain entity"),
                error_type="missed_by_general_ner",
                context=str(example["context"]),
                document_id=str(example["document_id"]),
                page_number=int(example["page_number"]),
                unit_id=str(example["unit_id"]),
                notes=f"{count} domain mentions of '{text}' had no general NER entity at all",
            )
        )

    # ---- manually reviewed subset ----
    reviewed_path = config.out_path("annotations_dir") / "ner_reviewed_annotations.csv"
    if reviewed_path.is_file():
        import csv

        with open(reviewed_path, "r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                if record.get("expected_label") == record.get("default_label"):
                    continue
                rows.append(
                    _error_row(
                        text=record["entity_text"],
                        default_label=record["default_label"],
                        expected=record["expected_label"],
                        error_type=record.get("error_type", "manual_review"),
                        context=record.get("context", ""),
                        document_id=record.get("document_id", ""),
                        page_number=int(record.get("page_number") or 0),
                        unit_id=record.get("unit_id", ""),
                        notes="manually reviewed; annotator: " + record.get("annotator", "single annotator"),
                    )
                )

    if logger is not None:
        manual = sum(1 for row in rows if row["error_type"] == "manual_review")
        log_event(
            logger,
            "INFO",
            "ner",
            f"error analysis: {len(rows)} rows "
            f"({sum(1 for r in rows if r['error_type'] == 'missed_by_general_ner')} missed, "
            f"{sum(1 for r in rows if r['error_type'] == 'domain_category_mismatch')} mismatched, "
            f"{manual} manually reviewed)",
        )
    return rows


def _error_row(**kwargs) -> Dict[str, object]:
    return {
        "text": kwargs["text"],
        "default_label": kwargs["default_label"],
        "expected_or_interpreted_label": kwargs["expected"],
        "error_type": kwargs["error_type"],
        "context": kwargs["context"][:200],
        "document_id": kwargs["document_id"],
        "page_number": kwargs["page_number"],
        "unit_id": kwargs["unit_id"],
        "notes": kwargs["notes"],
    }
