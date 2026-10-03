"""Load the Phase 1 structured corpus into a traceable in-memory model.

Structure produced by this module::

    Corpus
      └── Document (document_id, source_id, filename, registry metadata)
            └── Page (page_id, page_number)
                  └── Section (section_id, section_number, section_title)
                        └── Unit (unit_id, unit_type, text, provenance)

Every :class:`Unit` keeps the full ``source -> document -> page -> section ->
unit`` chain so that any Phase 2 example can be traced back to its page and
section, which is what Phase 3 retrieval will need to cite.

The loader also owns the *text selection policy* (prose only / prose+tables /
all textual units) and the shared experiment sample used by every tokenizer so
that tokenizers are always compared on identical input text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.utils import read_csv_rows, read_jsonl

from .config import Phase2Config

#: Phase 1 can emit U+FFFD for publisher glyphs it could not map. It is kept in
#: the text (Phase 1 artefact is untouched) but treated as a separator by every
#: Phase 2 tokenizer so it can never become part of a token.
REPLACEMENT_CHAR = "\ufffd"

_WS_SPLIT = re.compile(r"[ \t\u00a0\u2007\u202f]+")


@dataclass(frozen=True)
class Unit:
    """One Phase 1 content unit with complete provenance."""

    unit_id: str
    document_id: str
    source_id: str
    filename: str
    page_id: str
    page_number: int
    section_id: str
    section_number: str
    section_title: str
    unit_type: str
    unit_index: int
    text: str
    char_count: int
    word_count: int
    bbox: Tuple[float, ...] = ()
    font_size: Optional[float] = None

    # ---------------- convenience ----------------
    @property
    def provenance(self) -> str:
        """Human readable citation string, e.g. ``D07 | p24 | 3.2 | D07_P024_U009``."""
        return (
            f"{self.document_id} | p{self.page_number} | {self.section_number} | {self.unit_id}"
        )

    def to_trace(self) -> Dict[str, object]:
        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "page_number": self.page_number,
            "unit_id": self.unit_id,
            "section_id": self.section_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "unit_type": self.unit_type,
        }


@dataclass
class Section:
    section_id: str
    section_number: str
    section_title: str
    page_number: int
    units: List[Unit] = field(default_factory=list)

    @property
    def tables(self) -> List[Unit]:
        return [u for u in self.units if u.unit_type == "table"]

    @property
    def figures(self) -> List[Unit]:
        return [u for u in self.units if u.unit_type in ("figure", "box")]


@dataclass
class Page:
    page_id: str
    page_number: int
    document_id: str
    sections: List[Section] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)


@dataclass
class Document:
    document_id: str
    source_id: str
    filename: str
    title: str = ""
    publisher: str = ""
    year: str = ""
    document_type: str = ""
    page_count: int = 0
    sha256: str = ""
    pages: List[Page] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)

    @property
    def source_type(self) -> str:
        """Group label used in the manifest (``economic_survey`` / ...)."""
        return {
            "SRC01": "economic_survey",
            "SRC02": "financial_stability_report",
            "SRC03": "regulator_annual_report",
        }.get(self.source_id, "other_source")


@dataclass
class Corpus:
    documents: List[Document]
    units: List[Unit]                 # every unit, in corpus.jsonl order
    policy: str
    policy_unit_types: List[str]
    document_registry: List[Dict[str, str]] = field(default_factory=list)
    source_registry: List[Dict[str, str]] = field(default_factory=list)

    # ---------------- corpus level helpers ----------------
    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def page_count(self) -> int:
        return len({u.page_id for u in self.units})

    def selected_units(self, policy: Optional[str] = None, unit_types: Optional[Sequence[str]] = None) -> List[Unit]:
        """Units admitted by the active (or supplied) text selection policy."""
        types = set(unit_types) if unit_types is not None else set(self.policy_unit_types)
        return [u for u in self.units if u.unit_type in types and u.text.strip()]

    def documents_by_id(self) -> Dict[str, Document]:
        return {d.document_id: d for d in self.documents}

    def summary(self) -> Dict[str, object]:
        unit_types: Dict[str, int] = {}
        for unit in self.units:
            unit_types[unit.unit_type] = unit_types.get(unit.unit_type, 0) + 1
        return {
            "documents": self.document_count,
            "pages": self.page_count,
            "units_total": len(self.units),
            "units_by_type": dict(sorted(unit_types.items())),
            "text_selection_policy": self.policy,
            "policy_unit_types": list(self.policy_unit_types),
            "selected_units": len(self.selected_units()),
            "selected_characters": sum(len(u.text) for u in self.selected_units()),
        }


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_corpus(config: Phase2Config, logger: Optional[logging.Logger] = None) -> Corpus:
    """Read ``corpus.jsonl`` + registries and build the traceable corpus model."""
    corpus_path = config.corpus_jsonl
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 structured corpus not found: {corpus_path}\n"
            "Run Phase 1 first:  python -m src.phase1.run"
        )

    raw_records = read_jsonl(corpus_path)
    if not raw_records:
        raise ValueError(f"Phase 1 corpus is empty: {corpus_path}")

    document_registry = _read_registry(config, "document_registry")
    source_registry = _read_registry(config, "source_registry")
    doc_meta = {row["document_id"]: row for row in document_registry}

    documents: Dict[str, Document] = {}
    page_index: Dict[str, Page] = {}
    units: List[Unit] = []

    for record in raw_records:
        document_id = str(record.get("document_id", "UNKNOWN"))
        if document_id not in documents:
            meta = doc_meta.get(document_id, {})
            documents[document_id] = Document(
                document_id=document_id,
                source_id=str(record.get("source_id", meta.get("source_id", "UNKNOWN"))),
                filename=str(record.get("filename", meta.get("filename", ""))),
                title=str(meta.get("title", "")),
                publisher=str(meta.get("publisher", "")),
                year=str(meta.get("year", "")),
                document_type=str(meta.get("document_type", "")),
                page_count=int(meta.get("page_count", 0) or 0),
                sha256=str(meta.get("sha256", "")),
            )
        document = documents[document_id]

        page_id = str(record.get("page_id", f"{document_id}_P{record.get('page_number', 0)}"))
        page_number = int(record.get("page_number", 0) or 0)
        page = page_index.get(page_id)
        if page is None:
            page = Page(page_id=page_id, page_number=page_number, document_id=document_id)
            page_index[page_id] = page
            document.pages.append(page)

        unit = Unit(
            unit_id=str(record.get("unit_id", "")),
            document_id=document_id,
            source_id=str(record.get("source_id", document.source_id)),
            filename=str(record.get("filename", document.filename)),
            page_id=page_id,
            page_number=page_number,
            section_id=str(record.get("section_id", "UNKNOWN")),
            section_number=str(record.get("section_number", "UNKNOWN")),
            section_title=str(record.get("section_title", "UNKNOWN")),
            unit_type=str(record.get("unit_type", "other")),
            unit_index=int(record.get("unit_index", 0) or 0),
            text=str(record.get("text", "")),
            char_count=int(record.get("char_count", 0) or 0),
            word_count=int(record.get("word_count", 0) or 0),
            bbox=tuple(record.get("bbox", ()) or ()),
            font_size=record.get("font_size"),
        )
        units.append(unit)
        document.units.append(unit)
        page.units.append(unit)

        section = _section_for(page, unit)
        section.units.append(unit)

    corpus = Corpus(
        documents=[documents[k] for k in sorted(documents)],
        units=units,
        policy=config.active_policy,
        policy_unit_types=config.policy_unit_types(),
        document_registry=document_registry,
        source_registry=source_registry,
    )

    if logger is not None:
        from .config import log_event

        summary = corpus.summary()
        log_event(
            logger,
            "INFO",
            "load_corpus",
            f"corpus loaded from {corpus_path.name}: {summary['documents']} documents, "
            f"{summary['pages']} pages, {summary['units_total']} units; "
            f"policy '{corpus.policy}' selects {summary['selected_units']} units "
            f"({summary['selected_characters']} characters)",
        )
    return corpus


def _read_registry(config: Phase2Config, key: str) -> List[Dict[str, str]]:
    path = config.input_path(key)
    if not path.is_file():
        return []
    return read_csv_rows(path)


def _section_for(page: Page, unit: Unit) -> Section:
    for section in page.sections:
        if section.section_id == unit.section_id:
            return section
    section = Section(
        section_id=unit.section_id,
        section_number=unit.section_number,
        section_title=unit.section_title,
        page_number=unit.page_number,
    )
    page.sections.append(section)
    return section


# ----------------------------------------------------------------------
# Shared experiment sample
# ----------------------------------------------------------------------
@dataclass
class ExperimentSample:
    """The one text sample every tokenizer / BPE experiment is run on.

    Fair comparison is a hard requirement of the assignment, so the sample is
    fixed once (ordered by unit id) and reused by NLTK, spaCy, custom, hybrid
    and BPE tokenization.
    """

    policy: str
    units: List[Unit]
    documents: Dict[str, Document] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len({u.document_id for u in self.units})

    @property
    def character_count(self) -> int:
        return sum(len(u.text) for u in self.units)

    def texts(self) -> List[str]:
        return [u.text for u in self.units]

    def source_type(self, unit: Unit) -> str:
        doc = self.documents.get(unit.document_id)
        return doc.source_type if doc else "unknown"


def build_experiment_sample(corpus: Corpus, units: Optional[Sequence[Unit]] = None) -> ExperimentSample:
    """Freeze the common experiment sample (default: the policy-selected units)."""
    sample_units = list(units) if units is not None else corpus.selected_units()
    sample_units = sorted(sample_units, key=lambda u: u.unit_id)
    return ExperimentSample(
        policy=corpus.policy,
        units=sample_units,
        documents=corpus.documents_by_id(),
    )


def representative_sample(corpus: Corpus, logger: Optional[logging.Logger] = None) -> List[Dict[str, str]]:
    """Select 10 illustrative units, one per category required by the assignment.

    Selection is deterministic and corpus-driven: within each category the
    *median-length* candidate is chosen so the examples are typical rather than
    outliers.
    """
    selected = corpus.selected_units()
    by_type: Dict[str, List[Unit]] = {}
    for unit in selected:
        by_type.setdefault(unit.unit_type, []).append(unit)

    doc_types = corpus.documents_by_id()

    def median_pick(units: List[Unit]) -> Optional[Unit]:
        if not units:
            return None
        ordered = sorted(units, key=lambda u: (len(u.text), u.unit_id))
        return ordered[len(ordered) // 2]

    def first_matching(predicate, pool: Sequence[Unit]) -> Optional[Unit]:
        candidates = [u for u in pool if predicate(u.text)]
        return median_pick(candidates)

    patterns = {
        "normal_economic_paragraph": lambda t: bool(re.search(r"econom|growth|inflation", t, re.I)) and len(t) > 200,
        "paragraph_containing_dates": lambda t: bool(re.search(r"\b(20[12]\d|January|February|March|April|September|October|November|December|quarter)\b", t)),
        "paragraph_containing_percentages": lambda t: "%" in t or "per cent" in t,
        "paragraph_containing_currency": lambda t: bool(re.search(r"[\u20b9$€£]|Rs\.?\s|₹", t)),
        "paragraph_containing_abbreviations": lambda t: bool(re.search(r"\b[A-Z]{2,6}\b", t)) and bool(re.search(r"\b(GDP|GVA|CPI|WPI|FDI|RBI|SEBI|IMF|NP|GST)\b", t)),
        "paragraph_containing_financial_institutions": lambda t: bool(re.search(r"\b(bank|Bank|RBI|SEBI|NBFC|commercial bank|financial institution|IMF|World Bank)\b", t)),
        "paragraph_containing_hyphenated_terminology": lambda t: "-" in t and bool(re.search(r"[a-zA-Z]{3,}-[a-zA-Z]{3,}", t)),
        "paragraph_containing_numerical_expressions": lambda t: bool(re.search(r"\d[\d,.]*\s*(lakh|crore|billion|million|trillion|mn|bn)", t, re.I)),
        "table_text": lambda t: t is not None,
        "rare_domain_specific_terminology": lambda t: bool(
            re.search(r"\b(demonetisation|disinflation|securitisation|macroeconomic|financialisation|capitalisation|nowcasting|insolvency|procyclical)\b", t, re.I)
        ),
    }

    rows: List[Dict[str, str]] = []
    for category, predicate in patterns.items():
        if category == "table_text":
            unit = median_pick(by_type.get("table", []))
        else:
            pool = [u for u in selected if u.unit_type in ("paragraph", "box")]
            unit = first_matching(predicate, pool)
        if unit is None:
            continue
        doc = doc_types.get(unit.document_id)
        rows.append(
            {
                "category": category,
                "document_id": unit.document_id,
                "source_id": unit.source_id,
                "source_type": doc.source_type if doc else "unknown",
                "page_number": unit.page_number,
                "section_id": unit.section_id,
                "section_number": unit.section_number,
                "section_title": unit.section_title,
                "unit_id": unit.unit_id,
                "unit_type": unit.unit_type,
                "char_count": len(unit.text),
                "text": unit.text.replace("\n", " | "),
            }
        )

    if logger is not None:
        from .config import log_event

        log_event(logger, "INFO", "load_corpus", f"representative sample: {len(rows)}/10 categories filled")
    return rows


def experiment_manifest(
    sample: ExperimentSample,
    experiments: Iterable[str],
    policy: str,
    max_rows_per_experiment: int = 0,
) -> List[Dict[str, object]]:
    """One row per (experiment, unit) so every token is attributable to a unit."""
    rows: List[Dict[str, object]] = []
    for experiment_id in experiments:
        for index, unit in enumerate(sample.units, start=1):
            if max_rows_per_experiment and index > max_rows_per_experiment:
                break
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "document_id": unit.document_id,
                    "source_id": unit.source_id,
                    "source_type": sample.source_type(unit),
                    "page_number": unit.page_number,
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type,
                    "section_id": unit.section_id,
                    "section_number": unit.section_number,
                    "section_title": unit.section_title,
                    "text_selection_policy": policy,
                    "text_length": len(unit.text),
                    "unit_index_in_sample": index,
                }
            )
    return rows


# ----------------------------------------------------------------------
# Text normalisation shared by the experiment modules
# ----------------------------------------------------------------------
def normalise_for_tokenisation(text: str) -> str:
    """Phase 1 emits newline-separated table cells and ``\\ufffd`` for lost glyphs.

    Both are converted into plain whitespace here so every tokenizer sees the
    same surface text and no tokenizer can accidentally produce a token that
    contains a newline or an unmapped glyph.
    """
    return re.sub(r"[\r\n]+", " ", text.replace(REPLACEMENT_CHAR, " "))


def split_paragraphs(text: str) -> List[str]:
    return [p for p in _WS_SPLIT.split(text.strip()) if p]
