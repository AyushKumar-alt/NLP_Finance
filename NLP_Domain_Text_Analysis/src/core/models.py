"""Data model for the Phase 1 corpus.

The hierarchy enforced here is the backbone of the whole project:

    SOURCE  ->  DOCUMENT  ->  PAGE  ->  SECTION  ->  CONTENT UNIT  ->  TEXT

Every content unit therefore carries enough information to trace a token or a
retrieval hit all the way back to the originating PDF file and page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------
# unit_type vocabulary (kept as a constant so later phases can rely on it)
# ----------------------------------------------------------------------
UNIT_TYPES = (
    "paragraph",
    "heading",
    "table",
    "figure",
    "caption",
    "box",
    "footnote",
    "reference",
    "other",
)


@dataclass
class Span:
    """A positioned text run extracted from a PDF page."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str = ""
    bold: bool = False
    italic: bool = False
    rotated: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)],
            "font_size": round(self.size, 2),
            "font": self.font,
            "bold": self.bold,
            "italic": self.italic,
            "rotated": self.rotated,
        }


@dataclass
class Block:
    """A PDF text block (PyMuPDF) - the natural paragraph container."""

    spans: List[Span] = field(default_factory=list)
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    block_no: int = -1
    kind: str = "text"  # text | image
    text: str = ""
    lines: List[TextLine] = field(default_factory=list)

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def max_size(self) -> float:
        return max((s.size for s in self.spans), default=0.0)

    @property
    def min_size(self) -> float:
        sizes = [s.size for s in self.spans]
        return min(sizes) if sizes else 0.0

    @property
    def all_rotated(self) -> bool:
        return bool(self.spans) and all(s.rotated for s in self.spans)

    @property
    def any_rotated(self) -> bool:
        return any(s.rotated for s in self.spans)


@dataclass
class TextLine:
    """A visual line inside a block, keeping span geometry."""

    spans: List[Span]
    bbox: tuple

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def x0(self) -> float:
        return min((s.x0 for s in self.spans), default=0.0)

    @property
    def x1(self) -> float:
        return max((s.x1 for s in self.spans), default=0.0)

    @property
    def y0(self) -> float:
        return min((s.y0 for s in self.spans), default=0.0)

    @property
    def y1(self) -> float:
        return max((s.y1 for s in self.spans), default=0.0)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def size(self) -> float:
        return max((s.size for s in self.spans), default=0.0)

    @property
    def rotated(self) -> bool:
        return bool(self.spans) and all(s.rotated for s in self.spans)


@dataclass
class Token:
    """A whitespace-delimited token with exact character-level geometry.

    Produced from PyMuPDF ``rawdict`` output, which exposes a bounding box per
    character. This is what makes borderless (rule-free) financial table
    reconstruction possible: every cell token keeps its true x position.
    """

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 0.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class CaptionAnchor:
    """A detected ``Table I.2:`` / ``Chart I.3:`` / ``Box I.1:`` label."""

    kind: str            # table | figure | box
    label: str           # e.g. "Table I.2a"
    caption: str         # full caption text
    bbox: tuple
    line_index: int = -1


@dataclass
class TableRecord:
    table_id: str
    document_id: str
    source_id: str
    page_number: int
    section_id: str
    caption: str = ""
    label: str = ""
    raw_text: str = ""
    rows: int = 0
    columns: int = 0
    source_note: str = ""
    table_data: List[List[str]] = field(default_factory=list)
    quality: str = "LOW"
    quality_reason: str = ""
    data_extracted: bool = False
    bbox: Optional[List[float]] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class FigureRecord:
    figure_id: str
    document_id: str
    source_id: str
    page_number: int
    section_id: str
    caption: str = ""
    label: str = ""
    kind: str = "chart"     # chart | figure | exhibit | box
    surrounding_text: str = ""
    source_note: str = ""
    axis_labels: List[str] = field(default_factory=list)
    data_extracted: bool = False
    data_note: str = ""
    bbox: Optional[List[float]] = None


@dataclass
class SectionRecord:
    section_id: str
    document_id: str
    source_id: str
    section_number: str
    section_title: str
    page_start: int
    page_end: int
    start_y: float = 0.0
    detection_method: str = "numbered_pattern"
    confidence: float = 1.0
    is_confirmed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "start_y": round(self.start_y, 2),
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "is_confirmed": self.is_confirmed,
        }


@dataclass
class UnitRecord:
    """A retrieval unit - the atomic, traceable record of the corpus."""

    unit_id: str
    document_id: str
    source_id: str
    filename: str = ""
    page_id: str = ""
    page_number: int = 0
    section_id: str = "UNKNOWN"
    section_number: str = "UNKNOWN"
    section_title: str = "UNKNOWN"
    unit_type: str = "paragraph"
    text: str = ""
    char_count: int = 0
    word_count: int = 0
    unit_index: int = 0
    caption: str = ""
    label: str = ""
    source_note: str = ""
    table_data: Optional[List[List[str]]] = None
    table_rows: int = 0
    table_columns: int = 0
    table_extraction_quality: str = ""
    axis_labels: Optional[List[str]] = None
    data_extracted: Optional[bool] = None
    bbox: Optional[List[float]] = None
    font_size: Optional[float] = None
    extraction_note: str = ""

    def to_jsonl_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "filename": self.filename,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "section_id": self.section_id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "unit_type": self.unit_type,
            "unit_index": self.unit_index,
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
        }
        optional = {
            "caption": self.caption,
            "label": self.label,
            "source_note": self.source_note,
            "table_data": self.table_data,
            "table_rows": self.table_rows if self.unit_type == "table" else None,
            "table_columns": self.table_columns if self.unit_type == "table" else None,
            "table_extraction_quality": self.table_extraction_quality,
            "axis_labels": self.axis_labels,
            "data_extracted": self.data_extracted,
            "bbox": self.bbox,
            "font_size": self.font_size,
            "extraction_note": self.extraction_note,
        }
        for key, value in optional.items():
            if value not in (None, "", [], {}):
                record[key] = value
        return record


@dataclass
class PageRecord:
    document_id: str
    source_id: str
    page_id: str
    page_number: int
    text: str = ""
    cleaned_text: str = ""
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    block_count: int = 0
    image_count: int = 0
    is_text_extractable: bool = True
    is_likely_scanned: bool = False
    extraction_warning: str = ""
    body_font_size: float = 0.0
    width: float = 0.0
    height: float = 0.0
    unit_count: int = 0
    section_id: str = "UNKNOWN"
    extraction_engine: str = ""

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "block_count": self.block_count,
            "image_count": self.image_count,
            "is_text_extractable": self.is_text_extractable,
            "is_likely_scanned": self.is_likely_scanned,
            "extraction_warning": self.extraction_warning,
            "body_font_size": round(self.body_font_size, 2),
            "page_width": round(self.width, 2),
            "page_height": round(self.height, 2),
            "unit_count": self.unit_count,
            "section_id": self.section_id,
            "extraction_engine": self.extraction_engine,
        }
        if include_text:
            payload["text"] = self.text
            payload["cleaned_text"] = self.cleaned_text
        return payload
