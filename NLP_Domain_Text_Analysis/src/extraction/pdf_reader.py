"""PDF opening and page-level geometric extraction.

Engine choice (documented decision for the report)
--------------------------------------------------
* **PyMuPDF (fitz) - primary.** It returns a layout tree of
  ``block -> line -> span`` with font name, font size, and the writing
  direction of every line. Those three signals are what the structural
  detectors need (headings = larger font, footnotes = smaller font in the
  lower band, chart axis labels = rotated lines). A plain
  ``extract_text()`` call throws all of that away, which is exactly the
  "one giant string" approach this project must avoid.
* **pdfplumber - secondary.** Used for ruled-table detection (it understands
  drawn lines/rectangles) and as a word-level fallback when PyMuPDF yields
  no text for a page.

Neither library ever writes to the source file; documents are opened in
read-only mode only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..core.models import Block, Span, TextLine, Token

try:  # pragma: no cover - import guard
    import fitz  # type: ignore

    PYMUPDF_AVAILABLE = True
except Exception:  # pragma: no cover
    fitz = None
    PYMUPDF_AVAILABLE = False

try:  # pragma: no cover
    import pdfplumber  # type: ignore

    PDFPLUMBER_AVAILABLE = True
except Exception:  # pragma: no cover
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False


class PdfOpenError(RuntimeError):
    """Raised when a document cannot be opened or decrypted."""


@dataclass
class PageGeometry:
    """Everything extracted from one page before any interpretation."""

    page_number: int
    width: float
    height: float
    blocks: List[Block] = field(default_factory=list)
    lines: List[TextLine] = field(default_factory=list)
    image_count: int = 0
    drawing_count: int = 0
    engine: str = ""
    raw_text: str = ""
    rotated_char_ratio: float = 0.0
    body_font_size: float = 0.0
    tokens: List[Token] = field(default_factory=list)

    @property
    def text_blocks(self) -> List[Block]:
        return [b for b in self.blocks if b.kind == "text"]

    @property
    def horizontal_lines(self) -> List[TextLine]:
        return [ln for ln in self.lines if not ln.rotated]

    @property
    def rotated_lines(self) -> List[TextLine]:
        return [ln for ln in self.lines if ln.rotated]


def _is_rotated(direction: Tuple[float, float]) -> bool:
    dx, dy = direction
    # A horizontal line has direction (1, 0). Anything else is rotated text,
    # which in these reports means chart axis labels, not body prose.
    return abs(dx - 1.0) > 1e-3 or abs(dy) > 1e-3


def open_document(path: Path):
    """Open a PDF read-only. Raises :class:`PdfOpenError` on failure."""
    if not PYMUPDF_AVAILABLE:
        raise PdfOpenError("PyMuPDF is not installed")
    try:
        doc = fitz.open(str(path))
        if doc.needs_pass:
            raise PdfOpenError("PDF is password protected")
        if doc.page_count == 0:
            raise PdfOpenError("PDF contains no pages")
        return doc
    except PdfOpenError:
        raise
    except Exception as exc:
        raise PdfOpenError(f"{type(exc).__name__}: {exc}") from exc


def document_metadata(doc) -> Dict[str, str]:
    try:
        meta = doc.metadata or {}
    except Exception:  # pragma: no cover
        meta = {}
    return {str(k): str(v) for k, v in meta.items() if v not in (None, "")}


def page_count(doc) -> int:
    try:
        return int(doc.page_count)
    except Exception as exc:  # pragma: no cover
        raise PdfOpenError(f"page count unavailable: {exc}") from exc


def iter_page_geometry(doc, page_number_zero_based: int) -> PageGeometry:
    """Extract the layout geometry of a single page with PyMuPDF."""
    page = doc[page_number_zero_based]
    rect = page.rect
    geometry = PageGeometry(
        page_number=page_number_zero_based + 1,
        width=float(rect.width),
        height=float(rect.height),
        image_count=len(page.get_images(full=True)) if hasattr(page, "get_images") else 0,
        drawing_count=len(page.get_drawings()) if hasattr(page, "get_drawings") else 0,
        engine="pymupdf",
    )

    page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
    char_total = 0
    rotated_chars = 0

    for block_no, raw_block in enumerate(page_dict.get("blocks", [])):
        if raw_block.get("type") == 1:
            geometry.blocks.append(
                Block(bbox=tuple(raw_block.get("bbox", (0, 0, 0, 0))), block_no=block_no, kind="image")
            )
            continue

        spans: List[Span] = []
        lines: List[TextLine] = []
        for raw_line in raw_block.get("lines", []):
            line_spans: List[Span] = []
            rotated = _is_rotated(tuple(raw_line.get("dir", (1.0, 0.0))))
            for raw_span in raw_line.get("spans", []):
                text = raw_span.get("text", "")
                if not text:
                    continue
                x0, y0, x1, y1 = raw_span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                span = Span(
                    text=text,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    size=float(raw_span.get("size", 0.0)),
                    font=str(raw_span.get("font", "")),
                    bold=bool(raw_span.get("flags", 0) & 2 ** 4),
                    italic=bool(raw_span.get("flags", 0) & 2 ** 1),
                    rotated=rotated,
                )
                line_spans.append(span)
                spans.append(span)
                char_total += len(text.strip())
                if rotated:
                    rotated_chars += len(text.strip())
            if line_spans:
                line = TextLine(spans=line_spans, bbox=tuple(raw_line.get("bbox", (0, 0, 0, 0))))
                lines.append(line)

        if spans:
            text = "\n".join(line.text for line in lines)
            geometry.blocks.append(
                Block(
                    spans=spans,
                    bbox=tuple(raw_block.get("bbox", (0, 0, 0, 0))),
                    block_no=block_no,
                    kind="text",
                    text=text,
                    lines=lines,
                )
            )
            geometry.lines.extend(lines)

    geometry.rotated_char_ratio = (rotated_chars / char_total) if char_total else 0.0
    geometry.body_font_size = _body_font_size(geometry.blocks)
    geometry.raw_text = "\n".join(b.text for b in geometry.text_blocks if b.text)
    geometry.tokens = _extract_tokens(page, geometry)
    return geometry


def _extract_tokens(page, geometry: "PageGeometry") -> List[Token]:
    """Character-accurate whitespace tokens with their own bounding boxes."""
    tokens: List[Token] = []
    try:
        raw_dict = page.get_text(
            "rawdict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
        )
    except Exception:  # pragma: no cover - defensive
        return tokens
    for raw_block in raw_dict.get("blocks", []):
        if raw_block.get("type") == 1:
            continue
        for raw_line in raw_block.get("lines", []):
            rotated = _is_rotated(tuple(raw_line.get("dir", (1.0, 0.0))))
            if rotated:
                # Chart axis labels: captured separately, never mixed into prose.
                continue
            for raw_span in raw_line.get("spans", []):
                buffer: List[str] = []
                first_bbox: Optional[List[float]] = None
                last_bbox: Optional[List[float]] = None
                size = float(raw_span.get("size", 0.0))
                for ch in raw_span.get("chars", []):
                    char = ch.get("c", "")
                    bbox = ch.get("bbox")
                    if char.isspace() or not char.strip():
                        if buffer and bbox and first_bbox and last_bbox:
                            tokens.append(
                                Token(
                                    text="".join(buffer),
                                    x0=first_bbox[0],
                                    y0=first_bbox[1],
                                    x1=last_bbox[2],
                                    y1=last_bbox[3],
                                    size=size,
                                )
                            )
                        buffer, first_bbox, last_bbox = [], None, None
                        continue
                    if not buffer:
                        first_bbox = bbox
                    buffer.append(char)
                    last_bbox = bbox
                if buffer and first_bbox and last_bbox:
                    tokens.append(
                        Token(
                            text="".join(buffer),
                            x0=first_bbox[0],
                            y0=first_bbox[1],
                            x1=last_bbox[2],
                            y1=last_bbox[3],
                            size=size,
                        )
                    )
    tokens.sort(key=lambda t: (round(t.y0, 1), t.x0))
    return tokens


def _body_font_size(blocks: List[Block]) -> float:
    """Most common font size weighted by character count = body font."""
    if not blocks:
        return 0.0
    histogram: Dict[float, int] = {}
    for block in blocks:
        for span in block.spans:
            if span.rotated or not span.text.strip():
                continue
            key = round(span.size * 2) / 2.0
            histogram[key] = histogram.get(key, 0) + len(span.text.strip())
    if not histogram:
        return 0.0
    return max(histogram.items(), key=lambda kv: kv[1])[0]


def fallback_page_geometry(path: Path, page_number_zero_based: int) -> Optional[PageGeometry]:
    """Secondary engine: word-level extraction with pdfplumber."""
    if not PDFPLUMBER_AVAILABLE:
        return None
    try:
        with pdfplumber.open(str(path)) as pdf:
            if page_number_zero_based >= len(pdf.pages):
                return None
            page = pdf.pages[page_number_zero_based]
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            geometry = PageGeometry(
                page_number=page_number_zero_based + 1,
                width=float(page.width or 0.0),
                height=float(page.height or 0.0),
                engine="pdfplumber",
            )
            # Regroup words into visual lines by their top coordinate.
            lines_by_top: Dict[int, List[Dict]] = {}
            for word in words:
                key = int(round(float(word["top"]) / 3.0))
                lines_by_top.setdefault(key, []).append(word)
            for key in sorted(lines_by_top):
                group = sorted(lines_by_top[key], key=lambda w: float(w["x0"]))
                spans = [
                    Span(
                        text=str(w["text"]) + " ",
                        x0=float(w["x0"]),
                        y0=float(w["top"]),
                        x1=float(w["x1"]),
                        y1=float(w["bottom"]),
                        size=float(w.get("size", 0.0) or 0.0),
                        font=str(w.get("fontname", "")),
                    )
                    for w in group
                ]
                if not spans:
                    continue
                text = "".join(s.text for s in spans).strip()
                if not text:
                    continue
                bbox = (
                    min(s.x0 for s in spans),
                    min(s.y0 for s in spans),
                    max(s.x1 for s in spans),
                    max(s.y1 for s in spans),
                )
                line = TextLine(spans=spans, bbox=bbox)
                geometry.lines.append(line)
                geometry.blocks.append(
                    Block(spans=spans, bbox=bbox, block_no=len(geometry.blocks), kind="text", text=text)
                )
            geometry.raw_text = "\n".join(b.text for b in geometry.blocks)
            geometry.body_font_size = _body_font_size(geometry.blocks)
            geometry.image_count = len(page.images or [])
            return geometry
    except Exception:  # pragma: no cover - defensive
        return None


def page_has_graphics(path: Path, page_number_zero_based: int) -> bool:
    """True when a page contains drawings/images but almost no text."""
    geometry = None
    try:
        doc = open_document(path)
    except PdfOpenError:
        return False
    try:
        geometry = iter_page_geometry(doc, page_number_zero_based)
    finally:
        doc.close()
    return bool(geometry and (geometry.image_count or geometry.drawing_count))


def iter_pages(path: Path, use_fallback: bool = True) -> Iterator[PageGeometry]:
    """Yield page geometry for every page, with the fallback engine on failure."""
    doc = open_document(path)
    try:
        for index in range(doc.page_count):
            try:
                geometry = iter_page_geometry(doc, index)
                if use_fallback and not geometry.raw_text.strip():
                    fallback = fallback_page_geometry(path, index)
                    if fallback and fallback.raw_text.strip():
                        geometry.raw_text = fallback.raw_text
                        geometry.engine = "pymupdf+pdfplumber"
                yield geometry
            except Exception:
                geometry = fallback_page_geometry(path, index) if use_fallback else None
                if geometry is None:
                    geometry = PageGeometry(
                        page_number=index + 1, width=0.0, height=0.0, engine="failed"
                    )
                yield geometry
    finally:
        doc.close()
