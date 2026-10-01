"""Page level structural interpretation.

This module turns positioned PDF geometry into typed page elements:

* caption anchors  - ``Table I.2:``, ``Chart I.3:``, ``Box I.1:``
* table regions    - caption -> body -> ``Source:`` note
* figure regions   - caption + surrounding text + axis labels
* box regions      - caption + body text
* footnote bands   - small type in the lower part of the page
* paragraph blocks - everything left over

Nothing is guessed silently: when a detector is not confident it returns a
``quality``/``note`` field that is carried into the corpus record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.config import Phase1Config
from ..core.models import Block, CaptionAnchor, Span, TextLine, Token
from ..core.utils import count_words, is_rotation_private, strip_private_use

NUMERIC_TOKEN = re.compile(
    r"^[-(+]?[$₹€£¥]?\d[\d,\.]*\s?(?:%|per\s?cent|pct|bps|bn|mn|cr|tr|lakh|crore|usd|inr|eur)?[.,;\)]?$",
    re.IGNORECASE,
)
SOURCE_NOTE = re.compile(r"(?i)^\s*(source|sources|note|notes)\s*[:.\-]")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")


@dataclass
class PageElement:
    """An interpreted element of a page (before ids are assigned)."""

    kind: str
    text: str
    bbox: Tuple[float, float, float, float]
    blocks: List[Block] = field(default_factory=list)
    caption: str = ""
    label: str = ""
    source_note: str = ""
    tokens: List[Token] = field(default_factory=list)
    axis_labels: List[str] = field(default_factory=list)
    table_data: List[List[str]] = field(default_factory=list)
    quality: str = ""
    quality_reason: str = ""
    data_extracted: bool = False
    note: str = ""
    font_size: float = 0.0
    consumed: List[Block] = field(default_factory=list)
    body_on_next_page: bool = False


# ----------------------------------------------------------------------
# caption detection
# ----------------------------------------------------------------------
class CaptionDetector:
    def __init__(self, config: Phase1Config):
        self.table = re.compile(config.get("structure.tables.caption_pattern"))
        self.box = re.compile(config.get("structure.boxes.caption_pattern"))
        self.figures = [re.compile(p) for p in config.get("structure.figures.caption_patterns", [])]
        self.source = re.compile(config.get("structure.tables.source_note_pattern"))
        # A body sentence such as "Table 3.7). The market ..." must not be
        # mistaken for a caption, even though it starts with the label "Table 3".
        self.in_text_reference = re.compile(
            r"(?i)^\s*(?:table|tab\.?|chart|figure|fig\.?|box|exhibit)\s+[0-9IVXA-Z][0-9A-Za-z.\s]{0,24}\)"
        )

    def classify(self, text: str) -> Optional[Tuple[str, str]]:
        """Return ``(kind, label)`` for a caption, or ``None``."""
        stripped = re.sub(r"\s+", " ", text).strip()
        if not stripped or len(stripped) > 400:
            return None
        if self.in_text_reference.match(stripped):
            return None
        match = self.table.match(stripped)
        if match:
            return ("table", stripped.split(":", 1)[0].strip())
        match = self.box.match(stripped)
        if match:
            return ("box", stripped.split(":", 1)[0].strip())
        for pattern in self.figures:
            match = pattern.match(stripped)
            if match:
                return ("figure", stripped.split(":", 1)[0].strip())
        return None

    def is_source_note(self, text: str) -> bool:
        return bool(self.source.match(re.sub(r"\s+", " ", text).strip()))


# ----------------------------------------------------------------------
# region helpers
# ----------------------------------------------------------------------
def _normalise(text: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", strip_private_use(text or "")).strip()


def _join_wrapped(parts: Sequence[str]) -> str:
    """Re-join hard-wrapped PDF lines into a single logical line.

    Hyphenated line breaks (``customs-`` / ``duty``) are repaired, because the
    PDF stores them as two lines. Everything else is joined with a space.
    """
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not out:
            out = part
            continue
        if out.endswith("-") and not out.endswith(("--", "-")) and part[:1].islower():
            out = out[:-1] + part
        else:
            out = f"{out} {part}"
    return out


def _region_bbox(blocks: Sequence[Block], tokens: Sequence[Token] = ()) -> Tuple[float, float, float, float]:
    xs0 = [b.bbox[0] for b in blocks] or ([t.x0 for t in tokens] or [0.0])
    ys0 = [b.bbox[1] for b in blocks] or ([t.y0 for t in tokens] or [0.0])
    xs1 = [b.bbox[2] for b in blocks] or ([t.x1 for t in tokens] or [0.0])
    ys1 = [b.bbox[3] for b in blocks] or ([t.y1 for t in tokens] or [0.0])
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / min(area_a, area_b)


# ----------------------------------------------------------------------
# table reconstruction
# ----------------------------------------------------------------------
class TableReconstructor:
    """Rebuilds a grid from character-accurate token geometry.

    Borderless financial tables (the Economic Survey style) have no ruling
    lines, so pdfplumber's line strategy cannot see them at all. The
    reconstruction therefore works in two steps:

    1. **Cell segmentation** - inside every visual row, a horizontal gap wider
       than ``column_gap_min_pt`` separates two cells. Narrow gaps (ordinary
       word spacing) are ignored.
    2. **Column alignment** - the midpoints of all detected gaps across all
       rows are clustered; consecutive clusters define the column bands of the
       table. Every token is then assigned to the band that contains it.

    A long row label that would span several bands is kept whole, which is the
    correct behaviour for financial row labels such as
    ``Private Final Consumption Expenditure (PFCE)``.
    """

    def __init__(self, config: Phase1Config):
        self.gap_min = float(config.get("structure.tables.column_gap_min_pt", 6.0))
        self.boundary_tol = float(config.get("structure.tables.column_tolerance_pt", 9.0))
        self.row_tol = float(config.get("structure.tables.row_tolerance_pt", 5.0))
        self.min_rows = int(config.get("structure.tables.min_rows", 2))
        self.min_cols = int(config.get("structure.tables.min_cols", 2))
        self.max_rows = int(config.get("structure.tables.max_rows_stored", 400))
        self.max_cols = int(config.get("structure.tables.max_cols_stored", 40))

    # -- step 1 --------------------------------------------------------
    def _row_cells(self, row: Sequence[Token]) -> List[List[Token]]:
        cells: List[List[Token]] = [[row[0]]]
        for previous, current in zip(row, row[1:]):
            gap = current.x0 - previous.x1
            if gap >= self.gap_min:
                cells.append([current])
            else:
                cells[-1].append(current)
        return cells

    # -- step 2 --------------------------------------------------------
    def _column_bands(self, rows: Sequence[Sequence[Token]]) -> List[Tuple[float, float]]:
        boundaries: List[float] = []
        for row in rows:
            for previous, current in zip(row, row[1:]):
                gap = current.x0 - previous.x1
                if gap >= self.gap_min:
                    boundaries.append((previous.x1 + current.x0) / 2.0)
        if not boundaries:
            return []
        boundaries.sort()
        clusters: List[List[float]] = [[boundaries[0]]]
        for value in boundaries[1:]:
            if value - clusters[-1][-1] <= self.boundary_tol:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        centers = [sum(c) / len(c) for c in clusters]
        left = min(t.x0 for row in rows for t in row)
        right = max(t.x1 for row in rows for t in row)
        bands: List[Tuple[float, float]] = []
        previous_edge = left
        for center in centers:
            bands.append((previous_edge, center))
            previous_edge = center
        bands.append((previous_edge, right + 1.0))
        return bands

    def _rows(self, tokens: Sequence[Token]) -> List[List[Token]]:
        ordered = sorted(tokens, key=lambda t: (t.cy, t.x0))
        rows: List[List[Token]] = [[ordered[0]]]
        for token in ordered[1:]:
            reference = rows[-1][0].cy
            if abs(token.cy - reference) <= self.row_tol:
                rows[-1].append(token)
            else:
                rows.append([token])
        return [sorted(row, key=lambda t: t.x0) for row in rows]

    def build(self, tokens: Sequence[Token]) -> Tuple[List[List[str]], int, int, str, str]:
        """Return ``(grid, n_rows, n_cols, quality, reason)``."""
        if not tokens:
            return [], 0, 0, "LOW", "no positioned tokens in table region"

        rows = self._rows(tokens)
        if len(rows) < self.min_rows:
            return [], len(rows), 0, "LOW", (
                f"only {len(rows)} row(s) detected; table body is not on this page "
                "or is an image"
            )

        bands = self._column_bands(rows)
        if len(bands) < self.min_cols:
            # No reliable multi-column structure - fall back to one column of
            # visual rows so nothing is lost.
            grid = [[_normalise(" ".join(t.text for t in row))] for row in rows]
            return grid, len(grid), 1, "LOW", (
                "no column gaps found; preserved as a single-column row listing"
            )

        truncated = False
        if len(bands) > self.max_cols:
            bands = bands[: self.max_cols]
            bands[-1] = (bands[-1][0], max(t.x1 for row in rows for t in row) + 1.0)
            truncated = True

        grid: List[List[str]] = []
        multi_cell_rows = 0
        for row in rows:
            cells: List[str] = ["" for _ in bands]
            # Cells are segmented per row FIRST so that a wide row label such as
            # "Private Final Consumption Expenditure (PFCE)" is never split by
            # column bands that were derived from other rows.
            for group in self._row_cells(row):
                centre = sum(t.cx for t in group) / len(group)
                index = len(bands) - 1
                for band_index, (x0, x1) in enumerate(bands):
                    if centre < x1 or band_index == len(bands) - 1:
                        index = band_index
                        break
                text = _normalise(" ".join(t.text for t in group))
                cells[index] = f"{cells[index]} {text}".strip() if cells[index] else text
            if sum(1 for cell in cells if cell) >= 2:
                multi_cell_rows += 1
            grid.append([_normalise(cell) for cell in cells])

        n_rows, n_cols = len(grid), len(bands)
        density = multi_cell_rows / max(1, n_rows)
        if n_rows >= 3 and n_cols >= 2 and density >= 0.5:
            quality, reason = "HIGH", f"{n_rows}x{n_cols} aligned grid, {density:.0%} multi-cell rows"
        elif n_rows >= 2 and n_cols >= 2 and density >= 0.25:
            quality, reason = "MEDIUM", f"partial grid, only {density:.0%} multi-cell rows"
        else:
            quality, reason = "LOW", f"sparse grid ({density:.0%} multi-cell rows); raw text preserved"

        if truncated:
            reason += f"; truncated to {n_rows}x{n_cols}"
        if len(grid) > self.max_rows:
            grid = grid[: self.max_rows]
            reason += f"; truncated to {len(grid)} rows"
        return grid, len(grid), n_cols, quality, reason


# ----------------------------------------------------------------------
# main page analyzer
# ----------------------------------------------------------------------
class PageAnalyzer:
    def __init__(self, config: Phase1Config):
        self.config = config
        self.captions = CaptionDetector(config)
        self.tables = TableReconstructor(config)
        fn = config.section("structure").get("footnotes", {}) or {}
        self.fn_bottom = float(fn.get("bottom_band_fraction", 0.72))
        self.fn_max_font_ratio = float(fn.get("max_font_ratio", 0.93))
        self.fn_pattern = re.compile(fn.get("numbered_pattern", r"^\s*(?:\(?\d{1,3}\)?[.\)])\s+\S"))
        self.fn_citation = re.compile(fn.get("citation_pattern", r"\(\d{4}\)|https?://"))
        self.fn_max_words = int(fn.get("max_words", 90))

    # ------------------------------------------------------------------
    def analyze(self, geometry) -> List[PageElement]:
        body = self.config.section("structure")
        elements: List[PageElement] = []

        horizontal_blocks = self._horizontal_blocks(geometry)
        rotated_blocks = [b for b in geometry.text_blocks if b.any_rotated]
        flow = self._order_blocks(horizontal_blocks)

        anchors = self._find_anchors(flow)
        consumed: set = set()
        columns = self._anchor_columns(anchors, geometry.width)

        # ---- table / figure / box regions ----
        for index, anchor in enumerate(anchors):
            # A side-by-side anchor shares the vertical band of its neighbour, so
            # it must not be clipped by the neighbour's top coordinate.
            if index + 1 < len(anchors) and anchors[index + 1].bbox[1] < anchor.bbox[3] - 1.0:
                next_top = geometry.height
            else:
                next_top = anchors[index + 1].bbox[1] if index + 1 < len(anchors) else geometry.height
            x_low, x_high = columns.get(index, (0.0, geometry.width))

            candidates: List[Block] = []
            for block in flow:
                if block.block_no in consumed:
                    continue
                if block.bbox[1] < anchor.bbox[3] - 1.0:
                    continue
                if block.bbox[3] > next_top + 1.0:
                    continue
                if self._is_running_head_or_foot(block, geometry):
                    continue
                # Keep only blocks whose centre lies in this anchor's column.
                # Short numeric labels (chart data callouts) that merely touch
                # the column are kept too, so no chart value is orphaned.
                cx = (block.bbox[0] + block.bbox[2]) / 2.0
                if not (x_low - 2.0 <= cx <= x_high + 2.0):
                    width = block.bbox[2] - block.bbox[0]
                    if width > 40.0 or block.bbox[2] < x_low - 2.0 or block.bbox[0] > x_high + 2.0:
                        continue
                candidates.append(block)

            if anchor.kind == "table":
                region_blocks = candidates
            elif anchor.kind == "box":
                region_blocks = self._stop_at_prose(candidates, geometry, int(self.config.get("structure.figures.max_box_words", 80)))
            else:
                region_blocks = self._stop_at_prose(
                    candidates, geometry, int(self.config.get("structure.figures.max_surrounding_words", 60))
                )

            if anchor.kind == "table":
                element = self._build_table(geometry, anchor, region_blocks)
            elif anchor.kind == "box":
                element = self._build_box(anchor, region_blocks)
            else:
                element = self._build_figure(geometry, anchor, region_blocks, rotated_blocks)
            if element is None:
                continue
            consumed.add(anchor.line_index)
            for block in element.consumed or region_blocks:
                consumed.add(block.block_no)
            elements.append(element)

        # ---- footnotes (bottom band, small type) ----
        for block in self._footnote_blocks(flow, geometry):
            if block.block_no in consumed:
                continue
            consumed.add(block.block_no)
            elements.append(self._build_footnote(block))

        # ---- remaining text = paragraphs / headings ----
        for block in flow:
            if block.block_no in consumed:
                continue
            for piece in self._split_block(block):
                if piece.kind == "other" and self._is_page_number_label(piece, geometry):
                    # Page-number furniture: kept in raw/cleaned text, excluded
                    # from the retrieval units so it cannot pollute the corpus.
                    continue
                if piece.kind == "other" and self._is_reference_marker(piece, geometry):
                    piece.kind = "reference"
                    piece.quality = "MEDIUM"
                    piece.quality_reason = (
                        "isolated footnote/reference marker in the lower page band"
                    )
                elements.append(piece)

        elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
        return elements

    def _stop_at_prose(self, blocks: Sequence[Block], geometry, max_words: int) -> List[Block]:
        """Truncate a figure/box region when real body prose starts."""
        text_width = max(1.0, (geometry.width or 1.0))
        kept: List[Block] = []
        for block in blocks:
            words = len(_normalise(block.text).split())
            width_ratio = (block.bbox[2] - block.bbox[0]) / text_width
            if kept and words >= max_words and width_ratio >= 0.55:
                break
            kept.append(block)
        return kept

    def _is_page_number_label(self, element: PageElement, geometry) -> bool:
        """Folio / page-number furniture, including repeated folios such as ``4 4``."""
        text = _normalise(element.text)
        if not re.fullmatch(r"[\d\s.,;|/\u2013\u2014-]{1,12}", text):
            return False
        numbers = re.findall(r"\d{1,4}", text)
        if not numbers or len(set(numbers)) > 1:
            return False
        height = geometry.height or 1.0
        return element.bbox[3] > height * 0.88 or element.bbox[1] < height * 0.08

    def _is_reference_marker(self, element: PageElement, geometry) -> bool:
        text = _normalise(element.text)
        if not re.fullmatch(r"[\d\s.,;|/\u2013\u2014-]{1,12}", text):
            return False
        height = geometry.height or 1.0
        return element.bbox[1] > height * self.fn_bottom

    # ------------------------------------------------------------------
    def _horizontal_blocks(self, geometry) -> List[Block]:
        result: List[Block] = []
        for block in geometry.text_blocks:
            spans = [s for s in block.spans if not s.rotated and s.text.strip()]
            if not spans:
                continue
            lines = self._lines_of(block, spans)
            text = "\n".join(line.text for line in lines)
            bbox = (
                min(s.x0 for s in spans),
                min(s.y0 for s in spans),
                max(s.x1 for s in spans),
                max(s.y1 for s in spans),
            )
            result.append(
                Block(
                    spans=spans,
                    bbox=bbox,
                    block_no=block.block_no,
                    kind="text",
                    text=text,
                    lines=[ln for ln in (block.lines or []) if not ln.rotated] or lines,
                )
            )
        return result

    @staticmethod
    def _lines_of(block: Block, spans: Sequence[Span]) -> List[TextLine]:
        if block.lines:
            return [ln for ln in block.lines if not ln.rotated]
        grouped: Dict[int, List[Span]] = {}
        for span in spans:
            grouped.setdefault(int(round(span.cy / 3.0)), []).append(span)
        lines = []
        for key in sorted(grouped):
            group = sorted(grouped[key], key=lambda s: s.x0)
            lines.append(
                TextLine(
                    spans=group,
                    bbox=(
                        min(s.x0 for s in group),
                        min(s.y0 for s in group),
                        max(s.x1 for s in group),
                        max(s.y1 for s in group),
                    ),
                )
            )
        return lines

    @staticmethod
    def _order_blocks(blocks: Sequence[Block]) -> List[Block]:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    def _is_running_head_or_foot(self, block: Block, geometry) -> bool:
        """Page numbers and other marginal furniture in the top/bottom bands."""
        height = geometry.height or 1.0
        top_band = block.y1 < height * 0.08
        bottom_band = block.y0 > height * 0.92
        if not (top_band or bottom_band):
            return False
        if block.y1 - block.y0 > height * 0.05:
            return False
        text = _normalise(block.text)
        if not text:
            return True
        return bool(
            re.fullmatch(r"(?:page\s*)?\d{1,4}|[ivxlcdm]{1,7}|\d{1,4}\s*\|\s*[A-Za-z ]{0,30}", text, re.IGNORECASE)
        )

    # ------------------------------------------------------------------
    def _find_anchors(self, flow: Sequence[Block]) -> List[CaptionAnchor]:
        """Detect caption anchors, splitting side-by-side captions.

        Economic Survey pages place two charts (or two tables) next to each
        other, and each caption lives on its own visual line. Splitting per
        line keeps ``Chart I.7`` and ``Chart I.8`` as two separate units
        instead of one merged caption.
        """
        anchors: List[CaptionAnchor] = []
        for block in flow:
            lines = self._lines_of(block, block.spans)
            if not lines:
                continue
            per_line: List[Tuple[TextLine, str, str]] = []
            for line in lines:
                text = _normalise(line.text)
                classified = self.captions.classify(text)
                if classified:
                    per_line.append((line, classified[0], classified[1]))
            if per_line:
                for line, kind, label in per_line:
                    text = _normalise(line.text)
                    anchors.append(
                        CaptionAnchor(
                            kind=kind,
                            label=label,
                            caption=self._complete_caption(text, lines, line),
                            bbox=(line.x0, line.y0, line.x1, line.y1),
                            line_index=block.block_no,
                        )
                    )
                continue
            joined = _normalise(" ".join(line.text for line in lines))
            classified = self.captions.classify(joined)
            if classified:
                anchors.append(
                    CaptionAnchor(
                        kind=classified[0],
                        label=classified[1],
                        caption=joined,
                        bbox=block.bbox,
                        line_index=block.block_no,
                    )
                )
        anchors.sort(key=lambda a: (round(a.bbox[1], 1), a.bbox[0]))
        return anchors

    def _complete_caption(self, text: str, lines: Sequence[TextLine], target: TextLine) -> str:
        """Re-attach the wrapped continuation lines of a caption."""
        parts = [text]
        seen = False
        for line in lines:
            if line is target:
                seen = True
                continue
            if not seen:
                continue
            if self.captions.classify(_normalise(line.text)):
                break
            parts.append(_normalise(line.text))
        return _join_wrapped(parts)

    @staticmethod
    def _anchor_columns(anchors: Sequence[CaptionAnchor], page_width: float) -> Dict[int, Tuple[float, float]]:
        """Horizontal band owned by each anchor (handles side-by-side items).

        A caption with no horizontal neighbour owns the full text width - its own
        caption bbox is only as wide as the caption, not as wide as the content.
        """
        result: Dict[int, Tuple[float, float]] = {}
        for index, anchor in enumerate(anchors):
            left, right = 0.0, page_width
            neighbours = [
                other
                for other_index, other in enumerate(anchors)
                if other_index != index
                and other.bbox[1] < anchor.bbox[3]
                and other.bbox[3] > anchor.bbox[1]
            ]
            for other in neighbours:
                if other.bbox[2] <= anchor.bbox[0]:
                    # Neighbour sits entirely to the left of this anchor.
                    left = (other.bbox[2] + anchor.bbox[0]) / 2.0 - 1.0
                else:
                    right = (other.bbox[2] + anchor.bbox[2]) / 2.0 + 1.0
            result[index] = (min(left, 0.0), max(right, page_width if right >= page_width else right))
        return result

    # ------------------------------------------------------------------
    def _tokens_in(self, geometry, bbox, exclude_rotated: bool = True) -> List[Token]:
        return [
            token
            for token in geometry.tokens
            if token.y0 >= bbox[1] - 2.0
            and token.y1 <= bbox[3] + 2.0
            and token.x0 >= bbox[0] - 4.0
            and token.x1 <= bbox[2] + 4.0
        ]

    def _build_table(self, geometry, anchor: CaptionAnchor, region_blocks: List[Block]) -> Optional[PageElement]:
        # Split the region at a trailing "Source:" note.
        source_note = ""
        body_blocks = list(region_blocks)
        consumed_blocks = list(region_blocks)
        for block in region_blocks:
            text = _normalise(block.text)
            if self.captions.is_source_note(text) or SOURCE_NOTE.match(text):
                source_note = text
                body_blocks = [b for b in region_blocks if b.bbox[1] < block.bbox[1] - 1.0]
                # Everything up to and including the source note belongs to the
                # table; anything below (footnotes, page furniture) does not.
                consumed_blocks = [
                    b for b in region_blocks if b.bbox[1] <= block.bbox[1] + 1.0
                ]
                break

        if not body_blocks:
            caption_is_last = True
            for block in region_blocks:
                if block.bbox[1] < anchor.bbox[3] - 1.0:
                    caption_is_last = False
                    break
            reason = (
                "caption found but no table body was positioned on the page "
                "(table body is on the following page or is a pure image)"
            )
            return PageElement(
                kind="table",
                text=anchor.caption,
                bbox=anchor.bbox,
                caption=anchor.caption,
                label=anchor.label,
                source_note=source_note,
                quality="LOW",
                quality_reason=reason,
                note="caption-only preservation" + (
                    "; caption is the last element on the page so the grid continues "
                    "on the next page" if caption_is_last else ""
                ),
                body_on_next_page=caption_is_last,
                consumed=consumed_blocks,
            )

        region = _region_bbox(body_blocks)
        tokens = self._tokens_in(geometry, region)
        if not tokens:
            tokens = [
                Token(text=_normalise(line.text), x0=line.x0, y0=line.y0, x1=line.x1, y1=line.y1, size=line.size)
                for block in body_blocks
                for line in self._lines_of(block, block.spans)
            ]
        grid, n_rows, n_cols, quality, reason = self.tables.build(tokens)
        raw_text = "\n".join(_normalise(b.text) for b in body_blocks)
        raw_text = re.sub(r"\n{2,}", "\n", raw_text)
        data_extracted = quality == "HIGH"
        return PageElement(
            kind="table",
            text=raw_text,
            bbox=region,
            blocks=body_blocks,
            caption=anchor.caption,
            label=anchor.label,
            source_note=source_note,
            tokens=tokens,
            table_data=grid,
            quality=quality,
            quality_reason=reason,
            data_extracted=data_extracted,
            note="" if data_extracted else "numeric cell layout not reliably recovered; raw text preserved",
            consumed=consumed_blocks,
        )

    def _build_box(self, anchor: CaptionAnchor, region_blocks: List[Block]) -> Optional[PageElement]:
        body = [b for b in region_blocks if not self.captions.is_source_note(_normalise(b.text))]
        text = self._join_blocks(body)
        return PageElement(
            kind="box",
            text=text,
            bbox=_region_bbox(body) if body else anchor.bbox,
            blocks=body,
            caption=anchor.caption,
            label=anchor.label,
            quality="HIGH" if text else "LOW",
            quality_reason="box body captured" if text else "box caption only",
            note="" if text else "caption-only preservation",
        )

    def _build_figure(
        self,
        geometry,
        anchor: CaptionAnchor,
        region_blocks: List[Block],
        rotated_blocks: Sequence[Block],
    ) -> Optional[PageElement]:
        source_note = ""
        body = list(region_blocks)
        consumed_blocks = list(region_blocks)
        for block in region_blocks:
            text = _normalise(block.text)
            if self.captions.is_source_note(text) or SOURCE_NOTE.match(text):
                source_note = text
                body = [b for b in region_blocks if b.bbox[1] < block.bbox[1] - 1.0]
                consumed_blocks = [
                    b for b in region_blocks if b.bbox[1] <= block.bbox[1] + 1.0
                ]
                break

        region = _region_bbox(body) if body else anchor.bbox
        axis_labels: List[str] = []
        if bool(self.config.get("structure.figures.collect_axis_labels", True)):
            axis_labels = self._axis_labels(region, rotated_blocks)

        surrounding = self._join_blocks(body)
        surrounding = re.sub(r"\s+", " ", surrounding).strip()
        axis_text = " ".join(axis_labels)

        # A figure only counts as "data extracted" when a genuine tabular
        # layout of numbers was recovered - never merely because numbers exist.
        data_extracted = False
        note = (
            "only caption/source/axis labels recovered; chart data values are NOT "
            "claimed as accurately extracted"
        )
        grid, n_rows, n_cols, quality, reason = self.tables.build(self._tokens_in(geometry, region))
        if quality == "HIGH" and n_rows >= 3 and n_cols >= 2:
            data_extracted = True
            note = f"tabular labels recovered ({n_rows}x{n_cols}); verify before use"

        return PageElement(
            kind="figure",
            text=surrounding or anchor.caption,
            bbox=region,
            blocks=body,
            caption=anchor.caption,
            label=anchor.label,
            source_note=source_note,
            axis_labels=axis_labels,
            quality=quality,
            quality_reason=reason,
            data_extracted=data_extracted,
            note="" if data_extracted else note,
            consumed=consumed_blocks,
        )

    def _axis_labels(self, region, rotated_blocks: Sequence[Block]) -> List[str]:
        labels: List[str] = []
        for block in rotated_blocks:
            if not self._overlaps(block.bbox, region):
                continue
            text = _normalise(block.text)
            if text and not is_rotation_private(text):
                labels.append(text)
        # Rotated text written bottom-up decodes as reversed characters; keep it
        # but flag it rather than mixing it into prose.
        return labels[:120]

    @staticmethod
    def _overlaps(a, b, tolerance: float = 6.0) -> bool:
        return not (
            a[2] < b[0] - tolerance
            or a[0] > b[2] + tolerance
            or a[3] < b[1] - tolerance
            or a[1] > b[3] + tolerance
        )

    @staticmethod
    def _join_blocks(blocks: Sequence[Block]) -> str:
        parts = [_normalise(b.text).replace("\n", " ") for b in blocks]
        return _join_wrapped([p for p in parts if p])

    # ------------------------------------------------------------------
    def _footnote_blocks(self, flow: Sequence[Block], geometry) -> List[Block]:
        band = geometry.height * self.fn_bottom
        body_size = geometry.body_font_size or 10.0
        selected: List[Block] = []
        for block in flow:
            if block.y0 < band:
                continue
            if block.max_size > body_size * self.fn_max_font_ratio:
                continue
            text = _normalise(block.text)
            if self._looks_like_citation(text):
                selected.append(block)
        return selected

    def _looks_like_citation(self, text: str) -> bool:
        if not text:
            return False
        if self.fn_pattern.match(text):
            return True
        if text.startswith(("Note", "Notes", "Source", "SOURCE")):
            return True
        words = len(text.split())
        if words <= int(self.fn_max_words) and self.fn_citation.search(text):
            return True
        return False

    def _build_footnote(self, block: Block) -> PageElement:
        text = _join_wrapped([_normalise(block.text).replace("\n", " ")])
        has_url = bool(URL_RE.search(text))
        return PageElement(
            kind="footnote",
            text=text,
            bbox=block.bbox,
            blocks=[block],
            quality="HIGH",
            quality_reason="small type in the lower page band",
            note="URL/reference preserved verbatim" if has_url else "",
            font_size=block.max_size,
        )

    # ------------------------------------------------------------------
    def _split_block(self, block: Block) -> List[PageElement]:
        """Split one PDF block into paragraph elements and heading elements."""
        lines = self._lines_of(block, block.spans)
        if not lines:
            return []
        groups: List[List[TextLine]] = [[lines[0]]]
        gap_factor = float(self.config.get("structure.paragraphs.paragraph_gap_factor", 1.45))
        for previous, current in zip(lines, lines[1:]):
            line_gap = current.y0 - previous.y1
            typical = max(1.0, previous.size * 1.35)
            if line_gap > typical * gap_factor:
                groups.append([current])
            else:
                groups[-1].append(current)

        elements: List[PageElement] = []
        for group in groups:
            text = _join_wrapped([_normalise(line.text) for line in group])
            if not text:
                continue
            size = max(line.size for line in group)
            group_bbox = (
                min(l.x0 for l in group),
                min(l.y0 for l in group),
                max(l.x1 for l in group),
                max(l.y1 for l in group),
            )
            if self._looks_like_heading(text, group, size):
                elements.append(
                    PageElement(
                        kind="heading",
                        text=text,
                        bbox=group_bbox,
                        blocks=[block],
                        font_size=size,
                        quality="HIGH",
                        quality_reason="standalone short line above body text",
                    )
                )
                continue

            min_words = int(self.config.get("structure.paragraphs.min_paragraph_words", 4))
            min_chars = int(self.config.get("structure.paragraphs.min_paragraph_chars", 40))
            if count_words(text) < min_words and len(text) < min_chars:
                # A short line that is not a heading: keep it as ``other`` so no
                # text is ever dropped.
                elements.append(
                    PageElement(
                        kind="other",
                        text=text,
                        bbox=group_bbox,
                        blocks=[block],
                        font_size=size,
                        quality="MEDIUM",
                        quality_reason="short line: kept verbatim, not promoted to a paragraph",
                    )
                )
                continue

            elements.append(
                PageElement(
                kind="paragraph",
                text=text,
                bbox=group_bbox,
                blocks=[block],
                    font_size=size,
                    quality="HIGH",
                    quality_reason="wrapped lines merged inside one PDF block",
                )
            )
        return elements

    def _looks_like_heading(self, text: str, lines: Sequence[TextLine], size: float) -> bool:
        sec = self.config.section("structure").get("sections", {}) or {}
        max_words = int(sec.get("max_heading_words", 20))
        min_ratio = float(sec.get("min_heading_font_ratio", 1.0))
        words = text.split()
        if not words or len(words) > max_words:
            return False
        if text.endswith((".", ",", ";")) and len(words) > 3:
            return False
        standalone = len(lines) == 1
        all_caps = sum(1 for w in words if w.isupper() and any(c.isalpha() for c in w)) >= max(
            1, int(0.7 * len(words))
        )
        titlecase = text.istitle()
        large_enough = size >= min_ratio * 9.0
        numbered = bool(re.match(r"^(?:\(?\d{1,2}[.)]|[IVXLC]{1,6}[.)])\s+\S", text))
        if numbered and standalone:
            return True
        if standalone and large_enough and (all_caps or titlecase):
            return True
        return False
