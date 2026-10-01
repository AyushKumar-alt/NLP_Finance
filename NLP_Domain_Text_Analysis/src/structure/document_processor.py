"""Document processor: PDF -> pages -> sections -> traceable content units.

This is the module that enforces the canonical hierarchy of the project::

    SOURCE -> DOCUMENT -> PAGE -> SECTION -> CONTENT UNIT -> TEXT

Every unit id encodes its ancestry, so a retrieval hit can be walked back to
the page, the section, the document and finally the original PDF file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..cleaning.cleaner import ConservativeCleaner, detect_running_furniture
from ..core.config import Phase1Config
from ..core.models import (
    FigureRecord,
    PageRecord,
    SectionRecord,
    TableRecord,
    UnitRecord,
)
from ..extraction.page_analyzer import PageAnalyzer, PageElement, _normalise
from ..extraction.pdf_reader import (
    PdfOpenError,
    document_metadata,
    iter_page_geometry,
    open_document,
)
from ..ingestion.discovery import DiscoveredFile
from ..structure.sections import SECTION_ID_SAFE, SectionDetector
from ..core.utils import count_words, strip_private_use

UNIT_SUFFIX = {
    "paragraph": "PAR",
    "heading": "HEAD",
    "table": "T",
    "figure": "F",
    "caption": "CAP",
    "box": "BOX",
    "footnote": "FN",
    "reference": "REF",
    "other": "OTH",
}


@dataclass
class DocumentResult:
    """Everything Phase 1 produced for one PDF."""

    document_id: str
    source_id: str
    status: str = "PENDING"
    page_count: int = 0
    pages: List[PageRecord] = field(default_factory=list)
    sections: List[SectionRecord] = field(default_factory=list)
    units: List[UnitRecord] = field(default_factory=list)
    tables: List[TableRecord] = field(default_factory=list)
    figures: List[FigureRecord] = field(default_factory=list)
    raw_text: str = ""
    cleaned_text: str = ""
    page_validation: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    running_furniture: List[str] = field(default_factory=list)

    def total_words(self) -> int:
        return sum(page.word_count for page in self.pages)


class DocumentProcessor:
    def __init__(self, config: Phase1Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.analyzer = PageAnalyzer(config)
        self.sections = SectionDetector(config)
        self.cleaner = ConservativeCleaner(config)
        self.unknown = config.unknown_label

    # ------------------------------------------------------------------
    def process(self, record: DiscoveredFile) -> DocumentResult:
        doc_id = record.document_id
        result = DocumentResult(document_id=doc_id, source_id=record.source_id)

        try:
            document = open_document(Path(record.source_path))
        except PdfOpenError as exc:
            result.status = "FAILED"
            result.warnings.append(f"open failed: {exc}")
            self.logger.error(
                "Failed to open document: %s", exc,
                extra={"document_id": doc_id, "operation": "pdf_open"},
            )
            return result

        try:
            record.pdf_metadata.update(document_metadata(document))
            record.page_count = int(document.page_count)
            result.page_count = record.page_count
            geometries = []
            for index in range(record.page_count):
                geometry = iter_page_geometry(document, index)
                geometries.append(geometry)
            if geometries:
                self.logger.info(
                    "PDF extraction SUCCESS (%s pages)", record.page_count,
                    extra={"document_id": doc_id, "operation": "pdf_extraction"},
                )
            else:
                self.logger.warning(
                    "document reported zero pages",
                    extra={"document_id": doc_id, "operation": "pdf_extraction"},
                )
        except Exception as exc:
            result.status = "FAILED"
            result.warnings.append(f"page iteration failed: {exc}")
            self.logger.error(
                "PDF extraction FAILED: %s", exc,
                extra={"document_id": doc_id, "operation": "pdf_extraction"},
            )
            document.close()
            return result
        finally:
            try:
                document.close()
            except Exception:  # pragma: no cover
                pass

        # ---- per page interpretation -----------------------------------
        per_page_elements: List[List[PageElement]] = []
        raw_pages: List[str] = []
        for geometry in geometries:
            try:
                elements = self.analyzer.analyze(geometry)
            except Exception as exc:  # never let one page kill the document
                result.warnings.append(f"page {geometry.page_number} analysis failed: {exc}")
                elements = []
                self.logger.warning(
                    "page %d analysis failed: %s", geometry.page_number, exc,
                    extra={"document_id": doc_id, "operation": "page_analysis"},
                )
            per_page_elements.append(elements)
            raw_pages.append(self._raw_page_text(geometry))
            result.page_validation.append(
                self._page_validation_row(geometry, len(elements))
            )

        # ---- running headers / footers (document level) ---------------
        furniture = set()
        if bool(self.config.get("cleaning.remove_repeated_headers_footers", True)):
            furniture = self._running_furniture(geometries, per_page_elements)
            result.running_furniture = sorted(furniture)
        if furniture:
            per_page_elements = [
                [e for e in elements if _normalise(e.text).casefold() not in furniture]
                for elements in per_page_elements
            ]

        # ---- sections --------------------------------------------------
        ordered = self._ordered_elements(per_page_elements)
        result.sections = self.sections.detect(doc_id, record.source_id, ordered)

        # ---- pages + units --------------------------------------------
        table_counter = 0
        figure_counter = 0
        counters: Dict[str, int] = {}

        for geometry, raw_text, elements in zip(geometries, raw_pages, per_page_elements):
            page_id = f"{doc_id}_P{geometry.page_number:03d}"
            page_units: List[UnitRecord] = []
            scoped: Dict[str, int] = {}

            for index, element in enumerate(elements, start=1):
                section_id, section_number, section_title = self.sections.assign(
                    result.sections, geometry.page_number, element.bbox[1], self.unknown
                )
                if element.kind == "table":
                    table_counter += 1
                    element_id = f"{doc_id}_T{table_counter:03d}"
                    unit = self._table_unit(
                        element_id, record, page_id, geometry, element,
                        section_id, section_number, section_title, index,
                    )
                    result.tables.append(
                        TableRecord(
                            table_id=element_id,
                            document_id=doc_id,
                            source_id=record.source_id,
                            page_number=geometry.page_number,
                            section_id=section_id,
                            caption=element.caption,
                            label=element.label,
                            raw_text=element.text,
                            rows=len(element.table_data),
                            columns=(len(element.table_data[0]) if element.table_data else 0),
                            source_note=element.source_note,
                            table_data=element.table_data,
                            quality=element.quality,
                            quality_reason=element.quality_reason,
                            data_extracted=element.data_extracted,
                            bbox=[round(v, 2) for v in element.bbox],
                            page_start=geometry.page_number,
                            page_end=geometry.page_number + 1 if element.body_on_next_page else geometry.page_number,
                        )
                    )
                elif element.kind == "figure":
                    figure_counter += 1
                    element_id = f"{doc_id}_F{figure_counter:03d}"
                    unit = self._figure_unit(
                        element_id, record, page_id, geometry, element,
                        section_id, section_number, section_title, index,
                    )
                    result.figures.append(
                        FigureRecord(
                            figure_id=element_id,
                            document_id=doc_id,
                            source_id=record.source_id,
                            page_number=geometry.page_number,
                            section_id=section_id,
                            caption=element.caption,
                            label=element.label,
                            kind="chart" if "chart" in element.label.casefold() else "figure",
                            surrounding_text=element.text,
                            source_note=element.source_note,
                            axis_labels=element.axis_labels,
                            data_extracted=element.data_extracted,
                            data_note=element.note,
                            bbox=[round(v, 2) for v in element.bbox],
                        )
                    )
                elif element.kind == "box":
                    box_number = self._next(counters, "box")
                    unit = self._simple_unit(
                        record, page_id, geometry, element, section_id,
                        section_number, section_title, index,
                        unit_id=f"{doc_id}_BOX{box_number:03d}",
                    )
                    unit.caption = element.caption
                    unit.label = element.label
                else:
                    unit_id = self._scoped_unit_id(
                        doc_id, geometry.page_number, section_id, section_number,
                        element.kind, scoped,
                    )
                    unit = self._simple_unit(
                        record, page_id, geometry, element, section_id,
                        section_number, section_title, index, unit_id=unit_id,
                    )

                page_units.append(unit)
                counters[element.kind] = counters.get(element.kind, 0) + 1

            cleaned_page = self.cleaner.clean_text(raw_text)
            if furniture:
                from ..cleaning.cleaner import strip_running_furniture

                cleaned_page = strip_running_furniture(cleaned_page, furniture)

            page = PageRecord(
                document_id=doc_id,
                source_id=record.source_id,
                page_id=page_id,
                page_number=geometry.page_number,
                text=raw_text,
                cleaned_text=cleaned_page,
                char_count=len(raw_text),
                word_count=count_words(raw_text),
                line_count=len(raw_text.split("\n")) if raw_text else 0,
                block_count=len(geometry.blocks),
                image_count=geometry.image_count,
                body_font_size=geometry.body_font_size,
                width=geometry.width,
                height=geometry.height,
                unit_count=len(page_units),
                section_id=page_units[0].section_id if page_units else self.unknown,
                extraction_engine=geometry.engine,
            )
            result.pages.append(page)
            result.units.extend(page_units)

        result.counters = counters
        result.raw_text = self._join_pages(raw_pages, furniture)
        result.cleaned_text = self.cleaner.clean_text(result.raw_text)
        result.status = "SUCCESS" if result.pages else "EMPTY"
        if result.status == "EMPTY":
            result.warnings.append("no extractable text produced")
            self.logger.warning(
                "document produced no extractable text",
                extra={"document_id": doc_id, "operation": "document_processing"},
            )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _next(counters: Dict[str, int], key: str) -> int:
        counters[key] = counters.get(key, 0) + 1
        return counters[key]

    @staticmethod
    def _scoped_unit_id(
        document_id: str,
        page_number: int,
        section_id: str,
        section_number: str,
        unit_type: str,
        scoped: Dict[str, int],
    ) -> str:
        """Stable unit id encoding document + page + section + type + ordinal.

        ``D01_P002_SEC_1_1_PAR_001`` is the canonical form requested by the
        assignment: any unit id can be parsed back into its ancestry.
        """
        suffix = UNIT_SUFFIX.get(unit_type, "OTH")
        if unit_type in ("footnote", "reference"):
            safe_section = ""
        else:
            safe_section = SECTION_ID_SAFE.sub("_", str(section_number or "")).strip("_") or "NA"
            if safe_section.isdigit() or re.fullmatch(r"[0-9_]+", safe_section):
                safe_section = f"SEC_{safe_section}"
        key = f"{page_number}|{safe_section}|{suffix}"
        scoped[key] = scoped.get(key, 0) + 1
        ordinal = scoped[key]
        if unit_type in ("footnote", "reference"):
            return f"{document_id}_P{page_number:03d}_{suffix}{ordinal:03d}"
        return f"{document_id}_P{page_number:03d}_{safe_section}_{suffix}{ordinal:03d}"

    def _raw_page_text(self, geometry) -> str:
        """Faithful extractor output for the page (raw, never cleaned)."""
        return geometry.raw_text

    def _page_validation_row(self, geometry, unit_count: int) -> Dict[str, object]:
        text = geometry.raw_text
        char_count = len(text.strip())
        word_count = count_words(text)
        low_char = int(self.config.get("validation.low_text_char_threshold", 120))
        low_word = int(self.config.get("validation.low_text_word_threshold", 20))
        scan_char = int(self.config.get("validation.scanned_char_threshold", 60))
        scan_img_char = int(self.config.get("validation.scanned_image_char_threshold", 15))

        warnings: List[str] = []
        is_scanned = False
        if char_count <= scan_char:
            if geometry.image_count > 0 or geometry.drawing_count > 20:
                is_scanned = True
                warnings.append(
                    "almost no text but page carries images/vector drawings: "
                    "likely a scanned or image-only page"
                )
            else:
                warnings.append("page contains (almost) no extractable text")
        elif char_count <= scan_img_char and geometry.image_count > 0:
            is_scanned = True
            warnings.append("text below the scanned-page threshold with embedded images")
        if char_count < low_char or word_count < low_word:
            warnings.append("low text density (possible image/figure-only page)")
        if unit_count == 0 and char_count > 0:
            warnings.append("text was extracted but no content unit could be typed")

        return {
            "page_number": geometry.page_number,
            "text_char_count": char_count,
            "word_count": word_count,
            "is_text_extractable": char_count > 0,
            "is_likely_scanned": is_scanned,
            "image_count": geometry.image_count,
            "drawing_count": geometry.drawing_count,
            "unit_count": unit_count,
            "extraction_warning": "; ".join(warnings),
        }

    # ------------------------------------------------------------------
    def _running_furniture(
        self, geometries: Sequence, per_page_elements: Sequence[Sequence[PageElement]]
    ) -> set:
        candidates: List[Tuple[int, List[str]]] = []
        for geometry, elements in zip(geometries, per_page_elements):
            height = geometry.height or 1.0
            lines: List[str] = []
            for element in elements:
                # Running heads and folios arrive either as an isolated short
                # line or as a heading repeated at the top/bottom of every page.
                if element.kind not in ("other", "heading"):
                    continue
                if len(element.text.split()) > 8:
                    continue
                if element.bbox[1] < height * 0.09 or element.bbox[3] > height * 0.90:
                    lines.append(_normalise(element.text))
            candidates.append((geometry.page_number, [l for l in lines if l]))
        min_pages = int(self.config.get("cleaning.header_footer_match_count", 2))
        fraction = float(self.config.get("cleaning.header_footer_min_page_fraction", 0.5))
        if len(candidates) < 4:
            return set()
        return detect_running_furniture(candidates, min_pages, fraction)

    @staticmethod
    def _ordered_elements(per_page_elements: Sequence[Sequence[PageElement]]) -> List[Tuple[int, float, PageElement]]:
        ordered: List[Tuple[int, float, PageElement]] = []
        page_number = 0
        for elements in per_page_elements:
            page_number += 1
            for element in sorted(elements, key=lambda e: (round(e.bbox[1], 1), e.bbox[0])):
                ordered.append((page_number, element.bbox[1], element))
        return ordered

    def _join_pages(self, raw_pages: Sequence[str], furniture: set) -> str:
        parts = []
        for index, text in enumerate(raw_pages, start=1):
            parts.append(f"\n\n===== PAGE {index} =====\n{text}")
        joined = "".join(parts)
        return strip_private_use(joined)

    # ------------------------------------------------------------------
    def _unit_common(
        self,
        unit_id: str,
        record: DiscoveredFile,
        page_id: str,
        page_number: int,
        section_id: str,
        section_number: str,
        section_title: str,
        unit_type: str,
        text: str,
        index: int,
        element: PageElement,
    ) -> UnitRecord:
        return UnitRecord(
            unit_id=unit_id,
            document_id=record.document_id,
            source_id=record.source_id,
            filename=record.filename,
            page_id=page_id,
            page_number=page_number,
            section_id=section_id,
            section_number=section_number,
            section_title=section_title,
            unit_type=unit_type,
            text=text,
            char_count=len(text),
            word_count=count_words(text),
            unit_index=index,
            bbox=[round(v, 2) for v in element.bbox],
            font_size=round(element.font_size, 2) if element.font_size else None,
        )

    def _simple_unit(self, record, page_id, geometry, element, section_id, section_number, section_title, index, unit_id) -> UnitRecord:
        unit = self._unit_common(
            unit_id, record, page_id, geometry.page_number, section_id,
            section_number, section_title, element.kind, element.text, index, element,
        )
        unit.extraction_note = element.note
        return unit

    def _table_unit(self, unit_id, record, page_id, geometry, element, section_id, section_number, section_title, index) -> UnitRecord:
        unit = self._unit_common(
            unit_id, record, page_id, geometry.page_number, section_id,
            section_number, section_title, "table", element.text, index, element,
        )
        unit.caption = element.caption
        unit.label = element.label
        unit.source_note = element.source_note
        unit.table_data = element.table_data
        unit.table_rows = len(element.table_data)
        unit.table_columns = len(element.table_data[0]) if element.table_data else 0
        unit.table_extraction_quality = element.quality
        unit.data_extracted = element.data_extracted
        unit.extraction_note = element.note or element.quality_reason
        if element.body_on_next_page:
            unit.extraction_note = (
                f"{unit.extraction_note}; caption is the last element on the page "
                "so the grid continues on the next page"
            )
        return unit

    def _figure_unit(self, unit_id, record, page_id, geometry, element, section_id, section_number, section_title, index) -> UnitRecord:
        unit = self._unit_common(
            unit_id, record, page_id, geometry.page_number, section_id,
            section_number, section_title, "figure", element.text, index, element,
        )
        unit.caption = element.caption
        unit.label = element.label
        unit.source_note = element.source_note
        unit.axis_labels = element.axis_labels
        unit.data_extracted = element.data_extracted
        unit.extraction_note = element.note
        return unit
