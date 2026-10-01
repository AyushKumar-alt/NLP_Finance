"""Section detection for financial / economic reports.

Detected patterns
-----------------
``1.``        ``1.1.``      ``1.1.1``
``I.``        ``IV.1``      ``Chapter 2``
uppercase standalone headings such as
``GLOBAL ECONOMIC GROWTH - FRAGILE AND DIVERGING``
title-case standalone headings such as
``Review Of Financial Markets``

Policy
------
* A line is only promoted to a section when the evidence is unambiguous.
* When nothing matches, ``section_id`` stays ``UNKNOWN`` - the pipeline never
  invents a section number to make the output look tidy.
* In the Economic Survey the numbered paragraph opener (``1.1.``) sits *below*
  the chapter/section title, so the nearest preceding standalone heading is
  used as ``section_title``, which reproduces the layout a reader sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.config import Phase1Config
from ..core.models import SectionRecord
from ..extraction.page_analyzer import PageElement, _normalise

DOTTED_NUMBER = re.compile(r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2}){1,3}\.?)\s+(?P<rest>\S.*)$")
SIMPLE_NUMBER = re.compile(r"^\s*(?P<number>(?:\d{1,2}|[IVXLC]{1,6}|[A-Z]{1,3})[.)])\s+(?P<rest>\S.*)$")
CHAPTER = re.compile(r"^\s*(?P<number>chapter\s*\d{1,2})\s*[:.\-]?\s*(?P<rest>\S.*)$", re.IGNORECASE)
SECTION_ID_SAFE = re.compile(r"[^0-9A-Za-z]+")


@dataclass
class SectionCandidate:
    number: str
    title: str
    page_number: int
    y: float
    method: str
    confidence: float
    is_heading: bool


class SectionDetector:
    def __init__(self, config: Phase1Config):
        sec = config.section("structure").get("sections", {}) or {}
        self.unknown = str(sec.get("fallback_section_id", "UNKNOWN"))
        self.max_words = int(sec.get("max_heading_words", 20))
        self.allow_upper = bool(sec.get("allow_uppercase_heading", True))
        self.allow_title = bool(sec.get("allow_titlecase_heading", True))
        self.max_title_chars = 220

    # ------------------------------------------------------------------
    def _is_title_like(self, text: str) -> bool:
        words = text.split()
        if not words or len(words) > self.max_words:
            return False
        if len(text) > self.max_title_chars:
            return False
        letter_words = [w for w in words if any(c.isalpha() for c in w)]
        if not letter_words:
            return False
        # A bare short all-caps token ("AE", "EMD", "GDP", "H1") is a table row
        # label or a chart series name, never a section title.
        if len(words) == 1 and len(letter_words[0]) <= 4 and letter_words[0].isupper():
            return False
        if text.endswith((".", ",", ";", ":", "%")) and len(words) > 3:
            return False
        letter_words = [w for w in words if any(c.isalpha() for c in w)]
        if not letter_words:
            return False
        upper_ratio = sum(1 for w in letter_words if w.isupper()) / len(letter_words)
        if self.allow_upper and upper_ratio >= 0.7:
            return True
        if self.allow_title and text.istitle():
            return True
        # Title Case With Small Words e.g. "Review Of Financial Markets"
        if self.allow_title:
            alpha = [c for c in text if c.isalpha()]
            if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) < 0.25 and len(words) <= 12:
                if sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.6:
                    return True
        return False

    def _candidates(self, ordered_elements: Sequence[Tuple[int, float, PageElement]]) -> List[SectionCandidate]:
        found: List[SectionCandidate] = []
        for page_number, y, element in ordered_elements:
            if element.kind not in ("heading", "paragraph", "other", "box"):
                continue
            text = _normalise(element.text)
            if not text:
                continue
            standalone = element.kind == "heading"

            dotted = DOTTED_NUMBER.match(text)
            if dotted and (standalone or len(text.split()) > 3):
                number = dotted.group("number").rstrip(".")
                found.append(
                    SectionCandidate(
                        number=number,
                        title="",
                        page_number=page_number,
                        y=y,
                        method="numbered_dotted_pattern",
                        confidence=0.95 if standalone else 0.8,
                        is_heading=False,
                    )
                )
                continue

            chapter = CHAPTER.match(text)
            if chapter and standalone:
                found.append(
                    SectionCandidate(
                        number=chapter.group("number"),
                        title=chapter.group("rest").strip(),
                        page_number=page_number,
                        y=y,
                        method="chapter_pattern",
                        confidence=0.9,
                        is_heading=True,
                    )
                )
                continue

            if standalone and self._is_title_like(text):
                simple = SIMPLE_NUMBER.match(text)
                found.append(
                    SectionCandidate(
                        number=simple.group("number").rstrip(").") if simple else "",
                        title=(simple.group("rest") if simple else text).strip(),
                        page_number=page_number,
                        y=y,
                        method="standalone_heading",
                        confidence=0.85 if simple else 0.7,
                        is_heading=True,
                    )
                )
                continue

            simple = SIMPLE_NUMBER.match(text)
            if simple and standalone:
                found.append(
                    SectionCandidate(
                        number=simple.group("number").rstrip(")."),
                        title=simple.group("rest").strip(),
                        page_number=page_number,
                        y=y,
                        method="standalone_numbered_heading",
                        confidence=0.8,
                        is_heading=True,
                    )
                )
        return found

    # ------------------------------------------------------------------
    def detect(
        self,
        document_id: str,
        source_id: str,
        ordered_elements: Sequence[Tuple[int, float, PageElement]],
    ) -> List[SectionRecord]:
        candidates = self._candidates(ordered_elements)
        if not candidates:
            return []

        # Attach a title to dotted-number candidates from the nearest preceding
        # standalone heading. In the Economic Survey the chapter title is printed
        # once at the top of the chapter, so the inherited title is carried for
        # the whole document until another standalone heading supersedes it.
        last_heading_title = ""
        last_heading_page = -1
        for candidate in candidates:
            if candidate.is_heading and candidate.title:
                last_heading_title = candidate.title
                last_heading_page = candidate.page_number
            elif not candidate.title:
                if last_heading_title and candidate.page_number >= last_heading_page:
                    candidate.title = last_heading_title
                else:
                    candidate.title = self.unknown
                    candidate.confidence = min(candidate.confidence, 0.5)

        sections: List[SectionRecord] = []
        used_numbers: Dict[str, int] = {}
        for candidate in candidates:
            base = candidate.number or ""
            safe = SECTION_ID_SAFE.sub("_", base).strip("_") or "NA"
            if not base:
                safe = f"S{len(sections) + 1:02d}"
            used_numbers[safe] = used_numbers.get(safe, 0) + 1
            suffix = "" if used_numbers[safe] == 1 else f"_B{used_numbers[safe]}"
            sections.append(
                SectionRecord(
                    section_id=f"{document_id}_SEC_{safe}{suffix}",
                    document_id=document_id,
                    source_id=source_id,
                    section_number=base or self.unknown,
                    section_title=candidate.title or self.unknown,
                    page_start=candidate.page_number,
                    page_end=candidate.page_number,
                    start_y=candidate.y,
                    detection_method=candidate.method,
                    confidence=round(candidate.confidence, 2),
                    is_confirmed=candidate.confidence >= 0.7,
                )
            )

        self._close_ranges(sections, ordered_elements)
        return sections

    def _close_ranges(
        self,
        sections: List[SectionRecord],
        ordered_elements: Sequence[Tuple[int, float, PageElement]],
    ) -> None:
        if not ordered_elements:
            return
        last_page = ordered_elements[-1][0]
        for index, section in enumerate(sections):
            if index + 1 < len(sections):
                next_start = sections[index + 1].page_start
                section.page_end = max(section.page_start, next_start - 1) if next_start > section.page_start else section.page_start
            else:
                section.page_end = last_page

    # ------------------------------------------------------------------
    @staticmethod
    def assign(
        sections: Sequence[SectionRecord],
        page_number: int,
        y: float,
        unknown_label: str = "UNKNOWN",
    ) -> Tuple[str, str, str]:
        """Return ``(section_id, section_number, section_title)`` for a position.

        Uses the section start coordinates ``(page_start, start_y)`` so a section
        that starts halfway down a page does not steal the text above it.
        """
        if not sections:
            return (unknown_label, unknown_label, unknown_label)
        chosen: Optional[SectionRecord] = None
        for section in sections:
            key = (section.page_start, section.start_y)
            if key <= (page_number, y):
                chosen = section
            else:
                break
        if chosen is None:
            return (unknown_label, unknown_label, unknown_label)
        return (chosen.section_id, chosen.section_number, chosen.section_title)
