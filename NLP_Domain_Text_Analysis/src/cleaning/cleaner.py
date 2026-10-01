"""Light structural cleaning (Phase 1 only).

Hard rule of this project: cleaning must be **conservative**. Phase 1 fixes
obvious extraction artefacts and nothing else. Everything that changes the
meaning of the text - lowercasing, stopword removal, stemming, lemmatization,
punctuation stripping, number removal, URL removal - belongs to the Phase 2
experiments and is switched off in ``config/phase1_config.yaml``.

What *is* repaired here:

* line endings and repeated whitespace
* repeated blank lines produced by the PDF layout
* hard-wrapped lines inside a paragraph
* publisher symbol-font glyphs (Wingdings spacing characters)
* running headers / footers that repeat on many pages of the same document

What must survive untouched:

``7.4 per cent`` ``2 per cent`` ``₹`` ``$`` ``%`` ``2025`` ``2026`` ``FY26``
``FY27`` ``Q3`` ``H1`` ``GDP`` ``GVA`` ``GFCF`` ``PFCE`` ``FDI`` ``CPI`` ``WPI``
``IIP`` ``RBI`` ``SEBI`` ``IMF`` ``UNCTAD`` ``MoSPI`` ``USD`` ``EUR`` and every
URL found in a footnote.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from ..core.config import Phase1Config
from ..core.utils import collapse_whitespace, strip_private_use

MULTI_BLANK = re.compile(r"\n{2,}")
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


class ConservativeCleaner:
    """Applies only the repairs listed in ``config: cleaning``."""

    def __init__(self, config: Phase1Config):
        self.config = config
        options = config.section("cleaning")
        self.enabled = bool(options.get("enabled", True))
        self.normalize_newlines = bool(options.get("normalize_line_endings", True))
        self.collapse_ws = bool(options.get("collapse_internal_whitespace", True))
        self.strip_trailing = bool(options.get("strip_trailing_whitespace", True))
        self.remove_blank_runs = bool(options.get("remove_repeated_blank_lines", True))
        self.max_blanks = int(options.get("max_consecutive_blank_lines", 1))
        self.join_wrapped = bool(options.get("join_wrapped_lines", True))
        self.strip_pua = bool(options.get("strip_private_use_glyphs", True))

    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        out = text
        if self.normalize_newlines:
            out = out.replace("\r\n", "\n").replace("\r", "\n")
        out = out.replace("\x00", "")
        if self.strip_pua:
            out = strip_private_use(out)
        if self.strip_trailing:
            out = "\n".join(line.rstrip() for line in out.split("\n"))
        if self.collapse_ws:
            out = collapse_whitespace(out)
            if not self.join_wrapped:
                out = "\n".join(line.strip() for line in out.split("\n"))
        if self.remove_blank_runs:
            out = MULTI_BLANK.sub("\n\n", out)
            while "\n" * (self.max_blanks + 2) in out:
                out = out.replace("\n" * (self.max_blanks + 2), "\n" * (self.max_blanks + 1))
        return out.strip()

    # ------------------------------------------------------------------
    def join_wrapped_paragraph(self, lines: Sequence[str]) -> str:
        """Repair safe intra-paragraph line wrapping of one logical paragraph."""
        parts = [line.strip() for line in lines if line.strip()]
        if not parts:
            return ""
        out = parts[0]
        for part in parts[1:]:
            if out.endswith("-") and part[:1].islower():
                out = out[:-1] + part
            else:
                out = f"{out} {part}"
        return collapse_whitespace(out)


# ----------------------------------------------------------------------
# running header / footer detection (document level)
# ----------------------------------------------------------------------
def detect_running_furniture(
    page_lines: Sequence[Tuple[int, Sequence[str]]],
    min_pages: int,
    min_page_fraction: float,
) -> Set[str]:
    """Return the normalised texts that behave like running headers/footers.

    ``page_lines`` is ``[(page_number, [line, ...]), ...]`` using *candidates*
    only - short lines that sit in the top or bottom band of the page. A line
    is furniture when it repeats on at least ``min_pages`` pages and on at least
    ``min_page_fraction`` of the document.
    """
    counter: Counter = Counter()
    for _page, lines in page_lines:
        seen_here: Set[str] = set()
        for line in lines:
            key = re.sub(r"\s+", " ", line).strip().casefold()
            if not key:
                continue
            seen_here.add(key)
        for key in seen_here:
            counter[key] += 1

    total_pages = max(1, len(page_lines))
    threshold = max(min_pages, int(total_pages * min_page_fraction))
    return {text for text, count in counter.items() if count >= threshold}


def strip_running_furniture(text: str, furniture: Set[str]) -> str:
    if not furniture:
        return text
    kept: List[str] = []
    for line in text.split("\n"):
        key = re.sub(r"\s+", " ", line).strip().casefold()
        if key and key in furniture:
            continue
        kept.append(line)
    return "\n".join(kept)
