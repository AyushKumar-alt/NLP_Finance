"""Cleaning policy tests - Phase 1 must NOT perform Phase 2 transformations."""

from __future__ import annotations

from src.cleaning.cleaner import ConservativeCleaner, detect_running_furniture


def test_cleaning_keeps_numbers_currency_and_abbreviations(pipeline):
    cleaner = ConservativeCleaner(pipeline["config"])
    source = (
        "GDP  grew  at  7.4 per cent\u00a0in FY26.\r\n"
        "GFCF \u20b9346 lakh crore, USD 1.2 bn, CPI 2%.\r\n"
        "See https://www.rbi.org.in for Q3 / H1 / WPI / IIP / FDI / MoSPI / UNCTAD.\r\n"
    )
    out = cleaner.clean_text(source)
    for token in ("7.4", "per cent", "FY26", "GFCF", "₹346", "USD", "CPI", "2%", "Q3", "H1", "WPI", "IIP", "FDI", "MoSPI", "UNCTAD"):
        assert token in out, f"{token} must survive Phase 1 cleaning"
    assert "https://www.rbi.org.in" in out
    assert out == out.lower() or "GDP" in out  # case is never destroyed


def test_cleaning_does_not_lowercase_or_strip_punctuation(pipeline):
    cleaner = ConservativeCleaner(pipeline["config"])
    out = cleaner.clean_text("RBI, IMF and SEBI: policy rates!\nU.S. GDP rose.")
    assert out == "RBI, IMF and SEBI: policy rates!\nU.S. GDP rose."


def test_cleaning_repairs_line_endings_and_duplicate_blank_lines(pipeline):
    cleaner = ConservativeCleaner(pipeline["config"])
    out = cleaner.clean_text("line one\r\n\r\n\r\n\r\nline two\r\n\r\n")
    assert "\r" not in out
    assert out == "line one\n\nline two"


def test_private_use_glyphs_are_replaced(pipeline):
    cleaner = ConservativeCleaner(pipeline["config"])
    assert cleaner.clean_text("Chapter 2 \uf038 Review") == "Chapter 2 Review"


def test_running_furniture_detection(pipeline):
    pages = [
        (1, ["Economic Survey 2025-26", "Body line one"]),
        (2, ["Economic Survey 2025-26", "Body line two"]),
        (3, ["Economic Survey 2025-26", "Body line three"]),
        (4, ["Economic Survey 2025-26", "Body line four"]),
    ]
    furniture = detect_running_furniture(pages, min_pages=3, min_page_fraction=0.5)
    assert furniture == {"economic survey 2025-26"}
