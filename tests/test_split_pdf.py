"""Unit tests for the split_pdf CLI tool.

We construct a synthetic PDF in-memory rather than relying on a fixture file —
this keeps the tests hermetic and lets us assert exact page counts.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


pytest.importorskip("pypdf")

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.split_pdf import (  # noqa: E402
    find_chapter_range,
    parse_ranges,
    walk_outline,
    write_subset,
)


def _make_pdf(num_pages: int) -> bytes:
    """Build an N-page PDF with blank pages in memory."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_pdf_with_outline(tmp_path: Path) -> Path:
    """Build a 20-page PDF with a 3-entry outline at pages 1, 6, and 14."""
    src = tmp_path / "book.pdf"
    src.write_bytes(_make_pdf(20))

    reader = PdfReader(str(src))
    writer = PdfWriter(clone_from=reader)
    writer.add_outline_item("Chapter 1: Intro", 0)
    writer.add_outline_item("Chapter 2: Methods", 5)
    writer.add_outline_item("Chapter 3: Results", 13)

    out = tmp_path / "book_with_outline.pdf"
    with out.open("wb") as f:
        writer.write(f)
    return out


def test_parse_ranges_single():
    assert parse_ranges("30-60", 100) == [(30, 60)]


def test_parse_ranges_single_page():
    assert parse_ranges("42", 100) == [(42, 42)]


def test_parse_ranges_multi():
    assert parse_ranges("1-5,30-60,100-110", 200) == [(1, 5), (30, 60), (100, 110)]


def test_parse_ranges_whitespace_tolerant():
    assert parse_ranges(" 1 - 5 , 30 - 60 ", 100) == [(1, 5), (30, 60)]


def test_parse_ranges_rejects_zero():
    with pytest.raises(ValueError):
        parse_ranges("0-5", 100)


def test_parse_ranges_rejects_inverted():
    with pytest.raises(ValueError):
        parse_ranges("10-5", 100)


def test_parse_ranges_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        parse_ranges("1-200", 100)


def test_parse_ranges_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ranges("abc", 100)


def test_write_subset_single_range(tmp_path: Path):
    src = tmp_path / "src.pdf"
    src.write_bytes(_make_pdf(50))
    out = tmp_path / "out.pdf"

    reader = PdfReader(str(src))
    n = write_subset(reader, [(10, 20)], out)

    assert n == 11
    result = PdfReader(str(out))
    assert len(result.pages) == 11


def test_write_subset_multi_range(tmp_path: Path):
    src = tmp_path / "src.pdf"
    src.write_bytes(_make_pdf(100))
    out = tmp_path / "out.pdf"

    reader = PdfReader(str(src))
    n = write_subset(reader, [(1, 5), (50, 55), (95, 100)], out)

    assert n == 17  # 5 + 6 + 6
    result = PdfReader(str(out))
    assert len(result.pages) == 17


def test_walk_outline(tmp_path: Path):
    pdf = _make_pdf_with_outline(tmp_path)
    reader = PdfReader(str(pdf))

    rows = walk_outline(reader)
    titles = [(title, page) for _depth, title, page in rows]
    assert titles == [
        ("Chapter 1: Intro", 1),
        ("Chapter 2: Methods", 6),
        ("Chapter 3: Results", 14),
    ]


def test_find_chapter_range_substring(tmp_path: Path):
    pdf = _make_pdf_with_outline(tmp_path)
    reader = PdfReader(str(pdf))

    # First chapter spans up to the page before the next one.
    assert find_chapter_range(reader, "intro") == (1, 5)
    # Middle chapter likewise.
    assert find_chapter_range(reader, "Methods") == (6, 13)
    # Last chapter runs to the end of the document.
    assert find_chapter_range(reader, "Results") == (14, 20)


def test_find_chapter_range_no_match(tmp_path: Path):
    pdf = _make_pdf_with_outline(tmp_path)
    reader = PdfReader(str(pdf))
    assert find_chapter_range(reader, "Nonexistent") is None


def test_walk_outline_empty(tmp_path: Path):
    src = tmp_path / "no_outline.pdf"
    src.write_bytes(_make_pdf(5))
    reader = PdfReader(str(src))
    assert walk_outline(reader) == []
