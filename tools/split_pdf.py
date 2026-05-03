#!/usr/bin/env python3
"""Split a PDF into a subset of pages, or dump its outline.

Usage:
    split_pdf.py book.pdf --outline
    split_pdf.py book.pdf out.pdf --pages 30-60
    split_pdf.py book.pdf out.pdf --pages 1-5,30-60,100-110
    split_pdf.py book.pdf out.pdf --chapter "Chapter 1"

Page numbers are 1-indexed and inclusive (matches what your PDF reader shows).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import Destination


_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def parse_ranges(spec: str, total_pages: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for piece in spec.split(","):
        m = _RANGE_RE.match(piece)
        if not m:
            raise ValueError(f"bad page range {piece!r} (expected N or N-M)")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if start < 1 or end < start:
            raise ValueError(f"bad page range {piece!r} (start>=1, end>=start)")
        if end > total_pages:
            raise ValueError(f"page {end} out of bounds (PDF has {total_pages} pages)")
        ranges.append((start, end))
    return ranges


def walk_outline(reader: PdfReader, outline=None, depth: int = 0) -> list[tuple[int, str, int]]:
    """Flatten the nested outline into [(depth, title, page_number_1indexed), ...]."""
    if outline is None:
        outline = reader.outline
    rows: list[tuple[int, str, int]] = []
    for item in outline:
        if isinstance(item, list):
            rows.extend(walk_outline(reader, item, depth + 1))
            continue
        if not isinstance(item, Destination):
            continue
        try:
            page_idx = reader.get_destination_page_number(item)
            page_num = page_idx + 1 if page_idx is not None else 0
        except Exception:
            page_num = 0
        title = (getattr(item, "title", "") or "").strip()
        if title:
            rows.append((depth, title, page_num))
    return rows


def find_chapter_range(
    reader: PdfReader, query: str
) -> tuple[int, int] | None:
    """Find a chapter by title (case-insensitive substring) and return its page range.

    The end page is one before the next outline entry at the same or shallower
    depth, or the last page of the document.
    """
    rows = walk_outline(reader)
    if not rows:
        return None
    q = query.lower().strip()
    for i, (depth, title, page) in enumerate(rows):
        if q in title.lower() and page > 0:
            end = len(reader.pages)
            for depth2, _title2, page2 in rows[i + 1 :]:
                if depth2 <= depth and page2 > page:
                    end = page2 - 1
                    break
            return (page, end)
    return None


def write_subset(
    reader: PdfReader, ranges: list[tuple[int, int]], out_path: Path
) -> int:
    writer = PdfWriter()
    pages_written = 0
    for start, end in ranges:
        for i in range(start - 1, end):
            writer.add_page(reader.pages[i])
            pages_written += 1
    with out_path.open("wb") as f:
        writer.write(f)
    return pages_written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split a PDF by page ranges or chapter title.")
    p.add_argument("input", type=Path, help="Source PDF")
    p.add_argument("output", type=Path, nargs="?", help="Destination PDF (omit with --outline)")
    p.add_argument("--pages", help="Page range(s), e.g. '30-60' or '1-5,30-60,100-110'")
    p.add_argument("--chapter", help="Extract pages spanning the first outline entry whose title contains this string")
    p.add_argument("--outline", action="store_true", help="Print the PDF outline (TOC) and exit")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"error: {args.input} does not exist", file=sys.stderr)
        return 2

    reader = PdfReader(str(args.input))
    total = len(reader.pages)

    if args.outline:
        rows = walk_outline(reader)
        if not rows:
            print(f"(no outline embedded in {args.input.name}; total pages: {total})")
            return 0
        for depth, title, page in rows:
            indent = "  " * depth
            print(f"{indent}{title}  [p.{page}]")
        return 0

    if not args.output:
        print("error: output path required (or use --outline)", file=sys.stderr)
        return 2

    if args.pages and args.chapter:
        print("error: pass --pages or --chapter, not both", file=sys.stderr)
        return 2

    if args.chapter:
        rng = find_chapter_range(reader, args.chapter)
        if rng is None:
            print(f"error: no outline entry matched {args.chapter!r}", file=sys.stderr)
            return 1
        ranges = [rng]
    elif args.pages:
        try:
            ranges = parse_ranges(args.pages, total)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        print("error: pass --pages or --chapter (or --outline)", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = write_subset(reader, ranges, args.output)
    summary = ", ".join(f"{s}-{e}" if s != e else f"{s}" for s, e in ranges)
    print(f"wrote {n} pages ({summary}) from {args.input.name} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
