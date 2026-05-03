from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger("claude_wrapper.pdf")


class PDFExtractResult(NamedTuple):
    text: str
    num_pages: int
    truncated: bool
    error: str | None


def extract_pdf_text(path: Path, max_chars: int) -> PDFExtractResult:
    """Extract text from a PDF, returning concatenated page text with markers.

    Returns truncated=True when the cumulative text exceeds max_chars (we stop
    extracting further pages once over budget so we don't hold the whole book
    in memory just to throw most of it away).
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        return PDFExtractResult("", 0, False, f"pypdf not installed: {e}")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        return PDFExtractResult("", 0, False, f"failed to open PDF: {e}")

    n_pages = len(reader.pages)
    parts: list[str] = []
    total = 0
    truncated = False

    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            page_text = f"[page {i}: extraction error: {e}]"

        page_text = page_text.strip()
        chunk = f"\n--- page {i} ---\n{page_text}\n" if page_text else f"\n--- page {i} ---\n[empty]\n"

        if total + len(chunk) > max_chars:
            truncated = True
            remaining = max(0, max_chars - total)
            if remaining > 0:
                parts.append(chunk[:remaining])
                total += remaining
            break

        parts.append(chunk)
        total += len(chunk)

    return PDFExtractResult("".join(parts), n_pages, truncated, None)
