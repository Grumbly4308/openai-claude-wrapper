"""Regression tests for FilePayload.file_data accepting both data URLs and raw base64.

OpenAI's spec for inline file content uses a data URL
(``data:application/pdf;base64,...``); some clients send raw base64 with the
mime in a sibling field. The wrapper must accept both forms or it'll silently
fail when OWUI's "OpenAI PDF" / native-file mode is enabled.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Match the env setup the other test file uses so config doesn't try to write to /data.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-pdftest-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.converters import MessagePreparer  # noqa: E402
from src.file_store import FileStore  # noqa: E402


# A minimal valid PDF — pypdf can open this even though it has no real pages.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
    b"xref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n"
    b"trailer<</Root 1 0 R/Size 3>>\n"
    b"startxref\n95\n%%EOF\n"
)


def _make_preparer(tmp: Path) -> MessagePreparer:
    store = FileStore(tmp / "files")
    return MessagePreparer(file_store=store, workspace_root=tmp / "workspace")


def test_materialize_file_data_accepts_raw_base64(tmp_path: Path) -> None:
    prep = _make_preparer(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "uploads").mkdir(parents=True)

    raw_b64 = base64.b64encode(MINIMAL_PDF).decode()
    path, mime = asyncio.run(
        prep._materialize_file_data(raw_b64, "application/pdf", "doc.pdf", workspace)
    )

    assert path.read_bytes() == MINIMAL_PDF
    assert mime == "application/pdf"
    assert path.suffix == ".pdf"


def test_materialize_file_data_accepts_data_url(tmp_path: Path) -> None:
    prep = _make_preparer(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "uploads").mkdir(parents=True)

    data_url = "data:application/pdf;base64," + base64.b64encode(MINIMAL_PDF).decode()
    # No mime passed in sibling field — must be picked up from the data URL.
    path, mime = asyncio.run(
        prep._materialize_file_data(data_url, None, "doc.pdf", workspace)
    )

    assert path.read_bytes() == MINIMAL_PDF
    assert mime == "application/pdf"
    assert path.suffix == ".pdf"


def test_materialize_file_data_data_url_overrides_default_octet_stream(tmp_path: Path) -> None:
    """When the caller passes no mime hint, the data-URL mime must win."""
    prep = _make_preparer(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "uploads").mkdir(parents=True)

    data_url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n").decode()
    _path, mime = asyncio.run(
        prep._materialize_file_data(data_url, None, "img.png", workspace)
    )
    assert mime == "image/png"


def test_materialize_file_data_explicit_mime_wins_over_data_url(tmp_path: Path) -> None:
    """If the caller passes mime in the sibling field, prefer it (caller knows best)."""
    prep = _make_preparer(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "uploads").mkdir(parents=True)

    data_url = "data:application/octet-stream;base64," + base64.b64encode(MINIMAL_PDF).decode()
    _path, mime = asyncio.run(
        prep._materialize_file_data(data_url, "application/pdf", "doc.pdf", workspace)
    )
    assert mime == "application/pdf"


def test_materialize_file_data_rejects_garbage(tmp_path: Path) -> None:
    prep = _make_preparer(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "uploads").mkdir(parents=True)

    with pytest.raises(ValueError):
        asyncio.run(
            prep._materialize_file_data("not-base64!!!@@@###", "application/pdf", "x.pdf", workspace)
        )
