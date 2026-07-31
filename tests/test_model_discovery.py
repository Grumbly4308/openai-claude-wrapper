"""Tests for model_discovery.filter_canonical."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_discovery import filter_canonical


def test_versioned_families_kept():
    ids = {"claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"}
    assert filter_canonical(ids) == sorted(ids)


def test_claude5_minorless_ids_kept():
    ids = {"claude-opus-5", "claude-opus-5[1m]", "claude-sonnet-5"}
    assert filter_canonical(ids) == sorted(ids)


def test_bare_family_aliases_below_5_dropped():
    assert filter_canonical({"claude-opus-4", "claude-sonnet-4", "claude-haiku-3"}) == []


def test_codename_families_kept():
    ids = {"claude-fable-5", "claude-mythos-5"}
    assert filter_canonical(ids) == sorted(ids)


def test_retired_majors_dropped():
    assert filter_canonical({"claude-sonnet-3-7", "claude-haiku-3-5", "claude-opus-3-0"}) == []


def test_deprecated_denylist_dropped():
    assert filter_canonical({"claude-opus-4-0", "claude-sonnet-4-0", "claude-sonnet-4-0[1m]"}) == []


def test_noise_tokens_dropped():
    noise = {
        "claude-opus-4-20250514",  # dated snapshot
        "claude-sonnet-4.6",  # dotted alias
        "claude-opus-4-8-fast",  # deployment id
        "claude-sonnet-4-6-v1",  # internal routing id
    }
    assert filter_canonical(noise) == []
