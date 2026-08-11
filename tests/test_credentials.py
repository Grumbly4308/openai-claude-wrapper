"""Credential-state reporting: what the wrapper thinks it will authenticate with.

The failure this guards against is silent. A Claude Code login expires, the CLI
starts answering every turn with `exited 1` and empty stderr, and nothing in the
logs points at the file on disk — so read_credential_status has to agree with
tool_bridge.resolve_auth about *which* credential wins, and log_credential_status
has to escalate an expired one loudly enough to be found.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-cred-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402

import pytest  # noqa: E402

from src.config import (  # noqa: E402
    _LONG_LIVED_SECONDS,
    CredentialStatus,
    log_credential_status,
    read_credential_status,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Both env credentials outrank the file, so clear them for every test."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN_MINTED", raising=False)


def _write(tmp_path: Path, oauth: dict) -> Path:
    path = tmp_path / ".credentials.json"
    path.write_text(json.dumps({"claudeAiOauth": oauth}))
    return path


def _in(seconds: float) -> int:
    import time

    return int((time.time() + seconds) * 1000)


# ---------- precedence: must mirror tool_bridge.resolve_auth ----------


def test_api_key_wins_over_everything(tmp_path, monkeypatch):
    path = _write(tmp_path, {"accessToken": "tok", "expiresAt": _in(3600)})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    status = read_credential_status(path)
    assert status.kind == "api-key"
    # An API key has no expiry, so it must never be reported as short-lived.
    assert not status.expired and not status.short_lived


def test_env_token_wins_over_file(tmp_path, monkeypatch):
    path = _write(tmp_path, {"accessToken": "tok", "expiresAt": _in(3600)})
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    assert read_credential_status(path).kind == "env-token"


def test_blank_env_credential_falls_through_to_the_file(tmp_path, monkeypatch):
    path = _write(tmp_path, {"accessToken": "tok", "expiresAt": _in(3600)})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert read_credential_status(path).kind == "oauth-file"


# ---------- the file ----------


def test_missing_file_is_none(tmp_path):
    assert read_credential_status(tmp_path / "nope.json").kind == "none"


def test_unparseable_file_is_none(tmp_path):
    path = tmp_path / ".credentials.json"
    path.write_text("{not json")
    assert read_credential_status(path).kind == "none"


def test_file_without_access_token_is_none(tmp_path):
    assert read_credential_status(_write(tmp_path, {"expiresAt": _in(3600)})).kind == "none"


def test_file_without_expiry_reports_no_expiry(tmp_path):
    status = read_credential_status(_write(tmp_path, {"accessToken": "tok"}))
    assert status.kind == "oauth-file"
    assert status.expires_in is None
    # Unknown is not the same as expiring: neither flag may fire.
    assert not status.expired and not status.short_lived


def test_non_numeric_expiry_is_treated_as_unknown(tmp_path):
    status = read_credential_status(_write(tmp_path, {"accessToken": "t", "expiresAt": "soon"}))
    assert status.expires_in is None


def test_expired_token(tmp_path):
    status = read_credential_status(_write(tmp_path, {"accessToken": "t", "expiresAt": _in(-3600)}))
    assert status.expired
    assert not status.short_lived  # expired is its own state, reported separately
    assert "EXPIRED" in status.describe()


def test_short_lived_token(tmp_path):
    status = read_credential_status(_write(tmp_path, {"accessToken": "t", "expiresAt": _in(8 * 3600)}))
    assert status.short_lived and not status.expired


def test_long_lived_token(tmp_path):
    expires = _LONG_LIVED_SECONDS + 86400
    status = read_credential_status(_write(tmp_path, {"accessToken": "t", "expiresAt": _in(expires)}))
    assert not status.short_lived and not status.expired
    assert "valid for" in status.describe()


# ---------- the minted token's estimated expiry ----------
#
# setup-token prints an opaque credential nothing can introspect, so the boot
# report's only handle on its ~1-year death is the recorded mint date. The
# failure this guards against is the same silent one as the file's: a working
# deployment that dies in a year with empty stderr and no warning anywhere.


def _minted(days_ago: int) -> str:
    import datetime
    import time

    return (datetime.date.fromtimestamp(time.time()) - datetime.timedelta(days=days_ago)).isoformat()


def test_minted_date_gives_the_env_token_an_expiry_estimate(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_MINTED", _minted(10))
    status = read_credential_status()
    assert status.kind == "env-token"
    assert status.expires_in is not None and status.expires_in > 300 * 86400
    assert not status.expired and not status.short_lived
    assert "~1y" in status.describe()


def test_minted_token_warns_inside_the_last_month(monkeypatch, caplog):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_MINTED", _minted(350))
    status = read_credential_status()
    assert status.short_lived and not status.expired
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        log_credential_status("Claude", Path("/nonexistent/creds.json"))
    # The remedy is a new mint, not a re-login — the generic short-lived text
    # would send the operator to /login, which renews nothing here.
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "setup-token" in caplog.text


def test_minted_token_past_a_year_is_presumed_expired(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_MINTED", _minted(400))
    status = read_credential_status()
    assert status.expired
    assert "EXPIRED" in status.describe()


def test_unparseable_minted_date_falls_back_to_unknown_expiry(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_MINTED", "sometime last spring")
    status = read_credential_status()
    assert status.expires_in is None
    # Unknown is not the same as expiring: neither flag may fire.
    assert not status.expired and not status.short_lived


def test_without_a_minted_date_the_describe_says_how_to_get_one(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    assert "CLAUDE_CODE_OAUTH_TOKEN_MINTED" in read_credential_status().describe()


# ---------- escalation ----------


@pytest.mark.parametrize(
    "status,level",
    [
        (CredentialStatus("none"), logging.WARNING),
        (CredentialStatus("oauth-file", -60), logging.ERROR),
        (CredentialStatus("oauth-file", 3600), logging.WARNING),
        (CredentialStatus("oauth-file", _LONG_LIVED_SECONDS * 2), logging.INFO),
        (CredentialStatus("api-key"), logging.INFO),
    ],
)
def test_log_level_escalates_with_severity(status, level, tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("src.config.read_credential_status", lambda path=None: status)
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        log_credential_status("Claude")
    assert [r.levelno for r in caplog.records] == [level]


# ---------- the refresh window ----------


def test_refresh_window_is_read_and_described(tmp_path):
    status = read_credential_status(
        _write(tmp_path, {"accessToken": "t", "expiresAt": _in(8 * 3600), "refreshTokenExpiresAt": _in(29 * 86400)})
    )
    assert status.refresh_in is not None
    assert "re-login needed in" in status.describe()


def test_healthy_login_is_not_short_lived(tmp_path):
    """An 8h access token renewed by a 29d refresh token must not warn.

    This warned on every boot before the predicate keyed off the refresh window,
    which is exactly how a log line gets ignored.
    """
    status = read_credential_status(
        _write(tmp_path, {"accessToken": "t", "expiresAt": _in(8 * 3600), "refreshTokenExpiresAt": _in(29 * 86400)})
    )
    assert not status.short_lived


def test_expiring_refresh_window_is_short_lived(tmp_path):
    status = read_credential_status(
        _write(tmp_path, {"accessToken": "t", "expiresAt": _in(8 * 3600), "refreshTokenExpiresAt": _in(2 * 86400)})
    )
    assert status.short_lived


def test_without_refresh_token_the_access_expiry_is_the_deadline(tmp_path):
    status = read_credential_status(_write(tmp_path, {"accessToken": "t", "expiresAt": _in(8 * 3600)}))
    assert status.refresh_in is None and status.short_lived


# ---------- shadowing ----------


def test_env_token_shadowing_a_login_is_warned_about(tmp_path, caplog, monkeypatch):
    """The env credential wins but never renews the file, which then decays."""
    path = _write(tmp_path, {"accessToken": "t", "expiresAt": _in(8 * 3600), "refreshTokenExpiresAt": _in(29 * 86400)})
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        log_credential_status("Claude", path)
    assert "SHADOWED" in caplog.text
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.WARNING]


def test_no_shadow_warning_when_there_is_no_login_to_shadow(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        log_credential_status("Claude", tmp_path / "absent.json")
    assert "SHADOWED" not in caplog.text


def test_expired_log_names_the_remedy(tmp_path, caplog):
    path = _write(tmp_path, {"accessToken": "t", "expiresAt": _in(-60)})
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        log_credential_status("Claude", path)
    assert "setup-token" in caplog.text
