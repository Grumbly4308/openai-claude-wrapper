"""Codex-side config: agent selection, model list, effort vocabulary, credentials.

Named test_codex_config (not test_agent_config) so it collects AFTER
test_budget.py: the suite relies on test_budget being the first module to
import src.config (see the test_sandbox_shim docstring), and an earlier-sorting
module that imports src.* would freeze SETTINGS before its env preamble runs.

The default (claude) deployment's behavior is pinned by the rest of the suite;
the pins here are (a) the selector fails closed instead of serving the wrong
vendor's models, (b) codex mode swaps the model list, effort vocabulary and
owned_by wholesale, and (c) the codex credential report mirrors the claude one
without moving a byte of the claude wordings.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import datetime
import importlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-codex-config-test-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src import capabilities, config  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """The env key outranks the file, so clear it for every test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------- agent selector ----------


def _settings(monkeypatch, **env) -> config.Settings:
    for k in (
        "CLAUDE_WRAPPER_AGENT",
        "CLAUDE_WRAPPER_DEFAULT_MODEL",
        "CLAUDE_WRAPPER_CLAUDE_BIN",
        "CLAUDE_WRAPPER_CODEX_BIN",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return config.Settings.from_env()


@pytest.mark.parametrize("agent_env", [None, "", "claude"])
def test_agent_defaults_to_claude(monkeypatch, agent_env):
    """Unset AND empty select claude — compose ${VAR:-} interpolates to ""."""
    env = {} if agent_env is None else {"CLAUDE_WRAPPER_AGENT": agent_env}
    s = _settings(monkeypatch, **env)
    assert s.agent == "claude"
    assert s.agent_bin == s.claude_bin
    assert s.default_model == "claude-opus-4-8"


def test_codex_agent_selects_codex_defaults(monkeypatch):
    s = _settings(monkeypatch, CLAUDE_WRAPPER_AGENT="codex")
    assert s.agent == "codex"
    assert s.agent_bin == s.codex_bin == "codex"
    assert s.default_model == config.CODEX_DEFAULT_MODEL == "gpt-6-astra"


def test_codex_bin_env_is_honored(monkeypatch):
    s = _settings(monkeypatch, CLAUDE_WRAPPER_AGENT="codex", CLAUDE_WRAPPER_CODEX_BIN="/opt/codex")
    assert s.agent_bin == s.codex_bin == "/opt/codex"


def test_explicit_default_model_wins_under_codex(monkeypatch):
    s = _settings(
        monkeypatch, CLAUDE_WRAPPER_AGENT="codex", CLAUDE_WRAPPER_DEFAULT_MODEL="gpt-5.2"
    )
    assert s.default_model == "gpt-5.2"


def test_unknown_agent_fails_closed(monkeypatch):
    """A typo must kill the boot naming the variable, not serve claude models."""
    with pytest.raises(RuntimeError, match="CLAUDE_WRAPPER_AGENT"):
        _settings(monkeypatch, CLAUDE_WRAPPER_AGENT="gemini")


# ---------- codex model list ----------


def test_codex_model_list_defaults_to_static(monkeypatch):
    monkeypatch.delenv("CLAUDE_WRAPPER_CODEX_MODELS", raising=False)
    assert config._codex_models_from_env() == config.CODEX_FALLBACK_MODELS


def test_codex_model_list_env_override_tolerates_whitespace(monkeypatch):
    monkeypatch.setenv("CLAUDE_WRAPPER_CODEX_MODELS", " gpt-x , gpt-y,, ")
    assert config._codex_models_from_env() == ("gpt-x", "gpt-y")


# ---------- full-module behavior under codex (config reload) ----------


_RELOAD_ENV_KEYS = (
    "CLAUDE_WRAPPER_AGENT",
    "CLAUDE_WRAPPER_DEFAULT_MODEL",
    "CLAUDE_WRAPPER_CODEX_MODELS",
)


@contextlib.contextmanager
def _reloaded_codex_config(**env):
    """src.config rebuilt under CLAUDE_WRAPPER_AGENT=codex, restored after.

    The reload swaps the module-level SETTINGS the whole process shares, so
    teardown MUST reload again with the original env (and drop capabilities'
    memoized view of it) — later-sorting test modules import src.config
    expecting the frozen claude-mode SETTINGS.
    """
    saved = {k: os.environ.get(k) for k in _RELOAD_ENV_KEYS}
    orig_settings = config.SETTINGS
    for k in _RELOAD_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ["CLAUDE_WRAPPER_AGENT"] = "codex"
    os.environ.update(env)
    importlib.reload(config)
    capabilities.reset_profile_cache()
    try:
        yield config
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config)
        # Restore the exact frozen Settings object, not merely an equal one:
        # sibling test modules overwrite env vars (CLAUDE_WRAPPER_DATA et al.)
        # in their import preambles AFTER the process-wide SETTINGS froze, so a
        # from_env() rebuild here would drift from the instance wrapper_tools/
        # deps captured at import — e.g. memory writes landing in one data_dir
        # while test_tool_bridge globs another.
        config.SETTINGS = orig_settings
        config._supported_models_cache = None
        capabilities.reset_profile_cache()


@pytest.fixture
def codex_config():
    with _reloaded_codex_config() as cfg:
        yield cfg


def test_codex_mode_advertises_only_codex_models(codex_config):
    assert codex_config.supported_models() == codex_config.CODEX_FALLBACK_MODELS
    adv = codex_config.advertised_models()
    # Exactly the 5 codex effort variants per id — no (max), no (ultracode),
    # and "none" is accepted but never advertised.
    per_id = 1 + len(codex_config.CODEX_EFFORT_LEVELS)
    assert len(adv) == len(codex_config.CODEX_FALLBACK_MODELS) * per_id
    for base in codex_config.CODEX_FALLBACK_MODELS:
        assert base in adv
        for choice in codex_config.CODEX_EFFORT_LEVELS:
            assert f"{base} ({choice})" in adv
    assert not any("(max)" in m or "(ultracode)" in m or "(none)" in m for m in adv)


def test_codex_mode_env_models_and_default_are_selectable():
    with _reloaded_codex_config(
        CLAUDE_WRAPPER_CODEX_MODELS=" gpt-x , gpt-y ",
        CLAUDE_WRAPPER_DEFAULT_MODEL="gpt-z",
    ) as cfg:
        # The configured default is appended, mirroring the claude path.
        assert cfg.supported_models() == ("gpt-x", "gpt-y", "gpt-z")
        assert cfg.SETTINGS.default_model == "gpt-z"


def test_codex_mode_effort_vocabulary(codex_config):
    assert codex_config.effort_choices_for("gpt-5.2-codex") == codex_config.CODEX_EFFORT_LEVELS
    assert codex_config.split_model_effort("gpt-5.2-codex (minimal)") == (
        "gpt-5.2-codex",
        "minimal",
    )
    # ":none" is recognized (runner-acceptance path) even though never advertised.
    assert codex_config.split_model_effort("gpt-5.2:none") == ("gpt-5.2", "none")
    # ...and the claude-only tokens are NOT: an unrecognized suffix stays in
    # the model string and fails loudly downstream instead of being stripped.
    assert codex_config.split_model_effort("gpt-5.2:max") == ("gpt-5.2:max", None)


def test_effort_recognition_is_per_agent():
    # Under claude (the process default here), the codex-only tokens must not
    # parse: origin behavior was a loud CLI rejection / 404 on the variant id,
    # and stripping the suffix would flip that into a silent success at the
    # server default — a zero-behavior-change violation.
    assert config.split_model_effort("claude-sonnet-4-6:minimal") == (
        "claude-sonnet-4-6:minimal",
        None,
    )
    assert config.split_model_effort("claude-opus-4-8 (none)") == (
        "claude-opus-4-8 (none)",
        None,
    )
    # The shared tokens still parse.
    assert config.split_model_effort("claude-opus-4-8:high") == ("claude-opus-4-8", "high")


def test_codex_mode_owner(codex_config):
    assert codex_config.model_owner() == "openai"


def test_model_owner_defaults_to_anthropic():
    assert config.model_owner() == "anthropic"


# ---------- codex credentials ----------


def _jwt(exp_in: float) -> str:
    """An unsigned JWT whose payload carries exp = now + exp_in (epoch seconds)."""
    claims = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + exp_in}).encode())
    return "eyJhbGciOiJub25lIn0." + claims.rstrip(b"=").decode() + ".sig"


def _auth_json(tmp_path: Path, data) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_apikey_file_shape(tmp_path):
    path = _auth_json(tmp_path, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-test"})
    status = config.read_codex_file_credential(path)
    assert status.kind == "api-key" and status.provider == "codex"
    assert not status.expired and not status.short_lived
    assert status.describe() == f"Codex API key in {path} (no expiry)"


def test_chatgpt_tokens_shape(tmp_path):
    # last_refresh a second in the past, so the 28d estimate floors to 27d
    # regardless of clock jitter between read and describe.
    last = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
    path = _auth_json(
        tmp_path,
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": _jwt(7260)},
            "last_refresh": last.isoformat().replace("+00:00", "Z"),
        },
    )
    status = config.read_codex_file_credential(path)
    assert status.kind == "oauth-file" and status.provider == "codex"
    assert status.expires_in is not None and 7000 < status.expires_in <= 7260
    assert status.refresh_in is not None and 26 * 86400 < status.refresh_in < 29 * 86400
    # A healthy ~28d refresh estimate must not warn — same rule as claude's.
    assert not status.expired and not status.short_lived
    assert status.describe() == (
        f"Codex ChatGPT login ({path}), access valid for 2h, re-login estimated in 27d"
    )


def test_opaque_access_token_reports_no_expiry(tmp_path):
    path = _auth_json(tmp_path, {"tokens": {"access_token": "not-a-jwt"}})
    status = config.read_codex_file_credential(path)
    assert status.kind == "oauth-file"
    assert status.expires_in is None and status.refresh_in is None
    # Unknown is not the same as expiring: neither flag may fire.
    assert not status.expired and not status.short_lived


def test_garbage_auth_json_is_none(tmp_path):
    path = _auth_json(tmp_path, "{not json")
    status = config.read_codex_file_credential(path)
    assert status.kind == "none" and status.provider == "codex"
    assert status.describe() == f"NONE — no OPENAI_API_KEY and no usable codex login in {path}"


def test_env_key_wins_over_file(tmp_path, monkeypatch):
    path = _auth_json(tmp_path, {"tokens": {"access_token": _jwt(3600)}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    status = config.read_codex_credential_status(path)
    assert status.kind == "api-key" and status.path is None
    assert status.describe() == "OPENAI_API_KEY from the environment (no expiry)"


def test_blank_env_key_falls_through_to_file(tmp_path, monkeypatch):
    """Compose always delivers the var, usually as "" — that is not a key."""
    path = _auth_json(tmp_path, {"tokens": {"access_token": _jwt(3600)}})
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert config.read_codex_credential_status(path).kind == "oauth-file"


def test_env_key_shadowing_a_chatgpt_login_is_warned_about(tmp_path, caplog, monkeypatch):
    """The env key wins and codex then never refreshes the file, which decays."""
    path = _auth_json(tmp_path, {"tokens": {"access_token": _jwt(3600)}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        config.log_codex_credential_status("Codex", path)
    assert "SHADOWED" in caplog.text
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.WARNING]


def test_no_shadow_warning_without_a_chatgpt_login(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    with caplog.at_level(logging.INFO, logger="claude_wrapper.config"):
        config.log_codex_credential_status("Codex", tmp_path / "absent.json")
    assert "SHADOWED" not in caplog.text


def test_claude_wordings_are_byte_identical():
    """provider defaults to claude, and its describe() text is pinned verbatim
    — the codex vocabulary must not have moved a byte of it."""
    assert config.CredentialStatus("api-key").describe() == "ANTHROPIC_API_KEY (no expiry)"
    assert (
        config.CredentialStatus("oauth-file", 3600.0, Path("/x")).describe()
        == "Claude Code login (/x), access valid for 1h"
    )
    assert (
        config.CredentialStatus("none", path=Path("/x")).describe()
        == "NONE — no API key, no env token, and no usable login in /x"
    )


def test_log_agent_credential_status_dispatches_by_agent(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        config, "log_credential_status", lambda where, path=None: calls.append(("claude", where))
    )
    monkeypatch.setattr(
        config,
        "log_codex_credential_status",
        lambda where, path=None: calls.append(("codex", where)),
    )
    config.log_agent_credential_status("Claude")
    monkeypatch.setattr(
        config, "SETTINGS", dataclasses.replace(config.SETTINGS, agent="codex")
    )
    config.log_agent_credential_status("Claude")
    # The dispatcher also rewrites the role label so the boot line reads right.
    assert calls == [("claude", "Claude"), ("codex", "Codex")]
