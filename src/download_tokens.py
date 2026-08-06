"""Per-file capability tokens for browser-clickable download links.

A markdown link is opened by a browser, which sends no Authorization header.
Rather than put an API key in chat text (it would grant /v1/chat/completions
and enumeration + deletion of every stored file), each link carries an HMAC
over exactly one file id and one expiry. A leaked link costs one file until it
expires; forging one needs the signing key.

The expiry is *inside* the MAC and re-derived from the presented value, so a
holder can neither extend it nor forge ``exp=0`` onto a TTL'd deployment.

Pure functions taking the key and TTL as arguments — no SETTINGS import, so the
caller decides configuration and these stay trivially unit-testable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional

# Bumped if the signed payload's shape ever changes, so old links fail closed
# instead of being reinterpreted.
_ALG = "v1"


def signature(file_id: str, exp: int, key: str) -> str:
    mac = hmac.new(key.encode(), f"{_ALG}\n{file_id}\n{exp}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def mint(file_id: str, key: str, ttl: int) -> str:
    """Query string ("exp=…&sig=…") granting this one file; "" when unsigned.

    An empty key means the deployment has no API keys configured, i.e. auth is
    off and the link needs no capability to work.
    """
    if not key:
        return ""
    exp = int(time.time()) + ttl if ttl > 0 else 0  # 0 = never expires, and is signed
    return f"exp={exp}&sig={signature(file_id, exp, key)}"


def verify(file_id: str, exp_raw: Optional[str], sig: Optional[str], key: str) -> bool:
    """Whether (exp, sig) is a valid, unexpired capability for `file_id`."""
    # `sig` arrives percent-decoded off the query string and can hold anything.
    # hmac.compare_digest raises TypeError on a non-ASCII str, and this is the
    # one route reachable without a credential — so a mangled link would turn
    # into a 500 and a traceback per request instead of a clean 401.
    if not (key and sig and exp_raw) or not sig.isascii():
        return False
    try:
        exp = int(exp_raw)
    except (TypeError, ValueError):
        return False
    if exp and exp < time.time():
        return False
    return hmac.compare_digest(sig, signature(file_id, exp, key))
