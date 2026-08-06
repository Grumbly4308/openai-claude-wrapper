"""The origin the current HTTP client actually reached us on.

Generated-file links must be clickable in a chat UI, which means absolute.
CLAUDE_WRAPPER_PUBLIC_BASE_URL stays authoritative when set; when it is not,
the inbound request's own origin is a better answer than no link at all.
Carried in a ContextVar because _file_download_url runs deep inside the
response builders, which have no Request in scope.

The default is empty for callers that genuinely run outside a request, which is
what makes the trailer degrade to plain text rather than emit a broken link.
Note that the batches worker is NOT one of them: routes_batches launches it with
asyncio.create_task inside the POST handler, and create_task copies the current
context, so the worker keeps the submitting request's origin for the batch's
whole life -- long after the middleware's own reset has run. Batch outputs
therefore do get links, addressed to whatever origin submitted the batch.
"""

from __future__ import annotations

from contextvars import ContextVar

from .config import SETTINGS

_ORIGIN: ContextVar[str] = ContextVar("request_origin", default="")


def current_origin() -> str:
    """Origin of the request being served, or "" outside a request."""
    return _ORIGIN.get()


def _origin_from_scope(scope) -> str:
    """Derive "scheme://host[root_path]" from an ASGI scope.

    X-Forwarded-Proto is honored because uvicorn's --proxy-headers only trusts
    `forwarded_allow_ips` (default 127.0.0.1), which a Docker reverse proxy is
    not: behind a TLS terminator scope["scheme"] is "http", and a naive
    derivation would emit http:// links that browsers block as mixed content.
    A forged Host/X-Forwarded-Host only poisons the links in the forger's own
    reply — the download signature does not cover the host, so nothing is
    gained by steering it. Deployments that would rather not trust their proxy
    set public_base_url (which always wins) or CLAUDE_WRAPPER_DERIVE_BASE_URL=off.
    """
    headers = {k.lower(): v for k, v in scope.get("headers", ())}
    host = headers.get(b"x-forwarded-host") or headers.get(b"host")
    if not host:
        return ""
    host = host.decode("latin-1").split(",")[0].strip()
    if not host:
        return ""
    proto = headers.get(b"x-forwarded-proto")
    scheme = proto.decode("latin-1").split(",")[0].strip() if proto else scope.get("scheme", "http")
    return f"{scheme}://{host}{scope.get('root_path', '')}".rstrip("/")


class RequestOriginMiddleware:
    """Publishes the inbound origin into a ContextVar for the request's life.

    Pure ASGI, deliberately NOT BaseHTTPMiddleware / @app.middleware("http").
    BaseHTTPMiddleware wraps the response body in a memory stream and is a known
    source of SSE buffering regressions -- and this app's primary path is SSE.
    Pure ASGI runs the endpoint AND the StreamingResponse generator inside this
    call, so the ContextVar is live at all four trailer-emission sites.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not SETTINGS.derive_base_url:
            await self.app(scope, receive, send)
            return
        token = _ORIGIN.set(_origin_from_scope(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            _ORIGIN.reset(token)
