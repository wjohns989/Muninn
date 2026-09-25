"""Browser Origin checks for the local server.

Muninn listens on localhost and, unless a token is configured, accepts every
request. Browsers attach an Origin header to cross-site requests, so rejecting
foreign origins stops any web page the user visits (or a DNS-rebinding attack)
from reading or rewriting memories. The MCP Streamable HTTP spec requires this
check. Requests without an Origin header come from non-browser clients (MCP
hosts, the SDK, curl, tunnels) and are unaffected.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional, Tuple

from starlette.types import ASGIApp, Receive, Scope, Send

LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_LOCAL_ORIGIN = re.compile(LOCAL_ORIGIN_REGEX, re.IGNORECASE)


def configured_origins() -> Tuple[str, ...]:
    """Extra origins from MUNINN_ALLOWED_ORIGINS (comma-separated; "*" allows any, "null" allows file://)."""
    raw = os.environ.get("MUNINN_ALLOWED_ORIGINS", "")
    return tuple(part.strip().rstrip("/") for part in raw.split(",") if part.strip())


def origin_allowed(origin: Optional[str], extra: Iterable[str] = ()) -> bool:
    if origin is None:
        return True
    extra = tuple(extra)
    if "*" in extra:
        return True
    origin = origin.strip().rstrip("/")
    return bool(_LOCAL_ORIGIN.match(origin)) or origin in extra


class OriginGuardMiddleware:
    """Reject HTTP requests whose Origin is neither local nor allow-listed."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            origin = None
            for key, value in scope.get("headers", ()):
                if key == b"origin":
                    origin = value.decode("latin-1")
                    break
            if not origin_allowed(origin, configured_origins()):
                body = b'{"detail":"Origin not allowed. Add it to MUNINN_ALLOWED_ORIGINS to permit it."}'
                await send({
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
