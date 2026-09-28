"""Security guard for explicit local credential API operations."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import secrets
import threading
import time
from collections import deque
from typing import Any

from fastapi import HTTPException, Request

_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "::ffff:7f00:1"}
NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache", "X-Content-Type-Options": "nosniff"}


def authenticate_local(request: Request) -> tuple[str, str]:
    """Require a separate configured bearer token and the actual socket peer."""
    try:
        peer = str(ipaddress.ip_address(request.client.host if request.client else ""))
    except ValueError:
        raise HTTPException(404, "Unavailable") from None
    if peer not in _LOOPBACK:
        raise HTTPException(404, "Unavailable")
    expected = os.environ.get("MUNINN_CREDENTIAL_API_TOKEN", "")
    if len(expected) < 32:
        raise HTTPException(404, "Unavailable")
    supplied = request.headers.get("authorization", "")
    scheme, _, token = supplied.partition(" ")
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(token, expected):
        raise HTTPException(401, "Authentication required")
    return hashlib.sha256(expected.encode("utf-8")).hexdigest(), peer


async def read_passphrase(request: Request) -> str:
    """Read only a small JSON body without ever echoing it into errors/logs."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Invalid credential request")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > 4096 or int(length) < 0:
                raise HTTPException(413, "Invalid credential request")
        except ValueError:
            raise HTTPException(400, "Invalid credential request") from None
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > 4096:
            raise HTTPException(413, "Invalid credential request")
        chunks.extend(chunk)
    try:
        payload: Any = json.loads(chunks)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid credential request") from None
    if (not isinstance(payload, dict) or set(payload) != {"passphrase"}
            or not isinstance(payload["passphrase"], str)
            or not 12 <= len(payload["passphrase"]) <= 1024):
        raise HTTPException(400, "Invalid credential request")
    return payload["passphrase"]


class RevealLimiter:
    """Bound online passphrase guesses without retaining secrets or request bodies."""

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._per_key: dict[tuple[str, str, str, str], deque[float]] = {}
        self._global: deque[float] = deque()
        self._inflight: dict[tuple[str, str, str, str], int] = {}
        self._total_inflight = 0

    def _expire(self, now: float) -> None:
        cutoff = now - 300
        while self._global and self._global[0] <= cutoff:
            self._global.popleft()
        for key, attempts in list(self._per_key.items()):
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if not attempts:
                del self._per_key[key]

    def begin(self, key: tuple[str, str, str, str]) -> bool:
        """Atomically reserve a bounded Argon2/reveal slot."""
        with self._lock:
            self._expire(self._clock())
            if (len(self._global) >= 50 or len(self._per_key.get(key, ())) >= 5
                    or self._total_inflight >= 4 or self._inflight.get(key, 0) >= 1):
                return False
            self._total_inflight += 1
            self._inflight[key] = self._inflight.get(key, 0) + 1
            return True

    def finish(self, key: tuple[str, str, str, str], *, success: bool) -> None:
        with self._lock:
            self._total_inflight -= 1
            count = self._inflight[key] - 1
            if count:
                self._inflight[key] = count
            else:
                del self._inflight[key]
            if success:
                self._per_key.pop(key, None)
            else:
                now = self._clock()
                self._expire(now)
                self._global.append(now)
                self._per_key.setdefault(key, deque()).append(now)
