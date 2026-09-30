"""Capability-bound, asynchronous access to encrypted transcript pages."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_projection_store import (
    FORMAT,
    PARSER_REDACTOR,
    ProjectionIntegrityError,
    SecureProjectionStore,
)
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import ProjectionCancelled, UnsupportedTranscript, build_transcript_projection

_TTL = 600
_MAX_JOBS = 8
_MAX_SESSIONS = 128


class ProjectionAccess:
    """One CPU builder; completed projections remain reusable after restart."""

    def __init__(self, archive: Any) -> None:
        self.archive = archive
        self.store = SecureProjectionStore(archive)
        self.index = SecureHistoryBlindIndex(archive)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="muninn-projection")
        self._jobs: dict[tuple[str, str, int], Future[str]] = {}
        # One monotonic page position per live session; memory does not grow
        # with transcript length. The HTTP route separately rate-limits pages.
        self._sessions: dict[str, tuple[int, int]] = {}
        self._lock = threading.RLock()
        self._boot_key = os.urandom(32)
        self._stop = threading.Event()

    def _cursor_key(self) -> bytes:
        master = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                      info=b"muninn secure projection cursor v1").derive(self.archive._key)
        return hmac.new(master, self._boot_key, hashlib.sha256).digest()

    @staticmethod
    def _job_key(entry: dict[str, Any], version: int) -> tuple[str, str, int]:
        return entry["blob"], entry["sha256"], version

    def _cursor(self, entry: dict[str, Any], version: int, term: str,
                attempt: str, session: str, ordinal: int) -> str:
        payload = {"v": 1, "vault": self.archive.vault_id, "blob": entry["blob"],
                   "sha": entry["sha256"], "size": entry["size"], "version": version,
                   "term": term, "attempt": attempt, "session": session, "ordinal": ordinal,
                   "expires": int(time.time()) + _TTL, "format": FORMAT,
                   "parser_redactor": PARSER_REDACTOR}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        mac = hmac.new(self._cursor_key(), raw, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(raw + mac).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor: str) -> dict[str, Any]:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 768:
            raise ProjectionIntegrityError("invalid page cursor")
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != cursor or len(raw) < 33:
                raise ValueError
            payload, mac = raw[:-32], raw[-32:]
            if not hmac.compare_digest(mac, hmac.new(self._cursor_key(), payload, hashlib.sha256).digest()):
                raise ValueError
            data = json.loads(payload)
            expected = {"v", "vault", "blob", "sha", "size", "version", "term", "attempt", "session",
                        "ordinal", "expires", "format", "parser_redactor"}
            now = int(time.time())
            if (not isinstance(data, dict) or set(data) != expected
                    or data["v"] != 1 or data["vault"] != self.archive.vault_id
                    or data["format"] != FORMAT or data["parser_redactor"] != PARSER_REDACTOR
                    or not isinstance(data["ordinal"], int) or isinstance(data["ordinal"], bool)
                    or data["ordinal"] < 0 or not isinstance(data["expires"], int)
                    or isinstance(data["expires"], bool)
                    or not now < data["expires"] <= now + _TTL
                    or not isinstance(data["term"], str) or len(data["term"]) > 128
                    or not isinstance(data["session"], str) or len(data["session"]) != 32
                    or any(char not in "0123456789abcdef" for char in data["session"])):
                raise ValueError
            return data
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise ProjectionIntegrityError("invalid page cursor") from exc

    def _find_complete(self, entry: dict[str, Any], version: int
                       ) -> tuple[str, int, dict[str, int] | None] | None:
        with self.store._connect() as db:
            row = db.execute(
                "SELECT attempt FROM attempts WHERE vault=? AND blob=? AND sha=? AND size=? "
                "AND version=? AND state='complete' ORDER BY rowid DESC LIMIT 1",
                (self.archive.vault_id, entry["blob"], entry["sha256"], entry["size"], version),
            ).fetchone()
        if row is None:
            return None
        count, stats = self.store.projection_info(entry, version, row[0])
        return row[0], count, stats

    def _ready(self, entry: dict[str, Any], version: int, term: str,
               complete: tuple[str, int, dict[str, int] | None]) -> dict[str, Any]:
        attempt, count, stats = complete
        if count == 0:
            return {"state": "no_conversational_match", "coverage": stats}
        with self._lock:
            now = int(time.time())
            for session, (_position, expiry) in tuple(self._sessions.items()):
                if expiry <= now:
                    del self._sessions[session]
            if len(self._sessions) >= _MAX_SESSIONS:
                return {"state": "busy"}
            session = os.urandom(16).hex()
            self._sessions[session] = (0, now + _TTL)
        return {"state": "ready", "cursor": self._cursor(entry, version, term, attempt, session, 0),
                "pages": count, "redaction": "strict-best-effort", "coverage": stats,
                "scope": "supported_user_assistant_text_only"}

    def _job_state(self, key: tuple[str, str, int]) -> tuple[str, str | None]:
        with self._lock:
            future = self._jobs.get(key)
            if future is None:
                return "not_started", None
            if not future.done():
                return "pending", None
            del self._jobs[key]
        try:
            future.result()
        except UnsupportedTranscript:
            return "unsupported", None
        except ProjectionCancelled:
            return "cancelled", None
        except StreamingJSONError:
            return "unavailable", "malformed_json"
        except VaultIntegrityError:
            return "unavailable", "archive_integrity"
        except ProjectionIntegrityError:
            return "unavailable", "projection_integrity"
        except OSError:
            return "unavailable", "io"
        except MemoryError:
            return "unavailable", "resources"
        except Exception:
            return "unavailable", "other"
        return "finished", None

    def start(self, capability: str) -> dict[str, Any]:
        entry, version, data = self.index._entry_for_capability(capability)
        complete = self._find_complete(entry, version)
        if complete is not None:
            return self._ready(entry, version, data["term"], complete)
        key = self._job_key(entry, version)
        state, reason = self._job_state(key)
        if state == "pending":
            return {"state": "pending"}
        if state in {"unsupported", "unavailable"}:
            return {"state": state, **({"reason": reason} if reason else {})}
        with self._lock:
            if key in self._jobs:
                return {"state": "pending"}
            for old_key, future in tuple(self._jobs.items()):
                if old_key != key and future.done():
                    del self._jobs[old_key]
            if len(self._jobs) >= _MAX_JOBS:
                return {"state": "busy"}
            self._jobs[key] = self._executor.submit(
                build_transcript_projection, self.store, entry, version,
                should_cancel=self._stop.is_set)
        return {"state": "pending"}

    def poll(self, capability: str) -> dict[str, Any]:
        entry, version, data = self.index._entry_for_capability(capability)
        complete = self._find_complete(entry, version)
        if complete is not None:
            return self._ready(entry, version, data["term"], complete)
        state, reason = self._job_state(self._job_key(entry, version))
        return {"state": "unavailable" if state == "finished" else state,
                **({"reason": reason} if reason else {})}

    def page(self, cursor: str) -> dict[str, Any]:
        data = self._decode_cursor(cursor)
        with self._lock:
            now = int(time.time())
            for session, (_position, expiry) in tuple(self._sessions.items()):
                if expiry <= now:
                    del self._sessions[session]
            position = self._sessions.get(data["session"])
            if position is None or position[0] != data["ordinal"]:
                raise ProjectionIntegrityError("page cursor already used or session unavailable")
            self._sessions[data["session"]] = (data["ordinal"] + 1, now + _TTL)
        entry = None
        version = data["version"]
        for _source, current_version, candidate, _latest, _versions in self.index._current():
            if (candidate["blob"] == data["blob"] and candidate["sha256"] == data["sha"]
                    and candidate["size"] == data["size"] and current_version == version):
                entry = candidate
                break
        if entry is None:
            raise ProjectionIntegrityError("history snapshot is no longer available")
        complete = self._find_complete(entry, version)
        if complete is None or complete[0] != data["attempt"] or data["ordinal"] >= complete[1]:
            raise ProjectionIntegrityError("projection page unavailable")
        text = self.store.get_page(entry, version, data["attempt"], data["ordinal"])
        next_cursor = (self._cursor(entry, version, data["term"], data["attempt"], data["session"],
                                    data["ordinal"] + 1)
                       if data["ordinal"] + 1 < complete[1] else None)
        if next_cursor is None:
            with self._lock:
                self._sessions.pop(data["session"], None)
        return {"redacted_text": text, "next_cursor": next_cursor,
                "more": next_cursor is not None, "redaction": "strict-best-effort"}

    def close(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
