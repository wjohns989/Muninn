"""Encrypted, authenticated redacted transcript projections.

This module is deliberately independent of the history service.  A projection
is publishable only after the archive iterator has reached its authenticated
end and the completion record has been sealed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import aead
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.streaming_redaction import redacted_pages

MAX_PAGE_CHARS = 4000
FORMAT = "secure-projection-v1"
PARSER_REDACTOR = "parser-redactor-v1"


class ProjectionIntegrityError(RuntimeError):
    pass


class ProjectionBusyError(RuntimeError):
    """Another process owns projection construction or recovery."""


def _j(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class SecureProjectionStore:
    max_page_chars = MAX_PAGE_CHARS

    def __init__(self, archive: Any, root: Path | None = None) -> None:
        self.archive = archive
        self.root = Path(root) if root is not None else Path(archive.root) / "projections"
        if not self.root.exists():
            create_private_directory(self.root)
        verify_private(self.root)
        self.db_path = self.root / "projections.sqlite3"
        if not self.db_path.exists():
            create_private_file(self.db_path)
        verify_private(self.db_path)
        self.lock_path = self.root / "projection.lock"
        if not self.lock_path.exists():
            create_private_file(self.lock_path)
            self.lock_path.write_bytes(b"\0")
        verify_private(self.lock_path)
        with self._connect() as db:
            db.executescript("""CREATE TABLE IF NOT EXISTS attempts(
                attempt TEXT PRIMARY KEY, vault TEXT NOT NULL, blob TEXT NOT NULL,
                sha TEXT NOT NULL, size INTEGER NOT NULL, version INTEGER NOT NULL,
                state TEXT NOT NULL, count INTEGER NOT NULL, digest BLOB,
                completion BLOB)
            ;CREATE TABLE IF NOT EXISTS pages(
                attempt TEXT NOT NULL, ordinal INTEGER NOT NULL, length INTEGER NOT NULL,
                ciphertext BLOB NOT NULL, PRIMARY KEY(attempt, ordinal))""")
        try:
            self.recover_incomplete()
        except ProjectionBusyError:
            # A second service process may open the read path while the sole
            # writer is active. That writer will recover after acquiring its
            # own lock; readers still reject state=building.
            pass

    @contextmanager
    def _build_lock(self) -> Iterator[None]:
        verify_private(self.lock_path)
        with self.lock_path.open("r+b") as handle:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise ProjectionBusyError("projection builder is busy") from exc
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise ProjectionBusyError("projection builder is busy") from exc
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def recover_incomplete(self) -> int:
        """Remove only abandoned derived ciphertext after acquiring writer lock."""
        with self._build_lock():
            return self._recover_incomplete_locked()

    def _recover_incomplete_locked(self) -> int:
        removed = 0
        with self._connect() as db:
            while True:
                row = db.execute(
                    "SELECT attempt FROM attempts WHERE state='building' LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                attempt = row[0]
                while True:
                    ids = db.execute(
                        "SELECT rowid FROM pages WHERE attempt=? LIMIT 256", (attempt,)
                    ).fetchall()
                    if not ids:
                        break
                    db.executemany("DELETE FROM pages WHERE rowid=?", ids)
                    db.commit()
                db.execute("DELETE FROM attempts WHERE attempt=? AND state='building'", (attempt,))
                db.commit()
                removed += 1
        return removed

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        verify_private(self.db_path)
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA secure_delete=ON")
            with db:
                yield db
        finally:
            db.close()

    def _key(self) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"muninn secure projection key v1").derive(self.archive._key)

    def _identity(self, entry: dict[str, Any], version: int) -> dict[str, Any]:
        required = ("blob", "sha256", "size")
        if (any(k not in entry for k in required) or not isinstance(version, int)
                or isinstance(version, bool) or version < 0
                or not isinstance(entry["blob"], str) or len(entry["blob"]) != 32
                or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                or not isinstance(entry["size"], int) or entry["size"] < 0):
            raise ValueError("invalid immutable snapshot")
        return {"vault": self.archive.vault_id, "blob": entry["blob"], "hash": entry["sha256"],
                "size": int(entry["size"]), "version": version, "format": FORMAT,
                "parser_redactor": PARSER_REDACTOR}

    def _aad(self, ident: dict[str, Any], attempt: str, ordinal: int, length: int) -> bytes:
        return _j({**ident, "attempt": attempt, "page": ordinal, "plaintext_length": length})

    def _staged_pages(self, text: Iterable[str], page_chars: int) -> Iterable[str]:
        return redacted_pages(text, page_chars=page_chars)

    def build(self, entry: dict[str, Any], version: int,
              projector: Callable[[Iterable[bytes]], Iterable[str]], *,
              page_chars: int = MAX_PAGE_CHARS,
              stats: dict[str, int] | None = None) -> str:
        """Project selected conversation text; redact centrally before sealing.

        ``projector`` must extract only supported user/assistant fields, may
        yield arbitrary text fragments, and must consume the source normally.
        The store owns token-boundary redaction and output pagination.
        """
        with self._build_lock():
            self._recover_incomplete_locked()
            return self._build_locked(entry, version, projector, page_chars=page_chars, stats=stats)

    def _build_locked(self, entry: dict[str, Any], version: int,
                      projector: Callable[[Iterable[bytes]], Iterable[str]], *,
                      page_chars: int, stats: dict[str, int] | None) -> str:
        ident = self._identity(entry, version)
        attempt = uuid.uuid4().hex
        key = self._key()
        digest = hashlib.sha256()
        source_size = 0
        exhausted = False

        def tracked() -> Iterator[bytes]:
            nonlocal exhausted, source_size
            source = self.archive._iter_verified_entry(entry)
            try:
                for chunk in source:
                    digest.update(chunk)
                    source_size += len(chunk)
                    yield chunk
                exhausted = True
            finally:
                if not exhausted:
                    close = getattr(source, "close", None)
                    if close:
                        close()

        with self._connect() as db:
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (attempt, ident["vault"], ident["blob"], ident["hash"], ident["size"],
                        version, "building", 0, None, None))
            db.commit()
            try:
                ordinal = 0
                for page in self._staged_pages(projector(tracked()), page_chars):
                    if not isinstance(page, str) or not 1 <= len(page) <= self.max_page_chars:
                        raise ProjectionIntegrityError("projector yielded an invalid page")
                    raw = page.encode("utf-8")
                    nonce = os.urandom(12)
                    sealed = nonce + aead.AESGCM(key).encrypt(
                        nonce, raw, self._aad(ident, attempt, ordinal, len(raw)))
                    db.execute("INSERT INTO pages VALUES(?,?,?,?)", (attempt, ordinal, len(raw), sealed))
                    ordinal += 1
                    if ordinal % 64 == 0:
                        # Staging is not fetchable while state=building. Small
                        # transactions avoid a source-sized rollback journal
                        # and long reader/writer lock on multi-GB transcripts.
                        db.commit()
                if not exhausted:
                    raise ProjectionIntegrityError("projector did not exhaust authenticated source")
                if source_size != ident["size"] or digest.hexdigest() != ident["hash"]:
                    raise ProjectionIntegrityError("projected source does not match authenticated snapshot")
                if stats is not None and (set(stats) != {
                        "source_units", "conversational_units", "omitted_units"}
                        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                               for value in stats.values())
                        or stats["source_units"] != stats["conversational_units"] + stats["omitted_units"]):
                    raise ProjectionIntegrityError("invalid projection coverage")
                # The sealed completion binds the attempt and contiguous page
                # count. Each requested page is independently AEAD-bound to
                # this attempt, ordinal, snapshot, and plaintext length; a
                # whole-projection digest would require a multi-GB rescan on
                # every page request and is deliberately not claimed here.
                completion = _j({**ident, "attempt": attempt, "pages": ordinal,
                                 "source_sha256": digest.hexdigest(), "stats": stats})
                nonce = os.urandom(12)
                sealed_completion = nonce + aead.AESGCM(key).encrypt(nonce, completion, _j(ident))
                db.execute("UPDATE attempts SET state='complete',count=?,digest=?,completion=? WHERE attempt=?",
                           (ordinal, bytes.fromhex(digest.hexdigest()), sealed_completion, attempt))
                db.commit()
                return attempt
            except Exception:
                db.rollback()
                while True:
                    ids = db.execute(
                        "SELECT rowid FROM pages WHERE attempt=? LIMIT 256", (attempt,)
                    ).fetchall()
                    if not ids:
                        break
                    db.executemany("DELETE FROM pages WHERE rowid=?", ids)
                    db.commit()
                db.execute("DELETE FROM attempts WHERE attempt=?", (attempt,))
                db.commit()
                raise

    def _authenticated_count(self, db: sqlite3.Connection, ident: dict[str, Any],
                             attempt: str) -> tuple[int, dict[str, int] | None]:
        if (not isinstance(attempt, str) or len(attempt) != 32
                or any(char not in "0123456789abcdef" for char in attempt)):
            raise ProjectionIntegrityError("invalid projection reference")
        row = db.execute(
            "SELECT state,count,digest,completion FROM attempts WHERE attempt=?", (attempt,)
        ).fetchone()
        if not row or row[0] != "complete":
            raise ProjectionIntegrityError("projection is incomplete")
        try:
            completion = aead.AESGCM(self._key()).decrypt(row[3][:12], row[3][12:], _j(ident))
            record = json.loads(completion)
            expected_keys = set(ident) | {"attempt", "pages", "source_sha256", "stats"}
            if (set(record) != expected_keys
                    or any(record[key] != value for key, value in ident.items())
                    or record["attempt"] != attempt
                    or record["source_sha256"] != ident["hash"]
                    or not isinstance(record["pages"], int) or isinstance(record["pages"], bool)
                    or record["pages"] != row[1]
                    or row[2] != bytes.fromhex(ident["hash"])):
                raise ValueError
            count, first, last = db.execute(
                "SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM pages WHERE attempt=?", (attempt,)
            ).fetchone()
            if count != record["pages"] or (count and (first != 0 or last != count - 1)):
                raise ValueError
            stats = record["stats"]
            if stats is not None and (not isinstance(stats, dict)
                    or set(stats) != {"source_units", "conversational_units", "omitted_units"}
                    or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                           for value in stats.values())
                    or stats["source_units"] != stats["conversational_units"] + stats["omitted_units"]):
                raise ValueError
            return count, stats
        except (InvalidTag, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ProjectionIntegrityError("projection authentication failed") from exc

    def count_pages(self, entry: dict[str, Any], version: int, attempt: str) -> int:
        ident = self._identity(entry, version)
        with self._connect() as db:
            return self._authenticated_count(db, ident, attempt)[0]

    def projection_info(self, entry: dict[str, Any], version: int,
                        attempt: str) -> tuple[int, dict[str, int] | None]:
        ident = self._identity(entry, version)
        with self._connect() as db:
            return self._authenticated_count(db, ident, attempt)

    def get_page(self, entry: dict[str, Any], version: int, attempt: str, ordinal: int) -> str:
        ident = self._identity(entry, version)
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise ProjectionIntegrityError("invalid projection reference")
        with self._connect() as db:
            count, _stats = self._authenticated_count(db, ident, attempt)
            if ordinal >= count:
                raise ProjectionIntegrityError("projection page unavailable")
            page = db.execute(
                "SELECT length,ciphertext FROM pages WHERE attempt=? AND ordinal=?", (attempt, ordinal)
            ).fetchone()
            return self._decrypt_page(ident, attempt, ordinal, page)

    def _decrypt_page(self, ident: dict, attempt: str, ordinal: int, page,
                      cipher: aead.AESGCM | None = None) -> str:
        try:
            if not page:
                raise ValueError
            cipher = cipher or aead.AESGCM(self._key())
            raw = cipher.decrypt(page[1][:12], page[1][12:],
                                 self._aad(ident, attempt, ordinal, page[0]))
            if len(raw) != page[0]:
                raise ValueError
            return raw.decode("utf-8")
        except (InvalidTag, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ProjectionIntegrityError("projection authentication failed") from exc

    def _iter_sealed_pages(self, entry: dict, version: int, attempt: str) -> Iterator[str]:
        """Private linear-time read of one pinned, completed SQLite snapshot."""
        ident = self._identity(entry, version)
        with self._connect() as db:
            db.execute("BEGIN")
            count, _stats = self._authenticated_count(db, ident, attempt)
            cipher = aead.AESGCM(self._key())
            seen = 0
            for ordinal, length, ciphertext in db.execute(
                    "SELECT ordinal,length,ciphertext FROM pages WHERE attempt=? ORDER BY ordinal", (attempt,)):
                if ordinal != seen:
                    raise ProjectionIntegrityError("projection page sequence is incomplete")
                yield self._decrypt_page(ident, attempt, ordinal, (length, ciphertext), cipher)
                seen += 1
            if seen != count:
                raise ProjectionIntegrityError("projection page sequence is incomplete")
