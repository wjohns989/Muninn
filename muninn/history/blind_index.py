"""Rebuildable, encrypted lexical prefilter for strict history snapshots.

Only AEAD ciphertext and opaque blob IDs are persisted. A positive Bloom result
is always checked against the authenticated archive before metadata is returned.
No transcript text, tokens, snippets, or model prompts are stored here.
"""

from __future__ import annotations

import base64
import codecs
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_file, verify_private
from muninn.history.safe_span import sanitize_agent_span
from muninn.history.secure_archive import SafeHistoryMetadata, SecureHistoryArchive

_FORMAT = 1
_TOKENIZER = 1
_HASHES = 4
_TERM = re.compile(r"\w+", re.UNICODE)
_TEXT_KINDS = {"transcript", "prompt_history", "desktop_session", "export"}
_STRUCTURED_LINE_LIMIT = 256 * 1024
_STAGING_NAME = re.compile(r"muninn-chunks-[A-Za-z0-9_-]+\.db(?:-journal)?\Z")


def _terms(value: str) -> list[str]:
    return [word for word in _TERM.findall(value.casefold()) if 3 <= len(word) <= 64]


def _key(archive: SecureHistoryArchive, purpose: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32,
                salt=bytes.fromhex(archive.vault_id),
                info=b"muninn-history-blind-v1/" + purpose).derive(archive._key)


class SecureHistoryBlindIndex:
    """CPU-only per-snapshot Bloom filters, encrypted under a separated key."""

    def __init__(self, archive: SecureHistoryArchive, *, filter_bytes: int = 65536):
        if not 128 <= filter_bytes <= 1024 * 1024 or filter_bytes & (filter_bytes - 1):
            raise ValueError("Filter size must be a power of two between 128 B and 1 MiB")
        self.archive = archive
        self.filter_bytes = filter_bytes
        self.path = (archive.root / "blind_index.db").absolute()
        self.lock_path = (archive.root / "blind_index.lock").absolute()
        verify_private(archive.root)
        if not self.path.exists():
            create_private_file(self.path)
        if not self.lock_path.exists():
            create_private_file(self.lock_path)
            self.lock_path.write_bytes(b"\0")
        verify_private(self.path)
        verify_private(self.lock_path)
        self._cipher = AESGCM(_key(archive, b"filter-aead"))
        self._large_cipher = AESGCM(_key(archive, b"large-filter-aead"))
        self._token_key = _key(archive, b"token-positions")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY CHECK(id=1), "
                       "format INTEGER NOT NULL, vault_id TEXT NOT NULL, filter_bytes INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS filters (blob TEXT PRIMARY KEY, "
                       "nonce BLOB NOT NULL, sealed BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS large_filters (blob TEXT PRIMARY KEY, "
                       "filter_bytes INTEGER NOT NULL, nonce BLOB NOT NULL, sealed BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS chunk_filters (blob TEXT NOT NULL, ordinal INTEGER NOT NULL, "
                       "filter_bytes INTEGER NOT NULL, nonce BLOB NOT NULL, sealed BLOB NOT NULL, "
                       "PRIMARY KEY(blob, ordinal))")
            db.execute("CREATE TABLE IF NOT EXISTS chunk_completions (blob TEXT PRIMARY KEY, chunks INTEGER NOT NULL, "
                       "size INTEGER NOT NULL, sha256 TEXT NOT NULL, filter_bytes INTEGER NOT NULL, "
                       "parameters TEXT NOT NULL)")
            row = db.execute("SELECT format, vault_id, filter_bytes FROM meta WHERE id=1").fetchone()
            expected = (_FORMAT, archive.vault_id, filter_bytes)
            if row is None:
                db.execute("INSERT INTO meta (id, format, vault_id, filter_bytes) VALUES (1, ?, ?, ?)", expected)
            elif tuple(row) != expected:
                raise VaultIntegrityError("History search index identity or parameters differ")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        verify_private(self.path)
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def _build_lock(self) -> Iterator[None]:
        """Only one background builder may decrypt/index archive blobs at once."""
        verify_private(self.lock_path)
        with self.lock_path.open("r+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError("History index builder is busy") from exc
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
                    raise RuntimeError("History index builder is busy") from exc
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _cleanup_stale_staging(self) -> None:
        """Only the lock owner removes abandoned, rebuildable encrypted staging."""
        root = self.archive.root.resolve(strict=True)
        for candidate in root.glob("muninn-chunks-*.db*"):
            if not _STAGING_NAME.fullmatch(candidate.name):
                continue
            if candidate.is_symlink() or candidate.resolve(strict=True).parent != root:
                raise VaultIntegrityError("History index staging path is unsafe")
            verify_private(candidate)
            candidate.unlink()

    def _aad(self, entry: dict, version: int) -> bytes:
        return json.dumps({
            "format": _FORMAT, "vault_id": self.archive.vault_id,
            "blob": entry["blob"], "sha256": entry["sha256"], "size": entry["size"],
            "provider": entry["provider"], "kind": entry["kind"], "version": version,
            "filter_bytes": self.filter_bytes, "hashes": _HASHES, "tokenizer": _TOKENIZER,
        }, sort_keys=True, separators=(",", ":")).encode("ascii")

    def _large_filter_bytes(self, entry: dict) -> int:
        size = entry["size"]
        if size > 128 * 1024 * 1024:
            factor = 128
        elif size > 16 * 1024 * 1024:
            factor = 32
        elif size > 4 * 1024 * 1024:
            factor = 8
        else:
            factor = 1
        return min(8 * 1024 * 1024, self.filter_bytes * factor)

    def _large_aad(self, entry: dict, version: int, filter_bytes: int) -> bytes:
        return json.dumps({
            "format": 2, "vault_id": self.archive.vault_id,
            "blob": entry["blob"], "sha256": entry["sha256"], "size": entry["size"],
            "provider": entry["provider"], "kind": entry["kind"], "version": version,
            "filter_bytes": filter_bytes, "hashes": _HASHES, "tokenizer": _TOKENIZER,
        }, sort_keys=True, separators=(",", ":")).encode("ascii")

    def _chunk_aad(self, entry: dict, version: int, ordinal: int, filter_bytes: int) -> bytes:
        return json.dumps({"format": 3, "vault_id": self.archive.vault_id,
            "blob": entry["blob"], "sha256": entry["sha256"], "size": entry["size"],
            "provider": entry["provider"], "kind": entry["kind"], "version": version,
            "ordinal": ordinal, "filter_bytes": filter_bytes, "hashes": _HASHES,
            "tokenizer": _TOKENIZER}, sort_keys=True, separators=(",", ":")).encode("ascii")

    def _positions(self, term: str, filter_bytes: int | None = None) -> tuple[int, ...]:
        digest = hmac.new(self._token_key, term.encode("utf-8"), hashlib.sha256).digest()
        first = int.from_bytes(digest[:8], "big")
        step = int.from_bytes(digest[8:16], "big") | 1
        bits = (filter_bytes or self.filter_bytes) * 8
        return tuple((first + i * step) % bits for i in range(_HASHES))

    def _capability(self, entry: dict, version: int, term: str) -> str:
        payload = json.dumps({
            "v": 1, "vault": self.archive.vault_id, "blob": entry["blob"],
            "sha256": entry["sha256"], "version": version, "term": term,
            "expires": int(time.time()) + 600,
        }, sort_keys=True, separators=(",", ":")).encode("ascii")
        mac = hmac.new(_key(self.archive, b"fetch-capability"), payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(payload + mac).decode("ascii").rstrip("=")

    def _entry_for_capability(self, capability: str) -> tuple[dict, int, dict]:
        if not isinstance(capability, str) or not 1 <= len(capability) <= 512:
            raise ValueError("Invalid history fetch capability")
        try:
            raw = base64.urlsafe_b64decode(capability + "=" * (-len(capability) % 4))
            if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != capability:
                raise ValueError
            payload, mac = raw[:-32], raw[-32:]
            expected = hmac.new(_key(self.archive, b"fetch-capability"), payload, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise ValueError
            data = json.loads(payload)
            if (data["v"] != 1 or data["vault"] != self.archive.vault_id
                    or not int(time.time()) < data["expires"] <= int(time.time()) + 600
                    or data["term"] not in _terms(data["term"])):
                raise ValueError
            for source, version, entry, latest, versions in self._current():
                if (entry["blob"] == data["blob"] and entry["sha256"] == data["sha256"]
                        and version == data["version"]):
                    return entry, version, data
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            raise ValueError("Invalid history fetch capability") from exc
        raise ValueError("History snapshot is no longer available")

    def _current(self) -> list[tuple[str, int, dict, dict, int]]:
        manifest = self.archive._load_manifest()
        current = []
        for source, entries in manifest["files"].items():
            for version, entry in enumerate(entries):
                if entry["kind"] in _TEXT_KINDS:
                    current.append((source, version, entry, entries[-1], len(entries)))
        current.sort(key=lambda row: row[2]["captured_at"], reverse=True)
        return current

    def _open_filter(self, entry: dict, version: int, row: tuple[bytes, bytes]) -> bytes:
        nonce, sealed = row
        if len(nonce) != 12:
            raise VaultIntegrityError("History search filter integrity failed")
        try:
            value = self._cipher.decrypt(nonce, sealed, self._aad(entry, version))
        except (InvalidTag, ValueError) as exc:
            raise VaultIntegrityError("History search filter integrity failed") from exc
        if value == b"O":
            return value
        if len(value) != self.filter_bytes + 1 or value[:1] != b"R":
            raise VaultIntegrityError("History search filter format is invalid")
        return value

    def _open_large_filter(self, entry: dict, version: int,
                           row: tuple[int, bytes, bytes]) -> bytes:
        filter_bytes, nonce, sealed = row
        if (not isinstance(filter_bytes, int) or not self.filter_bytes <= filter_bytes <= 8 * 1024 * 1024
                or filter_bytes & (filter_bytes - 1) or filter_bytes != self._large_filter_bytes(entry)
                or len(nonce) != 12):
            raise VaultIntegrityError("Large history search filter parameters are invalid")
        try:
            value = self._large_cipher.decrypt(nonce, sealed, self._large_aad(entry, version, filter_bytes))
        except (InvalidTag, ValueError) as exc:
            raise VaultIntegrityError("Large history search filter integrity failed") from exc
        if value == b"O":
            return value
        if (len(value) != filter_bytes + 5 or value[:1] != b"L"
                or struct.unpack(">I", value[1:5])[0] != filter_bytes):
            raise VaultIntegrityError("Large history search filter format is invalid")
        return value

    def _lookup_filter(self, db: sqlite3.Connection, entry: dict,
                       version: int) -> tuple[bytes, int] | None:
        large = db.execute("SELECT filter_bytes, nonce, sealed FROM large_filters WHERE blob=?",
                           (entry["blob"],)).fetchone()
        if large is not None:
            value = self._open_large_filter(entry, version, large)
            return (value[5:], large[0]) if value != b"O" else (value, large[0])
        small = db.execute("SELECT nonce, sealed FROM filters WHERE blob=?", (entry["blob"],)).fetchone()
        if small is None:
            return None
        value = self._open_filter(entry, version, small)
        return (value[1:], self.filter_bytes) if value != b"O" else (value, self.filter_bytes)

    def _build_one(self, entry: dict, version: int,
                   filter_bytes: int | None = None) -> bytes:
        filter_bytes = filter_bytes or self.filter_bytes
        bits = bytearray(filter_bytes)
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        carry = ""
        overflow = False
        seen_terms: set[str] = set()

        def accept(chunk: bytes) -> None:
            nonlocal carry, overflow
            if overflow:
                return
            try:
                text = carry + decoder.decode(chunk)
            except UnicodeDecodeError:
                overflow = True
                return
            for term in _terms(text):
                if term in seen_terms:
                    continue
                seen_terms.add(term)
                if len(seen_terms) > 500000:
                    seen_terms.clear()
                    seen_terms.add(term)
                for position in self._positions(term, filter_bytes):
                    bits[position >> 3] |= 1 << (position & 7)
                if len(seen_terms) % 50000 == 0:
                    if sum(byte.bit_count() for byte in bits) > len(bits) * 8 * 0.60:
                        overflow = True
                        break
            carry = text[-128:]

        self.archive._verify_entry(entry, collect=False, on_chunk=accept)
        if not overflow:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                overflow = True
        if not overflow and sum(byte.bit_count() for byte in bits) > len(bits) * 8 * 0.60:
            overflow = True
        if overflow:
            return b"O"
        if filter_bytes == self.filter_bytes:
            return b"R" + bytes(bits)
        return b"L" + struct.pack(">I", filter_bytes) + bytes(bits)

    def _build_chunks(self, entry: dict, version: int, filter_bytes: int) -> tuple[str | None, int]:
        handle = tempfile.NamedTemporaryFile(prefix="muninn-chunks-", suffix=".db", dir=self.archive.root, delete=False)
        handle.close()
        os.unlink(handle.name)
        create_private_file(handle.name)
        staged = sqlite3.connect(handle.name)
        keep_staging = False
        try:
            staged.execute("CREATE TABLE chunks (ordinal INTEGER PRIMARY KEY, nonce BLOB, sealed BLOB)")
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            carry = ""
            ordinal = 0
            invalid = False
            def accept(chunk: bytes) -> None:
                nonlocal carry, ordinal, invalid
                if invalid:
                    return
                try:
                    text = carry + decoder.decode(chunk)
                except UnicodeDecodeError:
                    invalid = True
                    return
                bits = bytearray(filter_bytes)
                for term in set(_terms(text)):
                    for position in self._positions(term, filter_bytes):
                        bits[position >> 3] |= 1 << (position & 7)
                nonce = os.urandom(12)
                staged.execute("INSERT INTO chunks VALUES (?, ?, ?)", (ordinal, nonce, self._large_cipher.encrypt(
                    nonce, b"L" + struct.pack(">I", filter_bytes) + bytes(bits),
                    self._chunk_aad(entry, version, ordinal, filter_bytes))))
                ordinal += 1
                carry = text[-128:]
            self.archive._verify_entry(entry, collect=False, on_chunk=accept)
            if not invalid:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError:
                    invalid = True
            if invalid:
                staged.rollback()
                return None, -1
            if ordinal != entry["chunks"]:
                raise VaultIntegrityError("History chunk count does not match authenticated archive")
            staged.commit()
            keep_staging = True
            return handle.name, ordinal
        except Exception:
            staged.rollback()
            raise
        finally:
            staged.close()
            if not keep_staging:
                Path(handle.name).unlink(missing_ok=True)

    def _commit_chunks(self, entry: dict, staging_path: str,
                       chunk_count: int, filter_bytes: int) -> None:
        """Publish only a fully verified staged index, in one transaction."""
        staged = sqlite3.connect(f"{Path(staging_path).as_uri()}?mode=ro", uri=True)
        try:
            with self._connect() as db:
                db.execute("DELETE FROM chunk_filters WHERE blob=?", (entry["blob"],))
                db.execute("DELETE FROM chunk_completions WHERE blob=?", (entry["blob"],))
                count = 0
                for ordinal, nonce, sealed in staged.execute(
                        "SELECT ordinal, nonce, sealed FROM chunks ORDER BY ordinal"):
                    if ordinal != count:
                        raise VaultIntegrityError("History staged chunk index is incomplete")
                    db.execute("INSERT INTO chunk_filters VALUES (?, ?, ?, ?, ?)",
                               (entry["blob"], ordinal, filter_bytes, nonce, sealed))
                    count += 1
                if count != chunk_count or count != entry["chunks"]:
                    raise VaultIntegrityError("History staged chunk index is incomplete")
                db.execute("INSERT INTO chunk_completions VALUES (?, ?, ?, ?, ?, ?)",
                           (entry["blob"], chunk_count, entry["size"], entry["sha256"],
                            filter_bytes, json.dumps({"format": 3, "hashes": _HASHES,
                                                      "tokenizer": _TOKENIZER}, sort_keys=True)))
        finally:
            staged.close()
            Path(staging_path).unlink(missing_ok=True)

    def _completion(self, db: sqlite3.Connection, entry: dict) -> tuple | None:
        row = db.execute(
            "SELECT chunks, size, sha256, filter_bytes, parameters "
            "FROM chunk_completions WHERE blob=?", (entry["blob"],)
        ).fetchone()
        if row is None:
            return None
        try:
            parameters = json.loads(row[4])
        except (TypeError, ValueError) as exc:
            raise VaultIntegrityError("History chunk index metadata is invalid") from exc
        if (row[:4] != (entry["chunks"], entry["size"], entry["sha256"], self.filter_bytes)
                or parameters != {"format": 3, "hashes": _HASHES, "tokenizer": _TOKENIZER}):
            raise VaultIntegrityError("History chunk index metadata is invalid")
        count, first, last = db.execute(
            "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM chunk_filters WHERE blob=?",
            (entry["blob"],),
        ).fetchone()
        if (count != entry["chunks"] or
                (count and (first != 0 or last != count - 1))):
            raise VaultIntegrityError("History chunk index is incomplete")
        return row

    def build(self, *, max_snapshots: int | None = None,
              retry_unsearchable: bool = False) -> dict[str, int | bool]:
        """Index only missing immutable blobs; one authenticated commit each."""
        if max_snapshots is not None and max_snapshots < 1:
            raise ValueError("max_snapshots must be positive")
        if retry_unsearchable and (max_snapshots is None or max_snapshots > 10):
            raise ValueError("Retries require a batch of at most ten snapshots")
        with self._build_lock():
            self._cleanup_stale_staging()
            return self._build_locked(max_snapshots=max_snapshots,
                                      retry_unsearchable=retry_unsearchable)

    def _build_locked(self, *, max_snapshots: int | None,
                      retry_unsearchable: bool) -> dict[str, int | bool]:
        indexed = 0
        overflow = 0
        skipped = 0
        for _source, version, entry, _latest, _versions in self._current():
            desired = self._large_filter_bytes(entry)
            with self._connect() as db:
                completion = self._completion(db, entry)
                if completion is not None:
                    skipped += 1
                    continue
                existing = self._lookup_filter(db, entry, version)
            if retry_unsearchable and existing is None:
                skipped += 1
                continue
            if existing is not None:
                if not (retry_unsearchable and existing[0] == b"O"):
                    skipped += 1
                    continue
            if max_snapshots is not None and indexed + overflow >= max_snapshots:
                break
            if desired > self.filter_bytes:
                segment_bytes = self.filter_bytes
                projected = entry["chunks"] * (segment_bytes + 64)
                if shutil.disk_usage(self.archive.root).free < projected * 2:
                    raise RuntimeError("Insufficient free space for encrypted history index")
                staging_path, chunk_count = self._build_chunks(entry, version, segment_bytes)
                if chunk_count < 0:
                    # Authenticated binary/invalid-UTF8 snapshots remain explicitly
                    # unsearchable and retryable; they are never treated as a match.
                    nonce = os.urandom(12)
                    sealed = self._large_cipher.encrypt(nonce, b"O",
                        self._large_aad(entry, version, desired))
                    with self._connect() as db:
                        db.execute("INSERT OR REPLACE INTO large_filters VALUES (?, ?, ?, ?)",
                                   (entry["blob"], desired, nonce, sealed))
                    overflow += 1
                    continue
                assert staging_path is not None
                self._commit_chunks(entry, staging_path, chunk_count, segment_bytes)
                indexed += 1
                continue
            value = self._build_one(entry, version, desired)
            if value == b"O":
                staging_path, chunk_count = self._build_chunks(entry, version, self.filter_bytes)
                if chunk_count < 0:
                    nonce = os.urandom(12)
                    sealed = self._large_cipher.encrypt(nonce, b"O",
                        self._large_aad(entry, version, desired))
                    with self._connect() as db:
                        db.execute("INSERT OR REPLACE INTO large_filters VALUES (?, ?, ?, ?)",
                                   (entry["blob"], desired, nonce, sealed))
                    overflow += 1
                    continue
                assert staging_path is not None
                self._commit_chunks(entry, staging_path, chunk_count, self.filter_bytes)
                indexed += 1
                continue
            nonce = os.urandom(12)
            if desired > self.filter_bytes:
                sealed = self._large_cipher.encrypt(nonce, value, self._large_aad(entry, version, desired))
                with self._connect() as db:
                    db.execute("INSERT OR IGNORE INTO large_filters "
                               "(blob, filter_bytes, nonce, sealed) VALUES (?, ?, ?, ?)",
                               (entry["blob"], desired, nonce, sealed))
            else:
                sealed = self._cipher.encrypt(nonce, value, self._aad(entry, version))
                with self._connect() as db:
                    db.execute("INSERT OR REPLACE INTO filters (blob, nonce, sealed) VALUES (?, ?, ?)",
                               (entry["blob"], nonce, sealed))
            if value == b"O":
                overflow += 1
            else:
                indexed += 1
        coverage = self.coverage()
        return {"indexed": indexed, "overflow": overflow, "skipped": skipped, **coverage}

    def coverage(self) -> dict[str, int | bool]:
        total = ready = overflow = 0
        with self._connect() as db:
            for _source, version, entry, _latest, _versions in self._current():
                total += 1
                complete = self._completion(db, entry)
                if complete is not None:
                    ready += 1
                    continue
                selected = self._lookup_filter(db, entry, version)
                if selected is None:
                    continue
                if selected[0] == b"O":
                    overflow += 1
                else:
                    ready += 1
        missing = total - ready - overflow
        return {"total": total, "ready": ready, "missing": missing,
                "unsearchable": overflow, "complete": missing == 0 and overflow == 0}

    def retry_plan(self) -> dict[str, int]:
        """Estimate legacy overflow upgrades without exposing source paths."""
        snapshots = source_bytes = projected_index_bytes = 0
        with self._connect() as db:
            for _source, version, entry, _latest, _versions in self._current():
                if self._completion(db, entry) is not None:
                    continue
                large = db.execute("SELECT filter_bytes, nonce, sealed FROM large_filters WHERE blob=?",
                                   (entry["blob"],)).fetchone()
                if large is not None and self._open_large_filter(entry, version, large) != b"O":
                    continue
                row = db.execute("SELECT nonce, sealed FROM filters WHERE blob=?", (entry["blob"],)).fetchone()
                if ((large is not None and self._open_large_filter(entry, version, large) == b"O")
                        or (row is not None and self._open_filter(entry, version, row) == b"O")):
                    snapshots += 1
                    source_bytes += entry["size"]
                    projected_index_bytes += entry["chunks"] * (self.filter_bytes + 64)
        return {"retryable_snapshots": snapshots, "source_bytes": source_bytes,
                "projected_index_bytes": projected_index_bytes,
                "free_disk_bytes": shutil.disk_usage(self.archive.root).free}

    def search(self, query: str, *, limit: int = 20,
               max_candidates: int = 200) -> dict[str, object]:
        """Return metadata only after decrypting and authenticating candidates."""
        terms = list(dict.fromkeys(_terms(query)))
        if not terms or len(terms) > 8 or not 1 <= limit <= 100 or not 1 <= max_candidates <= 1000:
            raise ValueError("Invalid bounded history search query")
        matches: list[dict[str, str | int]] = []
        seen_refs: set[str] = set()
        total = ready = overflow = candidates = 0
        truncated = False
        with self._connect() as db:
            for source, version, entry, latest, versions in self._current():
                total += 1
                segmented = False
                completion = self._completion(db, entry)
                if completion is not None:
                    found = {term: False for term in terms}
                    row_count = 0
                    cursor = db.execute(
                        "SELECT ordinal, filter_bytes, nonce, sealed "
                        "FROM chunk_filters WHERE blob=? ORDER BY ordinal",
                                        (entry["blob"],))
                    for expected_ordinal, row in enumerate(cursor):
                        row_count += 1
                        ordinal, filter_bytes, nonce, sealed = row
                        if ordinal != expected_ordinal or filter_bytes != completion[3]:
                            raise VaultIntegrityError("History chunk index is incomplete")
                        try:
                            value = self._large_cipher.decrypt(nonce, sealed,
                                self._chunk_aad(entry, version, ordinal, filter_bytes))
                        except (InvalidTag, ValueError) as exc:
                            raise VaultIntegrityError("History chunk filter integrity failed") from exc
                        if (len(value) != filter_bytes + 5 or value[:1] != b"L"
                                or struct.unpack(">I", value[1:5])[0] != filter_bytes):
                            raise VaultIntegrityError("History chunk filter format is invalid")
                        bits = value[5:]
                        for term in terms:
                            if not found[term] and all(bits[p >> 3] & (1 << (p & 7))
                                                       for p in self._positions(term, filter_bytes)):
                                found[term] = True
                    if row_count != entry["chunks"]:
                        raise VaultIntegrityError("History chunk index is incomplete")
                    ready += 1
                    if not all(found.values()):
                        continue
                    segmented = True
                    selected = (b"R", completion[3])
                else:
                    selected = self._lookup_filter(db, entry, version)
                if selected is None:
                    continue
                bits, filter_bytes = selected
                if bits == b"O":
                    overflow += 1
                    continue
                if not segmented:
                    ready += 1
                if not segmented and not all(all(bits[p >> 3] & (1 << (p & 7))
                               for p in self._positions(term, filter_bytes))
                           for term in terms):
                    continue
                if candidates >= max_candidates or len(matches) >= limit:
                    truncated = True
                    continue
                ref = hmac.new(self.archive._key, b"history-metadata-ref-v1\0" + source.encode("utf-8"),
                               hashlib.sha256).hexdigest()
                if ref in seen_refs:
                    continue
                candidates += 1
                found: set[str] = set()
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                carry = ""

                def accept(chunk: bytes) -> None:
                    nonlocal carry
                    try:
                        text = carry + decoder.decode(chunk)
                    except UnicodeDecodeError as exc:
                        raise VaultIntegrityError("History search text is not valid UTF-8") from exc
                    found.update(term for term in _terms(text) if term in terms)
                    carry = text[-128:]

                self.archive._verify_entry(entry, collect=False, on_chunk=accept)
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise VaultIntegrityError("History search text is not valid UTF-8") from exc
                if not all(term in found for term in terms):
                    continue
                seen_refs.add(ref)
                metadata = SafeHistoryMetadata(
                    ref=ref, provider=latest["provider"], kind=latest["kind"],
                    captured_day_utc=time.strftime("%Y-%m-%d", time.gmtime(latest["captured_at"])),
                    size_bucket_kib=(latest["size"] + 1023) // 1024,
                    versions=versions,
                ).as_dict()
                metadata["fetch_capability"] = self._capability(entry, version, terms[0])
                matches.append(metadata)
        missing = total - ready - overflow
        return {"matches": matches, "total": total, "ready": ready,
                "missing": missing, "overflow": overflow,
                "complete": missing == 0 and overflow == 0 and not truncated,
                "truncated": truncated}

    def fetch_span(self, capability: str, *, max_chars: int = 3000) -> dict[str, object]:
        """Authenticate an entire selected snapshot, then release one redacted hit span."""
        if not 1 <= max_chars <= 4000:
            raise ValueError("Invalid bounded history span")
        entry, version, data = self._entry_for_capability(capability)
        term = data["term"]
        structured = self._structured_span(entry, term, max_chars=max_chars)
        if structured is not None:
            return {"redacted_text": structured, "redaction": "strict-best-effort",
                    "version": version, "truncated": False}
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        carry = ""
        snippet = ""
        found = False

        def accept(chunk: bytes) -> None:
            nonlocal carry, snippet, found
            try:
                text = carry + decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                raise VaultIntegrityError("History text is not valid UTF-8") from exc
            if not found:
                position = text.casefold().find(term)
                if position >= 0:
                    found = True
                    snippet = text[max(0, position - 1000):position + 7000]
            elif len(snippet) < 8000:
                snippet += text[len(carry):][:8000 - len(snippet)]
            carry = text[-128:]

        self.archive._verify_entry(entry, collect=False, on_chunk=accept)
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise VaultIntegrityError("History text is not valid UTF-8") from exc
        if not found:
            raise ValueError("History search hit is no longer available")
        redacted = sanitize_agent_span(snippet[:12000], max_chars=max_chars)
        return {"redacted_text": redacted, "redaction": "strict-best-effort",
                "version": version, "truncated": len(snippet) > max_chars}

    def _structured_span(self, entry: dict, term: str, *, max_chars: int) -> str | None:
        """Stream JSONL chat messages before sanitizing message text.

        JSONL metadata may contain a credential on the same physical line as a
        useful message. Sanitizing that raw line first would hide the message.
        The parser discards metadata and tool payloads before release.
        """
        if (entry.get("kind") != "transcript"
                or entry.get("provider") not in {"codex", "claude_code", "gemini_cli"}):
            return None
        provider = entry["provider"]
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        line = ""
        skipping_long_line = False
        candidate: str | None = None

        def message_parts(row: object) -> list[tuple[str, str]]:
            if not isinstance(row, dict):
                return []
            if provider == "codex":
                payload = row.get("payload")
                if not isinstance(payload, dict):
                    return []
                if row.get("type") == "event_msg":
                    role = {"user_message": "User", "agent_message": "Assistant"}.get(payload.get("type"))
                    value = payload.get("message")
                elif row.get("type") == "response_item" and payload.get("type") == "message":
                    role = {"user": "User", "assistant": "Assistant"}.get(payload.get("role"))
                    content = payload.get("content")
                    value = (content if isinstance(content, str) else "\n".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") in
                        {"input_text", "output_text", "text"} and isinstance(part.get("text"), str)
                    )) if isinstance(content, (str, list)) else None
                else:
                    return []
                return [(role, value)] if role and isinstance(value, str) else []
            role_value = row.get("type") or row.get("role")
            role = {"user": "User", "human": "User", "assistant": "Assistant",
                    "model": "Assistant", "gemini": "Assistant"}.get(role_value)
            message = row.get("message") if isinstance(row.get("message"), dict) else row
            value = message.get("content") if isinstance(message, dict) else None
            if isinstance(value, str):
                return [(role, value)] if role else []
            if isinstance(value, list):
                text = "\n".join(part.get("text", "") for part in value
                                  if isinstance(part, dict) and isinstance(part.get("text"), str)
                                  and (provider != "claude_code" or part.get("type") == "text"))
                return [(role, text)] if role and text else []
            parts = message.get("parts") if isinstance(message, dict) else None
            if isinstance(parts, list):
                text = "\n".join(part if isinstance(part, str) else part.get("text", "")
                                  for part in parts if isinstance(part, (str, dict)))
                return [(role, text)] if role and text else []
            return []

        def consume(raw: str) -> None:
            nonlocal line, candidate, skipping_long_line
            if candidate is not None:
                return
            segments = raw.split("\n")
            for number, segment in enumerate(segments):
                if not skipping_long_line and len(line) + len(segment) <= _STRUCTURED_LINE_LIMIT:
                    line += segment
                else:
                    line = ""
                    skipping_long_line = True
                if number == len(segments) - 1:
                    break
                if skipping_long_line:
                    skipping_long_line = False
                    line = ""
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    line = ""
                    continue
                line = ""
                for role, message in message_parts(row):
                    position = message.casefold().find(term)
                    if position >= 0 and candidate is None:
                        candidate = f"{role}: " + message[max(0, position - 1000):position + 7000]
                        return

        def accept(chunk: bytes) -> None:
            try:
                consume(decoder.decode(chunk))
            except UnicodeDecodeError as exc:
                raise VaultIntegrityError("History text is not valid UTF-8") from exc

        self.archive._verify_entry(entry, collect=False, on_chunk=accept)
        try:
            consume(decoder.decode(b"", final=True))
        except UnicodeDecodeError as exc:
            raise VaultIntegrityError("History text is not valid UTF-8") from exc
        if line and not skipping_long_line:
            consume("\n")
        if candidate is None:
            return None
        redacted = sanitize_agent_span(candidate[:12000], max_chars=max_chars)
        return redacted if redacted.strip() and redacted.strip() != "[REDACTED_SENSITIVE_LINE]" else None


def main() -> int:
    import argparse
    import getpass
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Local encrypted history search index")
    parser.add_argument("action", choices=("status", "plan", "build", "retry"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-snapshots", type=int)
    args = parser.parse_args()
    passphrase = getpass.getpass("History archive recovery passphrase (hidden): ") if os.name != "nt" else None
    index = SecureHistoryBlindIndex(SecureHistoryArchive(args.root, passphrase))
    if args.action == "status":
        result = index.coverage()
    elif args.action == "plan":
        result = index.retry_plan()
    elif args.action == "build":
        result = index.build(max_snapshots=args.max_snapshots or 20)
    else:
        result = index.build(max_snapshots=args.max_snapshots or 2, retry_unsearchable=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
