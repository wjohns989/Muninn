"""Encrypted, immutable source-unit evidence; no public raw-text endpoints.

Shares the archive's portable recovery envelope, deriving a distinct key and
AAD domain. The existing projection staging protocol publishes only after
full source authentication; source-unit metadata and text are ciphertext too.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history import streaming_redaction
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.secure_projection_store import ProjectionIntegrityError, SecureProjectionStore, _j
from muninn.history.structured_projector import ProjectionCancelled
from muninn.history.transcript_units import PARSER_VERSION, SourceUnit, UnitFragment, transcript_units


class SourceEvidenceStore(SecureProjectionStore):
    # Text + bounded cwd can each contain 4096 JSON-escaped control chars.
    # Size the encrypted envelope for worst-case escaping, not ordinary prose.
    max_page_chars = 65536

    def __init__(self, archive: Any, root: Path | None = None, *, read_only=False):
        self.read_only = read_only
        if read_only:
            self.archive = archive
            self.root = Path(root) if root is not None else Path(archive.root) / "source-evidence"
            self.db_path = self.root / "projections.sqlite3"
            self.lock_path = self.root / "projection.lock"
            verify_private(self.root)
            verify_private(self.db_path)
            with self._connect() as db:
                try:
                    db.execute("SELECT attempt,vault,blob,sha,size,version,state,count,digest,completion "
                               "FROM attempts LIMIT 0")
                    db.execute("SELECT attempt,ordinal,length,ciphertext FROM pages LIMIT 0")
                    db.execute("SELECT ref,ciphertext FROM unit_screens LIMIT 0")
                except sqlite3.DatabaseError as exc:
                    raise ProjectionIntegrityError("Source evidence schema is unavailable") from exc
            return
        super().__init__(archive, root or Path(archive.root) / "source-evidence")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS unit_screens("
                       "ref TEXT PRIMARY KEY, ciphertext BLOB NOT NULL)")

    @contextmanager
    def _connect(self):
        if not self.read_only:
            with super()._connect() as db:
                yield db
            return
        verify_private(self.db_path)
        db = sqlite3.connect(self.db_path.absolute().as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            db.execute("PRAGMA query_only=ON")
            yield db
        finally:
            db.close()

    def _screen_key(self):
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"muninn whole source unit screen v1").derive(self.archive._key)

    def _screen_binding(self, entry, version, attempt, unit):
        return {"source": self._identity(entry, version), "attempt": attempt,
                "unit": asdict(unit), "screen_version": streaming_redaction.UNIT_SCREEN_VERSION}

    def _screen_ref(self, binding):
        return hmac.new(self._screen_key(), b"index\0" + _j(binding), hashlib.sha256).hexdigest()

    def _decode_screen(self, ref, ciphertext):
        try:
            if (not isinstance(ref, str) or len(ref) != 64
                    or not isinstance(ciphertext, bytes) or not 28 < len(ciphertext) <= 65536):
                raise ValueError
            record = json.loads(AESGCM(self._screen_key()).decrypt(
                ciphertext[:12], ciphertext[12:], b"unit-screen-v1\0" + ref.encode("ascii")))
            if (set(record) != {"binding", "raw_sha", "screened_sha", "body_length", "body_sha"}
                    or type(record["body_length"]) is not int or record["body_length"] < 0
                    or any(not isinstance(record[k], str) or len(record[k]) != 64
                           or any(c not in "0123456789abcdef" for c in record[k])
                           for k in ("raw_sha", "screened_sha", "body_sha"))):
                raise ValueError
            binding = record["binding"]
            if (not isinstance(binding, dict)
                    or set(binding) != {"source", "attempt", "unit", "screen_version"}
                    or type(binding["screen_version"]) is not int or binding["screen_version"] < 1
                    or ref != self._screen_ref(binding)):
                raise ValueError
            return record
        except (InvalidTag, KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ProjectionIntegrityError("source-unit screening authentication failed") from exc

    def screen_info(self, entry, version, attempt, unit):
        """Read an original immutable-unit attestation, not a raw-file rescan.

        Selected pages and every serialized outgoing model request must still
        be authenticated/screened by callers. No read lock survives this call.
        """
        binding = self._screen_binding(entry, version, attempt, unit)
        ref = self._screen_ref(binding)
        with self._connect() as db:
            db.execute("BEGIN")
            _count, stats = self._authenticated_count(db, binding["source"], attempt)
            if stats is None or type(unit.ordinal) is not int or not 0 <= unit.ordinal < stats["source_units"]:
                raise ProjectionIntegrityError("source unit is unavailable")
            row = db.execute("SELECT CASE WHEN length(ciphertext)<=65536 THEN ciphertext ELSE NULL END "
                             "FROM unit_screens WHERE ref=?", (ref,)).fetchone()
            if row is None:
                return None
            record = self._decode_screen(ref, row[0])
            if record["binding"] != binding:
                raise ProjectionIntegrityError("source-unit screening binding changed")
        return (record["raw_sha"] == record["screened_sha"], record["body_length"],
                bytes.fromhex(record["body_sha"]))

    def _store_screen_info(self, entry, version, attempt, unit, *, raw_sha, screened_sha,
                           body_length, body_sha):
        """Private ledger writer: call only AFTER whole-unit iterator EOF/close."""
        binding = self._screen_binding(entry, version, attempt, unit)
        ref = self._screen_ref(binding)
        record = {"binding": binding, "raw_sha": raw_sha, "screened_sha": screened_sha,
                  "body_length": body_length, "body_sha": body_sha}
        nonce = os.urandom(12)
        ciphertext = nonce + AESGCM(self._screen_key()).encrypt(
            nonce, _j(record), b"unit-screen-v1\0" + ref.encode("ascii"))
        self._decode_screen(ref, ciphertext)  # Validate before any durable write.
        with self._connect() as db:
            # Optional cache writes must not stall paid interpretation behind
            # long-lived source readers. Durable source/ledger writes retain
            # their ordinary timeout and strict transaction behavior.
            db.execute("PRAGMA busy_timeout=100")
            db.execute("BEGIN IMMEDIATE")
            self._authenticated_count(db, binding["source"], attempt)
            db.execute("INSERT OR IGNORE INTO unit_screens VALUES(?,?)", (ref, ciphertext))
            stored = db.execute("SELECT CASE WHEN length(ciphertext)<=65536 THEN ciphertext ELSE NULL END "
                                "FROM unit_screens WHERE ref=?", (ref,)).fetchone()
            if self._decode_screen(ref, stored[0]) != record:
                raise ProjectionIntegrityError("source-unit screening proof conflicts")

    def _key(self) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"muninn source evidence key v1").derive(self.archive._key)

    def _identity(self, entry: dict, version: int) -> dict:
        ident = super()._identity(entry, version)
        ident.update(format="secure-source-evidence-v1", parser_redactor=f"source-units-v{PARSER_VERSION}",
                     provider=entry.get("provider"), kind=entry.get("kind"))
        return ident

    def _staged_pages(self, text: Iterable[str], page_chars: int) -> Iterable[str]:
        # Only this private store receives original units; its independently
        # derived key/format prevents raw pages being fetched as redacted ones.
        return text

    def build_snapshot(self, entry: dict, version: int, *,
                       should_cancel: Callable[[], bool] = lambda: False) -> str:
        existing = self.find_snapshot(entry, version)
        if existing is not None:
            return existing
        stats = {"source_units": 0, "conversational_units": 0, "omitted_units": 0}
        parent = self._append_parent(entry, version)

        def check_cancel():
            if should_cancel():
                raise ProjectionCancelled("source-unit extraction cancelled")

        def parent_parts():
            parent_entry, parent_attempt = parent
            ident = self._identity(parent_entry, version - 1)
            with self._connect() as db:
                count, parent_stats = self._authenticated_count(db, ident, parent_attempt)
            pages = self._iter_sealed_pages(parent_entry, version - 1, parent_attempt,
                                           check_cancel=check_cancel)
            try:
                yield from self._decoded_fragments(pages, count, parent_stats)
            finally:
                pages.close()

        def project(source: Iterable[bytes]) -> Iterator[str]:
            fragment = 0
            emitted = False
            for part in transcript_units(
                    self.archive, entry, source, should_cancel=should_cancel,
                    _prefix_entry=parent[0] if parent else None,
                    _parent_parts=parent_parts if parent else None):
                yield json.dumps({"unit": asdict(part.unit), "fragment": fragment,
                                  "text": part.text, "final": part.final},
                                 ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                fragment += 1
                emitted = emitted or bool(part.text)
                if part.final:
                    stats["source_units"] += 1
                    stats["conversational_units" if emitted else "omitted_units"] += 1
                    fragment, emitted = 0, False

        return super().build(entry, version, project, stats=stats)

    def _append_parent(self, entry: dict, version: int):
        """Resolve only an exact same-origin, sealed immediate JSONL prefix."""
        if (version == 0 or "prefix_of" not in entry or entry.get("kind") != "transcript"
                or entry.get("provider") not in {"codex", "claude_code"}):
            return None
        for versions in self.archive._load_manifest()["files"].values():
            if version >= len(versions) or versions[version] != entry:
                continue
            parent = self.archive._prefix_parent(versions, version)
            attempt = self.find_snapshot(parent, version - 1) if parent is not None else None
            return (parent, attempt) if attempt is not None else None
        return None

    def find_snapshot(self, entry: dict, version: int) -> str | None:
        ident = self._identity(entry, version)
        with self._connect() as db:
            row = db.execute("SELECT attempt FROM attempts WHERE vault=? AND blob=? AND sha=? "
                             "AND version=? AND state='complete' ORDER BY rowid DESC LIMIT 1",
                             (ident["vault"], ident["blob"], ident["hash"], version)).fetchone()
            if row is None:
                return None
            self._authenticated_count(db, ident, row[0])
            return row[0]

    def fragments(self, entry: dict, version: int, attempt: str) -> Iterator[UnitFragment]:
        # Pin one SQLite read snapshot and authenticate its seal/count once.
        # Revalidating COUNT(*) for every fragment would make this quadratic.
        ident = self._identity(entry, version)
        with self._connect() as db:
            db.execute("BEGIN")
            count, stats = self._authenticated_count(db, ident, attempt)
            cipher = AESGCM(self._key())
            rows = db.execute("SELECT ordinal,length,ciphertext FROM pages WHERE attempt=? ORDER BY ordinal",
                              (attempt,))
            yield from self._fragments(db, rows, ident, attempt, count, stats, cipher)

    def _fragments(self, db, rows, ident, attempt, count, stats, cipher) -> Iterator[UnitFragment]:
        def pages():
            seen = 0
            for ordinal, length, ciphertext in rows:
                if ordinal != seen:
                    raise ProjectionIntegrityError("source-unit fragment sequence is incomplete")
                seen += 1
                yield self._decrypt_page(ident, attempt, ordinal, (length, ciphertext), cipher)
        yield from self._decoded_fragments(pages(), count, stats)

    def _decoded_fragments(self, pages, count, stats) -> Iterator[UnitFragment]:
        expected_unit, expected_fragment = 0, 0
        unit = None
        seen = 0
        for page in pages:
            seen += 1
            try:
                data = json.loads(page)
                if (set(data) != {"unit", "fragment", "text", "final"}
                        or type(data["fragment"]) is not int or data["fragment"] != expected_fragment
                        or type(data["final"]) is not bool or not isinstance(data["text"], str)
                        or len(data["text"]) > 4096):
                    raise ValueError
                current = SourceUnit(**data["unit"])
                if current.ordinal != expected_unit or (unit is not None and current != unit):
                    raise ValueError
                if data["final"] and data["text"]:
                    raise ValueError
            except (TypeError, ValueError, KeyError) as exc:
                raise ProjectionIntegrityError("source-unit evidence authentication failed") from exc
            unit = current
            yield UnitFragment(unit, data["text"], data["final"])
            expected_fragment += 1
            if data["final"]:
                expected_unit, expected_fragment, unit = expected_unit + 1, 0, None
        if (seen != count or unit is not None or stats is None or expected_unit != stats["source_units"]):
            raise ProjectionIntegrityError("source-unit evidence coverage is incomplete")

    def unit_fragments(self, entry: dict, version: int, attempt: str,
                       unit_ordinal: int) -> Iterator[UnitFragment]:
        """Authenticate one complete unit with bounded memory and indexed seeks.

        Monotonic unit ordinals were generated before source publication. Binary
        seeks locate the range without scanning preceding multi-GB conversations.
        The caller must drain this iterator; do not hold it across model calls.
        """
        if type(unit_ordinal) is not int or unit_ordinal < 0:
            raise ProjectionIntegrityError("invalid source unit reference")
        ident = self._identity(entry, version)
        with self._connect() as db:
            db.execute("BEGIN")
            count, stats = self._authenticated_count(db, ident, attempt)
            if stats is None or unit_ordinal >= stats["source_units"]:
                raise ProjectionIntegrityError("source unit is unavailable")
            cipher = AESGCM(self._key())

            def read(ordinal):
                row = db.execute("SELECT length,CASE WHEN length BETWEEN 1 AND ? "
                                 "AND length(ciphertext)=length+28 THEN ciphertext ELSE NULL END "
                                 "FROM pages WHERE attempt=? AND ordinal=?",
                                 (self.max_page_chars * 4, attempt, ordinal)).fetchone()
                try:
                    data = json.loads(self._decrypt_page(ident, attempt, ordinal, row, cipher))
                    if (set(data) != {"unit", "fragment", "text", "final"}
                            or type(data["fragment"]) is not int or data["fragment"] < 0
                            or type(data["final"]) is not bool or not isinstance(data["text"], str)
                            or len(data["text"]) > 4096 or data["final"] and data["text"]):
                        raise ValueError
                    unit = SourceUnit(**data["unit"])
                    if type(unit.ordinal) is not int or not 0 <= unit.ordinal < stats["source_units"]:
                        raise ValueError
                    return unit, data
                except (ValueError, TypeError, KeyError) as exc:
                    raise ProjectionIntegrityError("source-unit evidence authentication failed") from exc

            def lower_bound(target):
                lo, hi = 0, count
                while lo < hi:
                    mid = (lo + hi) // 2
                    unit, _data = read(mid)
                    if unit.ordinal < target:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo

            first, end = lower_bound(unit_ordinal), lower_bound(unit_ordinal + 1)
            if first >= end:
                raise ProjectionIntegrityError("source unit is unavailable")
            expected, metadata = 0, None
            for ordinal in range(first, end):
                unit, data = read(ordinal)
                if (unit.ordinal != unit_ordinal or data["fragment"] != expected
                        or metadata is not None and unit != metadata
                        or data["final"] != (ordinal == end - 1)):
                    raise ProjectionIntegrityError("source-unit fragment sequence is incomplete")
                metadata = unit
                expected += 1
                yield UnitFragment(unit, data["text"], data["final"])

    def verify_all(self) -> dict[str, int]:
        entries = {(entry["blob"], entry["sha256"], version): entry
                   for versions in self.archive._load_manifest()["files"].values()
                   for version, entry in enumerate(versions)}
        report = {"snapshots": 0, "units": 0, "fragments": 0}
        with self._connect() as db:
            attempts = db.execute("SELECT attempt,blob,sha,version FROM attempts WHERE state='complete'").fetchall()
        for attempt, blob, sha, version in attempts:
            entry = entries.get((blob, sha, version))
            if entry is None:
                raise ProjectionIntegrityError("source evidence has no authenticated snapshot")
            for part in self.fragments(entry, version, attempt):
                report["fragments"] += 1
                report["units"] += int(part.final)
            report["snapshots"] += 1
        # Cache entries are ciphertext-only and included by SQLite backup. Even
        # obsolete screen-policy entries must authenticate against their exact
        # archived source/attempt/unit; corruption cannot hide behind a miss.
        with self._connect() as db:
            rows = db.execute("SELECT ref,CASE WHEN length(ciphertext)<=65536 THEN ciphertext ELSE NULL END "
                              "FROM unit_screens")
            for ref, ciphertext in rows:
                record = self._decode_screen(ref, ciphertext)
                try:
                    binding = record["binding"]
                    source = binding["source"]
                    entry = entries[(source["blob"], source["hash"], source["version"])]
                    if source != self._identity(entry, source["version"]):
                        raise ValueError
                    unit = SourceUnit(**binding["unit"])
                    parts = self.unit_fragments(entry, source["version"], binding["attempt"], unit.ordinal)
                    try:
                        if next(parts).unit != unit:
                            raise ValueError
                    finally:
                        parts.close()
                except (KeyError, TypeError, ValueError, StopIteration) as exc:
                    raise ProjectionIntegrityError("source-unit screening binding changed") from exc
        return report

    def backup_to(self, destination: Path) -> None:
        """Consistent ciphertext snapshot; never copies a live SQLite journal."""
        create_private_directory(destination)
        target = destination / "projections.sqlite3"
        create_private_file(target)
        verify_private(self.db_path)
        with self._connect() as source:
            copied = sqlite3.connect(target)
            try:
                source.backup(copied)
            finally:
                copied.close()
        verify_private(target)
