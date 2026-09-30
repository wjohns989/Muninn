"""Encrypted, immutable source-unit evidence; no public raw-text endpoints.

Shares the archive's portable recovery envelope, deriving a distinct key and
AAD domain. The existing projection staging protocol publishes only after
full source authentication; source-unit metadata and text are ciphertext too.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.secure_projection_store import ProjectionIntegrityError, SecureProjectionStore
from muninn.history.transcript_units import PARSER_VERSION, SourceUnit, UnitFragment, transcript_units


class SourceEvidenceStore(SecureProjectionStore):
    # Text + bounded cwd can each contain 4096 JSON-escaped control chars.
    # Size the encrypted envelope for worst-case escaping, not ordinary prose.
    max_page_chars = 65536

    def __init__(self, archive: Any, root: Path | None = None):
        super().__init__(archive, root or Path(archive.root) / "source-evidence")

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

        def project(source: Iterable[bytes]) -> Iterator[str]:
            fragment = 0
            emitted = False
            for part in transcript_units(self.archive, entry, source, should_cancel=should_cancel):
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
        expected_unit, expected_fragment = 0, 0
        unit = None
        seen = 0
        for ordinal, length, ciphertext in rows:
            if ordinal != seen:
                raise ProjectionIntegrityError("source-unit fragment sequence is incomplete")
            page = self._decrypt_page(ident, attempt, ordinal, (length, ciphertext), cipher)
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
