"""Encrypted query-independent window plans, not model-processing receipts.

Build once after authenticated source EOF. Indexed window reads avoid rescanning
an arbitrary-length transcript for each background inference. No model dispatch
or public plaintext endpoint belongs here.
"""
from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.secure_projection_store import SecureProjectionStore, ProjectionIntegrityError
from muninn.history.structured_projector import ProjectionCancelled
from muninn.history.transcript_units import PARSER_VERSION

WINDOW_CHARS = 3000
PLAN_VERSION = 1


class CitedWindowPlanStore(SecureProjectionStore):
    max_page_chars = 8192
    window_chars = WINDOW_CHARS

    def __init__(self, archive, root: Path | None = None):
        self.source = CitedAnalysisSource(archive)
        super().__init__(archive, root or Path(archive.root) / "cited-windows")

    def _key(self):
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"muninn cited window plans key v1").derive(self.archive._key)

    def _bound_identity(self, entry, version, source_attempt, *, geometry=None):
        if (not isinstance(source_attempt, str) or len(source_attempt) != 32
                or any(c not in "0123456789abcdef" for c in source_attempt)):
            raise ProjectionIntegrityError("Window plan source binding is unavailable")
        plan, parser, width = geometry or (PLAN_VERSION, PARSER_VERSION, self.window_chars)
        # Version-one plans support fragment partitions of any bounded width.
        # Unknown future parser/plan algorithms require an explicit migration.
        if (type(plan) is not int or plan != 1 or type(parser) is not int or parser != 1
                or type(width) is not int or not 1 <= width <= WINDOW_CHARS):
            raise ProjectionIntegrityError("Window plan geometry is unsupported")
        ident = super()._identity(entry, version)
        ident.update(vault=f"{self.archive.vault_id}:{source_attempt}:{plan}:{parser}:{width}",
                     source_attempt=source_attempt, format="secure-cited-window-plan-v1",
                     parser_redactor=f"source-units-v{parser}:windows-v{plan}:{width}")
        return ident

    def _identity(self, entry, version):
        attempt = self.source.ledger.units.find_snapshot(entry, version)
        return self._bound_identity(entry, version, attempt)

    def _staged_pages(self, text, page_chars):
        # Private descriptors, never prose to redact or expose as transcripts.
        return text

    @staticmethod
    def _cancel(should_cancel):
        if should_cancel():
            raise ProjectionCancelled("Window plan preparation cancelled")

    def _descriptors(self, entry, version, source_attempt, *, should_cancel=lambda: False, stats=None,
                     window_chars=None):
        width = self.window_chars if window_chars is None else window_chars
        emitted = False
        for page, part in enumerate(self.source.ledger.units.fragments(entry, version, source_attempt)):
            self._cancel(should_cancel)
            role_label = not emitted and part.text == f"\n\n{part.unit.role}: "
            emitted = emitted or bool(part.text)
            if (part.text and not part.final and not role_label
                    and (part.unit.role or "").casefold() in {"user", "assistant"}):
                for offset in range(0, len(part.text), width):
                    self._cancel(should_cancel)
                    text = part.text[offset:offset + width]
                    yield {"format": 1, "blob": entry["blob"], "sha256": entry["sha256"],
                           "version": version, "attempt": source_attempt, "page": page,
                           "offset": offset, "length": len(text), "parser_version": PARSER_VERSION,
                           "boundary_hit": False, "prefix": None,
                           "input_sha256": self.source._digest(self.source.content_window(part.unit, text))}
            if part.final:
                if stats is not None:
                    stats["source_units"] += 1
                    stats["conversational_units" if emitted else "omitted_units"] += 1
                emitted = False

    def find_snapshot(self, entry, version):
        source_attempt = self.source.ledger.units.find_snapshot(entry, version)
        if source_attempt is None:
            return None
        ident = self._bound_identity(entry, version, source_attempt)
        with self._connect() as db:
            row = db.execute("SELECT attempt FROM attempts WHERE vault=? AND blob=? AND sha=? "
                             "AND version=? AND state='complete' ORDER BY rowid DESC LIMIT 1",
                             (ident["vault"], ident["blob"], ident["hash"], version)).fetchone()
            if row is None:
                return None
            self._authenticated_count(db, ident, row[0])
            return row[0]

    def build_snapshot(self, entry, version, *, should_cancel=lambda: False):
        self._cancel(should_cancel)
        if self.source.ledger._entries.get((entry.get("blob"), version)) != entry:
            raise ProjectionIntegrityError("Window plan snapshot is not authenticated")
        source_attempt = self.source.ledger.units.build_snapshot(entry, version, should_cancel=should_cancel)
        existing = self.find_snapshot(entry, version)
        if existing is not None:
            return existing
        stats = {"source_units": 0, "conversational_units": 0, "omitted_units": 0}

        def project(raw):
            for _chunk in raw:
                self._cancel(should_cancel)
            if self._identity(entry, version)["source_attempt"] != source_attempt:
                raise ProjectionIntegrityError("Window plan source binding changed")
            for descriptor in self._descriptors(entry, version, source_attempt,
                    should_cancel=should_cancel, stats=stats):
                yield json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
            if self._identity(entry, version)["source_attempt"] != source_attempt:
                raise ProjectionIntegrityError("Window plan source binding changed")

        return super().build(entry, version, project, stats=stats)

    def window_at(self, entry, version, attempt, ordinal):
        descriptor = json.loads(self.get_page(entry, version, attempt, ordinal))
        CitedAnalysisSource.validate_descriptor(descriptor)
        if (descriptor["blob"] != entry["blob"] or descriptor["sha256"] != entry["sha256"]
                or descriptor["version"] != version
                or descriptor["attempt"] != self._identity(entry, version)["source_attempt"]):
            raise ProjectionIntegrityError("Window plan source binding is invalid")
        self.source.reopen(descriptor)
        return descriptor

    def verify_all(self):
        entries = self.source.ledger._entries
        report = {"snapshots": 0, "windows": 0}
        with self._connect() as db:
            attempts = db.execute("SELECT attempt,vault,blob,sha,version FROM attempts WHERE state='complete'").fetchall()
        prefix = self.archive.vault_id + ":"
        for attempt, vault, blob, sha, version in attempts:
            entry = entries.get((blob, version))
            if entry is None or entry["sha256"] != sha or not isinstance(vault, str) or not vault.startswith(prefix):
                raise ProjectionIntegrityError("Window plan snapshot is unavailable")
            try:
                source_attempt, plan, parser, width = vault[len(prefix):].split(":")
                geometry = tuple(int(value) for value in (plan, parser, width))
            except (ValueError, TypeError) as exc:
                raise ProjectionIntegrityError("Window plan identity is invalid") from exc
            ident = self._bound_identity(entry, version, source_attempt, geometry=geometry)
            if ident["vault"] != vault:
                raise ProjectionIntegrityError("Window plan identity is invalid")
            expected_stats = {"source_units": 0, "conversational_units": 0, "omitted_units": 0}
            expected = self._descriptors(entry, version, source_attempt, stats=expected_stats,
                                         window_chars=geometry[2])
            with self._connect() as db:
                db.execute("BEGIN")
                count, stats = self._authenticated_count(db, ident, attempt)
                rows = db.execute("SELECT ordinal,length,ciphertext FROM pages WHERE attempt=? ORDER BY ordinal", (attempt,))
                checked = 0
                for row, descriptor in zip_longest(rows, expected):
                    if row is None or descriptor is None or row[0] != checked:
                        raise ProjectionIntegrityError("Window plan coverage is incomplete")
                    actual = json.loads(self._decrypt_page(ident, attempt, row[0], row[1:]))
                    if actual != descriptor:
                        raise ProjectionIntegrityError("Window plan coverage is invalid")
                    checked += 1
                if checked != count or stats != expected_stats:
                    raise ProjectionIntegrityError("Window plan coverage is incomplete")
            report["snapshots"] += 1
            report["windows"] += checked
        return report
