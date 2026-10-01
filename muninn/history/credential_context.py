"""Private, encrypted replay of every original ambiguous assignment occurrence.

Legacy vault rows deduplicate a value within a snapshot. Replay preserves ALL
its raw occurrences, then joins physical JSONL lines to authenticated source
units; a single representative never stands in for different contexts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from muninn.history.credential_discovery import ExtractionStats, iter_transcript_findings
from muninn.history.credential_store import AmbiguousCandidate
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.secure_projection_store import ProjectionIntegrityError


class CredentialContextStore(SourceEvidenceStore):
    def __init__(self, archive, root: Path | None = None):
        super().__init__(archive, root or Path(archive.root) / "credential-context")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS context_reviews(attempt TEXT NOT NULL, "
                       "page INTEGER NOT NULL, model_identity TEXT NOT NULL, ciphertext BLOB NOT NULL, "
                       "PRIMARY KEY(attempt,page,model_identity))")

    def _key(self) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"muninn credential context key v1").derive(self.archive._key)

    def _identity(self, entry: dict, version: int) -> dict:
        ident = super()._identity(entry, version)
        ident.update(format="secure-credential-context-v1", parser_redactor="raw-ambiguity-replay-v1")
        return ident

    def build_snapshot(self, entry: dict, version: int, *, should_cancel=lambda: False) -> str:
        existing = self.find_snapshot(entry, version)
        if existing is not None:
            return existing

        def project(source: Iterable[bytes]) -> Iterator[str]:
            def checked():
                for chunk in source:
                    if should_cancel():
                        raise RuntimeError("Credential context replay cancelled")
                    yield chunk
            for item in iter_transcript_findings(checked(), ExtractionStats(),
                                                include_ambiguous=True, include_context=True):
                if isinstance(item, AmbiguousCandidate):
                    yield json.dumps(asdict(item), ensure_ascii=False, separators=(",", ":"))

        # Raw assignment scanning needs one fully authenticated stream; unit
        # metadata is joined from the separately completed source-unit store.
        return super(SourceEvidenceStore, self).build(entry, version, project)

    def contexts(self, entry: dict, version: int, attempt: str) -> Iterator[AmbiguousCandidate]:
        last_line = -1
        for page in self._iter_sealed_pages(entry, version, attempt):
            try:
                data = json.loads(page)
                item = AmbiguousCandidate(**data)
                if (type(item.source_line) is not int or item.source_line < last_line
                        or not isinstance(item.context, str) or len(item.context) > 1280
                        or not isinstance(item.candidate, str) or len(item.candidate) > 512):
                    raise ValueError
                last_line = item.source_line
            except (TypeError, ValueError, KeyError) as exc:
                raise ProjectionIntegrityError("Credential source context authentication failed") from exc
            yield item

    def _review_aad(self, entry: dict, version: int, attempt: str, page: int, model_identity: str) -> bytes:
        if (type(page) is not int or page < 0 or not isinstance(model_identity, str)
                or len(model_identity) != 64 or any(c not in "0123456789abcdef" for c in model_identity)):
            raise ValueError("Invalid credential context review reference")
        return json.dumps({**self._identity(entry, version), "domain": "credential-context-review-v1",
                           "attempt": attempt, "page": page, "model_identity": model_identity},
                          sort_keys=True, separators=(",", ":")).encode()

    def cached_review(self, entry: dict, version: int, attempt: str, page: int,
                      model_identity: str) -> str | None:
        aad = self._review_aad(entry, version, attempt, page, model_identity)
        with self._connect() as db:
            count, _stats = self._authenticated_count(db, self._identity(entry, version), attempt)
            if page >= count:
                raise ProjectionIntegrityError("Credential review source is unavailable")
            row = db.execute("SELECT ciphertext FROM context_reviews WHERE attempt=? AND page=? AND model_identity=?",
                             (attempt, page, model_identity)).fetchone()
        if row is None:
            return None
        try:
            decision = AESGCM(self._key()).decrypt(row[0][:12], row[0][12:], aad).decode()
            if decision not in {"rejected", "deferred"}:
                raise ValueError
            return decision
        except (InvalidTag, ValueError, TypeError, UnicodeError) as exc:
            raise ProjectionIntegrityError("Credential context review authentication failed") from exc

    def record_review(self, entry: dict, version: int, attempt: str, page: int,
                      model_identity: str, decision: str) -> None:
        if decision not in {"rejected", "deferred"}:
            raise ValueError("Invalid credential context review decision")
        aad = self._review_aad(entry, version, attempt, page, model_identity)
        # Confirm this exact raw context page exists under the completion seal.
        self.get_page(entry, version, attempt, page)
        nonce = os.urandom(12)
        sealed = nonce + AESGCM(self._key()).encrypt(nonce, decision.encode(), aad)
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO context_reviews VALUES(?,?,?,?)",
                       (attempt, page, model_identity, sealed))

    def verify_all(self) -> dict[str, int]:
        entries = {(entry["blob"], entry["sha256"], version): entry
                   for versions in self.archive._load_manifest()["files"].values()
                   for version, entry in enumerate(versions)}
        report = {"snapshots": 0, "contexts": 0}
        with self._connect() as db:
            attempts = db.execute("SELECT attempt,blob,sha,version FROM attempts WHERE state='complete'").fetchall()
            orphan = db.execute("SELECT 1 FROM context_reviews r LEFT JOIN attempts a "
                                "ON r.attempt=a.attempt WHERE a.attempt IS NULL OR a.state!='complete' LIMIT 1").fetchone()
            if orphan is not None:
                raise ProjectionIntegrityError("Credential review has no completed source context")
        for attempt, blob, sha, version in attempts:
            entry = entries.get((blob, sha, version))
            if entry is None:
                raise ProjectionIntegrityError("Credential context has no authenticated snapshot")
            report["contexts"] += sum(1 for _ in self.contexts(entry, version, attempt))
            with self._connect() as db:
                reviews = db.execute("SELECT page,model_identity FROM context_reviews WHERE attempt=?", (attempt,)).fetchall()
            for page, model_identity in reviews:
                self.cached_review(entry, version, attempt, page, model_identity)
            report["snapshots"] += 1
        return report
