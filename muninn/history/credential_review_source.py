"""Local source-context join for every occurrence of a legacy vault candidate."""

from __future__ import annotations

import hashlib
import hmac
import ntpath
import re
from pathlib import Path

from muninn.history.ambiguity_triage import CandidateForReview
from muninn.history.credential_context import CredentialContextStore
from muninn.history.credential_provenance import archive_source_index
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import UnsupportedTranscript


class CredentialReviewSource:
    def __init__(self, root: Path):
        self.archive = SecureHistoryArchive(root)
        self.index = archive_source_index(self.archive)
        self.manifest = self.archive._load_manifest()
        self.units = SourceEvidenceStore(self.archive)
        self.contexts = CredentialContextStore(self.archive)

    def prepare(self, row: dict):
        source = self.index.get(row.get("source_hash"))
        if source is None or row.get("origin") != "transcript":
            return None
        entry = self.manifest["files"][source.source_path][source.version]
        try:
            units = self.units.build_snapshot(entry, source.version)
            contexts = self.contexts.build_snapshot(entry, source.version)
        except (UnsupportedTranscript, StreamingJSONError):
            # Missing/contradictory provenance stays pending. Integrity and
            # storage failures must still abort, not masquerade as unknowns.
            return None
        return source, entry, units, contexts

    def inputs(self, prepared, row: dict, candidate: str):
        source, entry, units_attempt, context_attempt = prepared
        units = (part.unit for part in self.units.fragments(entry, source.version, units_attempt) if part.final)
        current = next(units, None)
        try:
            for page, context in enumerate(self.contexts.contexts(entry, source.version, context_attempt)):
                if (context.name, context.reason, context.candidate) != (row["name"], row["reason"], candidate):
                    continue
                while (current is not None and current.physical_line is not None
                       and current.physical_line < context.source_line):
                    current = next(units, None)
                if current is None or current.physical_line != context.source_line:
                    yield page, CandidateForReview(row["id"], row["name"], row["reason"], candidate)
                    continue
                label = ntpath.basename(ntpath.normpath(current.cwd)) if current.cwd else None
                label = label if label and re.fullmatch(r"[A-Za-z0-9._ -]{1,64}", label) else None
                project_ref = (hmac.new(self.archive._key, b"review-project-v1\0" +
                               ntpath.normcase(ntpath.normpath(current.cwd)).encode(),
                               hashlib.sha256).hexdigest() if current.cwd else None)
                yield page, CandidateForReview(row["id"], row["name"], row["reason"], candidate, {
                    "provider": current.provider, "record_ordinal": current.ordinal,
                    "record_type": current.kind, "role": current.role or "nonconversational",
                    "event_at": current.event_at, "time_basis": current.time_basis,
                    "cwd_label": label, "project_ref": project_ref,
                    "project_basis": current.project_basis,
                    "captured_at": source.captured_at, "source_mtime_ns": source.source_mtime_ns,
                    "context": context.context,
                })
        finally:
            units.close()

    def cached(self, prepared, page: int, model_identity: str):
        source, entry, _units, attempt = prepared
        return self.contexts.cached_review(entry, source.version, attempt, page, model_identity)

    def record(self, prepared, page: int, model_identity: str, decision: str):
        source, entry, _units, attempt = prepared
        self.contexts.record_review(entry, source.version, attempt, page, model_identity, decision)
