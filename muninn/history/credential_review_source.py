"""Local source-context join for every occurrence of a legacy vault candidate."""

from __future__ import annotations

import hashlib
import hmac
import ntpath
import re
from itertools import zip_longest
from pathlib import Path

from muninn.history.ambiguity_triage import CandidateForReview
from muninn.history.credential_context import CredentialContextStore
from muninn.history.credential_discovery import (
    ExtractionStats, _INLINE_ASSIGN, iter_transcript_findings,
)
from muninn.history.credential_store import AmbiguousCandidate
from muninn.history.credential_provenance import archive_source_index
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import UnsupportedTranscript


class CredentialReviewSource:
    def __init__(self, root: Path | SecureHistoryArchive):
        # Portable callers can reuse a passphrase-authenticated archive handle;
        # path-only Windows callers retain the existing local DPAPI unlock.
        self.archive = root if isinstance(root, SecureHistoryArchive) else SecureHistoryArchive(root)
        self.index = archive_source_index(self.archive)
        self.manifest = self.archive._load_manifest()
        self.units = SourceEvidenceStore(self.archive)
        self.contexts = CredentialContextStore(self.archive)

    @staticmethod
    def _matches_context(context, row: dict, candidate: str) -> bool:
        if (context.reason, context.candidate) != (row.get("reason"), candidate):
            return False
        name = row.get("name")
        if context.name == name:
            return True
        if not isinstance(name, str) or not isinstance(context.name, str):
            return False
        longer, shorter = sorted((name, context.name), key=len, reverse=True)
        if not longer or longer[0] not in "nrt" or longer[1:] != shorter:
            return False
        matches = list(_INLINE_ASSIGN.finditer(context.context))
        if any(match.group("name") == longer for match in matches):
            return False
        for match in matches:
            prefix = context.context[match.start():match.start("name")]
            if match.group("name") != shorter or prefix != "\\" + longer[0]:
                continue
            findings = iter_transcript_findings(
                [context.context[match.start():].encode("utf-8")], ExtractionStats(),
                include_ambiguous=True)
            first = next(findings, None)
            if (isinstance(first, AmbiguousCandidate)
                    and (first.name, first.reason, first.candidate)
                    == (shorter, row.get("reason"), candidate)):
                return True
        return False

    def prepare(self, row: dict, *, candidate: str | None = None):
        source = self.index.get(row.get("source_hash"))
        if source is None or row.get("origin") != "transcript":
            return None
        entry = self.manifest["files"][source.source_path][source.version]
        try:
            units = self.units.build_snapshot(entry, source.version)
            current = self.contexts.build_snapshot(entry, source.version)
            legacy = self.contexts.find_snapshot(entry, source.version, parser_revision=1)
            compatible = False
            if legacy is not None and candidate is not None:
                compatible, matched = True, False
                old_rows = self.contexts.contexts(entry, source.version, legacy)
                new_rows = self.contexts.contexts(entry, source.version, current)
                # Drain both authenticated streams: one old match cannot prove
                # that the updated parser has not found additional occurrences.
                for old, new in zip_longest(old_rows, new_rows):
                    same = (old is not None and new is not None
                            and old.source_line == new.source_line
                            and old.context == new.context
                            and self._matches_context(old, {
                                "name": new.name, "reason": new.reason}, new.candidate))
                    compatible &= same
                    matched |= old is not None and self._matches_context(old, row, candidate)
                compatible &= matched
            contexts = legacy if compatible else current
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
                if not self._matches_context(context, row, candidate):
                    continue
                while (current is not None and current.physical_line is not None
                       and current.physical_line < context.source_line):
                    current = next(units, None)
                if current is None or current.physical_line != context.source_line:
                    yield page, CandidateForReview(row["id"], context.name, row["reason"], candidate)
                    continue
                label = ntpath.basename(ntpath.normpath(current.cwd)) if current.cwd else None
                label = label if label and re.fullmatch(r"[A-Za-z0-9._ -]{1,64}", label) else None
                project_ref = (hmac.new(self.archive._key, b"review-project-v1\0" +
                               ntpath.normcase(ntpath.normpath(current.cwd)).encode(),
                               hashlib.sha256).hexdigest() if current.cwd else None)
                yield page, CandidateForReview(row["id"], context.name, row["reason"], candidate, {
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
