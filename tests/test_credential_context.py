import json

import pytest

from muninn.history.credential_context import CredentialContextStore
from muninn.history.credential_discovery import ExtractionStats, iter_transcript_findings
from muninn.history.credential_store import AmbiguousCandidate
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.source_evidence import SourceEvidenceStore


def _fixture(tmp_path):
    source = tmp_path / "chat.jsonl"
    source.write_text("\n" + "\n".join(json.dumps(row) for row in [
        {"type": "session_meta", "payload": {"cwd": "C:/one"}},
        {"type": "event_msg", "timestamp": "2026-09-30T12:00:00Z", "payload": {
            "type": "user_message", "message": "SERVICE_API_KEY=${SYNTHETIC_REF}"}},
        {"type": "turn_context", "payload": {"cwd": "D:/two"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message":
            "SERVICE_API_KEY=${SYNTHETIC_REF} means another source observation"}},
    ]) + "\n")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic portable recovery phrase")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    return archive, entry


def test_replay_keeps_all_raw_occurrences_and_correct_physical_line(tmp_path):
    archive, entry = _fixture(tmp_path)
    units = SourceEvidenceStore(archive)
    attempt = units.build_snapshot(entry, 0)
    ends = [part.unit for part in units.fragments(entry, 0, attempt) if part.final]
    assert [unit.physical_line for unit in ends] == [1, 2, 3, 4]
    contexts = CredentialContextStore(archive)
    ident = contexts.build_snapshot(entry, 0)
    rows = list(contexts.contexts(entry, 0, ident))
    assert len(rows) == 2 and [row.source_line for row in rows] == [2, 4]
    assert ends[1].cwd == "C:/one" and ends[3].cwd == "D:/two"
    assert all("SYNTHETIC_REF" in row.context for row in rows)
    assert b"SYNTHETIC_REF" not in contexts.db_path.read_bytes()


def test_replay_line_mapping_survives_arbitrary_chunk_edges():
    raw = b'header\n\nSERVICE_API_KEY=${FIRST_REF}\nSERVICE_API_KEY=${SECOND_REF}\n'
    for width in [1, 17, 1023, 1024, 2048]:
        chunks = (raw[start:start + width] for start in range(0, len(raw), width))
        rows = [row for row in iter_transcript_findings(chunks, ExtractionStats(),
                include_ambiguous=True, include_context=True) if isinstance(row, AmbiguousCandidate)]
        assert [row.source_line for row in rows] == [2, 3]
        assert all("\n" not in row.context for row in rows)


def test_context_decision_cache_binds_source_page_and_model_identity(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = CredentialContextStore(archive)
    attempt = store.build_snapshot(entry, 0)
    identity = "a" * 64
    assert store.cached_review(entry, 0, attempt, 0, identity) is None
    store.record_review(entry, 0, attempt, 0, identity, "rejected")
    assert store.cached_review(entry, 0, attempt, 0, identity) == "rejected"
    assert store.cached_review(entry, 0, attempt, 1, identity) is None
    assert store.cached_review(entry, 0, attempt, 0, "b" * 64) is None
    with store._connect() as db:
        db.execute("UPDATE context_reviews SET page=1 WHERE page=0")
    with pytest.raises(ProjectionIntegrityError):
        store.cached_review(entry, 0, attempt, 1, identity)


def test_orphan_review_fails_portable_backup_verification(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = CredentialContextStore(archive)
    attempt = store.build_snapshot(entry, 0)
    store.record_review(entry, 0, attempt, 0, "a" * 64, "rejected")
    with store._connect() as db:
        db.execute("UPDATE context_reviews SET attempt='unbound'")
    with pytest.raises(ProjectionIntegrityError, match="completed source"):
        store.verify_all()


def test_portable_archive_restore_preserves_context_and_decision_cache(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = CredentialContextStore(archive)
    attempt = store.build_snapshot(entry, 0)
    store.record_review(entry, 0, attempt, 0, "a" * 64, "deferred")
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "synthetic portable recovery phrase")
    reopened = CredentialContextStore(restored)
    assert reopened.verify_all() == {"snapshots": 1, "contexts": 2}
    assert reopened.cached_review(entry, 0, attempt, 0, "a" * 64) == "deferred"


@pytest.mark.parametrize("error_kind", ["unsupported", "malformed", "integrity"])
def test_review_prepare_defers_only_unsupported_provenance(tmp_path, monkeypatch, error_kind):
    from muninn.history.credential_review_source import CredentialReviewSource
    from muninn.history.credential_store import source_fingerprint
    from muninn.history.structured_projector import UnsupportedTranscript
    from muninn.history.streaming_jsonl import StreamingJSONError
    from muninn.history.credential_crypto import VaultIntegrityError
    archive, entry = _fixture(tmp_path)
    review = CredentialReviewSource(archive.root)
    errors = {"unsupported": UnsupportedTranscript, "malformed": StreamingJSONError,
              "integrity": VaultIntegrityError}
    def fail(*args, **kwargs):
        raise errors[error_kind]("synthetic test failure")
    monkeypatch.setattr(review.units, "build_snapshot", fail)
    row = {"origin": "transcript", "source_hash": source_fingerprint(
        f"{archive.vault_id}:{entry['blob']}:{entry['sha256']}")}
    if error_kind == "integrity":
        with pytest.raises(VaultIntegrityError):
            review.prepare(row)
    else:
        assert review.prepare(row) is None
