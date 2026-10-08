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


def test_record_review_while_context_iterator_is_open_does_not_hold_read_lock(tmp_path):
    import time
    archive, entry = _fixture(tmp_path)
    store = CredentialContextStore(archive)
    attempt = store.build_snapshot(entry, 0)
    contexts = store.contexts(entry, 0, attempt)
    assert next(contexts).source_line == 2
    started = time.monotonic()
    try:
        store.record_review(entry, 0, attempt, 0, "a" * 64, "rejected")
        assert store.cached_review(entry, 0, attempt, 0, "a" * 64) == "rejected"
        assert len(list(contexts)) == 1
        assert time.monotonic() - started < 5
    finally:
        contexts.close()


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


def test_parser_revision_upgrade_keeps_legacy_reviews_and_paid_receipts(tmp_path):
    archive, entry = _fixture(tmp_path)
    legacy = CredentialContextStore(archive)
    legacy._parser_revision = 1
    old = legacy.build_snapshot(entry, 0)
    identity = "a" * 64
    legacy.record_review(entry, 0, old, 0, identity, "deferred")
    receipt = {"state": "received", "admission": "b" * 32, "generation": 1,
               "body_hash": "c" * 64, "response": {"model": "synthetic-model",
                   "cost": "0.001", "decision": "deferred", "http_status": 200}}
    legacy.save_remote_receipt(entry, 0, old, 0, identity, receipt, expected=None)
    current = CredentialContextStore(archive)
    assert current.find_snapshot(entry, 0) is None
    assert current.find_snapshot(entry, 0, parser_revision=1) == old
    assert current.cached_review(entry, 0, old, 0, identity) == "deferred"
    assert current.remote_receipt(entry, 0, old, 0, identity) == receipt
    new = current.build_snapshot(entry, 0)
    assert new != old and current.find_snapshot(entry, 0) == new
    assert len(list(current.contexts(entry, 0, old))) == 2
    assert current.cached_review(entry, 0, new, 0, identity) is None
    assert current.remote_receipt(entry, 0, new, 0, identity) is None
    assert current.verify_all() == {"snapshots": 2, "contexts": 4}
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "synthetic portable recovery phrase")
    recovered = CredentialContextStore(restored)
    assert recovered.verify_all() == {"snapshots": 2, "contexts": 4}
    assert recovered.remote_receipt(entry, 0, old, 0, identity) == receipt


def test_context_revision_selection_rejects_mismatched_or_forged_completion(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = CredentialContextStore(archive)
    store._parser_revision = 1
    old = store.build_snapshot(entry, 0)
    current = CredentialContextStore(archive)
    with pytest.raises(ProjectionIntegrityError):
        list(current.contexts({**entry, "sha256": "f" * 64}, 0, old))
    with store._connect() as db:
        db.execute("UPDATE attempts SET completion=? WHERE attempt=?", (b"forged", old))
    with pytest.raises(ProjectionIntegrityError):
        current.find_snapshot(entry, 0)


@pytest.mark.parametrize("name,raw,expected", [
    ("nSERVICE_API_KEY", r"\nSERVICE_API_KEY=${SYNTHETIC_REF}", True),
    ("nSERVICE_API_KEY", "SERVICE_API_KEY=${SYNTHETIC_REF}", False),
    ("rSERVICE_API_KEY", r"\nSERVICE_API_KEY=${SYNTHETIC_REF}", False),
    ("nSERVICE_API_KEY", r"\nSERVICE_API_KEY=${OTHER_REF}", False),
    ("nSERVICE_API_KEY", r"\nSERVICE_API_KEY=${SYNTHETIC_REF} nSERVICE_API_KEY=${SYNTHETIC_REF}", False),
])
def test_legacy_queue_name_join_requires_exact_escaped_source_evidence(name, raw, expected):
    from muninn.history.credential_review_source import CredentialReviewSource
    context = AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value",
                                 "${SYNTHETIC_REF}", "", 1, raw)
    row = {"name": name, "reason": "unparsed_value"}
    assert CredentialReviewSource._matches_context(context, row, "${SYNTHETIC_REF}") is expected


def test_corrected_queue_row_reuses_matching_legacy_paid_context(tmp_path):
    import re
    import muninn.history.credential_context as module
    from muninn.history import credential_discovery as discovery
    from muninn.history.credential_review_source import CredentialReviewSource
    from muninn.history.credential_store import source_fingerprint
    archive, _ = _fixture(tmp_path)
    source = tmp_path / "escaped.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "\nSERVICE_API_KEY=${SYNTHETIC_REF}"}}) + "\n")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    legacy = CredentialContextStore(archive)
    legacy._parser_revision = 1
    old_regex = re.compile(r'(?<![A-Za-z0-9_])(?P<name>[A-Za-z_][A-Za-z0-9_]{2,63})'
                           r'[ \t]{0,16}(?:=|\\?"[ \t]{0,16}:)[ \t]{0,16}\\?"?')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "iter_transcript_findings", lambda chunks, stats, **kwargs:
                      discovery._iter_findings(chunks, "", stats, old_regex, **kwargs))
        old = legacy.build_snapshot(entry, 0)
    assert list(legacy.contexts(entry, 0, old))[0].name == "nSERVICE_API_KEY"
    identity = "a" * 64
    legacy.record_review(entry, 0, old, 0, identity, "deferred")
    row = {"id": "b" * 32, "origin": "transcript", "name": "SERVICE_API_KEY",
           "reason": "unparsed_value", "source_hash": source_fingerprint(
               f"{archive.vault_id}:{entry['blob']}:{entry['sha256']}")}
    review = CredentialReviewSource(archive)
    prepared = review.prepare(row, candidate="${SYNTHETIC_REF}")
    assert prepared[3] == old
    inputs = list(review.inputs(prepared, row, "${SYNTHETIC_REF}"))
    assert len(inputs) == 1 and inputs[0][1].id == row["id"]
    assert inputs[0][1].name == "nSERVICE_API_KEY"
    assert review.cached(prepared, 0, identity) == "deferred"


def test_legacy_matching_context_cannot_hide_a_newly_discovered_occurrence(tmp_path):
    import re
    import muninn.history.credential_context as module
    from muninn.history import credential_discovery as discovery
    from muninn.history.credential_review_source import CredentialReviewSource
    from muninn.history.credential_store import source_fingerprint
    archive, _ = _fixture(tmp_path)
    source = tmp_path / "missed.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message":
        "API_KEY=${SYNTHETIC_REF} documentation\nAPI_KEY=${SYNTHETIC_REF} uncertain"}}) + "\n")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    legacy = CredentialContextStore(archive)
    legacy._parser_revision = 1
    old_regex = re.compile(r'(?<![A-Za-z0-9_])(?P<name>[A-Za-z_][A-Za-z0-9_]{2,63})'
                           r'[ \t]{0,16}(?:=|\\?"[ \t]{0,16}:)[ \t]{0,16}\\?"?')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "iter_transcript_findings", lambda chunks, stats, **kwargs:
                      discovery._iter_findings(chunks, "", stats, old_regex, **kwargs))
        old = legacy.build_snapshot(entry, 0)
    assert len(list(legacy.contexts(entry, 0, old))) == 1
    row = {"id": "b" * 32, "origin": "transcript", "name": "API_KEY",
           "reason": "unparsed_value", "source_hash": source_fingerprint(
               f"{archive.vault_id}:{entry['blob']}:{entry['sha256']}")}
    review = CredentialReviewSource(archive)
    prepared = review.prepare(row, candidate="${SYNTHETIC_REF}")
    assert prepared[3] != old
    assert len(list(review.inputs(prepared, row, "${SYNTHETIC_REF}"))) == 2
    assert len(list(review.contexts.contexts(entry, 0, old))) == 1


@pytest.mark.parametrize("error_kind", ["unsupported", "malformed", "integrity"])
def test_review_prepare_defers_only_unsupported_provenance(tmp_path, monkeypatch, error_kind):
    from muninn.history.credential_review_source import CredentialReviewSource
    from muninn.history.credential_store import source_fingerprint
    from muninn.history.structured_projector import UnsupportedTranscript
    from muninn.history.streaming_jsonl import StreamingJSONError
    from muninn.history.credential_crypto import VaultIntegrityError
    archive, entry = _fixture(tmp_path)
    review = CredentialReviewSource(archive)
    assert review.archive is archive
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
