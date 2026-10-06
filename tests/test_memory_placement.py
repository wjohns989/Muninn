"""Isolated encrypted placement replay; synthetic receipts, no provider calls."""
import json

import pytest

from muninn.history.memory_classification import prepare_classification
from muninn.history.memory_ledger import MemoryLedger
from tests.test_memory_classification import peers, reply


def receipt(identity="b" * 64):
    return {"stage_id": identity, "admission_id": "c" * 32,
            "policy_generation": 1, "provider": "openrouter", "model": "synthetic-luna",
            "purpose": "memory-placement-v1"}


def commit(writer, reader, refs, *, peer_refs=None, stage="b" * 64):
    plan = prepare_classification(reader, refs, peer_refs=peer_refs)
    rows = [{"id": row["id"], "bucket": "preference", "disposition": "accepted",
        "evidence_refs": [row["id"], *[p["id"] for p in plan.payload()["peers"]]],
        "reason": "source_supported", "confidence": 0.99} for row in plan.payload()["candidates"]]
    raw = json.dumps({"items": rows})
    writer.commit_classification(plan, raw, model_identity="a" * 64, receipt=receipt(stage))
    return plan, raw


def test_durable_placement_visible_without_truth_or_candidate_mutation(tmp_path):
    writer, reader, refs = peers(tmp_path)
    before = {ref: writer._read_candidate(ref)[0] for ref in refs}
    plan, raw = commit(writer, reader, refs)
    for ref in refs:
        memory = MemoryLedger(writer.archive, read_only=True).get(ref)
        assert memory["state"] == "provisional" and memory["truth_status"] == "model_inferred"
        assert memory["placement"]["status"] == "accepted"
        assert memory["placement"]["bucket"] == "preference"
        assert memory["placement"]["global_conflict_coverage"] is False
        assert writer._read_candidate(ref)[0] == before[ref]
        assert writer.source(ref, include_transcript_capability=False)["memory"]["placement"] == memory["placement"]
    assert writer.review_page()["matches"] == []
    assert all(m["placement"]["status"] == "accepted" for m in writer.search("citations")["matches"])
    original = writer.verify_all()
    writer.commit_classification(plan, raw, model_identity="a" * 64, receipt=receipt())
    assert writer.verify_all() == original


def test_cohort_revision_and_human_decision_invalidate_without_resurrection(tmp_path):
    writer, reader, refs = peers(tmp_path)
    plan = prepare_classification(reader, refs)
    raw = json.dumps({"items": [{"id": row["id"], "bucket": "preference", "disposition": "accepted",
        "evidence_refs": [row["id"]], "reason": "source_supported", "confidence": 0.99}
        for row in plan.payload()["candidates"]]})
    writer.commit_classification(plan, raw, model_identity="a" * 64, receipt=receipt())
    writer.resolve_review(refs[1], state="filed", expected_state="provisional", reason="user_confirmed")
    writer.resolve_review(refs[1], state="filed", expected_state="filed", reason="user_confirmed")
    assert writer.get(refs[1])["placement"]["status"] == "stale"
    count = writer.verify_all()
    writer.commit_classification(plan, raw, model_identity="a" * 64, receipt=receipt())
    assert writer.verify_all() == count and writer.get(refs[1])["placement"]["status"] == "stale"
    assert writer.get(refs[0])["placement"]["status"] == "stale"
    assert [m["id"] for m in writer.review_page()["matches"]] == refs[:1]


def test_same_stage_identity_cannot_commit_different_result(tmp_path):
    writer, reader, refs = peers(tmp_path)
    plan, raw = commit(writer, reader, refs)
    with pytest.raises(ValueError, match="classification_stage_conflict"):
        writer.commit_classification(plan, raw.replace('"preference"', '"fact"'),
                                     model_identity="a" * 64, receipt=receipt())


def test_stale_peer_revision_rejected_before_write(tmp_path):
    writer, reader, refs = peers(tmp_path)
    plan = prepare_classification(reader, refs[:1], peer_refs=refs[1:])
    writer.resolve_review(refs[1], state="filed", expected_state="provisional", reason="user_confirmed")
    before = writer.verify_all()
    with pytest.raises(ValueError, match="classification_input_changed"):
        writer.commit_classification(plan, reply(plan), model_identity="a" * 64, receipt=receipt())
    assert writer.verify_all() == before


def test_replay_derived_peer_revision_and_transitive_invalidation(tmp_path):
    from tests.test_grouped_memory_review import add_source
    writer, reader, refs = peers(tmp_path)
    commit(writer, reader, refs)
    third = add_source(writer, tmp_path / "third.jsonl", cwd="C:/synthetic-project",
                       timestamp="2026-09-29T12:00:00Z")
    writer, reader = MemoryLedger(writer.archive), MemoryLedger(writer.archive, read_only=True)
    dependent, _raw = commit(writer, reader, [third], peer_refs=refs[:1], stage="d" * 64)
    assert dependent.bindings()[1]["decision_seq"] == 3  # cohort stored under A also revised B
    writer.resolve_review(refs[1], state="filed", expected_state="provisional", reason="user_confirmed")
    for ref in [refs[0], refs[1], third]:
        assert writer.get(ref)["placement"]["status"] == "stale"
        assert writer.source(ref, include_transcript_capability=False)["memory"]["placement"]["status"] == "stale"
    assert {m["id"] for m in writer.review_page()["matches"]} == {refs[0], third}
    assert all(m["placement"]["status"] == "stale" for m in writer.search("citations")["matches"])
    assert writer.verify_all()["decisions"] == 3


def test_placement_bucket_search_preserves_original_type_and_truth(tmp_path):
    writer, reader, refs = peers(tmp_path)
    plan = prepare_classification(reader, refs[:1])
    writer.commit_classification(plan, reply(plan, bucket="procedure"),
                                 model_identity="a" * 64, receipt=receipt())
    match = writer.search("procedure")["matches"][0]
    assert match["type"] == "preference" and match["truth_status"] == "model_inferred"
    assert match["placement"]["bucket"] == "procedure"


def test_conflicts_remain_visible_for_consultation_without_supersession(tmp_path):
    writer, reader, refs = peers(tmp_path)
    plan = prepare_classification(reader, refs[:1], peer_refs=refs[1:])
    writer.commit_classification(plan, reply(plan, disposition="conflict", reason="conflicting_evidence",
        evidence_refs=["m0", "m1"], bucket="conflict"), model_identity="a" * 64, receipt=receipt())
    page = writer.grouped_review_page()
    assert any(group["type"] == "conflict" for group in page["groups"])
    assert writer.get(refs[0])["placement"]["status"] == "conflict"
    assert writer.get(refs[0])["event_at"] == writer.get(refs[1])["event_at"]
    assert writer.get(refs[0])["truth_status"] == "model_inferred"


def test_portable_archive_restore_keeps_placement_and_human_invalidation(tmp_path, monkeypatch):
    import time
    from dataclasses import asdict
    from muninn.history.capture_journal import CaptureJournal
    from muninn.history.classification_jobs import PURPOSE
    from muninn.history.secure_archive import SecureHistoryArchive
    from tests.test_memory_ledger import PHRASE
    writer, reader, refs = peers(tmp_path)
    # Component fixture uses an authenticated synthetic journal owner; the
    # separate discovery tests prove enrollment from actual publication ACKs.
    journal = CaptureJournal(writer.archive, recover=False)
    plan = prepare_classification(reader, refs)
    rows = [{"id": row["id"], "bucket": "preference", "disposition": "accepted",
        "evidence_refs": [row["id"]], "reason": "source_supported", "confidence": 0.99}
        for row in plan.payload()["candidates"]]
    job_id, lease = "f" * 32, "e" * 32
    value = {"refs": refs, "state": "running", "lease": lease, "lease_until": time.time() + 120,
        "attempt": 1, "prepared": asdict(plan), "admission": "c" * 32, "generation": 1,
        "stage": None, "reason": ""}
    with journal._connect() as db:
        db.execute("INSERT INTO memory_classification_jobs VALUES(?,?,?,?)",
                   (job_id, journal._seal_search(value, job_id, PURPOSE), "running", time.time()))
    monkeypatch.setattr("muninn.history.remote_accounting.settled_response", lambda *a, **k: True)
    monkeypatch.setattr("muninn.history.remote_accounting.classification_admission_state", lambda *a, **k: "settled")
    journal.stage_classification(job_id, lease, json.dumps({"items": rows}), model="synthetic-luna")
    journal.publish_classification(job_id)
    writer.resolve_review(refs[1], state="filed", expected_state="provisional", reason="user_confirmed")
    expected = [writer.get(ref) for ref in refs]
    writer.archive.backup_to(tmp_path / "backup")
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "restored", PHRASE)
    actual = MemoryLedger(restored, read_only=True)
    assert [actual.get(ref) for ref in refs] == expected
    assert actual.verify_all() == writer.verify_all()
    assert [m["id"] for m in actual.review_page()["matches"]] == refs[:1]


def test_peer_invalidated_during_new_cohort_is_not_silently_rebound(tmp_path):
    from tests.test_grouped_memory_review import add_source
    writer, reader, refs = peers(tmp_path)
    # B depends on still-unclassified A. C then depends on B.
    commit(writer, reader, refs[1:], peer_refs=refs[:1])
    third = add_source(writer, tmp_path / "third.jsonl", cwd="C:/synthetic-project",
                       timestamp="2026-09-29T12:00:00Z")
    writer, reader = MemoryLedger(writer.archive), MemoryLedger(writer.archive, read_only=True)
    commit(writer, reader, [third], peer_refs=refs[1:], stage="d" * 64)
    # Placing A invalidates B. A's new dependency on B must remain pinned
    # to observed B, not acquire the new invalidated B revision silently.
    commit(writer, reader, refs[:1], peer_refs=refs[1:], stage="e" * 64)
    assert all(writer.get(ref)["placement"]["status"] == "stale" for ref in [*refs, third])
    assert {m["id"] for m in writer.review_page()["matches"]} == {*refs, third}


def test_invalid_authenticated_cohort_blocks_all_reads_and_publication(tmp_path):
    from muninn.history.memory_ledger import MemoryLedgerIntegrityError
    writer, reader, refs = peers(tmp_path)
    commit(writer, reader, refs)
    with writer._connect() as db:
        seq, ref, sealed = db.execute("SELECT seq,ref,ciphertext FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        event = writer._open(sealed, "event", seq, ref)
        event["payload"]["items"][0]["truth_status"] = "verified"
        malformed = writer._seal(event, "event", seq, ref)
        db.execute("UPDATE events SET ciphertext=? WHERE seq=?", (malformed, seq))
        writer._set_head(db, seq, writer._digest(seq, ref, malformed))
    before = writer.db_path.read_bytes()
    for check in (writer.verify_all, lambda: writer.verify_refs(refs), lambda: writer.get(refs[1]),
                  lambda: writer._append(refs[1], {"event": "decision", "state": "needs_user"})):
        with pytest.raises(MemoryLedgerIntegrityError, match="placement authentication"):
            check()
        assert writer.db_path.read_bytes() == before
