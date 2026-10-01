"""Isolated synthetic evidence; no provider call or live vault access."""
import json

import pytest

from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore


PHRASE = "synthetic portable recovery phrase"
MODEL = "a" * 64


def fixture(tmp_path, text="I want source citations kept.", role="user", timestamp=True, cwd=True):
    source = tmp_path / "chat.jsonl"
    row = {"type": "event_msg", "payload": {"type": role + "_message", "message": text}}
    if role == "assistant":
        row = {"type": "response_item", "payload": {"type": "message", "role": "assistant",
               "content": [{"type": "output_text", "text": text}]}}
    if timestamp:
        row["timestamp"] = "2026-09-30T12:00:00Z"
    row["payload"]["id"] = "native-sentinel-no-egress-42"
    rows = [{"type": "session_meta", "payload": {"cwd": "C:/synthetic-project"}}] if cwd else []
    source.write_text("\n".join(json.dumps(r) for r in [*rows, row]) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PHRASE)
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    units = SourceEvidenceStore(archive)
    attempt = units.build_snapshot(entry, 0)
    page = next(ordinal for ordinal in range(units.count_pages(entry, 0, attempt))
                if (part := json.loads(units.get_page(entry, 0, attempt, ordinal)))["fragment"] == 1
                and part["text"] and text.startswith(part["text"]))
    return archive, entry, attempt, page


def record(ledger, entry, attempt, page, text="I want source citations kept.", **changes):
    proposal = {"type": "observation", "text": text, "quote": text, "start": 0}
    proposal.update(changes)
    return ledger.record(entry, 0, attempt, page, proposal, model_identity=MODEL)


def test_user_observation_is_durable_but_not_verified_truth(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    result = MemoryLedger(archive).get(ident)
    assert result["state"] == "filed"
    assert result["epistemic_kind"] == "source_observation"
    assert result["truth_status"] == "unverified_assertion"
    assert result["event_at"] is not None and result["project_ref"]
    assert "C:/synthetic-project" not in json.dumps(result)
    assert b"source citations" not in ledger.db_path.read_bytes()


@pytest.mark.parametrize("changes", [{"role": "assistant"}, {"timestamp": False}, {"cwd": False}])
def test_missing_evidence_or_assistant_interpretation_stays_provisional(tmp_path, changes):
    archive, entry, attempt, page = fixture(tmp_path, **changes)
    ledger = MemoryLedger(archive)
    result = ledger.get(record(ledger, entry, attempt, page))
    assert result["state"] == "provisional"
    assert result["truth_status"] != "verified"


def test_type_proposal_or_paraphrase_does_not_grant_filing_authority(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    assert ledger.get(record(ledger, entry, attempt, page, type="decision"))["state"] == "provisional"
    assert ledger.get(record(ledger, entry, attempt, page,
                             text="Keep citations", quote="I want source citations kept."))["state"] == "provisional"
    before = ledger.verify_all()
    with pytest.raises(ValueError):
        record(ledger, entry, attempt, page, quote="Invented source quote")
    assert ledger.verify_all() == before


def test_secret_outside_claim_diverts_and_cannot_enter_public_record_or_remote_input(tmp_path):
    text = "Keep citations. SERVICE_API_KEY=sk-synthetic-987654321-secret"  # gitleaks:allow synthetic redaction canary, never a real key
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page, text="Keep citations.")
    result = ledger.get(ident)
    assert result["type"] == "possible_credential" and result["state"] == "pending"
    assert "text" not in result and "quote" not in result
    assert "987654321" not in json.dumps(result)
    assert ledger.remote_input(entry, 0, attempt, page) is None


def test_chunk_boundary_cannot_hide_secret_context_or_grant_remote_eligibility(tmp_path):
    text = "Keep citations. " + "ordinary words " * 1000 + " SERVICE_API_KEY=synthetic$secret"
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    result = ledger.get(record(ledger, entry, attempt, page, text="Keep citations."))
    assert result["state"] == "pending" and "text" not in result
    assert ledger.remote_input(entry, 0, attempt, page) is None
    assert ledger.verify_all()["candidates"] == 1


def test_long_benign_unit_is_streamed_and_does_not_have_a_total_size_cutoff(tmp_path):
    text = "Keep citations. " + "ordinary words " * 20000
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    result = ledger.get(record(ledger, entry, attempt, page, text="Keep citations."))
    assert result["state"] == "provisional" and result["text"] == "Keep citations."
    assert ledger.remote_input(entry, 0, attempt, page) is not None
    assert len(ledger._screen_cache) == 1


def test_quoted_or_negated_excerpt_is_not_filed_as_the_users_assertion(tmp_path):
    text = 'Someone said "Keep citations."; I disagree.'
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    result = ledger.get(record(ledger, entry, attempt, page, text="Keep citations.", start=14))
    assert result["state"] == "provisional" and result["epistemic_kind"] == "source_excerpt"
    assert result["truth_status"] == "unverified_excerpt"


def test_possible_credential_proposal_is_never_returned_as_ordinary_text(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    result = ledger.get(record(ledger, entry, attempt, page, type="possible_credential"))
    assert result["state"] == "pending" and "text" not in result and "quote" not in result


@pytest.mark.parametrize("private_path", [r"C:\Users\synthetic\private.env",
                                       "C:/Users/synthetic/private.env", "/home/synthetic/private.env"])
def test_json_escaping_or_slash_style_does_not_bypass_private_home_gate(tmp_path, private_path):
    text = "Keep citations. " + private_path
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    result = ledger.get(record(ledger, entry, attempt, page, text="Keep citations."))
    assert "text" not in result and ledger.remote_input(entry, 0, attempt, page) is None


def test_retry_is_idempotent_but_distinct_occurrences_and_contrary_observations_survive(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path, text="Keep citations. Drop citations.")
    ledger = MemoryLedger(archive)
    first = record(ledger, entry, attempt, page, text="Keep citations.")
    assert first == record(ledger, entry, attempt, page, text="Keep citations.")
    second = record(ledger, entry, attempt, page, text="Drop citations.", start=16)
    assert second != first and ledger.get(first)["text"] == "Keep citations."
    assert ledger.get(second)["text"] == "Drop citations."
    assert ledger.verify_all()["candidates"] == 2
    assert ledger.remote_input(entry, 0, attempt, page)["text"] == "Keep citations. Drop citations."
    serialized = json.dumps(ledger.remote_input(entry, 0, attempt, page))
    assert "synthetic-project" not in serialized and "native_id" not in serialized
    assert "native-sentinel-no-egress-42" not in serialized


def test_same_text_at_different_offsets_preserves_both_occurrences(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path, text="Keep citations. Keep citations.")
    ledger = MemoryLedger(archive)
    first = record(ledger, entry, attempt, page, text="Keep citations.")
    second = record(ledger, entry, attempt, page, text="Keep citations.", start=16)
    assert first != second
    assert first == record(ledger, entry, attempt, page, text="Keep citations.")
    assert second == record(ledger, entry, attempt, page, text="Keep citations.", start=16)
    assert ledger.verify_all()["candidates"] == 2


def test_unsealed_or_forged_source_cannot_publish(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    with pytest.raises(MemoryLedgerIntegrityError):
        record(ledger, {**entry, "size": entry["size"] + 1}, attempt, page)
    units = SourceEvidenceStore(archive)
    with units._connect() as db:
        db.execute("UPDATE attempts SET state='building' WHERE attempt=?", (attempt,))
    with pytest.raises(MemoryLedgerIntegrityError):
        record(ledger, entry, attempt, page)
    assert ledger.verify_all()["candidates"] == 0


@pytest.mark.parametrize("mutation", ["ciphertext", "delete", "reference", "head"])
def test_ledger_tamper_fails_closed(tmp_path, mutation):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    with ledger._connect() as db:
        if mutation == "head":
            db.execute("UPDATE head SET ciphertext=zeroblob(length(ciphertext))")
        elif mutation == "delete":
            db.execute("DELETE FROM events")
        elif mutation == "reference":
            db.execute("UPDATE events SET ref=?", ("b" * 64,))
        else:
            db.execute("UPDATE events SET ciphertext=zeroblob(length(ciphertext))")
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger.get(ident)
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger.verify_all()


def test_portable_restore_keeps_ledger_and_authenticates_citation(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored", PHRASE)
    reopened = MemoryLedger(restored)
    assert reopened.get(ident)["text"] == "I want source citations kept."
    assert reopened.verify_all()["candidates"] == 1


def test_late_source_page_tamper_prevents_public_read_and_restore_success(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    units = SourceEvidenceStore(archive)
    with units._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                   "WHERE attempt=? AND ordinal=?", (attempt, page))
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger.get(ident)
    with pytest.raises(Exception):
        SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored", PHRASE)


def test_failed_head_update_rolls_back_candidate_publication(tmp_path, monkeypatch):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    before = ledger.verify_all()
    def fail(*args):
        raise RuntimeError("isolated commit interruption")
    monkeypatch.setattr(ledger, "_set_head", fail)
    with pytest.raises(RuntimeError):
        record(ledger, entry, attempt, page)
    assert MemoryLedger(archive).verify_all() == before


def test_new_candidate_cannot_append_to_a_corrupted_prefix_with_valid_tail(tmp_path):
    text = "Keep citations. Drop citations. Third observation."
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    record(ledger, entry, attempt, page, text="Keep citations.")
    record(ledger, entry, attempt, page, text="Drop citations.", start=16)
    with ledger._connect() as db:
        db.execute("UPDATE events SET ciphertext=zeroblob(length(ciphertext)) WHERE seq=1")
        before_events = db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        before_head = db.execute("SELECT * FROM head").fetchall()
    with pytest.raises(MemoryLedgerIntegrityError):
        record(ledger, entry, attempt, page, text="Third observation.", start=32)
    with ledger._connect() as db:
        assert db.execute("SELECT * FROM events ORDER BY seq").fetchall() == before_events
        assert db.execute("SELECT * FROM head").fetchall() == before_head


def test_review_decision_appends_without_erasing_observation(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    ledger.mark_needs_user(ident, reason="possible_contradiction")
    result = ledger.get(ident)
    assert result["state"] == "needs_user" and result["text"] == "I want source citations kept."
    assert result["truth_status"] == "unverified_assertion"
    assert ledger.verify_all() == {"events": 2, "candidates": 1, "decisions": 1}
