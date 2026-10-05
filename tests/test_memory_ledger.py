"""Isolated synthetic evidence; no provider call or live vault access."""
import json
import sqlite3
from contextlib import contextmanager

import pytest

from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.private_acl import create_private_directory


PHRASE = "synthetic portable recovery phrase"
MODEL = "a" * 64


@pytest.mark.parametrize("private", [False, True])
def test_screen_cache_writer_contention_does_not_replace_source_proof(tmp_path, monkeypatch, private):
    text = ("Keep citations. SERVICE_API_KEY=sk-synthetic-only-secret" if private
            else "I want source citations kept.")  # gitleaks:allow synthetic canary
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    original = ledger.units._connect

    @contextmanager
    def fast_connections():
        with original() as db:
            db.execute("PRAGMA busy_timeout=50")
            yield db

    monkeypatch.setattr(ledger.units, "_connect", fast_connections)
    with original() as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM pages").fetchone()
        result = ledger.remote_input(entry, 0, attempt, page)
        assert (result is None) is private
        unit, _ = ledger._source(entry, 0, attempt, page)
        assert ledger.units.screen_info(entry, 0, attempt, unit) is None
        # A second reader must reauthenticate the source, not invent or reuse
        # an attestation whose commit failed.
        reopened = MemoryLedger(archive)
        monkeypatch.setattr(reopened.units, "_connect", fast_connections)
        calls = []
        fragments = reopened.units.unit_fragments

        def checked_fragments(*args, **kwargs):
            calls.append(True)
            return fragments(*args, **kwargs)

        monkeypatch.setattr(reopened.units, "unit_fragments", checked_fragments)
        assert (reopened.remote_input(entry, 0, attempt, page) is None) is private
        assert calls == [True]
    # Once contention ends, the ordinary encrypted cache can be committed.
    fresh = MemoryLedger(archive)
    assert (fresh.remote_input(entry, 0, attempt, page) is None) is private
    assert fresh.units.screen_info(entry, 0, attempt, unit) is not None


@pytest.mark.parametrize("code", [sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_FULL, sqlite3.SQLITE_IOERR])
def test_screen_cache_noncontention_errors_remain_fatal(tmp_path, monkeypatch, code):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)

    def fail(*args, **kwargs):
        error = sqlite3.OperationalError("synthetic failure")
        error.sqlite_errorcode = code
        raise error

    monkeypatch.setattr(ledger.units, "_store_screen_info", fail)
    with pytest.raises(sqlite3.OperationalError):
        ledger.remote_input(entry, 0, attempt, page)
    assert not ledger._screen_cache


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


@pytest.mark.parametrize("private_path", [r"C:\Users\user\private.env",
                                       "C:/Users/user/private.env", "/home/user/private.env"])
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


def test_human_review_requires_portable_unlock_and_never_verifies_truth(tmp_path, monkeypatch):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    before = ledger.get(ident)
    monkeypatch.setattr(archive, "_unlocked_with_passphrase", False)
    with pytest.raises(PermissionError):
        ledger.resolve_review(ident, state="filed", expected_state="filed", reason="user_confirmed")
    assert ledger.verify_all() == {"events": 1, "candidates": 1, "decisions": 0}
    monkeypatch.setattr(archive, "_unlocked_with_passphrase", True)
    ledger.resolve_review(ident, state="rejected", expected_state="filed", reason="user_rejected")
    after = MemoryLedger(archive).get(ident)
    assert after["state"] == "rejected"
    for key in ("type", "truth_status", "epistemic_kind", "text", "quote", "source_ref",
                "event_at", "time_basis", "project_ref", "project_basis"):
        assert after[key] == before[key]
    assert after["truth_status"] == "unverified_assertion"
    assert ledger.verify_all() == {"events": 2, "candidates": 1, "decisions": 1}


def test_human_review_rejects_credentials_and_stale_compare_without_appending(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path, text="SERVICE_API_KEY=synthetic-only-value")
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page, text="SERVICE_API_KEY=synthetic-only-value",
                   type="possible_credential")
    before = ledger.verify_all()
    with pytest.raises(ValueError):
        ledger.resolve_review(ident, state="filed", expected_state="pending", reason="user_confirmed")
    assert ledger.verify_all() == before

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    archive, entry, attempt, page = fixture(stale_root)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    ledger.resolve_review(ident, state="needs_user", expected_state="filed",
                          reason="insufficient_context")
    before = ledger.verify_all()
    with pytest.raises(ValueError):
        ledger.resolve_review(ident, state="filed", expected_state="filed", reason="user_confirmed")
    assert ledger.verify_all() == before


def test_rejected_human_memory_is_hidden_from_search_but_available_by_exact_reference(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    ledger.resolve_review(ident, state="rejected", expected_state="filed", reason="not_reliable")
    assert ledger.search("citations")["matches"] == []
    result = ledger.get(ident)
    assert result["state"] == "rejected" and result["text"] == "I want source citations kept."
    assert ledger.review_status() == {"provisional": 0, "filed": 0, "needs_user": 0, "rejected": 1}


def test_human_review_chain_binding_is_verified_and_legacy_needs_user_still_works(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    ledger.mark_needs_user(ident, reason="possible_contradiction")
    assert ledger.get(ident)["state"] == "needs_user"
    ledger.resolve_review(ident, state="filed", expected_state="needs_user", reason="source_supported")
    assert ledger.get(ident)["state"] == "filed"
    assert ledger.verify_all() == {"events": 3, "candidates": 1, "decisions": 2}

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    archive, entry, attempt, page = fixture(invalid_root)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    ledger._append(ident, {"event": "human_review", "state": "filed",
        "expected_state": "provisional", "reason": "user_confirmed", "actor": "local-user",
        "candidate_sha256": "0" * 64, "citation_sha256": "0" * 64})
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger.verify_all()
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger.get(ident)


def test_failed_human_review_head_update_rolls_back_transition(tmp_path, monkeypatch):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    before = ledger.verify_all()

    def fail(*_args):
        raise RuntimeError("isolated head interruption")

    monkeypatch.setattr(ledger, "_set_head", fail)
    with pytest.raises(RuntimeError):
        ledger.resolve_review(ident, state="rejected", expected_state="filed",
                              reason="user_rejected")
    reopened = MemoryLedger(archive)
    assert reopened.verify_all() == before
    assert reopened.get(ident)["state"] == "filed"


def test_review_preimage_is_new_private_encrypted_and_verified_before_return(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    ident = record(ledger, entry, attempt, page)
    ledger.resolve_review(ident, state="needs_user", expected_state="filed",
                          reason="possible_contradiction")
    private_parent = tmp_path / "private-preimages"
    create_private_directory(private_parent)
    result = ledger.backup_review_preimage(private_parent / "review-preimage")
    snapshot = private_parent / "review-preimage" / "memory-ledger.sqlite3"
    assert result["path"] == str(snapshot)
    assert result["events"] == 2 and result["candidates"] == 1 and result["decisions"] == 1
    assert b"I want source citations kept." not in snapshot.read_bytes()
    with sqlite3.connect(snapshot) as db:
        assert ledger._verify_snapshot(db) == {"events": 2, "candidates": 1, "decisions": 1}
    assert MemoryLedger(archive).get(ident)["truth_status"] == "unverified_assertion"


@pytest.mark.parametrize("destination_kind", ["existing", "inside_archive"])
def test_review_preimage_rejects_overwrite_and_archive_paths(tmp_path, destination_kind):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    if destination_kind == "existing":
        destination = tmp_path / "existing"
        destination.mkdir()
    else:
        destination = archive.root / "review-preimage"
    with pytest.raises(ValueError):
        ledger.backup_review_preimage(destination)


def test_bounded_batch_authenticates_old_chain_once_and_preserves_idempotency(tmp_path, monkeypatch):
    text = "Keep citations. Drop citations."
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    items = [{"page": page, "proposal": {"type": "observation", "text": quote,
              "quote": quote, "start": start}}
             for quote, start in [("Keep citations.", 0), ("Drop citations.", 16)]]
    original_walk = ledger._walk
    calls = []
    def walk(db):
        calls.append(True)
        yield from original_walk(db)
    monkeypatch.setattr(ledger, "_walk", walk)
    refs = ledger.record_batch(entry, 0, attempt, items + [items[0]], model_identity=MODEL)
    assert len(calls) == 1 and refs[0] != refs[1] and refs[0] == refs[2]
    assert ledger.record_batch(entry, 0, attempt, items, model_identity=MODEL) == refs[:2]
    assert len(calls) == 2
    assert ledger.verify_all() == {"events": 2, "candidates": 2, "decisions": 0}
    assert MemoryLedger(archive).get(refs[1])["text"] == "Drop citations."


@pytest.mark.parametrize("failure", ["bad_quote", "bad_page", "head", "prefix"])
def test_batch_publication_is_all_or_none(tmp_path, monkeypatch, failure):
    archive, entry, attempt, page = fixture(tmp_path, text="Keep citations. Drop citations.")
    ledger = MemoryLedger(archive)
    record(ledger, entry, attempt, page, text="Keep citations.")
    if failure == "prefix":
        with ledger._connect() as db:
            db.execute("UPDATE events SET ciphertext=zeroblob(length(ciphertext)) WHERE seq=1")
    with ledger._connect() as db:
        before_events = db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        before_head = db.execute("SELECT * FROM head").fetchall()
    items = [{"page": page, "proposal": {"type": "observation", "text": "Drop citations.",
              "quote": "Drop citations.", "start": 16}},
             {"page": page, "proposal": {"type": "decision", "text": "Keep citations.",
              "quote": "Keep citations.", "start": 0}}]
    if failure == "bad_quote":
        items[1]["proposal"]["quote"] = "invented quote"
    elif failure == "bad_page":
        items[1]["page"] = 999999
    elif failure == "head":
        def fail(*args):
            raise RuntimeError("isolated head interruption")
        monkeypatch.setattr(ledger, "_set_head", fail)
    with pytest.raises((ValueError, RuntimeError)):
        ledger.record_batch(entry, 0, attempt, items, model_identity=MODEL)
    with ledger._connect() as db:
        assert db.execute("SELECT * FROM events ORDER BY seq").fetchall() == before_events
        assert db.execute("SELECT * FROM head").fetchall() == before_head


@pytest.mark.parametrize("items", [[], [None], [{"page": 0}], [None] * 65, iter([])])
def test_batch_enforces_bounded_explicit_shape_without_publication(tmp_path, items):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    with pytest.raises(ValueError):
        ledger.record_batch(entry, 0, attempt, items, model_identity=MODEL)
    assert ledger.verify_all() == {"events": 0, "candidates": 0, "decisions": 0}


def test_model_origin_cannot_alias_or_promote_a_direct_source_observation(tmp_path):
    text = "I want source citations kept."
    archive, entry, attempt, page = fixture(tmp_path, text=text)
    ledger = MemoryLedger(archive)
    direct = record(ledger, entry, attempt, page)
    proposal = {"type": "observation", "text": text, "quote": text, "start": 0}
    refs = ledger.record_batch(entry, 0, attempt, [{"page": page, "proposal": proposal}],
                               model_identity=MODEL)
    assert refs[0] != direct
    model = MemoryLedger(archive).get(refs[0])
    assert model["state"] == "provisional" and model["proposal_origin"] == "model"
    assert model["epistemic_kind"] == "source_observation"
    assert model["truth_status"] == "unverified_assertion"
    assert ledger.get(direct)["state"] == "filed"
    assert ledger.get(direct)["proposal_origin"] == "source_rule"
    with pytest.raises(TypeError):
        ledger.record_batch(entry, 0, attempt, [{"page": page, "proposal": proposal}],
                            model_identity=MODEL, proposal_origin="source_rule")
    assert ledger.verify_all()["candidates"] == 2


def test_source_rule_identity_stays_compatible_with_existing_ledger(tmp_path):
    import hashlib
    import hmac
    from muninn.history.memory_ledger import POLICY, _json
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    unit, data = ledger._source(entry, 0, attempt, page)
    proposal = {"type": "observation", "text": "I want source citations kept.",
                "quote": "I want source citations kept.", "start": 0}
    old_ref = hmac.new(ledger._key, b"candidate\0" + _json({
        "blob": entry["blob"], "sha": entry["sha256"], "version": 0,
        "unit": unit.ordinal, "fragment": data["fragment"], "proposal": proposal,
        "policy": POLICY, "model": MODEL}), hashlib.sha256).hexdigest()
    assert record(ledger, entry, attempt, page) == old_ref


def test_legacy_origin_is_unknown_and_model_origin_survives_portable_restore(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    proposal = {"type": "observation", "text": "I want source citations kept.",
                "quote": "I want source citations kept.", "start": 0}
    old_ref, old_payload = ledger._prepare_record(entry, 0, attempt, page, proposal,
                                                  model_identity=MODEL, proposal_origin="source_rule")
    old_payload.pop("proposal_origin")
    ledger._append(old_ref, old_payload, idempotent=True)
    assert ledger.get(old_ref)["proposal_origin"] == "legacy_unrecorded"
    assert record(ledger, entry, attempt, page) == old_ref
    assert ledger.get(old_ref)["proposal_origin"] == "legacy_unrecorded"
    refs = ledger.record_batch(entry, 0, attempt, [{"page": page, "proposal": proposal}],
                               model_identity=MODEL)
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored", PHRASE)
    reopened = MemoryLedger(restored)
    assert reopened.get(old_ref)["proposal_origin"] == "legacy_unrecorded"
    assert reopened.get(refs[0])["proposal_origin"] == "model"
    assert reopened.get(refs[0])["state"] == "provisional"
    assert reopened.verify_all()["candidates"] == 2


@pytest.mark.parametrize("origin", [None, "source_rule", "agent_approved", "user", 1])
def test_callers_cannot_override_proposal_origin(tmp_path, origin):
    archive, entry, attempt, page = fixture(tmp_path)
    ledger = MemoryLedger(archive)
    proposal = {"type": "observation", "text": "I want source citations kept.",
                "quote": "I want source citations kept.", "start": 0}
    with pytest.raises(TypeError):
        ledger.record(entry, 0, attempt, page, proposal, model_identity=MODEL, proposal_origin=origin)
    with pytest.raises(TypeError):
        ledger.record_batch(entry, 0, attempt, [{"page": page, "proposal": proposal}],
                            model_identity=MODEL, proposal_origin=origin)
    assert ledger.verify_all()["events"] == 0


def test_agent_search_walks_chain_once_keeps_review_and_provisional_state(tmp_path, monkeypatch):
    archive, entry, attempt, page = fixture(tmp_path, text="Keep citations. Drop citations.")
    ledger = MemoryLedger(archive)
    items = [{"page": page, "proposal": {"type": "preference", "text": quote,
              "quote": quote, "start": start}} for quote, start in [("Keep citations.",0),("Drop citations.",16)]]
    refs = ledger.record_batch(entry, 0, attempt, items, model_identity=MODEL)
    ledger.mark_needs_user(refs[0], reason="possible_contradiction")
    walks=[]
    original=ledger._walk
    def walk(db):
        walks.append(True)
        yield from original(db)
    monkeypatch.setattr(ledger, "_walk", walk)
    result=ledger.search("citations",limit=2)
    assert len(walks)==1 and {m["id"] for m in result["matches"]}==set(refs)
    assert {m["state"] for m in result["matches"]}=={"provisional","needs_user"}
    assert all(m["proposal_origin"]=="model" and m["truth_status"]!="verified" for m in result["matches"])
    assert "synthetic-project" not in json.dumps(result)


def test_agent_source_follow_is_exact_version_with_safe_context_and_redacted_projection_grant(tmp_path):
    import base64
    from muninn.history.blind_index import SecureHistoryBlindIndex
    archive, entry, attempt, page=fixture(tmp_path)
    ledger=MemoryLedger(archive)
    ident=record(ledger,entry,attempt,page)
    source=ledger.source(ident,max_chars=100)
    assert source["context"]=="I want source citations kept."
    assert source["citation"]["quote_start"]==0
    assert source["citation"]["quote_length"]==len(source["context"])
    assert source["citation"]["unit"]>=0 and source["citation"]["version"]==0
    assert source["provider"]=="codex" and source["context_state"]=="available"
    actual,version,_=SecureHistoryBlindIndex(archive)._entry_for_capability(source["transcript_capability"])
    assert actual==entry and version==0
    cap=source["transcript_capability"]
    decoded=json.loads(base64.urlsafe_b64decode(cap+'='*(-len(cap)%4))[:-32])
    assert decoded["term"]=="transcript"  # the bearer must not contain the source quote
    assert "synthetic-project" not in json.dumps(source)


def test_agent_search_and_source_never_release_or_match_credential_values(tmp_path):
    text="Keep citations. SERVICE_API_KEY=synthetic$hiddenvalue"
    archive,entry,attempt,page=fixture(tmp_path,text=text)
    ledger=MemoryLedger(archive)
    ident=record(ledger,entry,attempt,page,text="Keep citations.")
    assert ledger.search("hiddenvalue")["matches"]==[]
    source=ledger.source(ident)
    assert source["context_state"]=="withheld" and "context" not in source
    assert "hiddenvalue" not in json.dumps(source)
    assert source["transcript_capability"]


def test_agent_search_authenticates_late_tail_even_after_limit(tmp_path):
    archive,entry,attempt,page=fixture(tmp_path,text="Keep citations. Drop citations.")
    ledger=MemoryLedger(archive)
    record(ledger,entry,attempt,page,text="Keep citations.")
    record(ledger,entry,attempt,page,text="Drop citations.",start=16)
    with ledger._connect() as db:
        db.execute("UPDATE events SET ciphertext=zeroblob(length(ciphertext)) WHERE seq=2")
    with pytest.raises(MemoryLedgerIntegrityError): ledger.search("citations",limit=1)


def test_agent_source_rechecks_whole_unit_privacy_not_only_stored_screen_flag(tmp_path, monkeypatch):
    archive,entry,attempt,page=fixture(tmp_path)
    ledger=MemoryLedger(archive)
    ident=record(ledger,entry,attempt,page)
    monkeypatch.setattr(ledger,"_unit_info",lambda *a,**kw:(False,0,b""))
    assert "text" not in ledger.get(ident)
    assert ledger.search("citations")["matches"]==[]
    assert ledger.source(ident)["context_state"]=="withheld"


@pytest.mark.parametrize("query,limit",[("",10),("x"*513,10),("one two three four five six seven eight nine",10),("citations",0),("citations",21),("citations",True)])
def test_agent_search_bounds_inputs(tmp_path,query,limit):
    archive,entry,attempt,page=fixture(tmp_path)
    ledger=MemoryLedger(archive)
    with pytest.raises(ValueError): ledger.search(query,limit=limit)


@pytest.mark.parametrize("query",["SERVICE_API_KEY=synthetic$secret",r"C:\Users\user\secret.env"])
def test_agent_search_rejects_sensitive_queries_without_echoing(tmp_path,query):
    archive,entry,attempt,page=fixture(tmp_path)
    ledger=MemoryLedger(archive)
    with pytest.raises(ValueError) as error: ledger.search(query)
    assert query not in str(error.value)


def test_agent_read_does_not_reuse_whole_unit_screen_after_late_fragment_tamper(tmp_path):
    archive,entry,attempt,page=fixture(tmp_path,text="Keep citations. "+"ordinary words "*1000)
    ledger=MemoryLedger(archive)
    ident=record(ledger,entry,attempt,page,text="Keep citations.")
    assert ledger.get(ident)["text"]=="Keep citations."
    with ledger.units._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=? AND ordinal=?",
                   (attempt,page+1))
    with pytest.raises(MemoryLedgerIntegrityError): ledger.get(ident)
