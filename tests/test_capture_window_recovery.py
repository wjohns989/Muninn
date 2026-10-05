"""Selective recovery uses isolated encrypted stores and never calls providers."""
import hashlib
import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from tests.test_capture_window_jobs import window_fixture


def failed_window(tmp_path, *, generation=-1, code="local_output_quote"):
    journal, archive, receipt = window_fixture(tmp_path)
    journal.queue_capture_windows(receipt, limit=1, remote_policy_generation=generation)
    job = journal.claim_analysis(include_capture=True, include_search=False)
    plans, entry = journal._capture_plan_source(receipt)
    window = plans.window_at(entry, receipt["version"], job.target["plan_attempt"], 0)
    assert journal.bind_analysis_window(job.job_id, job.lease_token, window)
    assert journal.fail_analysis(job.job_id, job.lease_token, code)
    return journal, archive, receipt, job, window


def test_selective_recovery_rebinds_window_without_replanning_or_fake_ack(tmp_path):
    journal, archive, receipt, job, window = failed_window(tmp_path)
    with journal._connect() as db:
        before = dict(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
        source_before = tuple(db.execute("SELECT * FROM capture_enrichment_sources").fetchone())
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt,
                                        remote_policy_generation=2) == "queued"
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt,
                                        remote_policy_generation=2) == "ineligible"
    with journal._connect() as db:
        after = db.execute("SELECT * FROM history_analysis_jobs").fetchone()
        assert tuple(db.execute("SELECT * FROM capture_enrichment_sources").fetchone()) == source_before
        target = journal._validated_analysis_target(after, db)
        assert {k: v for k, v in target.items() if k != "remote_policy_generation"} == job.target
        assert target["remote_policy_generation"] == 2
        assert journal._read_analysis_window(after) == window
        for key in ("job_id", "attempt", "created_at", "remote_dispatched"):
            assert after[key] == before[key]
        assert after["state"] == "pending" and after["error_code"] == ""
        assert after["dedup_key"] != before["dedup_key"]
    assert journal.capture_window_status(receipt)["acknowledged"] == 0
    reopened = CaptureJournal(archive, recover=False)
    claimed = reopened.claim_analysis(include_capture=True, include_search=False, capture_remote_only=True)
    assert claimed.job_id == job.job_id and claimed.attempt == job.attempt + 1
    assert claimed.window == window and claimed.remote_policy_generation == 2
    assert journal.verify_all() == 0


@pytest.mark.parametrize("code", ["model_unavailable", "local_output_invalid", "local_output_json",
                                  "local_output_cited_schema", "local_output_citation",
                                  "local_output_analysis_schema", "local_output_quote"])
def test_recoverable_codes(tmp_path, code):
    journal, _, _, job, _ = failed_window(tmp_path, code=code)
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt,
                                        remote_policy_generation=2) == "queued"


@pytest.mark.parametrize("field,value", [
    ("state", "outcome_unknown"), ("state", "succeeded"), ("state", "reused"),
    ("state", "retry"), ("state", "publication_pending"), ("state", "running"),
    ("remote_dispatched", 1), ("cancel_requested", 1), ("publication_started", 1),
    ("sealed_extraction", b"must preserve"), ("sealed_receipt", b"must preserve"),
    ("sealed_reuse", b"must preserve"), ("sealed_result", b"must preserve"),
    ("extraction_id", "f" * 64), ("lease_token", "f" * 32), ("lease_until", 1.0),
    ("error_code", "insufficient_context"), ("error_code", "vault_integrity"),
    ("error_code", "snapshot_unavailable"), ("error_code", "unknown"),
])
def test_ineligible_rows_are_byte_for_byte_unchanged(tmp_path, field, value):
    journal, _, _, job, _ = failed_window(tmp_path, generation=1)
    with journal._connect() as db:
        db.execute(f"UPDATE history_analysis_jobs SET {field}=?", (value,))
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt,
                                        remote_policy_generation=2) == "ineligible"
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before


@pytest.mark.parametrize("damage", ["target", "binding", "window"])
def test_authentication_failure_rolls_back_recovery(tmp_path, damage):
    journal, _, _, job, _ = failed_window(tmp_path)
    with journal._connect() as db:
        if damage == "binding":
            db.execute("UPDATE capture_enrichment_windows SET sealed_binding=zeroblob(30)")
        else:
            db.execute(f"UPDATE history_analysis_jobs SET sealed_{damage}=zeroblob(30)")
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
    with pytest.raises(VaultIntegrityError):
        journal.retry_capture_window(job.job_id, expected_attempt=job.attempt, remote_policy_generation=2)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before


def test_stale_attempt_or_full_queue_cannot_recover(tmp_path, monkeypatch):
    journal, _, _, job, _ = failed_window(tmp_path)
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt - 1,
                                        remote_policy_generation=2) == "ineligible"
    monkeypatch.setattr(journal, "_capture_window_capacity", lambda db: 0)
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt,
                                        remote_policy_generation=2) == "queue_full"
    assert journal.get_analysis_job(job.job_id)["state"] == "failed"


def test_recovery_rejects_consistently_sealed_wrong_plan_descriptor(tmp_path):
    journal, _, _, job, _ = failed_window(tmp_path)
    # A self-consistent seal is not proof that this is the admitted plan ordinal.
    target = {**job.target, "descriptor_sha256": "f" * 64}
    with journal._connect() as db:
        db.execute("UPDATE history_analysis_jobs SET sealed_target=?,sealed_window=NULL", (
            journal._seal_search(target, job.job_id, "analysis-target"),))
        db.execute("UPDATE capture_enrichment_windows SET sealed_binding=?", (
            journal._seal_search(target, job.job_id, "capture-window-binding-v1"),))
    with pytest.raises(VaultIntegrityError):
        journal.retry_capture_window(job.job_id, expected_attempt=job.attempt, remote_policy_generation=2)
    assert journal.get_analysis_job(job.job_id)["state"] == "failed"


def test_recovery_rolls_back_all_resealed_bindings_on_interruption(tmp_path, monkeypatch):
    journal, _, _, job, _ = failed_window(tmp_path)
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone())
        binding = tuple(db.execute("SELECT * FROM capture_enrichment_windows").fetchone())
    original = journal._seal_search

    def interrupted(value, ident, purpose):
        if purpose == "capture-window-binding-v1":
            raise RuntimeError("isolated interrupted recovery")
        return original(value, ident, purpose)

    monkeypatch.setattr(journal, "_seal_search", interrupted)
    with pytest.raises(RuntimeError):
        journal.retry_capture_window(job.job_id, expected_attempt=job.attempt, remote_policy_generation=2)
    with journal._connect() as db:
        assert tuple(db.execute("SELECT * FROM history_analysis_jobs").fetchone()) == before
        assert tuple(db.execute("SELECT * FROM capture_enrichment_windows").fetchone()) == binding


@pytest.mark.parametrize("attempt,generation", [(True, 2), (-1, 2), (1, True), (1, -1), (1, 0)])
def test_invalid_recovery_inputs(tmp_path, attempt, generation):
    journal, _, _, job, _ = failed_window(tmp_path)
    with pytest.raises(ValueError):
        journal.retry_capture_window(job.job_id, expected_attempt=attempt,
                                     remote_policy_generation=generation)


def test_operator_preview_is_read_only_and_apply_requires_current_consent(tmp_path, capsys, monkeypatch):
    from scripts.recover_capture_windows import main
    from muninn.history.remote_policy import write_policy
    journal, archive, _, job, _ = failed_window(tmp_path)
    monkeypatch.setattr("scripts.recover_capture_windows.getpass.getpass",
                        lambda prompt: "test-only portable passphrase")
    before = journal.path.read_bytes()
    assert main(["--archive-root", str(archive.root), "--prompt-passphrase"]) == 0
    assert journal.path.read_bytes() == before
    assert '"eligible_windows": 1' in capsys.readouterr().out
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    args = ["--archive-root", str(archive.root), "--apply", "--expected-generation", "2",
            "--backup-before", str(tmp_path / "preimage"), "--prompt-passphrase"]
    assert main(args) == 1
    assert journal.get_analysis_job(job.job_id)["state"] == "failed"
    assert not (tmp_path / "preimage").exists()
    args[args.index("2")] = "1"
    assert main(args) == 0
    with sqlite_connection(tmp_path / "preimage" / "capture-jobs.db") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT state FROM history_analysis_jobs").fetchone()[0] == "failed"
    assert journal.get_analysis_job(job.job_id)["state"] == "pending"
    assert '"model_calls": 0' in capsys.readouterr().out


def sqlite_connection(path):
    import sqlite3
    return sqlite3.connect(path)


def test_preview_target_change_blocks_same_attempt_recovery(tmp_path):
    journal, _, _, job, _ = failed_window(tmp_path)
    with journal._connect() as db:
        before = db.execute("SELECT sealed_target FROM history_analysis_jobs").fetchone()[0]
        db.execute("UPDATE history_analysis_jobs SET sealed_target=?", (
            journal._seal_search(job.target, job.job_id, "analysis-target"),))
    assert journal.retry_capture_window(job.job_id, expected_attempt=job.attempt,
        remote_policy_generation=2, expected_target_sha256=hashlib.sha256(before).hexdigest()) == "ineligible"
    assert journal.get_analysis_job(job.job_id)["state"] == "failed"


def test_preview_missing_store_creates_nothing(tmp_path):
    from scripts.recover_capture_windows import main
    missing = tmp_path / "nonexistent"
    assert main(["--archive-root", str(missing)]) == 1
    assert not missing.exists()


def test_post_backup_selected_row_change_aborts_before_any_recovery(tmp_path, monkeypatch):
    from scripts import recover_capture_windows as operator
    from muninn.history.remote_policy import write_policy
    journal, archive, _, job, _ = failed_window(tmp_path)
    monkeypatch.setattr(operator.getpass, "getpass", lambda prompt: "test-only portable passphrase")
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    backup = operator.backup_journal

    def raced(reader, destination):
        backup(reader, destination)
        with journal._connect() as db:
            db.execute("UPDATE history_analysis_jobs SET updated_at=updated_at+1")

    monkeypatch.setattr(operator, "backup_journal", raced)
    assert operator.main(["--archive-root", str(archive.root), "--prompt-passphrase", "--apply",
        "--expected-generation", "1", "--backup-before", str(tmp_path / "preimage")]) == 1
    assert journal.get_analysis_job(job.job_id)["state"] == "failed"
