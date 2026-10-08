"""Paid publication recovery must not depend on the original runtime directory."""
import os
import sqlite3

import pytest

from muninn.history.historical_batch_worker import HistoricalBatchWorker
from muninn.history.secure_archive import SecureHistoryArchive
from tests.test_historical_batch_jobs import fixture
from tests.test_historical_batch_worker import ready, responses


async def paid_history(tmp_path):
    journal, archive, outbox, ident, bindings = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    submitted, terminal = responses(archive, outbox, ident)
    clock = [0]
    async def send(method, **kwargs):
        return submitted if method == "POST" else terminal
    worker = HistoricalBatchWorker(journal, authorize_submit=lambda _: True,
                                   send=send, provider_status=ready, clock=lambda: clock[0])
    await worker.step()
    clock[0] = 61
    await worker.step()
    assert worker.status["state"] == "passed"
    return journal, archive, outbox, ident, bindings


@pytest.mark.asyncio
async def test_paid_archive_backup_and_restore_under_a_fresh_parent(tmp_path, recovery_copy):
    from muninn.history.batch_activation import read_batch_policy
    from muninn.history.remote_accounting import AdmissionError, reserve, status
    from muninn.history.remote_policy import read_policy, write_policy
    from tests.test_historical_batch_jobs import admission

    journal, archive, _outbox, _ident, bindings = await paid_history(tmp_path)
    held = admission(journal)
    source_policy = read_policy(journal.policy_root, lambda: (False, 1, 30, False))
    other_machine = tmp_path / "fresh-account"
    other_machine.mkdir()
    backup = other_machine / "backup"
    announced = []
    options = {"on_staging": announced.append} if recovery_copy.backend == "windows_unattended" else {}
    report = recovery_copy(archive, backup, "test-only portable passphrase", **options)
    if recovery_copy.backend == "windows_unattended":
        assert announced and announced[0].parent == other_machine
    assert report["runtime_bundle"] == 1
    assert report["publication_receipts_verified"] == 2
    restored = SecureHistoryArchive.restore_from_backup(
        backup, other_machine / "restored", "test-only portable passphrase")
    assert restored.root == other_machine / "restored" / "history_secure_archive"
    from muninn.history.capture_journal import CaptureJournal
    recovered = CaptureJournal(restored, recover=False)
    assert recovered.verify_publications(job_ids=[job for job, _ in bindings]) == 2
    policy = read_policy(recovered.policy_root, lambda: (True, 10, 100, False))
    assert not policy.enabled and not read_batch_policy(recovered.policy_root)["enabled"]
    assert read_policy(journal.policy_root, lambda: (False, 1, 30, False)) == source_policy
    assert status(recovered.policy_root)["daily_cost_usd"] == 0.001
    assert status(recovered.policy_root)["unresolved"] == 1
    with recovered._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE state='succeeded'").fetchone()[0] == 2
    with sqlite3.connect(recovered.policy_root / "remote_policy" / "policy.sqlite3") as db:
        assert db.execute("SELECT state,generation FROM remote_admissions WHERE id=?",
                          (held.identifier,)).fetchone() == ("unknown", 1)
    # Explicit reconsent cannot erase an unknown hold or reset paid cost floors.
    enabled = write_policy(recovered.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                           override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    with pytest.raises(AdmissionError, match="remote_admission_busy"):
        reserve(recovered.policy_root, enabled.generation, ready())


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["missing_cipher", "tampered_cipher", "missing_marker"])
async def test_runtime_restore_requires_authenticated_accounting(tmp_path, defect, recovery_copy):
    from muninn.history.credential_crypto import VaultIntegrityError
    from muninn.history.portable_accounting import MARKER, SNAPSHOT
    from muninn.history.private_acl import VaultPermissionError

    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    backup = tmp_path / "backup"
    recovery_copy(archive, backup, "test-only portable passphrase")
    snapshot = backup / "history_secure_archive" / SNAPSHOT
    if defect == "missing_cipher":
        snapshot.rename(snapshot.with_suffix(".retained"))
    elif defect == "missing_marker":
        marker = backup / MARKER
        marker.rename(marker.with_suffix(".retained"))
    else:
        sealed = bytearray(snapshot.read_bytes())
        sealed[-1] ^= 1
        snapshot.write_bytes(sealed)
    destination = tmp_path / "restored"
    with pytest.raises((VaultIntegrityError, VaultPermissionError)):
        SecureHistoryArchive.restore_from_backup(backup, destination, "test-only portable passphrase")
    assert not destination.exists()
    # The real/source policy and every batch remain untouched.
    assert (archive.root / "historical-batches.db").exists()


@pytest.mark.asyncio
async def test_plaintext_book_is_not_the_restore_authority(tmp_path, recovery_copy):
    from muninn.history.remote_accounting import status

    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    backup = tmp_path / "backup"
    recovery_copy(archive, backup, "test-only portable passphrase")
    with sqlite3.connect(backup / "remote_policy" / "policy.sqlite3") as db:
        db.execute("UPDATE remote_admissions SET cost_micro=99999999")
        db.execute("UPDATE policy SET enabled=1")
    restored = SecureHistoryArchive.restore_from_backup(backup, tmp_path / "restored",
                                                       "test-only portable passphrase")
    recovered_root = restored.root.parent
    assert status(recovered_root)["daily_cost_usd"] == 0.001
    from muninn.history.remote_policy import read_policy
    assert not read_policy(recovered_root, lambda: (True, 1, 30, False)).enabled


@pytest.mark.asyncio
async def test_restored_runtime_can_be_backed_up_again(tmp_path, recovery_copy):
    _journal, archive, _outbox, _ident, bindings = await paid_history(tmp_path)
    recovery_copy(archive, tmp_path / "first-backup", "test-only portable passphrase")
    restored = SecureHistoryArchive.restore_from_backup(
        tmp_path / "first-backup", tmp_path / "first-restore", "test-only portable passphrase")
    recovery_copy(restored, tmp_path / "second-backup", "test-only portable passphrase")
    again = SecureHistoryArchive.restore_from_backup(
        tmp_path / "second-backup", tmp_path / "second-restore", "test-only portable passphrase")
    from muninn.history.capture_journal import CaptureJournal
    assert CaptureJournal(again, recover=False).verify_publications(job_ids=[j for j, _ in bindings]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["sent", "passed"])
@pytest.mark.parametrize("defect", ["missing", "wrong_owner", "wrong_generation"])
async def test_batch_verification_requires_exact_paid_admission(tmp_path, phase, defect):
    from muninn.history.credential_crypto import VaultIntegrityError
    from tests.test_historical_batch_jobs import admission
    if phase == "passed":
        journal, _archive, _outbox, ident, _bindings = await paid_history(tmp_path)
    else:
        journal, _archive, outbox, ident, _bindings = fixture(tmp_path)
        journal.reserve_historical_batch(ident)
        paid = admission(journal, ident)
        outbox.begin_submission(ident, 0)
        journal.mark_historical_batch_dispatched(ident, paid.identifier)
    with sqlite3.connect(journal.policy_root / "remote_policy" / "policy.sqlite3") as db:
        if defect == "missing":
            # Damage only an isolated fixture. Live batches and ledgers are untouched.
            db.execute("DELETE FROM remote_admissions WHERE batch_owner=?", (ident,))
        elif defect == "wrong_owner":
            db.execute("UPDATE remote_admissions SET batch_owner=? WHERE batch_owner=?", ("f" * 32, ident))
        else:
            db.execute("UPDATE remote_admissions SET generation=generation+1 WHERE batch_owner=?", (ident,))
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Actual unattended backup requires Windows user protection")
async def test_corrupt_emitted_snapshot_is_not_published(tmp_path, monkeypatch):
    from muninn.history import portable_accounting
    from muninn.history.credential_crypto import VaultIntegrityError
    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    original = portable_accounting.snapshot_into
    def corrupt(copied, source, destination):
        original(copied, source, destination)
        path = copied.root / portable_accounting.SNAPSHOT
        value = bytearray(path.read_bytes())
        value[-1] ^= 1
        path.write_bytes(value)
    monkeypatch.setattr(portable_accounting, "snapshot_into", corrupt)
    stages = []
    with pytest.raises(VaultIntegrityError):
        archive.backup_to(tmp_path / "backup", on_staging=stages.append)
    assert not (tmp_path / "backup").exists()
    assert len(stages) == 1 and stages[0].is_dir()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Actual unattended backup requires Windows user protection")
async def test_unsupported_accounting_runtime_fails_before_copy(tmp_path, monkeypatch):
    from muninn.history import portable_accounting
    from muninn.history.credential_crypto import VaultIntegrityError
    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    archive.backup_to(tmp_path / "backup")
    calls = []
    def unavailable():
        raise VaultIntegrityError("SQLite serialization unavailable")
    monkeypatch.setattr(portable_accounting, "require_snapshot_support", unavailable)
    monkeypatch.setattr(SecureHistoryArchive, "_copy_archive_files", lambda *a, **kw: calls.append(kw))
    with pytest.raises(VaultIntegrityError, match="serialization unavailable"):
        archive.backup_to(tmp_path / "unsupported-backup")
    with pytest.raises(VaultIntegrityError, match="serialization unavailable"):
        SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "unsupported-restore",
                                                "test-only portable passphrase")
    assert not calls
    assert not list(tmp_path.glob(".unsupported-*.incomplete-*"))


def test_legacy_archive_does_not_require_sqlite_serialization(tmp_path, monkeypatch, recovery_copy):
    from muninn.history import portable_accounting
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable passphrase")
    def unavailable():
        raise AssertionError("Legacy path must not require serialization")
    monkeypatch.setattr(portable_accounting, "require_snapshot_support", unavailable)
    recovery_copy(archive, tmp_path / "backup", "test-only portable passphrase")
    restored = SecureHistoryArchive.restore_from_backup(tmp_path / "backup", tmp_path / "restore",
                                                       "test-only portable passphrase")
    assert restored.vault_id == archive.vault_id


@pytest.mark.asyncio
async def test_portable_runtime_restore_requires_serialization_before_any_copy(tmp_path, monkeypatch,
                                                                            recovery_copy):
    from muninn.history import portable_accounting
    from muninn.history.credential_crypto import VaultIntegrityError

    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    recovery_copy(archive, tmp_path / "backup", "test-only portable passphrase")

    def unavailable():
        raise VaultIntegrityError("SQLite serialization unavailable")

    monkeypatch.setattr(portable_accounting, "require_snapshot_support", unavailable)
    monkeypatch.setattr(SecureHistoryArchive, "_copy_archive_files",
                        lambda *a, **kw: pytest.fail("Unsupported runtime copied archive files"))
    destination = tmp_path / "unsupported-restore"
    with pytest.raises(VaultIntegrityError, match="serialization unavailable"):
        SecureHistoryArchive.restore_from_backup(tmp_path / "backup", destination,
                                                "test-only portable passphrase")
    assert not destination.exists()
    assert not list(tmp_path.glob(".unsupported-restore.incomplete-*"))
