"""Portable encrypted recovery inputs for isolated tests, not unattended backup.

The portable branch snapshots real encrypted stores and managed accounting,
then the caller exercises real restore/verification. It never mocks DPAPI or
claims that production backup_to supports Linux.
"""
from pathlib import Path

from muninn.history.capture_journal import CaptureJournal
from muninn.history.private_acl import _is_link, create_private_directory
from muninn.history.secure_archive import SecureHistoryArchive


def recovery_input(archive, destination, passphrase, *, temporary_root,
                   policy_root=None):
    boundary = Path(temporary_root).resolve(strict=True)
    destination = Path(destination)
    source = archive.root.resolve(strict=True)
    policy_root = Path(policy_root) if policy_root is not None else source.parent
    policy = policy_root.resolve(strict=True)
    resolved_destination = destination.resolve()
    if (source == boundary or not source.is_relative_to(boundary)
            or not policy.is_relative_to(boundary)
            or resolved_destination == boundary
            or not resolved_destination.is_relative_to(boundary)):
        raise ValueError("Recovery fixture must remain inside its temporary root")
    if destination.exists() or _is_link(destination):
        raise ValueError("Recovery fixture destination must be new")
    for target in (archive.root, policy_root, destination.parent):
        if _is_link(target):
            raise ValueError("Recovery fixture paths must not be linked")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Recovery fixture must not be inside its source archive")
    # Authenticate the explicit synthetic phrase before creating any copy.
    unlocked = SecureHistoryArchive(source, passphrase)
    assert unlocked.vault_id == archive.vault_id and unlocked._key == archive._key

    from muninn.history.portable_accounting import has_accounting, snapshot_into, verify_snapshot

    runtime = has_accounting(policy_root)
    if runtime:
        create_private_directory(destination)
    copied_root = destination / "history_secure_archive" if runtime else destination
    SecureHistoryArchive._copy_archive_files(source, copied_root,
                                            copy_journal=False, copy_accounting=False)
    CaptureJournal(archive, recover=False, policy_root=policy_root).backup_to(
        copied_root / "capture-jobs.db")
    copied = SecureHistoryArchive(copied_root, passphrase)
    if runtime:
        snapshot_into(copied, policy_root, destination)
        verify_snapshot(copied, destination)
    report = copied.verify_all()
    report["runtime_bundle"] = int(runtime)
    journal = CaptureJournal(copied, recover=False)
    journal.verify_all()
    report["publication_receipts_verified"] = journal.verify_publications()
    journal.verify_classifications()
    if (copied_root / "historical-batches-managed").exists():
        from muninn.history.historical_batch import BatchOutbox

        report["historical_batches_verified"] = BatchOutbox(copied).verify_all()["batches"]
    return report
