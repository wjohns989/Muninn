"""Small isolated Restic proof; no live runtime/provider/vault configuration."""
import hashlib
import os
from pathlib import Path

import pytest

from muninn.history.incremental_backup import IncrementalBackup
from muninn.history.private_acl import create_private_directory, create_private_file
from muninn.history.secure_archive import SecureHistoryArchive


def pinned_binary():
    binary = Path(os.environ.get("MUNINN_RESTIC_TEST_BINARY", ""))
    if not binary.is_file():
        pytest.skip("Explicit pinned test binary not supplied")
    return binary


@pytest.mark.skipif(os.name != "nt", reason="Pinned Windows binary")
def test_isolated_incremental_restore_and_portable_archive_key(tmp_path):
    binary = pinned_binary()
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic portable recovery phrase")
    bundle = tmp_path / "closed-bundle"
    create_private_directory(bundle)
    source = bundle / "synthetic.bin"
    create_private_file(source)
    source.write_bytes(os.urandom(128000))
    wanted = hashlib.sha256(source.read_bytes()).hexdigest()
    repo = IncrementalBackup(tmp_path / "incremental", binary, archive=archive)
    first = repo._backup_bundle(bundle)
    second = repo._backup_bundle(bundle)
    assert first["snapshot_id"] != second["snapshot_id"]
    assert second["data_added"] == 0
    assert repo.check()["repository_data_checked"]
    portable = IncrementalBackup(repo.root, binary, passphrase="synthetic portable recovery phrase")
    restored = tmp_path / "restored"
    assert portable.restore(first["snapshot_id"], restored)["restored_and_verified"]
    matches = list(restored.rglob("synthetic.bin"))
    assert len(matches) == 1 and hashlib.sha256(matches[0].read_bytes()).hexdigest() == wanted


@pytest.mark.skipif(os.name != "nt", reason="Pinned Windows binary")
@pytest.mark.asyncio
async def test_paid_bundle_roundtrip_preserves_batch_bytes_and_disabled_permissions(tmp_path):
    from tests.test_paid_history_recovery import paid_history
    from muninn.history.capture_journal import CaptureJournal
    from muninn.history.remote_policy import read_policy
    from muninn.history.batch_activation import read_batch_policy
    journal, archive, _outbox, _ident, bindings = await paid_history(tmp_path)
    bundle = tmp_path / "closed-bundle"
    repo = IncrementalBackup(tmp_path / "incremental", pinned_binary(), archive=archive)
    result = repo.backup_archive(archive, bundle)
    assert result["application_backup"]["runtime_bundle"] == 1
    original = bundle / "history_secure_archive" / "historical-batches.db"
    wanted = hashlib.sha256(original.read_bytes()).hexdigest()
    receipt = result["incremental_snapshot"]
    assert repo.check()["repository_data_checked"]
    extracted = tmp_path / "extracted"
    repo.restore(receipt["snapshot_id"], extracted)
    assert (extracted / "history_secure_archive" / "header.json").is_file()
    assert hashlib.sha256((extracted / "history_secure_archive" / "historical-batches.db").read_bytes()).hexdigest() == wanted
    restored = SecureHistoryArchive.restore_from_backup(
        extracted, tmp_path / "cold-restore", "test-only portable passphrase")
    recovered = CaptureJournal(restored, recover=False)
    assert recovered.verify_publications(job_ids=[job for job, _ in bindings]) == 2
    assert not read_policy(restored.root.parent, lambda: (True, 5, 50, False)).enabled
    assert not read_batch_policy(restored.root.parent)["enabled"]
    assert original.is_file() and (archive.root / "historical-batches.db").is_file()


def test_runner_excludes_inherited_credentials_and_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from muninn.history.credential_crypto import VaultIntegrityError
    import subprocess
    import io
    instance = IncrementalBackup.__new__(IncrementalBackup)
    instance.binary = tmp_path / "restic.exe"
    instance.store = tmp_path / "repository"
    instance._password = "synthetic-pipe-only-secret"
    monkeypatch.setenv("RESTIC_REPOSITORY", "s3:synthetic-unsafe-bucket")
    monkeypatch.setenv("RESTIC_PASSWORD", "synthetic-inherited-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-provider-secret")
    captured = []
    written = []
    class Incomplete:
        def __init__(self, command, **kwargs):
            captured.append((command, kwargs))
            self.stdin = SimpleNamespace(write=written.append, close=lambda: None)
            self.stdout = io.BytesIO()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self):
            return 3
    monkeypatch.setattr(subprocess, "Popen", Incomplete)
    with pytest.raises(VaultIntegrityError, match="exit 3"):
        instance._run("backup", ".")
    command, options = captured[0]
    assert instance._password not in " ".join(command)
    assert written == [(instance._password + "\n").encode()]
    assert options["stderr"] == subprocess.DEVNULL
    assert options["shell"] is False
    assert not any(name in options["env"] for name in (
        "RESTIC_REPOSITORY", "RESTIC_PASSWORD", "OPENROUTER_API_KEY"))


def test_enrollment_refuses_incomplete_overlapping_and_existing_restore_paths(tmp_path):
    from muninn.history.credential_crypto import VaultIntegrityError
    instance = IncrementalBackup.__new__(IncrementalBackup)
    instance.root = tmp_path / "repository"
    create_private_directory(instance.root)
    with pytest.raises(VaultIntegrityError, match="overlaps"):
        instance._backup_bundle(instance.root)
    incomplete = tmp_path / ".bundle.incomplete-synthetic"
    create_private_directory(incomplete)
    with pytest.raises(VaultIntegrityError, match="closed"):
        instance._backup_bundle(incomplete)
    with pytest.raises(FileExistsError):
        instance.restore("a" * 64, incomplete)


def test_public_enrollment_requires_authenticated_archive_not_directory(tmp_path):
    from muninn.history.credential_crypto import VaultIntegrityError
    instance = IncrementalBackup.__new__(IncrementalBackup)
    with pytest.raises(VaultIntegrityError, match="Authenticated source"):
        instance.backup_archive(tmp_path, tmp_path / "not-a-verified-bundle")


def test_public_enrollment_requires_same_key_not_only_vault_id(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from muninn.history.credential_crypto import VaultIntegrityError
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic portable recovery phrase")
    instance = IncrementalBackup.__new__(IncrementalBackup)
    instance.anchor = SimpleNamespace(vault_id=archive.vault_id, _key=b"x" * 32)
    instance.root = tmp_path / "repository"
    called = []
    monkeypatch.setattr(archive, "backup_to", lambda path: called.append(path) or {})
    monkeypatch.setattr(instance, "_backup_bundle", lambda path: {})
    with pytest.raises(VaultIntegrityError, match="Authenticated source"):
        instance.backup_archive(archive, tmp_path / "destination")
    assert called == [] and not (tmp_path / "destination").exists()


def test_local_only_path_checks_happen_before_filesystem_access(tmp_path, monkeypatch):
    from muninn.history import incremental_backup as module
    from muninn.history.credential_crypto import VaultIntegrityError
    seen = []
    monkeypatch.setattr(module, "unlinked", lambda path: seen.append(path) or Path(path))
    with pytest.raises(VaultIntegrityError, match="local disk"):
        module.local_path(r"\\synthetic-host\share\backup")
    assert seen == []
    monkeypatch.setattr(module, "_drive_type", lambda path: 4)
    with pytest.raises(VaultIntegrityError, match="local disk"):
        module.local_path(tmp_path / "network-mapped")
    assert seen == []
    monkeypatch.setattr(module, "_drive_type", lambda path: 3)
    assert module.local_path(tmp_path / "local") == tmp_path / "local"


def test_runner_drains_large_progress_with_bounded_reads_and_summary_retention(tmp_path, monkeypatch):
    import io
    import json
    import subprocess
    from types import SimpleNamespace
    instance = IncrementalBackup.__new__(IncrementalBackup)
    instance.binary, instance.store = tmp_path / "binary", tmp_path / "repository"
    instance._password = "synthetic-pipe-only-secret"
    summary = {"message_type": "summary", "snapshot_id": "a" * 64}
    class Output(io.BytesIO):
        def readline(self, size=-1):
            assert size == 65536
            return super().readline(size)
    class Completed:
        def __init__(self, *args, **kwargs):
            self.stdin = SimpleNamespace(write=lambda value: None, close=lambda: None)
            self.stdout = Output(b"x" * 200000 + b"\n" +
                b'{"message_type":"status","bytes_done":1}\n' * 10000 +
                json.dumps(summary).encode() + b"\n")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", Completed)
    assert instance._run("backup", ".") == [summary]
