import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive

PASSPHRASE = "correct horse battery archive recovery"


def test_portable_encrypted_snapshot_and_versions(tmp_path, monkeypatch):
    source = tmp_path / "conversation-secret-name.jsonl"
    source.write_bytes(b"my token is CANARY-VERY-PRIVATE-123\n")
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    assert archive.archive_file(source, "codex") == {"status": "captured", "versions": 1, "size": source.stat().st_size}
    assert archive.archive_file(source, "codex")["status"] == "unchanged"
    # Some writers preserve timestamps while rewriting a file in place.
    original_stat = source.stat()
    source.write_bytes(b"my token is CANARY-VERY-PRIVATE-999\n")
    __import__("os").utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert archive.archive_file(source, "codex")["versions"] == 2
    source.write_bytes(b"replacement CANARY-SECOND\n")
    assert archive.archive_file(source, "codex")["versions"] == 3
    assert archive.read_file(source, 0) == b"my token is CANARY-VERY-PRIVATE-123\n"
    assert archive.read_file(source, 1) == b"my token is CANARY-VERY-PRIVATE-999\n"
    assert archive.read_file(source) == b"replacement CANARY-SECOND\n"
    assert SecureHistoryArchive(archive.root, PASSPHRASE).read_file(source, 0) == archive.read_file(source, 0)
    with pytest.raises(VaultIntegrityError):
        SecureHistoryArchive(archive.root, "wrong passphrase with enough length")
    for path in archive.root.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            assert b"CANARY-VERY-PRIVATE-123" not in raw
            assert b"conversation-secret-name" not in raw


def test_blob_and_manifest_tampering_fail_closed(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"private payload\n")
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    archive.archive_file(source, "codex")
    blob = next((archive.root / "blobs").glob("*.enc"))
    raw = bytearray(blob.read_bytes())
    raw[-1] ^= 1
    blob.write_bytes(raw)
    with pytest.raises(VaultIntegrityError):
        archive.read_file(source)
    with pytest.raises(VaultIntegrityError):
        archive.verify_all()
    newest = sorted(archive.root.glob("manifest-*.enc"))[-1]
    raw = bytearray(newest.read_bytes())
    raw[-1] ^= 1
    newest.write_bytes(raw)
    with pytest.raises(VaultIntegrityError):
        SecureHistoryArchive(archive.root, PASSPHRASE)


def test_missing_manifest_fails_closed(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    manifest = next(archive.root.glob("manifest-*.enc"))
    manifest.rename(manifest.with_suffix(".saved"))
    with pytest.raises(VaultIntegrityError, match="no authenticated manifest"):
        SecureHistoryArchive(archive.root, PASSPHRASE)


def test_dpapi_failure_can_recover_with_portable_passphrase(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    source = tmp_path / "chat.jsonl"
    source.write_bytes(b"private")
    archive.archive_file(source, "codex")
    monkeypatch.setattr("muninn.history.secure_archive._dpapi_unwrap",
                        lambda unused: (_ for _ in ()).throw(VaultIntegrityError("wrong Windows user")))
    if __import__("os").name == "nt":
        with pytest.raises(VaultIntegrityError):
            SecureHistoryArchive(archive.root)
    assert SecureHistoryArchive(archive.root, PASSPHRASE).read_file(source) == b"private"


def test_no_legacy_plaintext_copy_in_strict_mode(tmp_path, monkeypatch):
    from muninn.history.vault import HistoryVault

    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    with pytest.raises(RuntimeError, match="strict history"):
        HistoryVault(tmp_path / "plaintext")
    assert not (tmp_path / "plaintext").exists()


def test_batch_commits_and_resume_without_duplicate_versions(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    sources = []
    for index in range(5):
        source = tmp_path / f"chat-{index}.jsonl"
        source.write_bytes(f"private-{index}".encode())
        sources.append((source, "codex", "transcript"))
    first = archive.archive_many(sources, commit_every=2)
    assert first == {"captured": 5, "unchanged": 0, "errors": [], "commits": 3}
    assert archive.status()["snapshots"] == 5
    second = SecureHistoryArchive(archive.root, PASSPHRASE).archive_many(sources, commit_every=2)
    assert second == {"captured": 0, "unchanged": 5, "errors": [], "commits": 0}
    assert archive.status()["snapshots"] == 5


def test_portable_restore_rewraps_only_on_explicit_request(tmp_path):
    source = tmp_path / "chat.jsonl"
    source.write_bytes(b"backup payload")
    original = SecureHistoryArchive.create(tmp_path / "original", PASSPHRASE)
    original.archive_file(source, "codex")
    restored = SecureHistoryArchive.restore_from_backup(original.root, tmp_path / "restored", PASSPHRASE)
    assert restored.read_file(source) == b"backup payload"
    assert original.read_file(source) == b"backup payload"
    assert restored.status() == original.status()
    assert restored.verify_all()["snapshots_verified"] == 1


@pytest.mark.skipif(__import__("os").name != "nt", reason="unattended backup uses Windows DPAPI")
def test_unattended_backup_is_new_verified_ciphertext_copy(tmp_path):
    source = tmp_path / "chat.jsonl"
    source.write_bytes(b"private backup payload")
    original = SecureHistoryArchive.create(tmp_path / "original", PASSPHRASE)
    original.archive_file(source, "codex")
    backup_root = tmp_path / "backup"

    report = original.backup_to(backup_root)

    assert report["snapshots_verified"] == 1
    assert SecureHistoryArchive(backup_root, PASSPHRASE).read_file(source) == source.read_bytes()
    assert original.read_file(source) == source.read_bytes()
    assert b"private backup payload" not in b"".join(
        path.read_bytes() for path in backup_root.rglob("*") if path.is_file()
    )
    with pytest.raises(ValueError, match="Invalid history archive restore locations"):
        original.backup_to(backup_root)
