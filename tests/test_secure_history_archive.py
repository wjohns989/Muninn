import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import _MAGIC, SecureHistoryArchive


def test_portable_plan_forwards_prompted_passphrase_to_service(tmp_path, monkeypatch, capsys):
    from muninn.history import secure_archive

    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    root = tmp_path / "archive"
    passphrase = "test-only portable passphrase"
    SecureHistoryArchive.create(root, passphrase)
    def refuse_unattended_unlock(_value):
        raise RuntimeError("test-only unattended unlock blocked")
    monkeypatch.setattr(secure_archive, "_dpapi_unwrap", refuse_unattended_unlock)
    from muninn.history.service import HistoryService
    without_passphrase = HistoryService(None, tmp_path / "no-unlock-vault", home=tmp_path,
                                        secure_archive_root=root)
    assert without_passphrase.status()["vault"]["ready"] is False
    monkeypatch.setattr(secure_archive.getpass, "getpass", lambda _: passphrase)
    monkeypatch.setattr(secure_archive.sys, "argv", [
        "secure_archive", "plan", "--root", str(root), "--home", str(tmp_path), "--portable",
    ])

    assert secure_archive.main() == 0
    assert passphrase not in capsys.readouterr().out

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


def test_verified_chunk_iterator_requires_full_exhaustion(tmp_path):
    source = tmp_path / "long.jsonl"
    source.write_bytes(b"a" * (2 * 1024 * 1024 + 73))
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    assert b"".join(archive._iter_verified_entry(entry)) == source.read_bytes()

    blob = archive.root / "blobs" / f"{entry['blob']}.enc"
    altered = bytearray(blob.read_bytes())
    altered[-1] ^= 1  # Late trailer failure, after all plaintext chunks.
    blob.write_bytes(altered)
    iterator = archive._iter_verified_entry(entry)
    assert next(iterator) == b"a" * (1024 * 1024)
    iterator.close()  # An early consumer has not authenticated the source.
    with pytest.raises(VaultIntegrityError):
        list(archive._iter_verified_entry(entry))
    with pytest.raises(VaultIntegrityError):
        archive._verify_entry(entry, collect=False)


def test_authenticated_frame_rejects_trailing_compressed_data(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"private message")
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    blob = archive.root / "blobs" / f"{entry['blob']}.enc"
    raw = blob.read_bytes()
    prefix_size = len(_MAGIC) + 8
    prefix = raw[:prefix_size]
    sealed_size = struct.unpack(">I", raw[prefix_size:prefix_size + 4])[0]
    sealed_start = prefix_size + 4
    sealed_end = sealed_start + sealed_size
    cipher = AESGCM(archive._key)
    nonce = prefix[len(_MAGIC):] + struct.pack(">I", 0)
    aad = archive._chunk_aad(entry["blob"], 0, False)
    compressed = cipher.decrypt(nonce, raw[sealed_start:sealed_end], aad)
    resealed = cipher.encrypt(nonce, compressed + b"TRAILING-GARBAGE", aad)
    blob.write_bytes(prefix + struct.pack(">I", len(resealed)) + resealed + raw[sealed_end:])
    with pytest.raises(VaultIntegrityError, match="authentication failed"):
        archive._verify_entry(entry, collect=False)


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


def test_archive_rejects_source_identity_changed_after_authorization(tmp_path):
    authorized = tmp_path / "allowed.jsonl"
    authorized.write_bytes(b"allowed content")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"OUTSIDE-CANARY")
    archive = SecureHistoryArchive.create(tmp_path / "secure", PASSPHRASE)

    with pytest.raises(ValueError, match="source identity changed"):
        archive.archive_file(outside, "codex", expected_source=authorized.resolve())
    assert archive.status()["snapshots"] == 0


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


@pytest.mark.skipif(__import__("os").name != "nt", reason="unattended backup uses Windows DPAPI")
def test_unattended_backup_accepts_relative_archive_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = SecureHistoryArchive.create(Path("original"), PASSPHRASE)

    report = original.backup_to(Path("backup"))

    assert report["snapshots_verified"] == 0
    assert (tmp_path / "backup" / "capture-jobs.db").is_file()


@pytest.mark.skipif(__import__("os").name != "nt", reason="unattended backup uses Windows DPAPI")
@pytest.mark.parametrize("verification_fails", [False, True])
def test_backup_validation_does_not_hold_live_capture_lock(tmp_path, monkeypatch, verification_fails):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from muninn.history.private_acl import verify_private

    source = tmp_path / "chat.jsonl"
    source.write_bytes(b"isolated pinned version\n")
    original = SecureHistoryArchive.create(tmp_path / "original", PASSPHRASE)
    original.archive_file(source, "codex")
    before = original.status()
    destination = tmp_path / "backup"
    verifying, release = threading.Event(), threading.Event()
    staging = []
    real_verify = SecureHistoryArchive.verify_all

    def paused_verify(archive):
        assert archive.root != original.root
        verifying.set()
        assert release.wait(10), "isolated validation release missing"
        if verification_fails:
            raise VaultIntegrityError("isolated backup proof rejected")
        return real_verify(archive)

    monkeypatch.setattr(SecureHistoryArchive, "verify_all", paused_verify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        backup = pool.submit(original.backup_to, destination, on_staging=staging.append)
        assert verifying.wait(10), "copied backup never reached validation"
        source.write_bytes(b"isolated pinned version\nnew live append\n")
        # A separate archive object contends for the actual live file lock.
        writer = SecureHistoryArchive(original.root, PASSPHRASE)
        capture = pool.submit(writer.archive_file, source, "codex")
        try:
            assert capture.result(timeout=1)["status"] == "captured"
            assert not destination.exists(), "unvalidated backup was published"
        finally:
            release.set()
        if verification_fails:
            with pytest.raises(VaultIntegrityError, match="isolated backup proof rejected"):
                backup.result(timeout=10)
        else:
            report = backup.result(timeout=10)
            assert report["generation"] == before["generation"]
            copied = SecureHistoryArchive(destination, PASSPHRASE)
            assert copied.read_file(source) == b"isolated pinned version\n"
    assert original.read_file(source) == source.read_bytes()
    assert original.status()["snapshots"] == 2
    if verification_fails:
        assert not destination.exists()
        assert len(staging) == 1 and staging[0].is_dir()
        verify_private(staging[0])
