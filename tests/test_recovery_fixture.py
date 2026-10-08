"""The portable test builder is bounded, encrypted and genuinely phrase-unlocked."""
import hashlib

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from tests.recovery_fixture import recovery_input


PHRASE = "synthetic portable recovery phrase"


def archive_in(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", PHRASE)
    source = tmp_path / "source.jsonl"
    source.write_text("synthetic private recovery payload", encoding="utf-8")
    archive.archive_file(source, "codex")
    from muninn.history.capture_journal import CaptureJournal

    CaptureJournal(archive)
    return archive, source


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).digest()
            for path in root.rglob("*") if path.is_file()}


def test_portable_input_uses_real_phrase_without_dpapi_or_source_changes(tmp_path, monkeypatch):
    from muninn.history import secure_archive

    archive, source = archive_in(tmp_path)
    before = fingerprint(archive.root)
    monkeypatch.setattr(secure_archive, "_dpapi_unwrap",
                        lambda *_: pytest.fail("Portable fixture used Windows unlock"))
    destination = tmp_path / "portable-input"
    report = recovery_input(archive, destination, PHRASE, temporary_root=tmp_path)
    restored = SecureHistoryArchive.restore_from_backup(destination, tmp_path / "restored", PHRASE)
    assert report["runtime_bundle"] == 0 and report["snapshots_verified"] == 1
    assert restored.read_file(source) == source.read_bytes()
    assert fingerprint(archive.root) == before
    assert b"synthetic private recovery payload" not in b"".join(
        path.read_bytes() for path in destination.rglob("*") if path.is_file())


def test_wrong_fixture_phrase_fails_before_creating_destination(tmp_path):
    archive, _ = archive_in(tmp_path)
    destination = tmp_path / "portable-input"
    with pytest.raises(VaultIntegrityError):
        recovery_input(archive, destination, "wrong synthetic recovery phrase", temporary_root=tmp_path)
    assert not destination.exists()


@pytest.mark.parametrize("target", ["outside", "source_child", "existing"])
def test_fixture_rejects_unsafe_destination_before_copy(tmp_path, target):
    archive, _ = archive_in(tmp_path)
    destination = {"outside": tmp_path.parent / (tmp_path.name + "-outside"),
                   "source_child": archive.root / "child", "existing": tmp_path / "existing"}[target]
    if target == "existing":
        destination.mkdir()
    before = fingerprint(archive.root)
    with pytest.raises(ValueError):
        recovery_input(archive, destination, PHRASE, temporary_root=tmp_path)
    assert fingerprint(archive.root) == before
    assert destination.exists() == (target == "existing")
