"""Same-origin append certificates; isolated ciphertext archives, no inference."""
import hashlib

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive, _CHUNK


@pytest.fixture
def archive(tmp_path):
    return SecureHistoryArchive.create(tmp_path / "archive", "test-only portable passphrase")


def entries(archive, path):
    return archive._load_manifest()["files"][str(path.resolve())]


def commit_manifest(archive, mutate):
    with archive._write_lock():
        manifest = archive._load_manifest()
        mutate(manifest)
        manifest["generation"] += 1
        archive._save_manifest(manifest)


@pytest.mark.parametrize("old_size", [0, 17, _CHUNK, _CHUNK + 37, 2 * _CHUNK + 19])
def test_append_certifies_exact_previous_prefix_in_existing_capture_pass(
        tmp_path, archive, monkeypatch, old_size):
    path = tmp_path / "session.jsonl"
    original = b"x" * old_size
    path.write_bytes(original)
    archive.archive_file(path, "codex")
    previous = entries(archive, path)[0]
    path.write_bytes(original + b"appended message\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Capturing an append must not decrypt an older snapshot")

    opens = []
    original_open = type(path).open

    def count_open(target, *args, **kwargs):
        if target == path and args and args[0] == "rb":
            opens.append(target)
        return original_open(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(archive, "_verify_entry", forbidden)
        patch.setattr(archive, "_iter_verified_entry", forbidden)
        patch.setattr(type(path), "open", count_open)
        report = archive.archive_file(path, "codex", include_snapshot_receipt=True)
    assert opens == [path]
    latest = entries(archive, path)[1]
    assert latest.get("prefix_of") == {
        "blob": previous["blob"], "sha256": previous["sha256"],
        "size": old_size, "version": 0}
    assert "prefix_of" not in report["snapshot_receipt"]
    assert archive._prefix_parent(entries(archive, path), 1) == previous
    assert archive.verify_all()["snapshots_verified"] == 2
    assert archive.read_file(path) == original + b"appended message\n"


@pytest.mark.parametrize("new_content,provider,kind", [
    (b"rewritten and longer", "codex", "transcript"),
    (b"ABC", "codex", "transcript"),
    (b"ab", "codex", "transcript"),
    (b"abc plus append", "claude_code", "transcript"),
    (b"abc plus append", "codex", "prompt_history"),
])
def test_rewrite_shrink_equal_size_and_changed_origin_do_not_certify(
        tmp_path, archive, new_content, provider, kind):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"abc")
    archive.archive_file(path, "codex")
    path.write_bytes(new_content)
    archive.archive_file(path, provider, kind)
    assert "prefix_of" not in entries(archive, path)[-1]


def test_different_path_identical_content_is_not_the_same_occurrence(tmp_path, archive):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_bytes(b"same text")
    archive.archive_file(first, "codex")
    second.write_bytes(b"same text then append")
    archive.archive_file(second, "codex")
    assert "prefix_of" not in entries(archive, second)[0]


@pytest.mark.parametrize("provider,kind", [("claude_code", "transcript"), ("codex", "prompt_history")])
def test_unchanged_bytes_do_not_hide_changed_origin(tmp_path, archive, provider, kind):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"same bytes and metadata")
    archive.archive_file(path, "codex")
    assert archive.archive_file(path, provider, kind)["status"] == "captured"
    versions = entries(archive, path)
    assert len(versions) == 2
    assert versions[1]["provider"] == provider and versions[1]["kind"] == kind
    assert "prefix_of" not in versions[1]


def test_chained_batched_and_unchanged_capture_keep_real_version_links(tmp_path, archive):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"first second")
    archive.archive_many([(path, "codex", "transcript")], commit_every=2)
    assert archive.archive_file(path, "codex") == {"status": "unchanged", "versions": 2}
    path.write_bytes(b"first second third")
    archive.archive_file(path, "codex")
    snapshots = entries(archive, path)
    for version in (1, 2):
        assert snapshots[version]["prefix_of"]["version"] == version - 1
        assert archive._prefix_parent(snapshots, version) == snapshots[version - 1]
    assert archive.verify_all()["snapshots_verified"] == 3


def test_source_growth_during_capture_cannot_publish_a_certificate(tmp_path, archive, monkeypatch):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"first second")
    original = archive._write_chunk
    changed = False

    def race(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            with path.open("ab") as handle:
                handle.write(b" concurrent append")
        return original(*args, **kwargs)

    monkeypatch.setattr(archive, "_write_chunk", race)
    with pytest.raises(RuntimeError, match="changed during encrypted capture"):
        archive.archive_file(path, "codex")
    assert len(entries(archive, path)) == 1


def test_open_source_identity_change_fails_even_without_explicit_expected_path(tmp_path, archive, monkeypatch):
    import os
    from types import SimpleNamespace

    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"first second")
    original_write = archive._write_chunk
    original_stat = os.fstat
    changed = False

    def race(*args, **kwargs):
        nonlocal changed
        changed = True
        return original_write(*args, **kwargs)

    def replaced_inode(fd):
        actual = original_stat(fd)
        if changed:
            return SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino + 1)
        return actual

    with monkeypatch.context() as patch:
        patch.setattr(archive, "_write_chunk", race)
        patch.setattr(os, "fstat", replaced_inode)
        with pytest.raises(ValueError, match="source identity changed"):
            archive.archive_file(path, "codex")
    assert len(entries(archive, path)) == 1


def test_legacy_entries_without_certificate_restore_without_invented_links(tmp_path, archive):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"first second")
    archive.archive_file(path, "codex")
    commit_manifest(archive, lambda manifest: manifest["files"][str(path.resolve())][1].pop("prefix_of", None))
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    assert "prefix_of" not in entries(restored, path)[1]
    assert restored.verify_all()["snapshots_verified"] == 2


@pytest.mark.parametrize("damage", ["version", "size", "blob", "sha256", "origin", "extra", "boolean"])
def test_invalid_prefix_relation_blocks_verification_and_portable_restore(tmp_path, archive, damage):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"first second")
    archive.archive_file(path, "codex")

    def corrupt(manifest):
        versions = manifest["files"][str(path.resolve())]
        certificate = versions[1]["prefix_of"]
        if damage == "origin":
            versions[1]["provider"] = "claude_code"
        elif damage == "extra":
            certificate["processed"] = True
        elif damage == "boolean":
            certificate["version"] = False
        else:
            certificate[damage] = {"version": 1, "size": 6, "blob": "0" * 32, "sha256": "0" * 64}[damage]

    commit_manifest(archive, corrupt)
    with pytest.raises(VaultIntegrityError, match="prefix"):
        archive.verify_all()
    with pytest.raises(VaultIntegrityError, match="prefix"):
        SecureHistoryArchive.restore_from_backup(
            archive.root, tmp_path / "restored", "test-only portable passphrase")


def test_authenticated_but_false_preserved_bytes_fail_even_direct_snapshot_read(tmp_path, archive):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"other longer")
    archive.archive_file(path, "codex")

    def forge(manifest):
        versions = manifest["files"][str(path.resolve())]
        old = versions[0]
        versions[1]["prefix_of"] = {key: old[key] for key in ("blob", "sha256", "size")}
        versions[1]["prefix_of"]["version"] = 0

    commit_manifest(archive, forge)
    with pytest.raises(VaultIntegrityError, match="prefix"):
        archive.read_file(path)
    with pytest.raises(VaultIntegrityError, match="prefix"):
        archive.verify_all()


def test_valid_prefix_is_preserved_by_portable_recovery(tmp_path, archive):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"first")
    archive.archive_file(path, "codex")
    path.write_bytes(b"first second")
    archive.archive_file(path, "codex")
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", "test-only portable passphrase")
    versions = entries(restored, path)
    assert restored._prefix_parent(versions, 1) == versions[0]
    assert restored.read_file(path) == b"first second"
    assert versions[1]["prefix_of"]["sha256"] == hashlib.sha256(b"first").hexdigest()
