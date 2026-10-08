"""Synthetic encrypted recovery fixtures; no live archives or providers."""
import hashlib
import os
import sqlite3
from contextlib import closing

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_directory, create_private_file
from muninn.history.recovery_pool import RecoveryPool, RELATIVE, MARKER
from muninn.history.secure_archive import SecureHistoryArchive


def fixture(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic recovery passphrase")
    parent = archive.root / "operator-preimages"
    create_private_directory(parent)
    pool = RecoveryPool(tmp_path / "pool", archive=archive)
    return archive, parent, pool


def snapshot(parent, index, payload=None):
    folder = parent / ("restart-20261007-010203-" + f"{index:032x}")
    create_private_directory(folder)
    create_private_directory(folder / "source-evidence")
    db_path = folder / RELATIVE
    create_private_file(db_path)
    with closing(sqlite3.connect(db_path)) as db:
        with db:
            db.execute("CREATE TABLE encrypted_records(value BLOB)")
            db.execute("INSERT INTO encrypted_records VALUES(?)", (payload or os.urandom(1500000),))
    return folder


def test_portable_roundtrip_dedup_preserves_snapshot_identities(tmp_path):
    archive, parent, pool = fixture(tmp_path)
    payload = os.urandom(1500000)
    first = snapshot(parent, 1, payload)
    second = snapshot(parent, 2, payload)
    original = (first / RELATIVE).read_bytes()
    a = pool.pack(archive, first, retire=True)
    b = pool.pack(archive, second, retire=True)
    assert a["retired_bytes"] == b["retired_bytes"] == len(original)
    assert a["new_chunk_bytes"] > 0 and b["new_chunk_bytes"] == 0
    assert a["snapshot"] != b["snapshot"]
    assert not (first / RELATIVE).exists() and not (second / RELATIVE).exists()
    portable = RecoveryPool(pool.root, passphrase="synthetic recovery passphrase")
    report = portable.restore(first, tmp_path / "restored")
    assert report["recovery_verified"]
    assert (tmp_path / "restored" / RELATIVE).read_bytes() == original
    with pytest.raises(FileExistsError):
        portable.restore(first, tmp_path / "restored")


@pytest.mark.parametrize("fault", ["missing", "tampered"])
def test_bad_chunk_preserves_original_and_rejects_restore(tmp_path, fault):
    archive, parent, pool = fixture(tmp_path)
    first = snapshot(parent, 1)
    pool.pack(archive, first)
    chunk = next((pool.root / "chunks").iterdir())
    if fault == "missing":
        chunk.unlink()
    else:
        raw = bytearray(chunk.read_bytes())
        raw[-1] ^= 1
        chunk.write_bytes(raw)
    with pytest.raises((VaultIntegrityError, PermissionError)):
        pool.pack(archive, first, retire=True)
    assert (first / RELATIVE).exists()
    with pytest.raises((VaultIntegrityError, PermissionError)):
        pool.restore(first, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("stage", ["manifest_published", "original_retired"])
def test_interrupted_retirement_keeps_a_recoverable_snapshot(tmp_path, stage):
    archive, parent, pool = fixture(tmp_path)
    first = snapshot(parent, 1)
    original = (first / RELATIVE).read_bytes()
    def interrupted(current):
        if current == stage:
            raise RuntimeError("synthetic crash")
    with pytest.raises(RuntimeError):
        pool.pack(archive, first, retire=True, on_stage=interrupted)
    assert (first / RELATIVE).exists() == (stage == "manifest_published")
    pool.pack(archive, first, retire=True)
    pool.restore(first, tmp_path / "recovered")
    assert (tmp_path / "recovered" / RELATIVE).read_bytes() == original


def test_changed_original_cannot_be_retired_using_a_prior_manifest(tmp_path):
    archive, parent, pool = fixture(tmp_path)
    first = snapshot(parent, 1)
    pool.pack(archive, first)
    with sqlite3.connect(first / RELATIVE) as db:
        db.execute("INSERT INTO encrypted_records VALUES(?)", (b"new user bytes",))
    with pytest.raises(VaultIntegrityError, match="Original preimage differs"):
        pool.pack(archive, first, retire=True)
    assert (first / RELATIVE).exists()


def test_foreign_archive_and_outside_snapshot_fail_closed(tmp_path):
    archive, parent, pool = fixture(tmp_path)
    foreign = SecureHistoryArchive.create(tmp_path / "foreign", "synthetic recovery passphrase")
    with pytest.raises(VaultIntegrityError):
        RecoveryPool(pool.root, archive=foreign)
    with pytest.raises(VaultIntegrityError):
        pool.pack(archive, tmp_path)


def test_manifest_tamper_and_path_traversal_rejected(tmp_path):
    archive, parent, pool = fixture(tmp_path)
    first = snapshot(parent, 1)
    pool.pack(archive, first)
    value = pool._read_manifest(first / MARKER, first.name)
    value["path"] = "../../live.db"
    import json
    (first / MARKER).write_bytes(pool._seal("manifest", first.name, json.dumps(value).encode()))
    with pytest.raises(VaultIntegrityError):
        pool.pack(archive, first, retire=True)
    assert (first / RELATIVE).exists()


def test_cli_keeps_latest_four_and_preserves_neighboring_batch_db(tmp_path):
    from scripts.compact_restart_recovery import eligible_snapshots
    archive, parent, pool = fixture(tmp_path)
    rows = [snapshot(parent, index) for index in range(6)]
    batch = rows[0] / "historical-batches.db"
    create_private_file(batch)
    batch.write_bytes(b"synthetic retained batch fixture")
    candidates = eligible_snapshots(archive)
    assert candidates == rows[:2]
    assert eligible_snapshots(archive, excluded=[rows[0].name]) == rows[1:2]
    pool.pack(archive, candidates[0], retire=True)
    assert batch.read_bytes() == b"synthetic retained batch fixture"
    assert all((row / RELATIVE).exists() for row in rows[-4:])
    with pytest.raises(ValueError):
        eligible_snapshots(archive, 3)


def test_publication_failure_leaves_original_untouched(tmp_path, monkeypatch):
    from muninn.history import recovery_pool
    archive, parent, pool = fixture(tmp_path)
    first = snapshot(parent, 1)
    def failed(source, target):
        raise OSError("synthetic durability failure")
    monkeypatch.setattr(recovery_pool, "durable_publish", failed)
    with pytest.raises(OSError):
        pool.pack(archive, first, retire=True)
    assert (first / RELATIVE).exists()
    assert not (first / MARKER).exists()


def test_existing_manifest_retry_requires_retirement_barrier(tmp_path, monkeypatch):
    from muninn.history import recovery_pool
    archive, parent, pool = fixture(tmp_path)
    first = snapshot(parent, 1)
    pool.pack(archive, first)
    barrier = recovery_pool.durability_barrier
    def failed(directory):
        raise OSError("synthetic post-rename durability failure")
    monkeypatch.setattr(recovery_pool, "durability_barrier", failed)
    with pytest.raises(OSError):
        pool.pack(archive, first, retire=True)
    assert (first / RELATIVE).exists()
    monkeypatch.setattr(recovery_pool, "durability_barrier", barrier)
    assert pool.pack(archive, first, retire=True)["retired_bytes"] > 0
