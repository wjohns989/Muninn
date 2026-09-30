import hashlib
import json

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import ProjectionIntegrityError, SecureProjectionStore


class FakeArchive:
    vault_id = "v" * 64
    _key = b"k" * 32

    def __init__(self, root, chunks):
        self.root = root
        self.chunks = chunks
        self.tamper = False

    def _iter_verified_entry(self, entry):
        for i, chunk in enumerate(self.chunks):
            if self.tamper and i == len(self.chunks) - 1:
                raise RuntimeError("late archive tamper")
            yield chunk


ENTRY = {"blob": "b" * 32, "sha256": hashlib.sha256(b"alpha beta gamma").hexdigest(), "size": 16}


def _store(tmp_path, chunks=(b"alpha", b" beta", b" gamma")):
    archive = FakeArchive(tmp_path / "archive", chunks)
    archive.root.mkdir(parents=True)
    return SecureProjectionStore(archive), archive


def pages(source):
    text = b"".join(source).decode()
    yield text[:10]
    yield text[10:]


def test_success_and_wrong_version(tmp_path):
    store, _ = _store(tmp_path)
    attempt = store.build(ENTRY, 3, pages, page_chars=10)
    assert store.get_page(ENTRY, 3, attempt, 0) == "alpha beta"
    with pytest.raises(ProjectionIntegrityError):
        store.get_page(ENTRY, 4, attempt, 0)


@pytest.mark.parametrize("mutation", ["already_yielded", "later_deleted", "later_length"])
def test_iterator_detects_changes_after_yield_even_with_same_count(tmp_path, mutation):
    store, _ = _store(tmp_path)
    attempt = store.build(ENTRY, 0, pages, page_chars=10)
    reader = store._iter_sealed_pages(ENTRY, 0, attempt)
    assert next(reader) == "alpha beta"
    with store._connect() as db:
        if mutation == "already_yielded":
            db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                       "WHERE attempt=? AND ordinal=0", (attempt,))
        elif mutation == "later_deleted":
            db.execute("DELETE FROM pages WHERE attempt=? AND ordinal=1", (attempt,))
        else:
            db.execute("UPDATE pages SET length=999999999 "
                       "WHERE attempt=? AND ordinal=1", (attempt,))
    with pytest.raises(ProjectionIntegrityError):
        list(reader)


def test_iterator_multiple_batches_release_reader_before_each_yield(tmp_path):
    store, _ = _store(tmp_path)

    def many(source):
        list(source)
        yield "alpha beta gamma " * 100

    attempt = store.build(ENTRY, 0, many, page_chars=10)
    expected = store.count_pages(ENTRY, 0, attempt)
    assert expected > 64
    seen = 0
    for page in store._iter_sealed_pages(ENTRY, 0, attempt):
        assert 1 <= len(page) <= 10
        # A committed write at every yield proves no consumer-spanning reader
        # survives even at a batch boundary under DELETE journal mode.
        with store._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS iterator_write_probe(n INTEGER)")
            db.execute("INSERT INTO iterator_write_probe VALUES(?)", (seen,))
        seen += 1
    assert seen == expected


def test_no_plaintext_and_early_exit_and_late_tamper(tmp_path):
    store, archive = _store(tmp_path)
    with pytest.raises(ProjectionIntegrityError):
        store.build(ENTRY, 0, lambda source: ("alpha",))
    assert b"alpha" not in store.db_path.read_bytes()
    archive.tamper = True
    with pytest.raises(RuntimeError):
        store.build(ENTRY, 1, lambda source: ("alpha" for _ in [*source]))


def test_page_and_completion_tamper_fail_closed(tmp_path):
    store, _ = _store(tmp_path)
    attempt = store.build(ENTRY, 0, pages, page_chars=10)
    with store._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=?", (attempt,))
    with pytest.raises(ProjectionIntegrityError):
        store.get_page(ENTRY, 0, attempt, 0)
    store2, _ = _store(tmp_path / "other")
    attempt2 = store2.build(ENTRY, 0, pages, page_chars=10)
    with store2._connect() as db:
        db.execute("UPDATE attempts SET completion=zeroblob(length(completion)) WHERE attempt=?", (attempt2,))
    with pytest.raises(ProjectionIntegrityError):
        store2.get_page(ENTRY, 0, attempt2, 0)


def test_interrupted_attempt_is_not_served(tmp_path):
    store, _ = _store(tmp_path)
    with store._connect() as db:
        db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                   ("interrupted", store.archive.vault_id, ENTRY["blob"], ENTRY["sha256"], 16, 0,
                    "building", 0, None, None))
    with pytest.raises(ProjectionIntegrityError):
        store.get_page(ENTRY, 0, "interrupted", 0)
    store = SecureProjectionStore(store.archive)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE state='building'").fetchone()[0] == 0


def test_failure_after_staging_commit_cleans_attempt(tmp_path):
    store, _ = _store(tmp_path)

    def interrupted(_source):
        for _ in range(70):
            yield "bounded page"
        raise RuntimeError("interrupted builder")

    with pytest.raises(RuntimeError, match="interrupted builder"):
        store.build(ENTRY, 0, interrupted)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0


def test_startup_recovery_removes_only_crashed_ciphertext_pages(tmp_path):
    store, archive = _store(tmp_path)
    complete = store.build(ENTRY, 0, pages, page_chars=10)
    crashed = "c" * 32
    with store._connect() as db:
        db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (crashed, archive.vault_id, ENTRY["blob"], ENTRY["sha256"], ENTRY["size"],
                    0, "building", 300, None, None))
        db.executemany("INSERT INTO pages VALUES(?,?,?,zeroblob(32))",
                       ((crashed, ordinal, 16) for ordinal in range(300)))
    with store._build_lock():
        # Opening a reader while a build owns the lock must not prune it.
        second = SecureProjectionStore(archive)
        with second._connect() as db:
            assert db.execute("SELECT COUNT(*) FROM pages WHERE attempt=?", (crashed,)).fetchone()[0] == 300
    recovered = SecureProjectionStore(archive)
    with recovered._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM pages WHERE attempt=?", (crashed,)).fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE attempt=?", (crashed,)).fetchone()[0] == 0
    assert recovered.get_page(ENTRY, 0, complete, 0) == "alpha beta"


def test_swapped_attempt_and_missing_page_fail_closed(tmp_path):
    store, _ = _store(tmp_path)
    first = store.build(ENTRY, 0, pages, page_chars=10)
    second = store.build(ENTRY, 0, pages, page_chars=10)
    with store._connect() as db:
        old = db.execute("SELECT ciphertext FROM pages WHERE attempt=? AND ordinal=0", (first,)).fetchone()[0]
        db.execute("UPDATE pages SET ciphertext=? WHERE attempt=? AND ordinal=0", (old, second))
    with pytest.raises(ProjectionIntegrityError):
        store.get_page(ENTRY, 0, second, 0)
    with store._connect() as db:
        db.execute("DELETE FROM pages WHERE attempt=? AND ordinal=1", (first,))
    with pytest.raises(ProjectionIntegrityError):
        store.get_page(ENTRY, 0, first, 0)


def test_real_archive_requires_final_auth_and_stores_only_redacted_pages(tmp_path):
    source = tmp_path / "chat.jsonl"
    original = json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "ordinary task api_key=tiny123 after"}}).encode()
    source.write_bytes(original)
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only portable recovery phrase")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    store = SecureProjectionStore(archive)

    def project(chunks):
        # This fixture exercises encrypted storage and the streaming redactor;
        # provider-specific field extraction is a separate integration gate.
        return (chunk.decode("utf-8") for chunk in chunks)

    attempt = store.build(entry, 0, project)
    safe = store.get_page(entry, 0, attempt, 0)
    assert "ordinary task" in safe and "after" in safe
    assert "tiny123" not in safe
    assert b"tiny123" not in store.db_path.read_bytes()
    blob = archive.root / "blobs" / f"{entry['blob']}.enc"
    damaged = bytearray(blob.read_bytes())
    damaged[-1] ^= 1
    blob.write_bytes(damaged)
    with pytest.raises(VaultIntegrityError):
        store.build(entry, 0, project)
