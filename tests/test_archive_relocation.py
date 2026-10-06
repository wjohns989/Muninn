"""Relocation acceptance on disposable archives, never the live installation."""
import json
import copy
import os
from pathlib import Path

import pytest

from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.credential_crypto import VaultIntegrityError

_SESSION = "b5828f79-4882-4f39-a2cb-759c353f68af"
_OTHER = "7a58fb79-befa-4e3e-8adb-1d192327ce34"
_PASSPHRASE = "isolated relocation recovery phrase"


def _source(root: Path, provider: str, session=_SESSION):
    root.mkdir()
    source = root / (f"rollout-{session}.jsonl" if provider == "codex" else f"{session}.jsonl")
    record = ({"type": "session_meta", "payload": {"id": session, "cwd": "isolated-project"}}
              if provider == "codex" else
              {"type": "user", "sessionId": session, "cwd": "isolated-project",
               "message": {"role": "user", "content": "isolated original observation"}})
    source.write_bytes((json.dumps(record) + "\n").encode())
    return source


@pytest.mark.parametrize("provider", ["codex", "claude_code"])
def test_move_reuses_original_snapshot_and_preserves_receipt(tmp_path, provider):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", provider)
    raw = source.read_bytes()
    before = archive.archive_file(source, provider, include_snapshot_receipt=True)
    old_entry = dict(archive._load_manifest()["files"][str(source.resolve())][0])
    destination = tmp_path / "relocated"
    destination.mkdir()
    moved = destination / source.name
    source.rename(moved)

    after = archive.archive_file(moved, provider, include_snapshot_receipt=True)

    assert after["status"] == "unchanged"
    assert after["snapshot_receipt"] == before["snapshot_receipt"]
    assert archive.status()["sources"] == 1
    assert archive.status()["snapshots"] == 1
    assert archive.read_file(source, 0) == archive.read_file(moved, 0) == raw
    assert archive._load_manifest()["files"][str(source.resolve())][0] == old_entry


def test_append_after_move_has_one_lineage_and_old_prefix_parent(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes()
    old = archive.archive_file(source, "codex", include_snapshot_receipt=True)
    destination = tmp_path / "relocated"
    destination.mkdir()
    moved = destination / source.name
    source.rename(moved)
    appended = raw + b'{"type":"event_msg","payload":{"type":"user_message","message":"later observation"}}\n'
    moved.write_bytes(appended)

    new = archive.archive_file(moved, "codex", include_snapshot_receipt=True)

    assert archive.status()["sources"] == 1
    assert archive.status()["snapshots"] == 2
    assert new["snapshot_receipt"]["version"] == 1
    assert archive.read_file(source, 0) == raw
    assert archive.read_file(moved, 1) == appended
    entries = archive._load_manifest()["files"][str(source.resolve())]
    assert archive._prefix_parent(entries, 1)["blob"] == old["snapshot_receipt"]["blob"]
    assert entries[1]["observed_source"] == str(moved.resolve())


def test_equal_bytes_different_native_sessions_do_not_merge(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    other = _source(tmp_path / "other", "codex", _OTHER)
    other.write_bytes(source.read_bytes())
    archive.archive_file(source, "codex")
    archive.archive_file(other, "codex")
    assert archive.status()["sources"] == 2
    assert archive.status()["snapshots"] == 2


def _copy(source, root):
    root.mkdir()
    target = root / source.name
    target.write_bytes(source.read_bytes())
    # Explicitly different metadata: relocation requires a digest, not mtime.
    os.utime(target, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 10000000))
    return target


def test_alias_only_commit_is_idempotent_and_never_enrolls_new_snapshot(tmp_path):
    from muninn.history.capture_journal import CaptureJournal

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    journal = CaptureJournal(archive)
    journal.configure_enrichment(0)
    source = _source(tmp_path / "original", "codex")
    original = archive.archive_file(source, "codex", include_snapshot_receipt=True)
    assert journal.reconcile_enrichment() == 1
    pending = journal.pending_enrichment()
    generation = archive.status()["generation"]
    catalog = archive.metadata_catalog()
    moved = _copy(source, tmp_path / "moved")
    copied = archive.archive_many([(moved, "codex", "transcript")])
    assert copied == {"captured": 0, "unchanged": 1, "errors": [], "commits": 1}
    manifest = archive._load_manifest()
    assert manifest["format"] == 2 and manifest["generation"] == generation + 1
    assert list(archive.iter_committed_receipts(after_generation=generation)) == []
    assert journal.reconcile_enrichment() == 0
    assert journal.pending_enrichment() == pending
    assert archive.metadata_catalog() == catalog
    assert archive.latest_source_signatures()[str(moved.resolve())] == (
        moved.stat().st_size, moved.stat().st_mtime_ns)
    repeated = archive.archive_file(moved, "codex", include_snapshot_receipt=True)
    assert repeated["status"] == "unchanged"
    assert repeated["snapshot_receipt"] == original["snapshot_receipt"]
    assert archive.status()["generation"] == generation + 1
    pinned = archive._load_manifest(generation=generation)
    assert pinned["format"] == 1 and str(moved.resolve()) not in pinned.get("aliases", {})


def test_existing_agent_fetch_capability_survives_move_and_append(tmp_path):
    from muninn.history.blind_index import SecureHistoryBlindIndex

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes() + b'{"type":"event_msg","payload":{"type":"user_message","message":"relocationneedle original observation"}}\n'
    source.write_bytes(raw)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    index.build()
    match = index.search("relocationneedle")["matches"][0]
    before = index.fetch_span(match["fetch_capability"])
    moved = _copy(source, tmp_path / "moved")
    source.unlink()  # Disposable fixture: emulate a real move with old path gone.
    archive.archive_file(moved, "codex")
    assert index.build()["indexed"] == 0
    assert index.fetch_span(match["fetch_capability"]) == before
    moved.write_bytes(raw + b'{"type":"event_msg","payload":{"type":"user_message","message":"later distinct observation"}}\n')
    archive.archive_file(moved, "codex")
    index.build()
    assert index.fetch_span(match["fetch_capability"]) == before
    assert "relocationneedle original observation" in before["redacted_text"]


def test_multichunk_relocation_append_keeps_exact_prefix(tmp_path):
    from muninn.history.secure_archive import _CHUNK

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes() + b"x" * (_CHUNK + 73)
    source.write_bytes(raw)
    archive.archive_file(source, "codex")
    moved = _copy(source, tmp_path / "moved")
    moved.write_bytes(raw + b"y" * (_CHUNK + 31))
    archive.archive_file(moved, "codex")
    entries = archive._load_manifest()["files"][str(source.resolve())]
    assert entries[1]["prefix_of"]["size"] == len(raw)
    assert archive.read_file(source, 0) == raw
    assert archive.read_file(moved) == moved.read_bytes()
    assert archive.verify_all()["snapshots_verified"] == 2


def test_replaced_handle_during_relocation_cannot_publish_alias_or_blob(tmp_path, monkeypatch):
    from types import SimpleNamespace

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    moved = _copy(source, tmp_path / "moved")
    moved.write_bytes(moved.read_bytes() + b"later observation\n")
    before = copy.deepcopy(archive._load_manifest())
    blobs = list(archive._blobs.glob("*.enc"))
    original_write, original_stat = archive._write_chunk, os.fstat
    changed = False

    def race(*args, **kwargs):
        nonlocal changed
        changed = True
        return original_write(*args, **kwargs)

    def replaced(fd):
        actual = original_stat(fd)
        return (SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino + 1)
                if changed else actual)

    with monkeypatch.context() as patch:
        patch.setattr(archive, "_write_chunk", race)
        patch.setattr(os, "fstat", replaced)
        with pytest.raises(ValueError, match="source identity changed"):
            archive.archive_file(moved, "codex")
    assert archive._load_manifest() == before
    assert list(archive._blobs.glob("*.enc")) == blobs


@pytest.mark.parametrize("content", [b"divergent and much longer observation" * 20, b"short"])
def test_divergent_native_copy_is_not_acknowledged_or_published(tmp_path, content):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    moved = _copy(source, tmp_path / "moved")
    moved.write_bytes(content)
    before = copy.deepcopy(archive._load_manifest())
    blobs = list(archive._blobs.glob("*.enc"))
    with pytest.raises(ValueError, match="preserve the latest"):
        archive.archive_file(moved, "codex")
    assert archive._load_manifest() == before
    assert list(archive._blobs.glob("*.enc")) == blobs
    assert moved.read_bytes() == content


def test_stale_alias_cannot_join_an_older_prefix(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes()
    archive.archive_file(source, "codex")
    first = _copy(source, tmp_path / "first")
    archive.archive_file(first, "codex")
    second = _copy(source, tmp_path / "second")
    second.write_bytes(raw + b"newer observation\n")
    archive.archive_file(second, "codex")
    assert str(first.resolve()) not in archive.latest_source_signatures()
    first.write_bytes(raw + b"different branch, longer than newer observation\n")
    before = copy.deepcopy(archive._load_manifest())
    with pytest.raises(ValueError, match="preserve the latest"):
        archive.archive_file(first, "codex")
    assert archive._load_manifest() == before


def test_retained_original_cannot_regress_aliased_lineage_or_reenroll_old_content(tmp_path):
    from muninn.history.capture_journal import CaptureJournal

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    journal = CaptureJournal(archive)
    journal.configure_enrichment(0)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    moved = _copy(source, tmp_path / "moved")
    archive.archive_file(moved, "codex")
    moved.write_bytes(moved.read_bytes() + b"newer observation\n")
    archive.archive_file(moved, "codex")
    assert journal.reconcile_enrichment() == 2
    pending = journal.pending_enrichment()
    before = copy.deepcopy(archive._load_manifest())
    blobs = list(archive._blobs.glob("*.enc"))
    with pytest.raises(ValueError, match="preserve the latest"):
        archive.archive_file(source, "codex")
    assert archive._load_manifest() == before
    assert list(archive._blobs.glob("*.enc")) == blobs
    assert journal.reconcile_enrichment() == 0
    assert journal.pending_enrichment() == pending
    # Returning to the original location is valid only with current bytes.
    source.write_bytes(moved.read_bytes() + b"return to original location\n")
    assert archive.archive_file(source, "codex")["versions"] == 3
    entries = archive._load_manifest()["files"][str(source.resolve())]
    assert archive._prefix_parent(entries, 2) == entries[1]


def test_conflicting_legacy_native_lineages_require_review(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    other = _copy(source, tmp_path / "legacy")
    manifest = archive._load_manifest()
    manifest["files"][str(other.resolve())] = copy.deepcopy(manifest["files"][str(source.resolve())])
    manifest["generation"] += 1
    archive._save_manifest(manifest)
    moved = _copy(source, tmp_path / "moved")
    before = copy.deepcopy(archive._load_manifest())
    with pytest.raises(ValueError, match="conflicting native lineages"):
        archive.archive_file(moved, "codex")
    assert archive._load_manifest() == before


@pytest.mark.parametrize("damage", ["chain", "cycle", "shadow", "missing", "foreign", "digest", "downgrade"])
def test_invalid_authenticated_aliases_fail_closed(tmp_path, damage):
    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    moved = _copy(source, tmp_path / "moved")
    archive.archive_file(moved, "codex")
    manifest = archive._load_manifest()
    aliases = manifest["aliases"]
    alias = aliases[str(moved.resolve())]
    if damage == "chain":
        third = str((tmp_path / "third" / moved.name).resolve())
        aliases[third] = dict(alias, anchor=str(moved.resolve()))
    elif damage == "cycle":
        alias["anchor"] = str(moved.resolve())
    elif damage == "shadow":
        aliases[str(source.resolve())] = dict(alias)
    elif damage == "missing":
        alias["anchor"] = str(tmp_path / "absent")
    elif damage == "foreign":
        aliases[str(moved.resolve()).replace(_SESSION, _OTHER)] = aliases.pop(str(moved.resolve()))
    elif damage == "digest":
        alias["sha256"] = "0" * 64
    else:
        manifest["format"] = 1
    manifest["generation"] += 1
    archive._save_manifest(manifest)
    with pytest.raises(VaultIntegrityError, match="alias|lineage"):
        archive._load_manifest()


def test_legacy_format_gate_rejects_new_alias_generation_before_capture(tmp_path, monkeypatch):
    import muninn.history.secure_archive as module

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    moved = _copy(source, tmp_path / "moved")
    archive.archive_file(moved, "codex")
    before = archive.status()

    def legacy_gate(manifest):
        # Exact old reader acceptance predicate; encryption envelope unchanged.
        if manifest.get("format") != 1:
            raise VaultIntegrityError("Legacy reader rejects unsupported manifest")

    with monkeypatch.context() as patch:
        patch.setattr(module, "validate_lineage", legacy_gate)
        with pytest.raises(VaultIntegrityError, match="Legacy reader"):
            SecureHistoryArchive(archive.root, _PASSPHRASE)
        with pytest.raises(VaultIntegrityError, match="Legacy reader"):
            archive.archive_file(moved, "codex")
    assert archive.status() == before


def test_portable_recovery_keeps_alias_receipts_prefix_and_original_project_time(tmp_path):
    from muninn.history.transcript_units import transcript_units

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes() + b'{"type":"event_msg","timestamp":"2026-01-02T03:04:05Z","payload":{"type":"user_message","message":"original project observation"}}\n'
    source.write_bytes(raw)
    old = archive.archive_file(source, "codex", include_snapshot_receipt=True)
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    old_units = list(transcript_units(archive, entry))
    moved = _copy(source, tmp_path / "different-physical-project")
    moved.write_bytes(raw + b'{"type":"session_meta","payload":{"id":"later","cwd":"actual-later-project"}}\n' + b'{"type":"event_msg","timestamp":"2026-02-03T04:05:06Z","payload":{"type":"user_message","message":"later project observation"}}\n')
    new = archive.archive_file(moved, "codex", include_snapshot_receipt=True)
    # A portable recovery uses the passphrase, never the originating DPAPI.
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored", _PASSPHRASE)
    entries = restored._load_manifest()["files"][str(source.resolve())]
    assert restored._snapshot_receipt(entries[0], 0) == old["snapshot_receipt"]
    assert restored._snapshot_receipt(entries[1], 1) == new["snapshot_receipt"]
    assert restored._prefix_parent(entries, 1) == entries[0]
    assert restored.read_file(source, 0) == restored.read_file(moved, 0) == raw
    assert list(transcript_units(restored, entries[0])) == old_units
    new_units = list(transcript_units(restored, entries[1]))
    assert new_units[:len(old_units)] == old_units
    assert any(part.unit.cwd == "actual-later-project" for part in new_units)
    assert all(part.unit.cwd != str(moved.parent) for part in new_units)
    assert restored.verify_all()["snapshots_verified"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_real_scan_does_not_strand_new_alias_behind_stale_original_after_checkpoint(
        tmp_path, monkeypatch, interrupt):
    from unittest.mock import Mock
    from muninn.history.service import HistoryService
    from muninn.history.vault import HistoryVault

    for name in ("MUNINN_HISTORY_HOMES", "CODEX_HOME", "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_CAPTURE_ENRICHMENT", "0")
    root = tmp_path / "archive"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    archive = SecureHistoryArchive.create(root, _PASSPHRASE)
    original_root = tmp_path / ".codex" / "sessions" / "original"
    original_root.parent.mkdir(parents=True)
    source = _source(original_root, "codex")
    source = source.rename(source.with_name(f"rollout-2026-01-02T03-04-05-{_SESSION}.jsonl"))
    archive.archive_file(source, "codex")
    moved = _copy(source, original_root.parent / "moved")
    archive.archive_file(moved, "codex")
    moved.write_bytes(moved.read_bytes() + b"already captured append\n")
    archive.archive_file(moved, "codex")
    moved.write_bytes(moved.read_bytes() + b"new missed-hook append\n")
    inventory = [{"path": source, "provider": "codex", "kind": "transcript"}]
    # Excluded inventory metadata forces the real durable scan checkpoint;
    # no filler file or capture job is required for this regression.
    inventory.extend({"path": source.with_name(f"excluded-{number}.db"),
                      "provider": "codex", "kind": "state_db"} for number in range(250))
    inventory.append({"path": moved, "provider": "codex", "kind": "transcript"})
    monkeypatch.setattr(HistoryVault, "_source_files", staticmethod(
        lambda provider: iter(inventory) if provider.provider == "codex" else iter(())))
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase=_PASSPHRASE)
    if interrupt:
        journal = service._require_capture_journal()
        record = journal.record_scan_batch

        def interrupted(generation, outcomes):
            record(generation, outcomes)
            raise RuntimeError("isolated scan interruption after durable checkpoint")

        with monkeypatch.context() as patch:
            patch.setattr(journal, "record_scan_batch", interrupted)
            with pytest.raises(RuntimeError, match="isolated scan interruption"):
                await service.scan_capture_sources()
        interrupted_generation = journal.begin_scan()
    report = await service.scan_capture_sources()
    if interrupt:
        assert report["generation"] == interrupted_generation
    assert report["queued"] == 1
    assert await service._process_capture_job_once()
    assert archive.read_file(moved) == moved.read_bytes()
    assert archive.status()["sources"] == 1 and archive.status()["snapshots"] == 3
    # The same stable inventory cannot replace B with A in a subsequent scan.
    again = await service.scan_capture_sources()
    for _ in range(again["queued"]):
        assert await service._process_capture_job_once()
    assert archive.status()["snapshots"] == 3


def test_same_fingerprint_move_refreshes_the_encrypted_journal_locator(tmp_path):
    from muninn.history.capture_journal import CaptureJournal

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    journal = CaptureJournal(archive)
    source = _source(tmp_path / "original", "codex")
    assert journal.enqueue(source, "codex", immediate=True) == "queued"
    destination = tmp_path / "moved"
    destination.mkdir()
    moved = source.rename(destination / source.name)
    assert journal.enqueue(moved, "codex", immediate=True) == "queued"
    job = journal.claim_due()
    assert job is not None and job.path == moved.resolve()


@pytest.mark.parametrize("mode", ["append_chain", "divergent_branches", "uncaptured_prefix", "no_anchor"])
def test_multiple_locator_selection_requires_verified_byte_continuity(tmp_path, mode):
    from muninn.history.capture_locator_selection import select_capture_locator

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes()
    if mode != "no_anchor":
        archive.archive_file(source, "codex")
    other = _copy(source, tmp_path / "other")
    if mode == "divergent_branches":
        source.write_bytes(raw + b"branch one\n")
        other.write_bytes(raw + b"branch two\n")
        with pytest.raises(ValueError, match="branches require review"):
            select_capture_locator(archive, archive._load_manifest(), [source, other], "codex")
        assert archive.status()["snapshots"] == 1
        return
    if mode == "uncaptured_prefix":
        source.write_bytes(raw[:20])
        other.write_bytes(raw[:25])
        assert select_capture_locator(archive, archive._load_manifest(), [source, other], "codex") is None
        return
    source.write_bytes(raw + b"first append\n")
    other.write_bytes(source.read_bytes() + b"second append\n")
    before = copy.deepcopy(archive._load_manifest())
    assert select_capture_locator(archive, before, [source, other], "codex") == other
    assert archive._load_manifest() == before
    assert archive.archive_file(other, "codex")["status"] == "captured"


def test_authenticated_historical_rewrite_is_not_a_new_viable_branch(tmp_path):
    from muninn.history.capture_locator_selection import select_capture_locator

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    archive.archive_file(source, "codex")
    older = _copy(source, tmp_path / "older")
    source.write_bytes(b"a legitimate pre-relocation rewrite\n")
    archive.archive_file(source, "codex")
    assert select_capture_locator(archive, archive._load_manifest(), [source, older], "codex") == source


def test_short_prefix_selection_authenticates_the_encrypted_trailer(tmp_path):
    from muninn.history.capture_locator_selection import select_capture_locator

    archive = SecureHistoryArchive.create(tmp_path / "archive", _PASSPHRASE)
    source = _source(tmp_path / "original", "codex")
    raw = source.read_bytes()
    archive.archive_file(source, "codex")
    other = _copy(source, tmp_path / "other")
    source.write_bytes(raw[:20])
    other.write_bytes(raw[:25])
    manifest = archive._load_manifest()
    entry = manifest["files"][str(source.resolve())][0]
    blob = archive._blobs / (entry["blob"] + ".enc")
    ciphertext = bytearray(blob.read_bytes())
    ciphertext[-1] ^= 1
    blob.write_bytes(ciphertext)
    with pytest.raises(VaultIntegrityError, match="authentication"):
        select_capture_locator(archive, manifest, [source, other], "codex")


@pytest.mark.asyncio
async def test_real_scan_conflicting_branches_enqueue_nothing(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from muninn.history.service import HistoryService
    from muninn.history.vault import HistoryVault

    for name in ("MUNINN_HISTORY_HOMES", "CODEX_HOME", "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "strict")
    monkeypatch.setenv("MUNINN_CAPTURE_ENRICHMENT", "0")
    root = tmp_path / "archive"
    monkeypatch.setenv("MUNINN_HISTORY_ARCHIVE_DIR", str(root))
    archive = SecureHistoryArchive.create(root, _PASSPHRASE)
    original_root = tmp_path / ".codex" / "sessions" / "original"
    original_root.parent.mkdir(parents=True)
    source = _source(original_root, "codex")
    source = source.rename(source.with_name(f"rollout-2026-01-02T03-04-05-{_SESSION}.jsonl"))
    archive.archive_file(source, "codex")
    moved = _copy(source, original_root.parent / "moved")
    source.write_bytes(source.read_bytes() + b"branch one\n")
    moved.write_bytes(moved.read_bytes() + b"branch two\n")
    inventory = [{"path": path, "provider": "codex", "kind": "transcript"}
                 for path in (source, moved)]
    monkeypatch.setattr(HistoryVault, "_source_files", staticmethod(
        lambda provider: iter(inventory) if provider.provider == "codex" else iter(())))
    service = HistoryService(Mock(), tmp_path / "unused", home=tmp_path,
                             archive_passphrase=_PASSPHRASE)
    report = await service.scan_capture_sources()
    assert report["errors"] == 1 and report["queued"] == 0
    assert service._require_capture_journal().status().get("pending", 0) == 0
    assert not await service._process_capture_job_once()
    assert archive.status()["snapshots"] == 1
