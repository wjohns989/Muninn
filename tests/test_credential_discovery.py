"""No live secret or project file is read by these scanner tests."""

from __future__ import annotations

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.credential_discovery import (
    ExtractionStats,
    iter_env_findings,
    iter_transcript_findings,
    scan_archive,
    scan_project_env,
    scan_project_files,
)
from muninn.history.credential_store import CredentialStore
from muninn.history.secure_archive import SecureHistoryArchive


class RecordingStore:
    def __init__(self):
        self.values = []
        self.projects = []

    def scan_source(self, *, findings, **_kwargs):
        # The production store commits only after the generator completes.
        proposed = list(findings)
        self.values.extend(proposed)
        self.projects.append(_kwargs["project"])
        return {"created": len(proposed), "rotated": 0, "staled": 0}


def test_boundary_spanning_value_and_ambiguous_examples():
    stats = ExtractionStats()
    chunks = [b"x" * 1000 + b"\nOPENROUTER_API_KEY=aaaabbbb", b"cccc11112222\nPASSWORD=${", b"NOT_SET}\n"]
    found = list(iter_env_findings(chunks, ".env.local", stats))
    assert found == [("OPENROUTER_API_KEY", "aaaabbbbcccc11112222", ".env.local")]
    assert stats.accepted == 1
    assert stats.ambiguous == 1


def test_large_noncredential_line_does_not_hide_later_assignment():
    stats = ExtractionStats()
    chunks = [b"x" * 65536, b"x" * 65536 + b"\nSERVICE_ACCESS_TOKEN=aaBBccDD12345678\n"]
    assert list(iter_env_findings(chunks, ".env", stats)) == [
        ("SERVICE_ACCESS_TOKEN", "aaBBccDD12345678", ".env")]
    assert stats.accepted == 1


def test_invalid_utf8_is_not_silently_skipped():
    with pytest.raises(UnicodeDecodeError):
        list(iter_env_findings([b"API_KEY=goodVALUE123\n", b"\xff"], ".env", ExtractionStats()))


def test_project_scan_excludes_templates_and_reports_bad_source(tmp_path):
    root = tmp_path / "my-project"
    root.mkdir()
    (root / ".env.example").write_text("SERVICE_API_KEY=exampleSECRET123\n")
    (root / ".env.local").write_text("SERVICE_API_KEY=realVALUE12345678\n")
    (root / ".env.bad").write_bytes(b"OTHER_API_KEY=validVALUE12345\n\xff")
    store = RecordingStore()
    report = scan_project_env(root, store, passphrase="test-only")
    assert report["files"] == 2
    assert report["succeeded"] == 1
    assert report["errors"] == 1
    assert report["complete"] is False
    assert len(store.values) == 1
    assert store.values[0][0] == "SERVICE_API_KEY"


def test_project_file_scan_covers_common_text_formats_without_values_in_report(tmp_path):
    root = tmp_path / "my-project"
    root.mkdir()
    (root / ".env.local").write_text("ENV_ACCESS_TOKEN=envVALUE12345678\n")
    (root / "settings.json").write_text('{"JSON_API_KEY":"jsonVALUE12345678"}')
    (root / "config.yaml").write_text("YAML_SECRET_KEY: yamlVALUE12345678\n")
    (root / "settings.toml").write_text("TOML_AUTH_TOKEN = 'tomlVALUE12345678'\n")
    (root / "client.py").write_text('PYTHON_API_KEY = "codeVALUE12345678"\n')
    (root / ".env.example").write_text("EXAMPLE_API_KEY=exampleVALUE12345678\n")
    ignored = root / ".git"
    ignored.mkdir()
    (ignored / "config.py").write_text('GIT_API_KEY = "gitVALUE12345678"\n')
    store = RecordingStore()

    report = scan_project_files(root, store, passphrase="test-only")

    assert report["files"] == 5
    assert report["succeeded"] == 5
    assert report["errors"] == 0
    assert report["complete"] is True
    assert {name for name, _, _ in store.values} == {
        "ENV_ACCESS_TOKEN", "JSON_API_KEY", "YAML_SECRET_KEY",
        "TOML_AUTH_TOKEN", "PYTHON_API_KEY",
    }
    for _, value, _ in store.values:
        assert value not in str(report)


def test_documentation_examples_remain_unverified_candidates(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("# Example\nAPI_KEY=docsVALUE12345678\n")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")

    report = scan_project_files(root, store, passphrase="synthetic vault passphrase")

    assert report["complete"] is True
    assert report["inserted"] == 1
    matches = store.search("API_KEY")
    assert matches[0]["source_hint"] == "README.md"
    assert matches[0]["candidate_status"] == "unverified"
    assert "docsVALUE12345678" not in str(matches)


def test_collection_root_labels_each_nested_git_project(tmp_path):
    collection = tmp_path / "projects"
    for name, value in (("alpha", "alphaVALUE12345678"), ("beta", "betaVALUE12345678")):
        project = collection / name
        (project / ".git").mkdir(parents=True)
        (project / ".env").write_text(f"SERVICE_API_KEY={value}\n")
    store = RecordingStore()

    report = scan_project_files(collection, store, passphrase="test-only")

    assert report["complete"] is True
    assert report["files"] == 2
    assert store.projects == ["alpha", "beta"]
    assert {hint for _, _, hint in store.values} == {"alpha/.env", "beta/.env"}


def test_symlinked_env_is_rejected_without_following(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("SERVICE_API_KEY=realVALUE12345678\n")
    link = root / ".env"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    store = RecordingStore()
    report = scan_project_env(root, store, passphrase="test-only")
    assert report["errors"] == 1
    assert report["complete"] is False
    assert store.values == []


def test_transcript_assignment_scans_across_chunks_without_claiming_current_use():
    stats = ExtractionStats()
    found = list(iter_transcript_findings(
        [b'{"content":"SERVICE_API_KEY=aaaabbbb', b'cccc11112222"}\n'], stats))
    assert found == [("SERVICE_API_KEY", "aaaabbbbcccc11112222", "")]


def test_archive_late_integrity_failure_rolls_back_findings(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    archive = SecureHistoryArchive.create(root, "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = RecordingStore()
    original = archive._verify_entry

    def fail_at_end(entry, *, collect, on_chunk=None):
        result = original(entry, collect=collect, on_chunk=on_chunk)
        if on_chunk is not None:
            raise VaultIntegrityError("synthetic late final digest failure")
        return result

    monkeypatch.setattr(archive, "_verify_entry", fail_at_end)
    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert report["attempted"] == 1
    assert report["succeeded"] == 0
    assert report["errors"] == 1
    assert report["complete"] is False
    assert store.values == []


def test_real_vault_archive_scan_is_idempotent_and_metadata_only(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    first = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert first["complete"] is True
    assert first["inserted"] == 1
    second = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert second["inserted"] == 0
    assert second["skipped"] == 1
    matches = store.search("SERVICE_API_KEY")
    assert len(matches) == 1
    assert matches[0]["origin"] == "transcript"
    assert "aaaabbbbcccc11112222" not in str(matches)
    with pytest.raises(ValueError, match="generation"):
        scan_archive(archive, store, passphrase="synthetic vault passphrase", offset=1)
    with pytest.raises(ValueError, match="generation changed"):
        scan_archive(archive, store, passphrase="synthetic vault passphrase",
                     offset=1, expected_generation=first["generation"] + 1)


def test_real_vault_rolls_back_when_final_archive_verify_fails(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    original = archive._verify_entry

    def fail_at_end(entry, *, collect, on_chunk=None):
        result = original(entry, collect=collect, on_chunk=on_chunk)
        if on_chunk is not None:
            raise VaultIntegrityError("synthetic late final digest failure")
        return result

    monkeypatch.setattr(archive, "_verify_entry", fail_at_end)
    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert report["complete"] is False
    assert store.search("SERVICE_API_KEY") == []
    with store._connect(readonly=True) as db:
        assert db.execute("SELECT count(*) FROM scan_receipts").fetchone()[0] == 0


def test_archive_receipts_resume_after_new_source_sorts_before_old_sources(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    for name in ("b", "c"):
        source = tmp_path / f"{name}.jsonl"
        source.write_text('{"content":"SERVICE_API_KEY=' + name + 'aaaabbbbcccc1111"}\n')
        archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    first = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert first["succeeded"] == 2
    assert first["skipped"] == 0

    original_verify = archive._verify_entry
    verified = []

    def count_verify(entry, *, collect, on_chunk=None):
        verified.append(entry["blob"])
        return original_verify(entry, collect=collect, on_chunk=on_chunk)

    monkeypatch.setattr(archive, "_verify_entry", count_verify)
    second = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert second["skipped"] == 2
    assert verified == []

    source = tmp_path / "a.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaaVALUE12345678"}\n')
    archive.archive_file(source, "codex")
    third = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert third["succeeded"] == 1
    assert third["skipped"] == 2
    assert len(verified) == 1
    assert third["complete"] is True


def test_archive_zero_finding_snapshot_receipt_is_resumable(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"No credential assignment here"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    assert scan_archive(archive, store, passphrase="synthetic vault passphrase")["succeeded"] == 1

    def unexpected_verify(*_args, **_kwargs):
        raise AssertionError("Previously verified snapshot should not be read again")

    monkeypatch.setattr(archive, "_verify_entry", unexpected_verify)
    resumed = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert resumed["complete"] is True
    assert resumed["skipped"] == 1


def test_archive_receipts_survive_portable_vault_backup_restore(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    assert scan_archive(archive, store, passphrase="synthetic vault passphrase")["inserted"] == 1
    assert store.backup(tmp_path / "backup", passphrase="synthetic vault passphrase") == 1
    restored = CredentialStore.restore(tmp_path / "backup", tmp_path / "restored",
                                       passphrase="synthetic vault passphrase")

    def unexpected_verify(*_args, **_kwargs):
        raise AssertionError("Backed-up receipt should prevent a repeat blob read")

    monkeypatch.setattr(archive, "_verify_entry", unexpected_verify)
    resumed = scan_archive(archive, restored, passphrase="synthetic vault passphrase")
    assert resumed["complete"] is True
    assert resumed["skipped"] == 1


def test_new_capture_during_scan_is_reported_as_incomplete(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = RecordingStore()
    original = archive._load_manifest
    calls = 0

    def changed_manifest():
        nonlocal calls
        calls += 1
        value = original()
        return {**value, "generation": value["generation"] + (calls > 1)}

    monkeypatch.setattr(archive, "_load_manifest", changed_manifest)
    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert report["succeeded"] == 1
    assert report["changed_during_scan"] is True
    assert report["complete"] is False
