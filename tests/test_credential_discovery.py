"""No live secret or project file is read by these scanner tests."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

import muninn.history.credential_discovery as discovery
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
        accepted = [item for item in proposed if isinstance(item, tuple)]
        self.values.extend(accepted)
        self.projects.append(_kwargs["project"])
        return {"created": len(accepted), "rotated": 0, "staled": 0,
                "ambiguities": sum(isinstance(item, discovery.AmbiguousCandidate)
                                   for item in proposed)}


def test_boundary_spanning_value_and_ambiguous_examples():
    stats = ExtractionStats()
    chunks = [b"x" * 1000 + b"\nOPENROUTER_API_KEY=aaaabbbb", b"cccc11112222\nPASSWORD=${", b"NOT_SET}\n"]
    found = list(iter_env_findings(chunks, ".env.local", stats))
    assert found == [("OPENROUTER_API_KEY", "aaaabbbbcccc11112222", ".env.local")]
    assert stats.accepted == 1
    assert stats.ambiguous == 1
    assert stats.ambiguous_reasons == {"unparsed_value": 1}


def test_ambiguous_reasons_do_not_expose_candidate_text():
    stats = ExtractionStats()
    payload = (b"FIRST_API_KEY=${SECRET_VALUE}\n"
               b"SECOND_API_KEY=sampleValue987654\n"
               b"THIRD_API_KEY=aaaaaaaaaaaaaaaa\n")
    assert list(iter_env_findings([payload], ".env", stats)) == []
    assert stats.ambiguous == 3
    assert stats.ambiguous_reasons == {
        "unparsed_value": 1, "placeholder_like": 1, "low_diversity": 1,
    }
    assert "SECRET_VALUE" not in str(stats.ambiguous_reasons)


def test_ambiguous_events_are_opt_in_and_bounded():
    payload = b"SERVICE_API_KEY=${MISSING_REFERENCE}\n"
    stats = ExtractionStats()
    events = list(iter_env_findings([payload], ".env", stats, include_ambiguous=True))
    assert len(events) == 1
    assert isinstance(events[0], discovery.AmbiguousCandidate)
    assert events[0].reason == "unparsed_value"
    assert events[0].candidate == "${MISSING_REFERENCE}"
    assert list(iter_env_findings([payload], ".env", ExtractionStats())) == []
    assert discovery._bounded_candidate("x" * 200, 0) == ""
    assert discovery._bounded_candidate("${REF}suffix ", 0) == "${REF}suffix"


def test_large_noncredential_line_does_not_hide_later_assignment():
    stats = ExtractionStats()
    chunks = [b"x" * 65536, b"x" * 65536 + b"\nSERVICE_ACCESS_TOKEN=aaBBccDD12345678\n"]
    assert list(iter_env_findings(chunks, ".env", stats)) == [
        ("SERVICE_ACCESS_TOKEN", "aaBBccDD12345678", ".env")]
    assert stats.accepted == 1


def test_invalid_utf8_is_not_silently_skipped():
    with pytest.raises(UnicodeDecodeError):
        list(iter_env_findings([b"API_KEY=goodVALUE123\n", b"\xff"], ".env", ExtractionStats()))


@pytest.mark.parametrize("payload", [
    b"legacy \x96 text\nSERVICE_API_KEY=realVALUE12345678\n",
    b"legacy \x81 text\nSERVICE_API_KEY=realVALUE12345678\n",
    "SERVICE_API_KEY=realVALUE12345678\n".encode("utf-16"),
    "SERVICE_API_KEY=realVALUE12345678\n".encode("utf-16-be").join([b"\xfe\xff", b""]),
    b"\xef\xbb\xbf" + "SERVICE_API_KEY=realVALUE12345678\n".encode("utf-16le"),
])
def test_project_text_scanner_accepts_legacy_bytes_and_utf16(payload):
    stats = ExtractionStats()
    found = list(discovery.iter_project_findings([payload[:9], payload[9:]], "settings.txt", stats))
    assert found == [("SERVICE_API_KEY", "realVALUE12345678", "settings.txt")]


def test_project_scanner_rejects_malformed_assignment_but_retains_safe_text():
    stats = ExtractionStats()
    payload = (b"BAD_API_KEY=realVALUE12345678\x81suffix\n"
               b"x" * 98 + b"\x00\n"
               b"GOOD_API_KEY=anotherVALUE12345678\n")
    found = list(discovery.iter_project_findings([payload], "notes.txt", stats))
    assert found == [("GOOD_API_KEY", "anotherVALUE12345678", "notes.txt")]


def test_project_queue_keeps_legacy_malformed_assignment_without_losing_valid_key(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_bytes(
        b"BAD_API_KEY=realVALUE12345678\x81suffix\n"
        b"GOOD_API_KEY=anotherVALUE12345678\n"
    )
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    report = scan_project_files(root, store, passphrase="synthetic vault passphrase")
    assert report["complete"] is True
    assert report["queued"] == 1
    assert report["inserted"] == 1
    assert len(store.search("GOOD_API_KEY")) == 1


def test_project_txt_scanner_recovers_ansi_colored_assignment_across_chunks():
    payload = (b"x" * 4094 + b"\x1b[31m" * 100
               + b"\nSERVICE_API_KEY=\x1b[32mrealVALUE12345678\x1b[0m\n")
    parts = [payload[:4096], payload[4096:4098], payload[4098:4101],
             payload[4101:]]
    found = list(discovery.iter_project_findings(parts, "notes.txt", ExtractionStats()))
    assert found == [("SERVICE_API_KEY", "realVALUE12345678", "notes.txt")]


def test_project_txt_scanner_recovers_colon_form_sgr():
    payload = (b"\x1b[38:2::255:0:0m" * 100
               + b"\nSERVICE_API_KEY=realVALUE12345678\n")
    found = list(discovery.iter_project_findings([payload], "notes.txt", ExtractionStats()))
    assert found == [("SERVICE_API_KEY", "realVALUE12345678", "notes.txt")]


def test_project_txt_scanner_keeps_malformed_escape_binary_as_coverage_gap():
    payload = b"\x1b[bad" * 100 + b"\nSERVICE_API_KEY=realVALUE12345678\n"
    with pytest.raises(discovery.CredentialScanBinaryError):
        list(discovery.iter_project_findings([payload], "notes.txt", ExtractionStats()))


def test_project_non_txt_scanner_does_not_strip_ansi():
    payload = b"\x1b[31m" * 100 + b"SERVICE_API_KEY=realVALUE12345678\n"
    with pytest.raises(discovery.CredentialScanBinaryError):
        list(discovery.iter_project_findings([payload], "code.py", ExtractionStats()))


@pytest.mark.parametrize("payload", [
    b"\x00\x05\x16\x07" + b"\x00" * 4096,  # AppleDouble-like metadata
    b"\x03\x00\x08\x00" + b"\x00" * 4096,  # binary Android XML-like data
])
def test_project_scanner_classifies_binary_data_as_coverage_gap(payload):
    with pytest.raises(discovery.CredentialScanBinaryError):
        list(discovery.iter_project_findings([payload], "binary.xml", ExtractionStats()))


def test_project_scanner_rejects_binary_late_without_partial_vault_write(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "settings.txt").write_bytes(
        b"\x1b[31mSERVICE_API_KEY=realVALUE12345678\x1b[0m\n"
        + b"a" * 65536 + b"\x00" * 65536
    )
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    report = scan_project_files(root, store, passphrase="synthetic vault passphrase")
    assert report["complete"] is False
    assert report["error_categories"]["unsupported_binary"] == 1
    assert report["succeeded"] == 0
    assert store.search("SERVICE_API_KEY") == []
    assert "realVALUE12345678" not in str(report)


def test_project_txt_scanner_keeps_overlong_sgr_and_rejects_affected_assignment():
    payload = b"SERVICE_API_KEY=\x1b[" + b"1" * 100 + b"mrealVALUE12345678\n"
    assert b"".join(discovery._strip_sgr_chunks([payload])) == payload
    assert list(discovery.iter_project_findings([payload], "notes.txt", ExtractionStats())) == []


def test_windows_reparse_directory_is_not_followed_when_not_reported_as_junction(tmp_path, monkeypatch):
    target = tmp_path / "foreign-reparse"
    target.mkdir()
    (target / "hidden.py").write_text('HIDDEN_API_KEY = "hiddenVALUE12345678"\n')
    accessible = tmp_path / "accessible"
    accessible.mkdir()
    (accessible / "visible.py").write_text('VISIBLE_API_KEY = "visibleVALUE12345678"\n')
    original_lstat = discovery.os.lstat

    def lstat(path, *args, **kwargs):
        if str(path) == str(target):
            return SimpleNamespace(st_file_attributes=0x400)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(discovery.os, "lstat", lstat)
    assert discovery._is_link_or_junction(target) is True

    def walk(_root, *, followlinks, onerror):
        assert followlinks is False
        children = ["foreign-reparse", "accessible"]
        yield str(tmp_path), children, []
        assert children == ["accessible"]
        onerror(OSError("unrelated inaccessible directory"))
        yield str(accessible), [], ["visible.py"]

    monkeypatch.setattr(discovery.os, "walk", walk)
    store = RecordingStore()
    report = scan_project_files(tmp_path, store, passphrase="test-only")
    assert report["files"] == report["succeeded"] == 1
    assert report["walk_errors"] == report["errors"] == 1
    assert [name for name, _, _ in store.values] == ["VISIBLE_API_KEY"]


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
    assert report["error_categories"]["utf8"] == 1
    assert sum(report["error_categories"].values()) == report["errors"]
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


def test_walk_error_is_counted_and_accessible_siblings_continue(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "first.py").write_text('FIRST_API_KEY = "firstVALUE12345678"\n')
    child = root / "accessible"
    child.mkdir()
    (child / "second.py").write_text('SECOND_API_KEY = "secondVALUE12345678"\n')

    def walk(_root, *, followlinks, onerror):
        assert followlinks is False
        yield str(root), ["accessible"], ["first.py"]
        onerror(OSError("private inaccessible path or credential text"))
        yield str(child), [], ["second.py"]

    monkeypatch.setattr(discovery.os, "walk", walk)
    progress = []
    store = RecordingStore()
    report = scan_project_files(root, store, passphrase="test-only", progress=progress.append)

    assert report["files"] == 2
    assert report["succeeded"] == 2
    assert report["walk_errors"] == 1
    assert report["errors"] == 1
    assert report["error_categories"]["walk"] == 1
    assert sum(report["error_categories"].values()) == report["errors"]
    assert report["complete"] is False
    assert len(store.values) == 2
    assert "private inaccessible" not in str(report) + str(progress)


def test_root_walk_error_returns_incomplete_report(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()

    def walk(_root, *, followlinks, onerror):
        onerror(OSError("private root path"))
        return iter(())

    monkeypatch.setattr(discovery.os, "walk", walk)
    progress = []
    report = scan_project_files(root, RecordingStore(), passphrase="test-only",
                                progress=progress.append)

    assert report["files"] == 0
    assert report["walk_errors"] == 1
    assert report["errors"] == 1
    assert report["complete"] is False
    assert "private root path" not in str(report) + str(progress)


def test_missing_project_root_returns_incomplete_report(tmp_path):
    progress = []
    report = scan_project_files(tmp_path / "missing", RecordingStore(),
                                passphrase="test-only", progress=progress.append)
    assert report["files"] == 0
    assert report["errors"] == report["walk_errors"] == 1
    assert report["error_categories"]["root"] == 1
    assert sum(report["error_categories"].values()) == report["errors"]
    assert report["complete"] is False
    assert "missing" not in str(report) + str(progress)


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


def test_scan_coverage_does_not_claim_ambiguity_is_resolved(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("SERVICE_API_KEY=${UNKNOWN}\n")
    report = scan_project_env(root, RecordingStore(), passphrase="test-only")
    assert report["complete"] is True
    assert report["ambiguity_free"] is False
    assert report["ambiguous"] == 1
    assert report["ambiguous_reasons"] == {"unparsed_value": 1}
    assert "UNKNOWN" not in str(report)


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


def test_project_scan_keeps_long_unicode_location_and_label(tmp_path):
    root = tmp_path / "projects"
    project_name = "project with spaces α " + "x" * 70
    project = root / project_name
    (project / ".git").mkdir(parents=True)
    (project / ".env").write_text("SERVICE_API_KEY=aaaabbbbcccc11112222\n")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")

    report = scan_project_files(root, store, passphrase="synthetic vault passphrase")

    assert report["complete"] is True
    matches = store.search("SERVICE_API_KEY")
    assert matches[0]["project"] == project_name
    assert matches[0]["source_hint"] == f"{project_name}/.env"


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
    assert report["error_categories"]["path"] == 1
    assert report["complete"] is False
    assert store.values == []


@pytest.mark.parametrize("failure, category", [
    (discovery.CredentialScanSourceChangedError("private source"), "source_changed"),
    (OSError("private path"), "io"),
    (RuntimeError("private internal detail"), "other"),
    (TypeError("private data shape"), "other"),
    (KeyError("private missing field"), "other"),
])
def test_project_scan_error_categories_hide_exception_text(tmp_path, failure, category):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("SERVICE_API_KEY=aaaabbbbcccc11112222\n")

    class FailingStore:
        def scan_source(self, **_kwargs):
            raise failure

    report = scan_project_env(root, FailingStore(), passphrase="synthetic vault passphrase")
    assert report["error_categories"][category] == 1
    assert sum(report["error_categories"].values()) == report["errors"] == 1
    assert "private" not in str(report)


def test_project_invalid_label_is_metadata_category(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("SERVICE_API_KEY=aaaabbbbcccc11112222\n")
    monkeypatch.setattr(discovery, "_valid_project_label", lambda _name: False)

    report = scan_project_env(root, RecordingStore(), passphrase="synthetic vault passphrase")
    assert report["error_categories"]["metadata"] == 1
    assert sum(report["error_categories"].values()) == report["errors"] == 1


def test_project_type_error_continues_to_accessible_sibling(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("FIRST_API_KEY=aaaabbbbcccc11112222\n")
    (root / "settings.json").write_text('SECOND_API_KEY="bbbbcccc111122223333"\n')

    class OneBadStore(RecordingStore):
        def scan_source(self, *, findings, **kwargs):
            if kwargs["source_hash"] == discovery.source_fingerprint(str((root / ".env").resolve())):
                raise TypeError("private data shape")
            return super().scan_source(findings=findings, **kwargs)

    store = OneBadStore()
    report = scan_project_files(root, store, passphrase="synthetic vault passphrase")
    assert report["files"] == 2
    assert report["succeeded"] == 1
    assert report["error_categories"]["other"] == 1
    assert sum(report["error_categories"].values()) == report["errors"] == 1
    assert "private data shape" not in str(report)


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
    assert report["error_categories"]["archive_integrity"] == 1
    assert sum(report["error_categories"].values()) == report["errors"]
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
    assert first["ambiguity_free"] is True
    assert first["inserted"] == 1
    second = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert second["inserted"] == 0
    assert second["skipped"] == 1
    assert second["snapshots_not_evaluated"] == 1
    assert second["ambiguity_free"] is False
    matches = store.search("SERVICE_API_KEY")
    assert len(matches) == 1
    assert matches[0]["origin"] == "transcript"
    assert "aaaabbbbcccc11112222" not in str(matches)
    with pytest.raises(ValueError, match="generation"):
        scan_archive(archive, store, passphrase="synthetic vault passphrase", offset=1)
    with pytest.raises(ValueError, match="generation changed"):
        scan_archive(archive, store, passphrase="synthetic vault passphrase",
                     offset=1, expected_generation=first["generation"] + 1)


def test_archive_ambiguity_enters_encrypted_review_queue(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=${MISSING_REFERENCE}"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")

    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")

    assert report["complete"] is True
    assert report["ambiguity_free"] is False
    assert report["queued"] == 1
    assert store.ambiguity_status() == {"pending": 1}
    assert "MISSING_REFERENCE" not in store.db_path.read_bytes().decode("utf-8", errors="ignore")
    assert "MISSING_REFERENCE" not in str(store.list_ambiguities())


def test_leading_underscore_secret_name_can_commit_to_vault(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text("_API_KEY=aaaabbbbcccc11112222\n")
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")

    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")

    assert report["complete"] is True
    assert report["inserted"] == 1
    assert store.search("_API_KEY")[0]["service"] == "_API_KEY"


def test_archive_utf8_failure_reports_only_category_and_rolls_back(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_bytes(b"SERVICE_API_KEY=aaaabbbbcccc11112222\n\xff")
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")

    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")

    assert report["error_categories"]["utf8"] == 1
    assert sum(report["error_categories"].values()) == report["errors"] == 1
    assert report["complete"] is False
    assert store.search("SERVICE_API_KEY") == []
    assert "aaaabbbbcccc11112222" not in str(report)


@pytest.mark.parametrize("failure, category", [
    (OSError("private path"), "io"),
    (VaultIntegrityError("private vault detail"), "vault"),
    (TypeError("private internal detail"), "other"),
])
def test_archive_scan_error_categories_hide_exception_text(tmp_path, failure, category):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text("No credential assignment here\n")
    archive.archive_file(source, "codex")

    class FailingStore:
        def scan_source(self, **_kwargs):
            raise failure

    progress = []
    report = scan_archive(archive, FailingStore(), passphrase="synthetic vault passphrase",
                          progress=progress.append)
    assert report["error_categories"][category] == 1
    assert sum(report["error_categories"].values()) == report["errors"] == 1
    assert progress[-1]["error_categories"][category] == 1
    assert "private" not in str(report) + str(progress)


def test_archive_invalid_provider_is_metadata_category(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text("No credential assignment here\n")
    archive.archive_file(source, "codex")
    original = archive._load_manifest

    def bad_provider():
        manifest = original()
        files = {name: [{**entry, "provider": "bad/path"} for entry in entries]
                 for name, entries in manifest["files"].items()}
        return {**manifest, "files": files}

    monkeypatch.setattr(archive, "_load_manifest", bad_provider)
    report = scan_archive(archive, RecordingStore(), passphrase="synthetic vault passphrase")
    assert report["error_categories"]["metadata"] == 1
    assert sum(report["error_categories"].values()) == report["errors"] == 1
    assert "bad/path" not in str(report)


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
