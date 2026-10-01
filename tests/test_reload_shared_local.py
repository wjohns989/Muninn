"""Operator procedure controls using isolated fakes, never live lifecycle."""
import pytest
import sqlite3
import sys
from types import SimpleNamespace

from scripts import reload_shared_local as reload
from scripts import local_runtime_preflight as preflight


def test_default_is_read_only_and_enable_requires_restart():
    args = reload.parse_args([])
    assert not args.restart and not args.enable_capture_auto
    with pytest.raises(SystemExit):
        reload.parse_args(["--enable-capture-auto"])


def test_enable_changes_only_two_copied_environment_flags():
    original = {"MUNINN_AUTH_TOKEN": "test-only-auth", "MUNINN_PORT": "42069",
                "MUNINN_CAPTURE_ENRICHMENT": "0", "MUNINN_CAPTURE_AUTO_ANALYSIS": "0",
                "MUNINN_OLLAMA_MODEL": "test-model"}
    updated = reload.launch_environment(original, enable_capture_auto=True)
    assert original["MUNINN_CAPTURE_AUTO_ANALYSIS"] == "0"
    assert updated["MUNINN_CAPTURE_ENRICHMENT"] == updated["MUNINN_CAPTURE_AUTO_ANALYSIS"] == "1"
    assert {key: value for key, value in updated.items() if key not in reload.CAPTURE_FLAGS} == {
        key: value for key, value in original.items() if key not in reload.CAPTURE_FLAGS}
    assert reload.launch_environment(original, enable_capture_auto=False) == original


@pytest.mark.parametrize("value", ["api-key-private-canary", 1, "banana", "true\nsecret"])
def test_only_nonsecret_flag_values_can_enter_preimages(value):
    with pytest.raises(ValueError):
        reload.validate_flag_value(value)


@pytest.mark.parametrize("value", [None, "0", "1", " false ", "YES"])
def test_valid_flag_preimage_is_preserved_exactly(value):
    assert reload.validate_flag_value(value) == value


def fake_registry(values, *, fail_on=None, after_write=False):
    state = dict(values)
    def read(name):
        return state.get(name)
    def write(name, value):
        if name == fail_on and not after_write:
            raise OSError("test-only write failure")
        if value is None:
            state.pop(name, None)
        else:
            state[name] = value
        if name == fail_on and after_write and value == "1":
            raise OSError("test-only interruption after write")
    return state, read, write


def test_user_flags_persist_together_and_remove_absent_preimage_on_rollback():
    first, second = reload.CAPTURE_FLAGS
    before = {first: None, second: "0"}
    state, read, write = fake_registry(before)
    reload.persist_capture_flags(before, read=read, write=write)
    assert state == {first: "1", second: "1"}


@pytest.mark.parametrize("after_write", [False, True])
def test_partial_registry_failure_compensates_prior_writes(after_write):
    first, second = reload.CAPTURE_FLAGS
    before = {first: None, second: "0"}
    state, read, write = fake_registry(before, fail_on=second, after_write=after_write)
    with pytest.raises(reload.PersistenceError):
        reload.persist_capture_flags(before, read=read, write=write)
    assert first not in state
    assert state[second] == "0"


def test_concurrent_user_revocation_is_preserved_during_compensation():
    first, second = reload.CAPTURE_FLAGS
    before = {first: "0", second: "0"}
    state = dict(before)
    def write(name, value):
        if name == second:
            state[first] = "0"  # The user revoked after our first write.
            raise OSError("test-only interruption")
        state[name] = value
    with pytest.raises(reload.PersistenceError):
        reload.persist_capture_flags(before, read=lambda name: state.get(name), write=write)
    assert state == before


def test_changed_user_preimage_rejects_persistence_without_overwrite():
    first, second = reload.CAPTURE_FLAGS
    before = {first: "0", second: "0"}
    state, read, write = fake_registry({first: "0", second: "1"})
    with pytest.raises(reload.PersistenceError):
        reload.persist_capture_flags(before, read=read, write=write)
    assert state == {first: "0", second: "1"}


def test_enable_mode_requires_authenticated_effective_mode():
    def report(enabled, remote=False):
        return {"capture_enrichment": {"capture_enabled": enabled,
                "automatic_analysis_enabled": enabled, "automatic_remote_enabled": remote}}
    assert reload.mode_matches(report(True), enable=True)
    assert not reload.mode_matches(report(False), enable=True)
    assert not reload.mode_matches(report(True, remote=True), enable=True)
    assert reload.mode_matches(report(False), enable=False)
    assert not reload.mode_matches({}, enable=True)


def test_idle_queue_rejects_active_or_claimable_jobs():
    idle = {table: {} for table in reload.TABLES}
    assert reload.queues_idle(idle)
    assert not reload.queues_idle({**idle, "history_analysis_jobs": {"publishing": 1}})
    assert not reload.queues_idle({**idle, "jobs": {"retry": 1}})


@pytest.mark.parametrize("report", [{}, {"capture_enrichment": None}])
def test_legacy_status_requires_owned_process_flags_off(report):
    assert not reload.mode_matches(report, enable=False)
    assert reload.pre_reload_mode_matches(report, {})
    assert reload.pre_reload_mode_matches(report, {name: "false" for name in reload.CAPTURE_FLAGS})
    for name in reload.CAPTURE_FLAGS:
        for value in ("1", "true", "banana", "private-nonboolean", 0):
            assert not reload.pre_reload_mode_matches(report, {name: value})


def test_legacy_mode_does_not_relax_post_reload_verification():
    assert not reload.mode_matches({"capture_enrichment": None}, enable=True)
    assert not reload.pre_reload_mode_matches({"capture_enrichment": []}, {})
    assert not reload.pre_reload_mode_matches({"capture_enrichment": {"capture_enabled": True}}, {})
    assert not reload.pre_reload_mode_matches({"capture_enrichment": {"capture_enabled": False}},
                                            {reload.CAPTURE_FLAGS[0]: "1"})


def isolated_installation(tmp_path, monkeypatch):
    archive = tmp_path / "history_secure_archive"
    archive.mkdir()
    with sqlite3.connect(archive / "capture-jobs.db") as db:
        for table in reload.TABLES:
            db.execute(f"CREATE TABLE {table}(state TEXT)")
    report = {"anonymous_protected_http": 401, "authenticated_protected_http": 200,
              "history_http": 200, "archive_ready": True, "history_security": "strict",
              "anonymous_root_contains_token": False, "listener_owners": [123],
              "muninn_processes": [{"pid": 123}],
              "capture_enrichment": {"capture_enabled": False}}
    process = SimpleNamespace(exe=lambda: sys.executable, cwd=lambda: str(tmp_path),
        create_time=lambda: 1.0, cmdline=lambda: [sys.executable, str(tmp_path / "server.py")],
        environ=lambda: {"MUNINN_NO_AUTH": "0", "MUNINN_AUTH_TOKEN": "fixture-only",
                        "MUNINN_HOST": "127.0.0.1", "MUNINN_PORT": "42069",
                        "MUNINN_HISTORY_ARCHIVE_DIR": str(archive)})
    monkeypatch.setattr(reload, "inspect_runtime", lambda *args, **kwargs: report)
    monkeypatch.setattr(reload.psutil, "Process", lambda pid: process)
    return report, process


def test_read_only_run_never_prepares_destinations_or_changes_processes(tmp_path, monkeypatch):
    isolated_installation(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("read-only inspection attempted mutation")
    monkeypatch.setattr(reload, "create_private_directory", forbidden)
    monkeypatch.setattr(reload.subprocess, "Popen", forbidden)
    monkeypatch.setattr(reload, "persist_capture_flags", forbidden)
    args = reload.parse_args(["--repo", str(tmp_path)])
    reload.run(args)
    assert not (tmp_path / ".muninn_runtime").exists()


def test_dirty_candidate_rejected_before_backup_stop_or_settings(tmp_path, monkeypatch):
    if reload.os.name != "nt":
        pytest.skip("Windows reload path")
    isolated_installation(tmp_path, monkeypatch)
    def dirty(*args):
        raise RuntimeError("Program source differs from approved revision")
    def forbidden(*args, **kwargs):
        pytest.fail("candidate rejection happened after mutation")
    monkeypatch.setattr(reload, "verify_candidate", dirty)
    monkeypatch.setattr(reload, "create_private_directory", forbidden)
    monkeypatch.setattr(reload.subprocess, "Popen", forbidden)
    monkeypatch.setattr(reload, "read_user_flag", forbidden)
    args = reload.parse_args(["--repo", str(tmp_path), "--restart",
                              "--enable-capture-auto", "--expected-revision", "abcdef0"])
    with pytest.raises(RuntimeError, match="source differs"):
        reload.run(args)


@pytest.mark.parametrize("revision", [None, "HEAD", "main", "abcdef0;echo", "123"])
def test_candidate_hash_validated_before_git_execution(tmp_path, monkeypatch, revision):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid commit input reached subprocess")
    monkeypatch.setattr(reload.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        reload.verify_candidate(tmp_path, revision)


@pytest.mark.parametrize("port", [True, 0, 65536, "42069"])
def test_invalid_preflight_port_rejected_before_process_or_network_reads(tmp_path, monkeypatch, port):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid port reached process inspection")
    monkeypatch.setattr(preflight.psutil, "process_iter", forbidden)
    with pytest.raises(ValueError, match="port"):
        preflight.inspect_runtime(tmp_path, port=port)


def test_preflight_uses_selected_port_for_listener_and_http(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.psutil, "process_iter", lambda *args: [])
    monkeypatch.setattr(preflight.psutil, "net_connections", lambda **kwargs: [
        SimpleNamespace(pid=123, laddr=SimpleNamespace(port=43123), status=preflight.psutil.CONN_LISTEN),
        SimpleNamespace(pid=456, laddr=SimpleNamespace(port=42069), status=preflight.psutil.CONN_LISTEN)])
    requested = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def get(self, url):
            requested.append(url)
            return SimpleNamespace(status_code=200)
    monkeypatch.setattr(preflight.httpx, "Client", Client)
    report = preflight.inspect_runtime(tmp_path, port=43123)
    assert report["listener_owners"] == [123]
    assert requested == ["http://127.0.0.1:43123/health"]


def test_untracked_runtime_files_reject_even_with_clean_tracked_diff(tmp_path, monkeypatch):
    results = iter([SimpleNamespace(stdout="abcdef012345\n"), SimpleNamespace(returncode=0),
                    SimpleNamespace(stdout=b"muninn/local_unreviewed.py\0")])
    monkeypatch.setattr(reload.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="Untracked program"):
        reload.verify_candidate(tmp_path, "abcdef0")


@pytest.mark.parametrize("source", [b"ignored_shadow.py", b"sourceless_shadow.pyc", b"shadow.pyd"])
def test_ignored_runtime_source_is_included_but_generated_cache_is_not(tmp_path, monkeypatch, source):
    def run(command, **kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout="abcdef012345\n")
        if "diff" in command:
            return SimpleNamespace(returncode=0)
        assert "--exclude-standard" not in command
        return SimpleNamespace(stdout=b"muninn/" + source + b"\0muninn/__pycache__/derived.pyc\0")
    monkeypatch.setattr(reload.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Untracked program"):
        reload.verify_candidate(tmp_path, "abcdef0")


def test_generated_cache_alone_does_not_block_candidate(tmp_path, monkeypatch):
    results = iter([SimpleNamespace(stdout="abcdef012345\n"), SimpleNamespace(returncode=0),
                    SimpleNamespace(stdout=b"muninn/__pycache__/derived.pyc\0")])
    monkeypatch.setattr(reload.subprocess, "run", lambda *args, **kwargs: next(results))
    reload.verify_candidate(tmp_path, "abcdef0")


@pytest.mark.parametrize("wrong", ["birth", "command", "listener", "binding"])
def test_stop_fence_rejects_changed_identity_or_listener(tmp_path, monkeypatch, wrong):
    process = SimpleNamespace(pid=123, is_running=lambda: True,
        create_time=lambda: 2.0 if wrong == "birth" else 1.0,
        cmdline=lambda: ["changed"] if wrong == "command" else [sys.executable, "server.py"],
        exe=lambda: sys.executable, cwd=lambda: str(tmp_path))
    monkeypatch.setattr(reload.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(reload.psutil, "net_connections", lambda **kwargs: [SimpleNamespace(
        pid=456 if wrong == "listener" else 123, status=reload.psutil.CONN_LISTEN,
        laddr=SimpleNamespace(port=42069, ip="0.0.0.0" if wrong == "binding" else "127.0.0.1"))])
    with pytest.raises(RuntimeError):
        reload.verify_stop_ownership(process, (1.0, (sys.executable, "server.py")),
                                     repo=tmp_path, port=42069)


def test_linked_ancestor_rejected_before_resolution(tmp_path, monkeypatch):
    from pathlib import Path
    import stat
    original = Path.lstat
    linked = tmp_path / "linked-parent"
    def lstat(path, *args, **kwargs):
        if path == linked:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(RuntimeError, match="ancestor"):
        reload.require_unlinked_path(linked / "missing-child")
