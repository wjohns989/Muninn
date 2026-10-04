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


def test_preserve_mode_requires_restart_and_cannot_activate_or_finalize():
    with pytest.raises(SystemExit):
        reload.parse_args(["--preserve-capture-auto"])
    for conflicting in ("--enable-capture-auto", "--finalize-capture-auto"):
        with pytest.raises(SystemExit):
            reload.parse_args(["--restart", "--preserve-capture-auto", conflicting,
                               "--expected-revision", "abcdef0"])
    args = reload.parse_args(["--restart", "--preserve-capture-auto",
                              "--expected-revision", "abcdef0"])
    assert args.preserve_capture_auto and not args.enable_capture_auto


def test_remote_capture_opt_in_requires_owned_preserve_reload():
    with pytest.raises(SystemExit):
        reload.parse_args(["--enable-capture-remote"])
    with pytest.raises(SystemExit):
        reload.parse_args(["--restart", "--enable-capture-remote",
                           "--expected-revision", "abcdef0"])
    args = reload.parse_args(["--restart", "--preserve-capture-auto",
                              "--enable-capture-remote", "--expected-revision", "abcdef0"])
    assert args.enable_capture_remote


def test_shared_launcher_reads_persisted_remote_capture_flag():
    launcher = (reload.REPO_DEFAULT / "scripts" / "start_shared_local.ps1").read_text(encoding="utf-8")
    assert '"MUNINN_CAPTURE_AUTO_REMOTE"' in launcher
    assert 'GetEnvironmentVariable($name, "User")' in launcher


def test_analysis_preflight_reports_only_nonsecret_problem_counts(tmp_path, monkeypatch):
    from muninn.history import auto_routing

    archive = tmp_path / "history_secure_archive"
    archive.mkdir()
    with sqlite3.connect(archive / "capture-jobs.db") as db:
        db.execute("CREATE TABLE history_analysis_jobs(state TEXT, error_code TEXT)")
        db.executemany("INSERT INTO history_analysis_jobs VALUES(?,?)", [
            ("failed", "private diagnostic"), ("failed", "another private diagnostic"),
            ("retry", "retry code"), ("succeeded", ""),
        ])
    monkeypatch.setattr(auto_routing, "_local_setting", lambda name: (
        str(archive) if name == "MUNINN_HISTORY_ARCHIVE_DIR" else None))
    assert preflight.inspect_analysis_errors(tmp_path) == [
        {"state": "failed", "count": 2}, {"state": "retry", "count": 1}]


def test_reload_failure_code_never_exposes_unreviewed_exception_text():
    assert reload.safe_failure_code(RuntimeError("Candidate ownership differs")) == "candidate_ownership"
    assert reload.safe_failure_code(RuntimeError("private path or token")) == "unspecified"
    assert reload.safe_failure_code(ValueError("Candidate ownership differs")) == "unspecified"
    assert reload.safe_failure_code(reload.PersistenceError("Candidate ownership differs")) == "unspecified"


def test_candidate_ownership_allows_only_the_just_retired_process():
    report = {"listener_owners": [22], "muninn_processes": [
        {"pid": 11, "create_time": 1.0}, {"pid": 22, "create_time": 2.0},
    ]}
    assert reload.candidate_ownership_matches(report, child_pid=22, retired_pid=11,
                                              retired_created_at=1.0)
    assert not reload.candidate_ownership_matches({**report, "listener_owners": [11]},
                                                  child_pid=22, retired_pid=11,
                                                  retired_created_at=1.0)
    assert not reload.candidate_ownership_matches({**report, "muninn_processes": [
        {"pid": 11, "create_time": 1.0}, {"pid": 22, "create_time": 2.0},
        {"pid": 33, "create_time": 3.0}]}, child_pid=22, retired_pid=11,
                                                  retired_created_at=1.0)
    assert not reload.candidate_ownership_matches({**report, "muninn_processes": [
        {"pid": 11, "create_time": 1.0}]}, child_pid=22, retired_pid=11,
                                                  retired_created_at=1.0)
    assert not reload.candidate_ownership_matches({**report, "muninn_processes": [
        {"pid": 11, "create_time": 3.0}, {"pid": 22, "create_time": 2.0}]},
        child_pid=22, retired_pid=11, retired_created_at=1.0)
    assert reload.candidate_ownership_matches({"listener_owners": [11],
        "muninn_processes": [{"pid": 11, "create_time": 2.0}]},
        child_pid=11, retired_pid=11, retired_created_at=1.0)


def test_startup_waits_for_missing_listener_but_not_a_foreign_owner():
    assert reload.startup_listener_pending({"listener_owners": []})
    assert reload.startup_listener_pending({"listener_owners": None})
    assert not reload.startup_listener_pending({"listener_owners": [33]})


@pytest.mark.parametrize("bad", ["missing_report", "remote", "flag_missing", "flag_false", "flag_invalid"])
def test_preserve_preflight_requires_modern_local_mode_and_owned_true_flags(bad):
    report = {"capture_enrichment": {"capture_enabled": True,
        "automatic_analysis_enabled": True, "automatic_remote_enabled": False}}
    environment = {name: "1" for name in reload.CAPTURE_FLAGS}
    assert reload.pre_reload_mode_matches(report, environment, preserve_capture_auto=True)
    if bad == "missing_report":
        report = {}
    elif bad == "remote":
        report["capture_enrichment"]["automatic_remote_enabled"] = True
    else:
        environment[reload.CAPTURE_FLAGS[0]] = {
            "flag_missing": None, "flag_false": "0", "flag_invalid": "not-a-boolean"}[bad]
    assert not reload.pre_reload_mode_matches(report, environment, preserve_capture_auto=True)


def test_preserve_preflight_accepts_matching_owned_remote_mode():
    report = {"capture_enrichment": {"capture_enabled": True,
        "automatic_analysis_enabled": True, "automatic_remote_enabled": True}}
    environment = {**{name: "1" for name in reload.CAPTURE_FLAGS}, reload.REMOTE_FLAG: "1"}
    assert reload.pre_reload_mode_matches(report, environment, preserve_capture_auto=True)
    environment[reload.REMOTE_FLAG] = "0"
    assert not reload.pre_reload_mode_matches(report, environment, preserve_capture_auto=True)


def test_preserve_queues_allow_only_unclaimed_durable_work():
    queued = {table: {"pending": 2, "retry": 1} for table in reload.TABLES}
    queued["history_analysis_jobs"]["publication_pending"] = 1
    assert not reload.queues_idle(queued)
    assert reload.queues_idle(queued, preserve_queued=True)
    for table, states in reload.ACTIVE.items():
        for state in states - {"pending", "retry", "publication_pending"}:
            assert not reload.queues_idle({**queued, table: {state: 1}}, preserve_queued=True)


def test_preserve_rejects_publication_pending_with_active_lease(tmp_path):
    with sqlite3.connect(tmp_path / "jobs.db") as db:
        db.execute("CREATE TABLE history_analysis_jobs(state TEXT, lease_token TEXT, lease_until REAL)")
        db.execute("INSERT INTO history_analysis_jobs VALUES('publication_pending',NULL,NULL)")
        assert reload.publication_queue_unclaimed(db)
        db.execute("INSERT INTO history_analysis_jobs VALUES('publication_pending','claim',123)")
        assert not reload.publication_queue_unclaimed(db)


@pytest.mark.skipif(reload.os.name != "nt", reason="Windows owned reload procedure")
@pytest.mark.parametrize("claim_race,foreign_owner,backup_failure", [
    (False, False, False), (True, False, False), (False, True, False),
    (False, False, True)])
@pytest.mark.parametrize("remote_enabled", [False, True])
def test_preserve_run_retains_queued_rows_and_rejects_claim_before_stop(
        tmp_path, monkeypatch, claim_race, foreign_owner, backup_failure, remote_enabled):
    report, process = isolated_installation(tmp_path, monkeypatch)
    report["capture_enrichment"] = {"capture_enabled": True,
        "automatic_analysis_enabled": True, "automatic_remote_enabled": remote_enabled}
    original = process.environ()
    environment = {**original, **{name: "1" for name in reload.CAPTURE_FLAGS},
                   reload.REMOTE_FLAG: "1" if remote_enabled else "0"}
    process.environ = lambda: dict(environment)
    journal = tmp_path / "history_secure_archive" / "capture-jobs.db"
    with sqlite3.connect(journal) as db:
        for table in reload.TABLES:
            db.execute(f"INSERT INTO {table}(state) VALUES('pending')")
        db.execute("INSERT INTO history_analysis_jobs(state) VALUES('publication_pending')")
        # Candidate startup also checks these schema fields; no private content.
        for column in ("sealed_window", "sealed_extraction", "extraction_id", "sealed_receipt",
                       "sealed_reuse", "cancel_requested", "publication_started"):
            db.execute(f"ALTER TABLE history_analysis_jobs ADD COLUMN {column}")
    stopped, launched = [], []
    def stop():
        # An independent claim writer cannot pass the held stop fence.
        with sqlite3.connect(journal, timeout=0) as competing:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing.execute("UPDATE jobs SET state='capturing'")
        stopped.append(True)

    process.terminate = stop
    process.wait = lambda **kwargs: None
    stopped_child = []
    child = SimpleNamespace(pid=456, poll=lambda: None,
                            terminate=lambda: stopped_child.append(True),
                            wait=lambda **kwargs: None)

    def start(command, **kwargs):
        launched.append(kwargs["env"])
        report["listener_owners"] = [456]
        report["muninn_processes"] = [{"pid": 123, "create_time": 1.0},
                                      {"pid": 456, "create_time": 2.0}]
        return child

    monkeypatch.setattr(reload.subprocess, "Popen", start)
    poststart_checks = []
    def inspect(*args, **kwargs):
        if launched and not poststart_checks:
            poststart_checks.append(True)
            if foreign_owner:
                return {**report, "listener_owners": [999], "history_http": 503}
            return {**report, "listener_owners": []}
        return report
    monkeypatch.setattr(reload, "inspect_runtime", inspect)
    monkeypatch.setattr(reload, "verify_private", lambda *args: None)
    monkeypatch.setattr(reload, "verify_stop_ownership", lambda *args, **kwargs: None)
    created_preimages_under_fence = []
    create_private_file = reload.create_private_file
    def traced_create_private_file(path):
        create_private_file(path)
        if path.name == "capture-jobs.db":
            with sqlite3.connect(journal, timeout=0) as competing:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    competing.execute("UPDATE jobs SET state='capturing'")
            created_preimages_under_fence.append(True)
            if backup_failure:
                raise OSError("fixture preimage failure")
    monkeypatch.setattr(reload, "create_private_file", traced_create_private_file)
    candidate_checks = []

    def candidate(*args):
        candidate_checks.append(True)
        if claim_race and len(candidate_checks) == 2:
            with sqlite3.connect(journal) as db:
                db.execute("UPDATE history_analysis_jobs SET state='running'")

    monkeypatch.setattr(reload, "verify_candidate", candidate)

    def forbidden(*args, **kwargs):
        pytest.fail("Preservation must not read or write registry settings")

    monkeypatch.setattr(reload, "read_user_flag", forbidden)
    monkeypatch.setattr(reload, "persist_capture_flags", forbidden)
    args = reload.parse_args(["--repo", str(tmp_path), "--restart", "--preserve-capture-auto",
                              "--expected-revision", "abcdef0"])
    if claim_race:
        with pytest.raises(RuntimeError, match="Queue changed"):
            reload.run(args)
        assert stopped == launched == []
    elif backup_failure:
        with pytest.raises(OSError, match="fixture preimage failure"):
            reload.run(args)
        assert stopped == launched == []
        with sqlite3.connect(journal, timeout=0) as db:
            db.execute("UPDATE jobs SET state='retry'")
    elif foreign_owner:
        with pytest.raises(RuntimeError, match="Candidate ownership differs"):
            reload.run(args)
        assert stopped == [True] and launched == [environment]
        assert stopped_child == ([True] if remote_enabled else [])
    else:
        reload.run(args)
        assert created_preimages_under_fence == [True]
        assert stopped == [True] and launched == [environment]
        assert stopped_child == []
        with sqlite3.connect(journal) as db:
            expected = {table: {"pending": 1} for table in reload.TABLES}
            expected["history_analysis_jobs"]["publication_pending"] = 1
            assert reload.queue_states(db) == expected


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


def test_remote_opt_in_changes_only_remote_flag_in_copied_environment():
    original = {"MUNINN_AUTH_TOKEN": "test-only-auth", reload.REMOTE_FLAG: "0"}
    updated = reload.launch_environment(original, enable_capture_auto=False,
                                        enable_capture_remote=True)
    assert original[reload.REMOTE_FLAG] == "0"
    assert updated == {**original, reload.REMOTE_FLAG: "1"}


def test_backlog_drain_is_explicit_and_expiry_is_only_in_child_environment(monkeypatch):
    for minutes in ("0", "181"):
        with pytest.raises(SystemExit):
            reload.parse_args(["--restart", "--preserve-capture-auto", "--enable-capture-remote",
                               "--expected-revision", "abcdef0", "--backlog-drain-minutes", minutes])
    with pytest.raises(SystemExit):
        reload.parse_args(["--restart", "--preserve-capture-auto", "--expected-revision", "abcdef0",
                           "--backlog-drain-minutes", "60"])
    args = reload.parse_args(["--restart", "--preserve-capture-auto", "--enable-capture-remote",
                              "--expected-revision", "abcdef0", "--backlog-drain-minutes", "60"])
    monkeypatch.setattr(reload.time, "time", lambda: 100.0)
    original = {reload.REMOTE_FLAG: "1"}
    updated = reload.launch_environment(original, enable_capture_auto=False,
                                        enable_capture_remote=True,
                                        backlog_drain_minutes=args.backlog_drain_minutes)
    assert updated["MUNINN_CAPTURE_BACKLOG_DRAIN_UNTIL"] == "3700.0"
    assert original == {reload.REMOTE_FLAG: "1"}


def test_remote_flag_persistence_preserves_preimage_and_compensates_failure():
    before = None
    state, read, write = fake_registry({})
    reload.persist_remote_flag(before, read=read, write=write)
    assert state[reload.REMOTE_FLAG] == "1"
    state, read, write = fake_registry({}, fail_on=reload.REMOTE_FLAG, after_write=True)
    with pytest.raises(reload.PersistenceError):
        reload.persist_remote_flag(before, read=read, write=write)
    assert reload.REMOTE_FLAG not in state


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
            if table == "history_analysis_jobs":
                db.execute(f"CREATE TABLE {table}(state TEXT, lease_token TEXT, lease_until REAL)")
            else:
                db.execute(f"CREATE TABLE {table}(state TEXT)")
    report = {"anonymous_protected_http": 401, "authenticated_protected_http": 200,
              "history_http": 200, "archive_ready": True, "history_security": "strict",
              "anonymous_root_contains_token": False, "listener_owners": [123],
              "muninn_processes": [{"pid": 123}],
              "capture_enrichment": {"capture_enabled": False}}
    process = SimpleNamespace(pid=123, exe=lambda: sys.executable, cwd=lambda: str(tmp_path),
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


def test_preimages_use_private_archive_not_public_runtime_parent(tmp_path):
    from muninn.history.private_acl import create_private_directory, verify_private

    archive = tmp_path / "archive"
    create_private_directory(archive)
    destination = reload.prepare_preimage_destination(archive)
    assert destination.parent == archive / "operator-preimages"
    verify_private(destination.parent)
    verify_private(destination)


def test_preimage_database_selection_never_recursively_copies_preimages(tmp_path):
    from muninn.history.private_acl import create_private_directory, create_private_file

    archive = tmp_path / "archive"
    create_private_directory(archive)
    journal = archive / "capture-jobs.db"
    create_private_file(journal)
    evidence = archive / "source-evidence"
    create_private_directory(evidence)
    database = evidence / "projections.sqlite3"
    create_private_file(database)
    previous = reload.prepare_preimage_destination(archive)
    create_private_file(previous / "old.sqlite3")
    assert reload.preimage_databases(archive, journal) == [journal, database]


def test_finalize_requires_candidate_and_preimage_without_restart():
    with pytest.raises(SystemExit):
        reload.parse_args(["--finalize-capture-auto"])
    with pytest.raises(SystemExit):
        reload.parse_args(["--restart", "--finalize-capture-auto", "--expected-revision", "abcdef0",
                           "--preimage-root", "fixture"])
    args = reload.parse_args(["--finalize-capture-auto", "--expected-revision", "abcdef0",
                              "--preimage-root", "fixture"])
    assert args.finalize_capture_auto and not args.restart


def test_flag_preimage_is_private_bounded_and_from_archive_namespace(tmp_path):
    import json
    from muninn.history.private_acl import create_private_directory, create_private_file

    archive = tmp_path / "archive"
    create_private_directory(archive)
    preimage = reload.prepare_preimage_destination(archive)
    flags = preimage / "capture-user-flags.json"
    create_private_file(flags)
    before = {name: None for name in reload.CAPTURE_FLAGS}
    flags.write_text(json.dumps(before))
    assert reload.load_capture_flag_preimage(archive, preimage) == before
    with pytest.raises(RuntimeError, match="namespace"):
        reload.load_capture_flag_preimage(archive, tmp_path)
    flags.write_text(json.dumps({**before, "unrelated": "0"}))
    with pytest.raises(RuntimeError, match="Incomplete"):
        reload.load_capture_flag_preimage(archive, preimage)
    flags.write_text(" " * 1025)
    with pytest.raises(RuntimeError, match="bounded"):
        reload.load_capture_flag_preimage(archive, preimage)


@pytest.mark.parametrize("report,pending", [({}, True), ({"capture_enrichment": None}, True),
    ({"capture_enrichment": {}}, False), ({"capture_enrichment": {"capture_enabled": False}}, False),
    ({"capture_enrichment": []}, False)])
def test_startup_waits_only_for_missing_mode_not_contradictory_mode(report, pending):
    assert reload.startup_mode_pending(report) is pending


def test_finalize_never_spawns_or_stops_and_uses_saved_compare_preimage(tmp_path, monkeypatch):
    report, process = isolated_installation(tmp_path, monkeypatch)
    report["capture_enrichment"] = {"capture_enabled": True,
        "automatic_analysis_enabled": True, "automatic_remote_enabled": False}
    original = process.environ
    process.environ = lambda: {**original(), **{name: "1" for name in reload.CAPTURE_FLAGS}}
    saved = {name: None for name in reload.CAPTURE_FLAGS}
    writes = []
    monkeypatch.setattr(reload, "verify_candidate", lambda *args: None)
    monkeypatch.setattr(reload, "verify_stop_ownership", lambda *args, **kwargs: None)
    monkeypatch.setattr(reload, "load_capture_flag_preimage", lambda *args: saved)
    monkeypatch.setattr(reload, "persist_capture_flags", lambda before: writes.append(before))
    def forbidden(*args, **kwargs):
        pytest.fail("finalization attempted lifecycle or preimage mutation")
    monkeypatch.setattr(reload.subprocess, "Popen", forbidden)
    monkeypatch.setattr(reload, "prepare_preimage_destination", forbidden)
    args = reload.parse_args(["--repo", str(tmp_path), "--finalize-capture-auto",
        "--expected-revision", "abcdef0", "--preimage-root", str(tmp_path / "fixture")])
    if reload.os.name == "nt":
        reload.run(args)
        assert writes == [saved]
