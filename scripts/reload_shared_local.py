"""Explicit Windows operator reload; defaults to read-only inspection.

No process identifiers or environment secrets are persisted. Mutation is limited
to the verified existing installation, encrypted database preimages and two
nonsecret opt-in flags. This is not an automatic startup or upgrade daemon.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DEFAULT))

import httpx
import psutil

from muninn.history.capture_cadence import SmallCaptureCadence
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from scripts.local_runtime_preflight import inspect_runtime

CAPTURE_FLAGS = ("MUNINN_CAPTURE_ENRICHMENT", "MUNINN_CAPTURE_AUTO_ANALYSIS")
REMOTE_FLAG = "MUNINN_CAPTURE_AUTO_REMOTE"
TABLES = ("jobs", "history_search_jobs", "history_analysis_jobs")
ACTIVE = {"jobs": {"pending", "retry", "capturing"},
          "history_search_jobs": {"pending", "retry", "running"},
          "history_analysis_jobs": {"pending", "retry", "running", "publishing", "publication_pending"}}
PROGRAM_PATHS = ("server.py", "muninn", "mcp.py", "mcp_wrapper.py", "pyproject.toml",
                 "uv.lock", "requirements*.txt", "scripts/start_shared_local.ps1",
                 "dashboard.html", "dashboard.css", "scripts/reload_shared_local.py",
                 "scripts/compact_restart_recovery.py")
RUNTIME_SOURCE_SUFFIXES = {".py", ".pyw", ".js", ".mjs", ".cjs", ".ps1", ".toml",
                           ".lock", ".txt", ".json", ".yaml", ".yml", ".pyd", ".so",
                           ".html", ".css"}


class PersistenceError(RuntimeError):
    """Runtime may be enabled; failed persistence must not imply rollback."""


_SAFE_FAILURE_CODES = {
    "Durable workers are not idle": "workers_busy",
    "Queue changed before owned stop": "workers_claimed",
    "Candidate exited; private preimages and logs preserved": "candidate_exited",
    "Candidate ownership differs": "candidate_ownership",
    "Requested capture mode is not effective": "capture_mode",
    "Candidate publication schema is incomplete": "publication_schema",
    "Candidate startup deadline reached; private preimages and logs preserved": "startup_deadline",
    "Paid batch stores are incomplete": "paid_stores_incomplete",
    "Paid submission identity is uncertain": "paid_submission_unknown",
    "Paid authorization is in flight": "paid_authorization_busy",
    "Paid batch admission binding differs": "paid_binding_mismatch",
}


def safe_failure_code(exc):
    """Expose only reviewed static check names, never exception text or paths."""
    return _SAFE_FAILURE_CODES.get(str(exc), "unspecified") if type(exc) is RuntimeError else "unspecified"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def paid_stop_fence(data, archive, journal):
    """Exclude new paid POSTs and refuse any unsaved submission identity.

    Lock order follows paid authorization, outbox and journal. Existing rows
    are authenticated under the locks; no schema or permission is changed.
    The caller keeps this context until the owned process has actually stopped.
    """
    from muninn.history.historical_batch import BatchOutbox, _provider_id
    from muninn.history.secure_archive import SecureHistoryArchive
    from scripts.enroll_history_backlog import ReadOnlyJournal

    policy_path = data / "remote_policy" / "policy.sqlite3"
    batch_path = archive / "historical-batches.db"
    with ExitStack() as stack:
        remote = batches = outbox = reader = None
        if policy_path.exists():
            require_unlinked_path(policy_path)
            verify_private(policy_path)
            remote = stack.enter_context(closing(sqlite3.connect(
                policy_path.as_uri() + "?mode=rw", uri=True, timeout=2)))
            remote.row_factory = sqlite3.Row
            remote.execute("BEGIN IMMEDIATE")
        if batch_path.exists():
            require(remote is not None, "Paid batch stores are incomplete")
            unlocked = SecureHistoryArchive(archive)
            # Existing pair required: constructor cannot initialize a new pair.
            require((archive / "historical-batches-managed").is_file(), "Paid batch stores are incomplete")
            outbox = BatchOutbox(unlocked)
            batches = stack.enter_context(closing(sqlite3.connect(
                batch_path.as_uri() + "?mode=rw", uri=True, timeout=2)))
            batches.execute("BEGIN IMMEDIATE")
            reader = ReadOnlyJournal(unlocked)
            # Plaintext state is rejection-only. Acceptance below authenticates
            # the exact paid record, not every retained request/result body.
            require(batches.execute("SELECT 1 FROM batches WHERE state='submission_unknown' LIMIT 1").fetchone() is None,
                    "Paid submission identity is uncertain")
        else:
            require(not (archive / "historical-batches-managed").exists(), "Paid batch stores are incomplete")
        fence = stack.enter_context(closing(sqlite3.connect(journal.as_uri() + "?mode=rw", uri=True, timeout=2)))
        fence.row_factory = sqlite3.Row
        fence.execute("BEGIN IMMEDIATE")
        if remote is not None:
            # A reservation is also left alone, rather than making startup
            # wait for its timeout after killing the in-flight authorization.
            pending = remote.execute(
                "SELECT * FROM remote_admissions WHERE state IN ('reserved','unknown') LIMIT 2").fetchall()
            require(len(pending) <= 1, "Paid authorization is in flight")
            for admission in pending:
                require(admission["state"] == "unknown" and "batch_owner" in admission.keys(),
                        "Paid authorization is in flight")
                row = batches.execute("SELECT * FROM batches WHERE id=?", (admission["batch_owner"],)).fetchone() if batches else None
                record = outbox._read(row) if row is not None else None
                require(record is not None and record["state"] in {"submitted", "terminal_saved", "cleaned"}
                        and _provider_id(record["provider_id"]), "Paid submission identity is uncertain")
                owner = reader._historical_batch_head(fence)[1]
                require(owner is not None and owner["phase"] == "sent"
                        and owner["generation"] == admission["generation"] == record["consent_generation"],
                        "Paid batch admission binding differs")
                if record.get("repair_parent"):
                    parent_row = batches.execute("SELECT * FROM batches WHERE id=?", (owner["id"],)).fetchone()
                    parent = outbox._read(parent_row) if parent_row is not None else None
                    require(record["repair_parent"] == owner["id"] and parent is not None
                            and parent.get("repairs", []) and parent["repairs"][-1] == record["id"]
                            and record.get("repair_admission") == admission["id"],
                            "Paid batch admission binding differs")
                else:
                    require(owner["id"] == record["id"] and owner["admission_id"] == admission["id"],
                            "Paid batch admission binding differs")
        yield fence


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--port", type=int, default=42069)
    parser.add_argument("--restart", action="store_true", help="Requires explicit operator authorization")
    parser.add_argument('--retire-one-old-recovery-preimage', action='store_true',
                        help='Separate explicit choice after verified restart: compact/retire at most one '
                             'old recovery preimage, preserving four full copies and the new preimage')
    parser.add_argument("--enable-capture-auto", action="store_true", help="Enable local-only new-capture processing")
    parser.add_argument("--preserve-capture-auto", action="store_true",
                        help="Reload already enabled local capture without changing settings or queued work")
    parser.add_argument("--recover-active-capture", action="store_true",
                        help="Replay interrupted CPU archive captures; model/search workers must be idle")
    parser.add_argument("--recover-active-capture-limit", type=int, default=1,
                        help="Explicit replay bound (1-32), default one; requires active-capture recovery")
    parser.add_argument("--enable-capture-remote", action="store_true",
                        help="Opt in to managed ZDR for new capture windows on an existing auto-capture service")
    parser.add_argument("--backlog-drain-minutes", type=int,
                        help="Temporary remote-only catch-up (1-180 minutes), then ordinary cadence resumes")
    parser.add_argument("--expected-revision", help="Approved, tested Git commit hash")
    parser.add_argument("--finalize-capture-auto", action="store_true",
                        help="Persist an already running approved local activation; never restart")
    parser.add_argument("--preimage-root", type=Path, help="Private preimage from the approved activation")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("invalid local port")
    if args.retire_one_old_recovery_preimage and not args.restart:
        parser.error('Recovery preimage retirement requires an explicitly authorized restart')
    if args.enable_capture_auto and not args.restart:
        parser.error("capture activation requires an explicitly authorized restart")
    if args.preserve_capture_auto and (not args.restart or args.enable_capture_auto or args.finalize_capture_auto):
        parser.error("capture preservation requires restart and cannot activate or finalize")
    if args.enable_capture_remote and not args.preserve_capture_auto:
        parser.error("remote capture requires restart with --preserve-capture-auto")
    if args.recover_active_capture and not (args.restart and args.preserve_capture_auto):
        parser.error("active capture recovery requires restart with --preserve-capture-auto")
    if (not 1 <= args.recover_active_capture_limit <= 32
            or args.recover_active_capture_limit != 1 and not args.recover_active_capture):
        parser.error("capture recovery limit requires active-capture recovery and must be 1-32")
    if args.backlog_drain_minutes is not None and (
            not args.enable_capture_remote or not 1 <= args.backlog_drain_minutes <= 180):
        parser.error("backlog drain requires remote capture and 1-180 minutes")
    if args.restart and not args.expected_revision:
        parser.error("restart requires an approved --expected-revision")
    if args.finalize_capture_auto and (args.restart or args.enable_capture_auto
                                      or not args.expected_revision or args.preimage_root is None):
        parser.error("finalization requires candidate/preimage and cannot restart")
    if args.preimage_root is not None and not args.finalize_capture_auto:
        parser.error("--preimage-root is only valid for finalization")
    return args


def launch_environment(original, *, enable_capture_auto, enable_capture_remote=False,
                       backlog_drain_minutes=None):
    copied = dict(original)
    if enable_capture_auto:
        copied.update({name: "1" for name in CAPTURE_FLAGS})
    if enable_capture_remote:
        copied[REMOTE_FLAG] = "1"
    if backlog_drain_minutes is not None:
        if not enable_capture_remote or type(backlog_drain_minutes) is not int or not 1 <= backlog_drain_minutes <= 180:
            raise ValueError("Invalid temporary backlog drain")
        copied["MUNINN_CAPTURE_BACKLOG_DRAIN_UNTIL"] = str(time.time() + 60 * backlog_drain_minutes)
    return copied


def validate_flag_value(value):
    if value is None:
        return None
    if (type(value) is not str or len(value) > 16
            or value.strip().lower() not in {"0", "1", "true", "false", "yes", "no", "on", "off"}):
        raise ValueError("Capture flag preimage is not a nonsecret Boolean setting")
    return value


def read_user_flag(name):
    import winreg
    if name not in (*CAPTURE_FLAGS, REMOTE_FLAG):
        raise ValueError("Setting is outside the capture flag whitelist")
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, kind = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    if kind != winreg.REG_SZ:
        raise ValueError("Capture flag registry type is unsupported")
    return validate_flag_value(value)


def write_user_flag(name, value):
    import winreg
    if name not in (*CAPTURE_FLAGS, REMOTE_FLAG):
        raise ValueError("Setting is outside the capture flag whitelist")
    validate_flag_value(value)
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, "Environment", access=winreg.KEY_SET_VALUE) as key:
        if value is None:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                pass
        else:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def persist_capture_flags(before, *, read=read_user_flag, write=write_user_flag):
    require(set(before) == set(CAPTURE_FLAGS), "Incomplete capture flag preimage")
    for value in before.values():
        validate_flag_value(value)
    if {name: read(name) for name in CAPTURE_FLAGS} != before:
        raise PersistenceError("Capture settings changed; existing user choice preserved")
    try:
        for name in CAPTURE_FLAGS:
            write(name, "1")
        if any(read(name) != "1" for name in CAPTURE_FLAGS):
            raise PersistenceError("Capture setting persistence was not verified")
    except Exception as exc:
        # Compensate even an interruption immediately after a registry write.
        # Never overwrite a concurrent user revocation to a non-enabled value.
        failures = 0
        for name in CAPTURE_FLAGS:
            try:
                if read(name) == "1" and before[name] != "1":
                    write(name, before[name])
            except Exception:
                failures += 1
        raise PersistenceError("Capture setting persistence failed; "
                               + ("compensation incomplete" if failures else "settings compensated")) from exc


def persist_remote_flag(before, *, read=read_user_flag, write=write_user_flag):
    validate_flag_value(before)
    require(read(REMOTE_FLAG) == before, "Remote capture setting changed before persistence")
    try:
        write(REMOTE_FLAG, "1")
        if read(REMOTE_FLAG) != "1":
            raise PersistenceError("Remote capture setting persistence was not verified")
    except Exception as exc:
        if before != "1" and read(REMOTE_FLAG) == "1":
            write(REMOTE_FLAG, before)
        raise PersistenceError("Remote capture setting persistence failed") from exc


def mode_matches(report, *, enable, remote=False):
    state = report.get("capture_enrichment")
    if not isinstance(state, dict):
        return False
    if enable:
        return (state.get("capture_enabled") is True
                and state.get("automatic_analysis_enabled") is True
                and state.get("automatic_remote_enabled") is remote)
    return state.get("capture_enabled") is False


def pre_reload_mode_matches(report, environment, *, preserve_capture_auto=False):
    """Old servers lack the mode field; require owned process flags off.

    Absence of an HTTP status field alone never authorizes a reload. The
    environment is from the already verified owned process and stays private.
    Post-reload mode verification still requires the modern effective report.
    """
    if preserve_capture_auto:
        try:
            remote_value = validate_flag_value(environment.get(REMOTE_FLAG))
        except ValueError:
            return False
        remote_enabled = bool(remote_value and remote_value.strip().lower() in {"1", "true", "yes", "on"})
        return (mode_matches(report, enable=True, remote=remote_enabled)
                and all(isinstance(environment.get(name), str)
                        and environment[name].strip().lower() in {"1", "true", "yes", "on"}
                        for name in CAPTURE_FLAGS))
    for name in CAPTURE_FLAGS:
        value = environment.get(name)
        try:
            validate_flag_value(value)
        except ValueError:
            return False
        if value is not None and value.strip().lower() not in {"0", "false", "no", "off"}:
            return False
    state = report.get("capture_enrichment")
    if state is None:
        return True
    return mode_matches(report, enable=False)


def queue_states(db):
    queues = {table: dict(db.execute(f"SELECT state,COUNT(*) FROM {table} GROUP BY state")) for table in TABLES}
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_classification_jobs'").fetchone():
        queues["memory_classification_jobs"] = dict(db.execute(
            "SELECT state,COUNT(*) FROM memory_classification_jobs GROUP BY state"))
    return queues


def queues_idle(queues, *, preserve_queued=False, allow_active_capture=False, active_capture_limit=1):
    """Default stops no claim; explicit bounded recovery permits replayable CPU captures."""
    if type(active_capture_limit) is not int or not 1 <= active_capture_limit <= 32:
        return False
    if any(queues.get("memory_classification_jobs", {}).get(state, 0) for state in ("running", "staged")):
        return False  # Preserve-queued cannot abandon a classification writer/stage.
    excluded = {"pending", "retry", "publication_pending"} if preserve_queued else set()
    recover_capture = (preserve_queued and allow_active_capture
                       and 0 <= queues["jobs"].get("capturing", 0) <= active_capture_limit)
    return not any(queues[table].get(state, 0)
                   for table in TABLES for state in ACTIVE[table] - excluded
                   if not (recover_capture and table == "jobs" and state == "capturing"))


def publication_queue_unclaimed(db):
    """A deferred publication is durable only when it holds no live lease."""
    return db.execute("SELECT 1 FROM history_analysis_jobs WHERE state='publication_pending' "
                      "AND (lease_token IS NOT NULL OR lease_until IS NOT NULL) LIMIT 1").fetchone() is None


def prepare_preimage_destination(archive):
    require_unlinked_path(archive)
    verify_private(archive)
    parent = archive / "operator-preimages"
    require_unlinked_path(parent)
    if not parent.exists():
        create_private_directory(parent)
    verify_private(parent)
    destination = parent / ("restart-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex)
    require_unlinked_path(destination)
    create_private_directory(destination)
    return destination


def preimage_databases(archive, journal):
    preimages = archive / "operator-preimages"
    databases = [journal]
    # The paid outbox uses .db, not .sqlite3. Preserve it under the same paid
    # stop fence as the journal; never omit retained inputs/results on reload.
    paid_outbox = archive / "historical-batches.db"
    if paid_outbox.exists():
        databases.append(paid_outbox)
    # Prune our own namespace before traversal, not just before copying.
    for directory, children, files in os.walk(archive, followlinks=False):
        if Path(directory) == archive:
            children[:] = [name for name in children if archive / name != preimages]
        databases.extend(Path(directory) / name for name in sorted(files)
                         if name.endswith(".sqlite3"))
    return databases


def launch_recovery_compaction(repo, archive, destination):
    """Bounded post-success maintenance; cannot reverse a verified restart.

    No service/model credentials are inherited. The helper retires only one old
    source-evidence preimage, preserving four full copies AND this exact new
    preimage, including when the local wall clock moved backward.
    """
    try:
        require_unlinked_path(destination)
        verify_private(destination)
        pool = Path.home() / "muninn_backups" / "restart-recovery-pool-v1"
        stdout_path = destination / "recovery-compaction.stdout.log"
        stderr_path = destination / "recovery-compaction.stderr.log"
        create_private_file(stdout_path)
        create_private_file(stderr_path)
        environment = {name: value for name, value in os.environ.items()
                       if name.upper() in {"SYSTEMROOT", "WINDIR", "USERPROFILE",
                                           "LOCALAPPDATA", "APPDATA", "TEMP", "TMP"}}
        environment["PYTHONUTF8"] = "1"
        command = [sys.executable, "-B", "-m", "scripts.compact_restart_recovery", "compact",
                   "--archive-root", str(archive), "--pool-root", str(pool),
                   "--keep-full", "4", "--limit", "1", "--retire",
                   "--exclude-snapshot", destination.name,
                   "--log-path", str(destination / "recovery-compaction.progress.jsonl")]
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            child = subprocess.Popen(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print(json.dumps({"stage": "recovery_compaction_started", "pid": child.pid,
                          "max_snapshots": 1, "full_preimages_preserved": 4}), flush=True)
    except Exception as exc:
        print(json.dumps({"stage": "recovery_compaction_deferred",
                          "error_category": type(exc).__name__}), flush=True)


def load_capture_flag_preimage(archive, preimage):
    require_unlinked_path(preimage)
    require(preimage.parent.resolve(strict=True) == (archive / "operator-preimages").resolve(strict=True),
            "Capture preimage is outside the private namespace")
    verify_private(archive)
    verify_private(preimage.parent)
    verify_private(preimage)
    flags = preimage / "capture-user-flags.json"
    verify_private(flags)
    require(flags.is_file() and flags.stat().st_size <= 1024, "Capture flag preimage is not bounded")
    before = json.loads(flags.read_text(encoding="utf-8"))
    require(isinstance(before, dict) and set(before) == set(CAPTURE_FLAGS),
            "Incomplete capture flag preimage")
    for value in before.values():
        validate_flag_value(value)
    return before


def startup_mode_pending(report):
    return report.get("capture_enrichment") is None


def verify_candidate(repo, revision):
    if not re.fullmatch(r"[0-9a-f]{7,64}", revision or ""):
        raise ValueError("Expected revision must be a Git commit hash")
    resolved = subprocess.run(["git", "rev-parse", "--verify", revision + "^{commit}"],
                              cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    require(resolved.startswith(revision), "Approved revision is unavailable")
    compared = subprocess.run(["git", "diff", "--quiet", resolved, "--", *PROGRAM_PATHS],
                              cwd=repo, capture_output=True)
    require(compared.returncode == 0, "Program source differs from approved revision")
    # Do not exclude ignored source: .gitignore cannot exempt executable code
    # from the approved candidate. Derived caches/logs are not source files.
    untracked = subprocess.run(["git", "ls-files", "--others", "-z",
                                "--", *PROGRAM_PATHS], cwd=repo, capture_output=True, check=True)
    unexpected = []
    for raw in untracked.stdout.split(b"\0"):
        if not raw:
            continue
        path = Path(os.fsdecode(raw))
        if (path.suffix.lower() in RUNTIME_SOURCE_SUFFIXES
                or (path.suffix.lower() in {".pyc", ".pyo"} and "__pycache__" not in path.parts)):
            unexpected.append(raw)
    require(not unexpected, "Untracked program files are outside approved revision")


def require_unlinked_path(path):
    """Inspect lexical ancestors before resolution can hide a junction/link."""
    path = Path(os.path.abspath(path))
    for ancestor in (path, *path.parents):
        try:
            details = ancestor.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(details.st_mode)
                and not (getattr(details, "st_file_attributes", 0)
                         & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
                "Operational path contains a linked or reparse ancestor")


def verify_stop_ownership(process, identity, *, repo, port):
    current = psutil.Process(process.pid)
    require(current.is_running() and current.create_time() == identity[0]
            and tuple(current.cmdline()) == identity[1], "Owned process identity changed before stop")
    require(Path(current.exe()).resolve() == Path(sys.executable).resolve()
            and Path(current.cwd()).resolve() == repo, "Owned installation changed before stop")
    listeners = [item for item in psutil.net_connections(kind="tcp")
                 if item.laddr and item.laddr.port == port and item.status == psutil.CONN_LISTEN]
    require(bool(listeners) and all(item.pid == process.pid and item.laddr.ip == "127.0.0.1"
                                   for item in listeners), "Owned loopback listener changed before stop")


def runtime_ready(report):
    return (report.get("anonymous_protected_http") == 401
            and report.get("authenticated_protected_http") == 200
            and report.get("history_http") == 200
            and report.get("archive_ready") is True
            and report.get("history_security") == "strict"
            and report.get("anonymous_root_contains_token") is False)


def candidate_ownership_matches(report, *, child_pid, retired_pid, retired_created_at):
    """Ignore only the process already confirmed terminated by process.wait()."""
    processes = report.get("muninn_processes")
    if (report.get("listener_owners") != [child_pid] or not isinstance(processes, list)
            or type(retired_created_at) not in (int, float)):
        return False
    seen_child = seen_retired = False
    for item in processes:
        if not isinstance(item, dict) or type(item.get("pid")) is not int:
            return False
        pid, created = item["pid"], item.get("create_time")
        if type(created) not in (int, float):
            return False
        if pid == child_pid:
            if seen_child or (pid == retired_pid and created == retired_created_at):
                return False
            seen_child = True
        elif pid == retired_pid and created == retired_created_at and not seen_retired:
            seen_retired = True
        else:
            return False
    return seen_child


def startup_listener_pending(report):
    """The socket census may precede the HTTP readiness checks in one probe."""
    return report.get("listener_owners") is None or report.get("listener_owners") == []


def run(args):
    require_unlinked_path(args.repo)
    repo = args.repo.resolve(strict=True)
    before = inspect_runtime(repo, authenticated=True, port=args.port)
    require(runtime_ready(before), "Existing authenticated strict installation is unavailable")
    require(len(before["muninn_processes"]) == 1, "Existing service ownership is not unique")
    item = before["muninn_processes"][0]
    require(before["listener_owners"] == [item["pid"]], "Listener does not match the owned service")
    process = psutil.Process(item["pid"])
    identity = (process.create_time(), tuple(process.cmdline()))
    require(Path(process.exe()).resolve() == Path(sys.executable).resolve(), "Caller must use the existing interpreter")
    require(Path(process.cwd()).resolve() == repo, "Existing service checkout differs")
    environment = process.environ()  # Values stay private and only in memory.
    require(environment.get("MUNINN_NO_AUTH", "0") == "0" and environment.get("MUNINN_AUTH_TOKEN"),
            "Existing authentication settings are unavailable")
    require(environment.get("MUNINN_HOST") == "127.0.0.1"
            and environment.get("MUNINN_PORT") == str(args.port), "Existing loopback binding differs")
    data = Path(environment.get("MUNINN_DATA_DIR") or ".muninn_runtime")
    if not data.is_absolute():
        data = repo / data
    archive = Path(environment.get("MUNINN_HISTORY_ARCHIVE_DIR") or data / "history_secure_archive")
    if not archive.is_absolute():
        archive = repo / archive
    require_unlinked_path(data)
    require_unlinked_path(archive)
    journal = archive / "capture-jobs.db"
    with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as db:
        initial = queue_states(db)
        initial_publication_unclaimed = publication_queue_unclaimed(db) if args.preserve_capture_auto else True
    print(json.dumps({"stage": "inspection", "queues": initial}), flush=True)
    if args.finalize_capture_auto:
        require(os.name == "nt", "Capture persistence is supported only on Windows")
        require(mode_matches(before, enable=True), "Approved local activation is not running")
        require(all(environment.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
                    for name in CAPTURE_FLAGS), "Owned process capture flags are not enabled")
        verify_candidate(repo, args.expected_revision)
        saved = load_capture_flag_preimage(archive, args.preimage_root)
        verify_stop_ownership(process, identity, repo=repo, port=args.port)
        persist_capture_flags(saved)
        print(json.dumps({"stage": "activation_persisted", "port": args.port,
                          "automatic_local_capture": True, "capture_settings_persisted": True,
                          "process_restarted": False}), flush=True)
        return
    if not args.restart:
        return
    require(os.name == "nt", "Owned forced reload is supported only on Windows")
    require(queues_idle(initial, preserve_queued=args.preserve_capture_auto,
                        allow_active_capture=args.recover_active_capture,
                        active_capture_limit=args.recover_active_capture_limit), "Durable workers are not idle")
    require(initial_publication_unclaimed, "Durable publication has a live lease")
    require(pre_reload_mode_matches(before, environment, preserve_capture_auto=args.preserve_capture_auto),
            "Existing capture mode differs from the requested reload procedure")
    verify_candidate(repo, args.expected_revision)
    launching = launch_environment(environment, enable_capture_auto=args.enable_capture_auto,
                                   enable_capture_remote=args.enable_capture_remote,
                                   backlog_drain_minutes=args.backlog_drain_minutes)
    expected_auto = args.enable_capture_auto or args.preserve_capture_auto
    expected_remote = (args.enable_capture_remote or
                       bool(launching.get(REMOTE_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}))
    user_before = None
    remote_user_before = read_user_flag(REMOTE_FLAG) if args.enable_capture_remote else None
    if args.enable_capture_auto or args.preserve_capture_auto:
        SmallCaptureCadence(quiet_seconds=float(launching.get("MUNINN_CAPTURE_QUIET_SECONDS", "300")),
                            interval_seconds=float(launching.get("MUNINN_CAPTURE_INTERVAL_SECONDS", "30")),
                            max_wait_seconds=float(launching.get("MUNINN_CAPTURE_MAX_WAIT_SECONDS", "1800")))
    if args.enable_capture_auto:
        user_before = {name: read_user_flag(name) for name in CAPTURE_FLAGS}
    destination = prepare_preimage_destination(archive)
    databases = preimage_databases(archive, journal)
    for path in databases:
        require_unlinked_path(path)
        verify_private(path)
    if user_before is not None:
        flags_path = destination / "capture-user-flags.json"
        create_private_file(flags_path)
        flags_path.write_text(json.dumps(user_before, sort_keys=True), encoding="utf-8")
        verify_private(flags_path)
    if args.enable_capture_remote:
        remote_path = destination / "capture-remote-user-flag.json"
        create_private_file(remote_path)
        remote_path.write_text(json.dumps({REMOTE_FLAG: remote_user_before}), encoding="utf-8")
        verify_private(remote_path)
    out, err = destination / "server.stdout.log", destination / "server.stderr.log"
    create_private_file(out)
    create_private_file(err)
    command = process.cmdline()  # Preserve interpreter options and service arguments.
    require(tuple(command) == identity[1], "Owned launch command changed")
    require(command and Path(command[0]).resolve() == Path(sys.executable).resolve(), "Owned launch command differs")
    # No registry mutation or process stop precedes all destination checks.
    verify_candidate(repo, args.expected_revision)
    with paid_stop_fence(data, archive, journal) as fence:
        fenced_queues = queue_states(fence)
        require(queues_idle(fenced_queues, preserve_queued=args.preserve_capture_auto,
                            allow_active_capture=args.recover_active_capture,
                            active_capture_limit=args.recover_active_capture_limit),
                "Queue changed before owned stop")
        if args.preserve_capture_auto:
            require(publication_queue_unclaimed(fence), "Durable publication gained a live lease")
        # Keep the journal writer fence through the preimages. Otherwise a
        # worker can claim the next job during a large encrypted DB backup.
        for path in databases:
            target = destination / path.relative_to(archive)
            if not target.parent.exists():
                create_private_directory(target.parent)
            create_private_file(target)
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
                with sqlite3.connect(target) as backup:
                    source.backup(backup)
                    require(backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
                            "Database preimage failed")
            verify_private(target)
        verify_stop_ownership(process, identity, repo=repo, port=args.port)
        captures = fenced_queues["jobs"].get("capturing", 0)
        print(json.dumps({"stage": "preimage_validated", "databases": len(databases),
                          "inflight_jobs": captures, "recoverable_cpu_captures": captures,
                          "stop_method": "owned_windows_forced_termination"}), flush=True)
        process.terminate()
        process.wait(timeout=15)
        fence.rollback()
    with out.open("ab") as stdout, err.open("ab") as stderr:
        child = subprocess.Popen(command, cwd=repo, env=launching, stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    verified = False
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            require(child.poll() is None, "Candidate exited; private preimages and logs preserved")
            try:
                report = inspect_runtime(repo, authenticated=True, port=args.port)
                owners = report.get("listener_owners")
                if owners is not None and owners != [] and owners != [child.pid]:
                    raise RuntimeError("Candidate ownership differs")
                if runtime_ready(report):
                    if startup_listener_pending(report):
                        time.sleep(1)
                        continue
                    owned = report.get("muninn_processes")
                    if not candidate_ownership_matches(report, child_pid=child.pid,
                                                       retired_pid=process.pid,
                                                       retired_created_at=identity[0]):
                        entries = owned if isinstance(owned, list) else []
                        print(json.dumps({"stage": "candidate_ownership_observation",
                                          "listener_matches_child": report.get("listener_owners") == [child.pid],
                                          "owned_process_count": len(entries),
                                          "child_visible": any(isinstance(item, dict) and item.get("pid") == child.pid
                                                               for item in entries),
                                          "retired_visible": any(isinstance(item, dict) and item.get("pid") == process.pid
                                                                 for item in entries),
                                          "retired_identity_matches": any(isinstance(item, dict)
                                              and item.get("pid") == process.pid
                                              and item.get("create_time") == identity[0] for item in entries)}),
                              flush=True)
                        raise RuntimeError("Candidate ownership differs")
                    if startup_mode_pending(report):
                        time.sleep(1)
                        continue
                    require(mode_matches(report, enable=expected_auto, remote=expected_remote),
                            "Requested capture mode is not effective")
                    if args.backlog_drain_minutes is not None:
                        require(report.get("capture_enrichment", {}).get("backlog_drain", {}).get("enabled") is True,
                                "Requested backlog drain is not effective")
                    with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as db:
                        columns = {row[1] for row in db.execute("PRAGMA table_info(history_analysis_jobs)")}
                    require({"sealed_window", "sealed_extraction", "extraction_id", "sealed_receipt",
                             "sealed_reuse", "cancel_requested", "publication_started"} <= columns,
                            "Candidate publication schema is incomplete")
                    if user_before is not None:
                        persist_capture_flags(user_before)
                    if args.enable_capture_remote:
                        persist_remote_flag(remote_user_before)
                    verified = True
                    print(json.dumps({"stage": "restarted_verified", "port": args.port, "process_count": 1,
                                      "strict_archive_ready": True, "automatic_local_capture": expected_auto,
                                      "automatic_remote_capture": expected_remote,
                                      "capture_settings_persisted": user_before is not None}), flush=True)
                    if args.retire_one_old_recovery_preimage:
                        launch_recovery_compaction(repo, archive, destination)
                    else:
                        print(json.dumps({'stage': 'recovery_preimages_preserved',
                                          'retirement_requested': False,
                                          'maintenance_started': False}), flush=True)
                    return
            except (httpx.HTTPError, ConnectionError):
                pass
            time.sleep(1)
        raise RuntimeError("Candidate startup deadline reached; private preimages and logs preserved")
    finally:
        if expected_remote and not verified and child.poll() is None:
            # Popen's Windows process handle identifies only our new candidate.
            # Leave sealed queue recovery to the journal; never keep an
            # unverified remote-enabled child running after a failed reload.
            child.terminate()
            child.wait(timeout=15)


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception as exc:
        print(json.dumps({"stage": "failed", "error_category": type(exc).__name__,
                          "failure_code": safe_failure_code(exc),
                          "runtime_rollback_claimed": False}), flush=True)
        raise SystemExit(2)
