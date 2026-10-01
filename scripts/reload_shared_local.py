"""Explicit Windows operator reload; defaults to read-only inspection.

No process identifiers or environment secrets are persisted. Mutation is limited
to the verified existing installation, encrypted database preimages and two
nonsecret opt-in flags. This is not an automatic startup or upgrade daemon.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
import time
import uuid

REPO_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DEFAULT))

import httpx
import psutil

from muninn.history.capture_cadence import SmallCaptureCadence
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from scripts.local_runtime_preflight import inspect_runtime

CAPTURE_FLAGS = ("MUNINN_CAPTURE_ENRICHMENT", "MUNINN_CAPTURE_AUTO_ANALYSIS")
TABLES = ("jobs", "history_search_jobs", "history_analysis_jobs")
ACTIVE = {"jobs": {"pending", "retry", "capturing"},
          "history_search_jobs": {"pending", "retry", "running"},
          "history_analysis_jobs": {"pending", "retry", "running", "publishing", "publication_pending"}}
PROGRAM_PATHS = ("server.py", "muninn", "mcp.py", "mcp_wrapper.py", "pyproject.toml",
                 "uv.lock", "requirements*.txt", "scripts/start_shared_local.ps1")
RUNTIME_SOURCE_SUFFIXES = {".py", ".pyw", ".js", ".mjs", ".cjs", ".ps1", ".toml",
                           ".lock", ".txt", ".json", ".yaml", ".yml", ".pyd", ".so"}


class PersistenceError(RuntimeError):
    """Runtime may be enabled; failed persistence must not imply rollback."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--port", type=int, default=42069)
    parser.add_argument("--restart", action="store_true", help="Requires explicit operator authorization")
    parser.add_argument("--enable-capture-auto", action="store_true", help="Enable local-only new-capture processing")
    parser.add_argument("--preserve-capture-auto", action="store_true",
                        help="Reload already enabled local capture without changing settings or queued work")
    parser.add_argument("--expected-revision", help="Approved, tested Git commit hash")
    parser.add_argument("--finalize-capture-auto", action="store_true",
                        help="Persist an already running approved local activation; never restart")
    parser.add_argument("--preimage-root", type=Path, help="Private preimage from the approved activation")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("invalid local port")
    if args.enable_capture_auto and not args.restart:
        parser.error("capture activation requires an explicitly authorized restart")
    if args.preserve_capture_auto and (not args.restart or args.enable_capture_auto or args.finalize_capture_auto):
        parser.error("capture preservation requires restart and cannot activate or finalize")
    if args.restart and not args.expected_revision:
        parser.error("restart requires an approved --expected-revision")
    if args.finalize_capture_auto and (args.restart or args.enable_capture_auto
                                      or not args.expected_revision or args.preimage_root is None):
        parser.error("finalization requires candidate/preimage and cannot restart")
    if args.preimage_root is not None and not args.finalize_capture_auto:
        parser.error("--preimage-root is only valid for finalization")
    return args


def launch_environment(original, *, enable_capture_auto):
    copied = dict(original)
    if enable_capture_auto:
        copied.update({name: "1" for name in CAPTURE_FLAGS})
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
    if name not in CAPTURE_FLAGS:
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
    if name not in CAPTURE_FLAGS:
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


def mode_matches(report, *, enable):
    state = report.get("capture_enrichment")
    if not isinstance(state, dict):
        return False
    if enable:
        return (state.get("capture_enabled") is True
                and state.get("automatic_analysis_enabled") is True
                and state.get("automatic_remote_enabled") is False)
    return state.get("capture_enabled") is False


def pre_reload_mode_matches(report, environment, *, preserve_capture_auto=False):
    """Old servers lack the mode field; require owned process flags off.

    Absence of an HTTP status field alone never authorizes a reload. The
    environment is from the already verified owned process and stays private.
    Post-reload mode verification still requires the modern effective report.
    """
    if preserve_capture_auto:
        return (mode_matches(report, enable=True)
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
    return {table: dict(db.execute(f"SELECT state,COUNT(*) FROM {table} GROUP BY state")) for table in TABLES}


def queues_idle(queues, *, preserve_queued=False):
    """Preserve-mode stops no claimed worker and leaves durable queued work intact."""
    excluded = {"pending", "retry"} if preserve_queued else set()
    return not any(queues[table].get(state, 0)
                   for table in TABLES for state in ACTIVE[table] - excluded)


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
    # Prune our own namespace before traversal, not just before copying.
    for directory, children, files in os.walk(archive, followlinks=False):
        if Path(directory) == archive:
            children[:] = [name for name in children if archive / name != preimages]
        databases.extend(Path(directory) / name for name in sorted(files)
                         if name.endswith(".sqlite3"))
    return databases


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
    require(queues_idle(initial, preserve_queued=args.preserve_capture_auto), "Durable workers are not idle")
    require(pre_reload_mode_matches(before, environment, preserve_capture_auto=args.preserve_capture_auto),
            "Existing capture mode differs from the requested reload procedure")
    verify_candidate(repo, args.expected_revision)
    launching = launch_environment(environment, enable_capture_auto=args.enable_capture_auto)
    expected_auto = args.enable_capture_auto or args.preserve_capture_auto
    user_before = None
    if args.enable_capture_auto:
        SmallCaptureCadence(quiet_seconds=float(launching.get("MUNINN_CAPTURE_QUIET_SECONDS", "300")),
                            interval_seconds=float(launching.get("MUNINN_CAPTURE_INTERVAL_SECONDS", "30")))
        user_before = {name: read_user_flag(name) for name in CAPTURE_FLAGS}
    destination = prepare_preimage_destination(archive)
    databases = preimage_databases(archive, journal)
    for path in databases:
        require_unlinked_path(path)
        verify_private(path)
        target = destination / path.relative_to(archive)
        if not target.parent.exists():
            create_private_directory(target.parent)
        create_private_file(target)
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(target) as backup:
                source.backup(backup)
                require(backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Database preimage failed")
        verify_private(target)
    if user_before is not None:
        flags_path = destination / "capture-user-flags.json"
        create_private_file(flags_path)
        flags_path.write_text(json.dumps(user_before, sort_keys=True), encoding="utf-8")
        verify_private(flags_path)
    out, err = destination / "server.stdout.log", destination / "server.stderr.log"
    create_private_file(out)
    create_private_file(err)
    command = process.cmdline()  # Preserve interpreter options and service arguments.
    require(tuple(command) == identity[1], "Owned launch command changed")
    require(command and Path(command[0]).resolve() == Path(sys.executable).resolve(), "Owned launch command differs")
    # No registry mutation or process stop precedes all destination checks.
    verify_candidate(repo, args.expected_revision)
    with sqlite3.connect(journal.as_uri() + "?mode=rw", uri=True, timeout=2) as fence:
        fence.execute("BEGIN IMMEDIATE")
        require(queues_idle(queue_states(fence), preserve_queued=args.preserve_capture_auto),
                "Queue changed before owned stop")
        verify_stop_ownership(process, identity, repo=repo, port=args.port)
        print(json.dumps({"stage": "preimage_validated", "databases": len(databases), "inflight_jobs": 0,
                          "stop_method": "owned_windows_forced_termination"}), flush=True)
        process.terminate()
        process.wait(timeout=15)
        fence.rollback()
    with out.open("ab") as stdout, err.open("ab") as stderr:
        child = subprocess.Popen(command, cwd=repo, env=launching, stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        require(child.poll() is None, "Candidate exited; private preimages and logs preserved")
        try:
            report = inspect_runtime(repo, authenticated=True, port=args.port)
            if runtime_ready(report):
                require(report["listener_owners"] == [child.pid]
                        and len(report["muninn_processes"]) == 1, "Candidate ownership differs")
                if startup_mode_pending(report):
                    time.sleep(1)
                    continue
                require(mode_matches(report, enable=expected_auto), "Requested capture mode is not effective")
                with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as db:
                    columns = {row[1] for row in db.execute("PRAGMA table_info(history_analysis_jobs)")}
                require({"sealed_window", "sealed_extraction", "extraction_id", "sealed_receipt",
                         "sealed_reuse", "cancel_requested", "publication_started"} <= columns,
                        "Candidate publication schema is incomplete")
                if user_before is not None:
                    persist_capture_flags(user_before)
                print(json.dumps({"stage": "restarted_verified", "port": args.port, "process_count": 1,
                                  "strict_archive_ready": True, "automatic_local_capture": expected_auto,
                                  "capture_settings_persisted": user_before is not None}), flush=True)
                return
        except (httpx.HTTPError, ConnectionError):
            pass
        time.sleep(1)
    raise RuntimeError("Candidate startup deadline reached; private preimages and logs preserved")


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception as exc:
        print(json.dumps({"stage": "failed", "error_category": type(exc).__name__,
                          "runtime_rollback_claimed": False}), flush=True)
        raise SystemExit(2)
