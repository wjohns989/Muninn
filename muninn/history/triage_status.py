"""Bounded read-only credential-review status. Never open a credential vault.

The record is operational evidence, not a credential/unlock receipt. Process
arguments are used only for local identity checks and never returned to clients.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path

import psutil

from .private_acl import verify_private

STAGES = frozenset({'zdr_readiness', 'waiting_for_remote_admission', 'awaiting_passphrase',
    'passphrase_received', 'opening_pre_backup_vault', 'validating_pre_backup',
    'opening_post_backup_vault', 'validating_post_backup', 'starting_review',
    'reading_triage_status', 'validated_pre_triage_backup', 'validated_post_triage_backup',
    'review_page', 'source_context_prepare', 'local_context_review', 'zdr_context_review',
    'remote_context_review', 'triage_status'})
COUNTERS = frozenset({'rows', 'page', 'model_calls', 'contexts_reused', 'groups_seen',
    'left_pending', 'deferred_for_user', 'model_rejected', 'rule_rejected',
    'source_context_pending', 'contexts_quota_deferred', 'contexts_route_deferred',
    'credential_records', 'remaining_seconds', 'unresolved_admissions', 'http_status_code'})


def runtime_binding(vault: Path, archive: Path, policy: Path, repo: Path, interpreter: Path) -> str:
    """Opaque local binding; it includes no credential values or input material."""
    paths = [os.path.normcase(str(Path(p).resolve())) for p in
             (vault, archive, policy, repo, interpreter)]
    return hashlib.sha256(json.dumps(['triage-runtime-v2', *paths], separators=(',', ':')).encode()).hexdigest()


def progress_identity(binding: str | None) -> dict:
    process = psutil.Process()
    return {'format': 2, 'worker_pid': process.pid,
            'worker_started_at': process.create_time(), 'recorded_at': time.time(),
            'runtime_binding': binding}


def _number(value) -> bool:
    # Avoid converting an arbitrary JSON integer to a C double (may overflow).
    return (type(value) in (int, float) and 0 <= value < 2**63
            and (type(value) is int or math.isfinite(value)))


def _unlinked(path: Path) -> Path:
    if '..' in path.parts:
        raise ValueError('Invalid progress path')
    path = path.absolute()
    for component in (*reversed(path.parents), path):
        details = component.lstat()
        if (stat.S_ISLNK(details.st_mode) or getattr(details, 'st_file_attributes', 0) & 0x400
                or (hasattr(component, 'is_junction') and component.is_junction())):
            raise ValueError('Linked progress path')
    return path


def _file_identity(details):
    return (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns,
            details.st_nlink)


def _read_latest(path: Path) -> tuple[dict, int]:
    _unlinked(path)
    verify_private(path.parent)
    verify_private(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError('Invalid progress file')
    with path.open('rb') as stream:
        if _file_identity(os.fstat(stream.fileno())) != _file_identity(before):
            raise ValueError('Progress changed')
        stream.seek(max(0, before.st_size - 65536))
        tail = stream.read(65536)
        after = os.fstat(stream.fileno())
    if (_file_identity(before) != _file_identity(after)
            or _file_identity(before) != _file_identity(path.lstat())):
        raise ValueError('Progress changed')
    _unlinked(path)
    verify_private(path)
    if not tail or not tail.endswith(b'\n'):
        raise ValueError('Incomplete progress')
    latest = tail[:-1].rsplit(b'\n', 1)[-1]
    if not latest or len(latest) > 16384:
        raise ValueError('Invalid progress record')
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate progress field')
            result[key] = value
        return result
    record = json.loads(latest, object_pairs_hook=unique_pairs)
    if not isinstance(record, dict):
        raise ValueError('Invalid progress record')
    return record, before.st_mtime_ns


def _processes() -> tuple[list[dict], int]:
    candidates, inaccessible = [], 0
    for process in psutil.process_iter():
        try:
            if 'python' not in process.name().lower():
                continue
            args = process.cmdline()
            if 'scripts.triage_credential_ambiguity' not in args:
                continue
            candidates.append({'pid': process.pid, 'create_time': process.create_time(),
                               'exe': process.exe(), 'cwd': process.cwd(), 'cmdline': args})
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError, ValueError):
            inaccessible += 1
    return candidates, inaccessible


def _still_same_process(worker: dict) -> bool:
    try:
        process = psutil.Process(worker['pid'])
        return (process.create_time() == worker['create_time']
                and process.exe() == worker['exe'] and process.cwd() == worker['cwd']
                and process.cmdline() == worker['cmdline'])
    except (psutil.Error, OSError, ValueError):
        return False


def _path_arg(args: list[str], flag: str, cwd: Path) -> Path:
    # Do not permit duplicate, missing, equals-style, or traversal bindings.
    if args.count(flag) != 1 or any(arg.startswith(flag + '=') for arg in args):
        raise ValueError('Ambiguous runtime')
    index = args.index(flag) + 1
    if index >= len(args) or args[index].startswith('--'):
        raise ValueError('Missing runtime')
    path = Path(args[index])
    if '..' in path.parts:
        raise ValueError('Ambiguous runtime')
    return (cwd / path).absolute() if not path.is_absolute() else path.absolute()


def operational_status(runtime: Path, *, repo: Path, interpreter: Path,
                       archive_root: Path) -> dict:
    """Only operational counts/codes leave this function; no dispatch or writes."""
    result = {'state': 'unknown', 'input_needed': False, 'worker_count': None,
              'observed_at': time.time(), 'basis': 'process_and_private_progress'}
    try:
        runtime, repo, interpreter = (Path(p).resolve() for p in (runtime, repo, interpreter))
        binding = runtime_binding(runtime / 'credential_vault', archive_root, runtime,
                                  repo, interpreter)
        processes, inaccessible = _processes()
        workers = [p for p in processes if Path(p['cwd']).resolve() == repo]
        result['worker_count'] = len(workers)
        if inaccessible or len(workers) > 1:
            return result
        if workers and Path(workers[0]['exe']).resolve() != interpreter:
            return result
        directory = runtime / 'remote_policy' / 'triage-progress'
        if not directory.exists():
            return {**result, 'state': 'idle'} if not workers else result
        _unlinked(directory)
        verify_private(directory)
        if workers:
            worker = workers[0]
            args = worker['cmdline']
            index = args.index('scripts.triage_credential_ambiguity')
            if args.count('scripts.triage_credential_ambiguity') != 1 or index < 1 or args[index - 1] != '-m':
                return result
            for flag, expected in (('--root', runtime / 'credential_vault'),
                                   ('--archive-root', Path(archive_root).resolve()),
                                   ('--policy-root', runtime)):
                if _path_arg(args, flag, repo).resolve() != expected:
                    return result
            log = _path_arg(args, '--progress-log', repo)
            if log.parent != directory or not log.name.endswith('.progress.jsonl'):
                return result
            record, _ = _read_latest(log)
        else:
            records = []
            with os.scandir(directory) as entries:
                for count, entry in enumerate(entries, 1):
                    if count > 64:
                        return result
                    if entry.name.endswith('.progress.jsonl'):
                        records.append(_read_latest(Path(entry.path)))
            if not records:
                return {**result, 'state': 'idle'}
            record, _ = max(records, key=lambda item: item[1])
            if 'format' not in record:
                # Legacy terminal failure is historical only; old wait is idle.
                return {**result, 'state': 'previous_run_failed' if record.get('state') == 'failed' else 'idle'}
        if (type(record.get('format')) is not int or record['format'] != 2
                or type(record.get('worker_pid')) is not int or not 0 < record['worker_pid'] < 2**31
                or not _number(record.get('worker_started_at'))
                or not _number(record.get('recorded_at'))
                or not record['worker_started_at'] <= record['recorded_at'] <= time.time() + 5
                or record.get('runtime_binding') != binding):
            return result
        if workers and (record['worker_pid'] != worker['pid']
                        or record['worker_started_at'] != worker['create_time']
                        or not _still_same_process(worker)):
            return result
        stage = record.get('stage') if record.get('stage') in STAGES else None
        failed = record.get('state') == 'failed'
        input_needed = bool(workers and not failed and stage == 'awaiting_passphrase'
                            and record.get('passphrase_needed') is True)
        state = ('previous_run_failed' if failed else
                 ('finished' if stage == 'triage_status' else 'worker_exited') if not workers else
                 'awaiting_passphrase' if input_needed else
                 'waiting_for_admission' if stage == 'waiting_for_remote_admission' else
                 'backup_validation' if stage in {'opening_pre_backup_vault', 'validating_pre_backup',
                    'opening_post_backup_vault', 'validating_post_backup'} else 'reviewing')
        result.update(state=state, input_needed=input_needed, last_record_at=record['recorded_at'],
                      stage=stage, counters={key: value for key, value in record.items()
                        if key in COUNTERS and type(value) is int and 0 <= value < 2**63})
        if record.get('failure_stage') in STAGES:
            result['failure_stage'] = record['failure_stage']
        if type(record.get('review_resolved')) is bool:
            result['review_resolved'] = record['review_resolved']
        return result
    except (OSError, psutil.Error, ValueError, TypeError, KeyError, IndexError, RuntimeError):
        return result
