"""Private, read-only operational status; no credential values or inference."""
import json
import sys
from pathlib import Path

import pytest

from muninn.history import triage_status as status
from muninn.history.private_acl import create_private_directory, create_private_file


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    policy = runtime / 'remote_policy'
    policy.mkdir()
    directory = policy / 'triage-progress'
    create_private_directory(directory)
    log = directory / 'test.progress.jsonl'
    create_private_file(log)
    archive = runtime / 'history_secure_archive'
    repo = tmp_path / 'repo'
    repo.mkdir()
    worker = dict(pid=123, create_time=100.0, cwd=str(repo), exe=sys.executable,
        cmdline=[sys.executable, '-m', 'scripts.triage_credential_ambiguity',
            '--root', str(runtime / 'credential_vault'), '--archive-root', str(archive),
            '--policy-root', str(runtime), '--progress-log', str(log)])
    monkeypatch.setattr(status, '_processes', lambda: ([worker], 0))
    monkeypatch.setattr(status, '_still_same_process', lambda p: True)
    monkeypatch.setattr(status.time, 'time', lambda: 200.0)
    record = dict(format=2, worker_pid=123, worker_started_at=100.0, recorded_at=150.0,
        runtime_binding=status.runtime_binding(runtime / 'credential_vault', archive, runtime,
                                                repo, Path(sys.executable)),
        stage='awaiting_passphrase', passphrase_needed=True)
    def write(value):
        log.write_text(json.dumps(value) + '\n', encoding='utf-8')
    def read():
        return status.operational_status(runtime, repo=repo, interpreter=Path(sys.executable),
                                         archive_root=archive)
    write(record)
    return worker, record, write, read, log


def test_live_identity_bound_prompt_and_read_only_redaction(fixture):
    worker, record, write, read, log = fixture
    record.update(secret='synthetic-private', error_category='synthetic-private', rows=4)
    write(record)
    before = log.read_bytes()
    result = read()
    assert result['state'] == 'awaiting_passphrase'
    assert result['input_needed'] is True
    assert result['worker_count'] == 1
    assert result['counters'] == {'rows': 4}
    assert 'synthetic-private' not in json.dumps(result)
    assert 'cmdline' not in result and 'runtime_binding' not in result
    assert log.read_bytes() == before


@pytest.mark.parametrize('field', ['--root', '--archive-root', '--policy-root'])
@pytest.mark.parametrize('change', ['different', 'missing', 'duplicate'])
def test_wrong_or_ambiguous_runtime_never_prompts(fixture, field, change):
    worker, record, write, read, log = fixture
    index = worker['cmdline'].index(field)
    if change == 'different':
        worker['cmdline'][index + 1] = str(log.parent / 'other')
    elif change == 'missing':
        del worker['cmdline'][index:index + 2]
    else:
        worker['cmdline'] += worker['cmdline'][index:index + 2]
    assert read()['state'] == 'unknown'
    assert read()['input_needed'] is False


@pytest.mark.parametrize('field,value', [
    ('worker_pid', 124), ('worker_pid', True), ('worker_started_at', 101.0),
    ('recorded_at', float('nan')), ('recorded_at', 300), ('runtime_binding', 'wrong'),
    ('worker_started_at', 10**400), ('recorded_at', 10**400),
])
def test_bad_record_identity_never_prompts(fixture, field, value):
    worker, record, write, read, log = fixture
    record[field] = value
    write(record)
    assert read()['state'] == 'unknown'
    assert read()['input_needed'] is False


def test_pid_reused_between_snapshot_and_read(fixture, monkeypatch):
    monkeypatch.setattr(status, '_still_same_process', lambda p: False)
    assert fixture[3]()['input_needed'] is False
    assert fixture[3]()['state'] == 'unknown'


@pytest.mark.parametrize('tail', [b'{"stage":', b'not json\n', b'x' * 17000 + b'\n'])
def test_newer_bad_tail_never_falls_back_to_awaiting(fixture, tail):
    log = fixture[4]
    with log.open('ab') as stream:
        stream.write(tail)
    assert fixture[3]()['state'] == 'unknown'
    assert fixture[3]()['input_needed'] is False


def test_legacy_stale_wait_is_not_active(fixture, monkeypatch):
    monkeypatch.setattr(status, '_processes', lambda: ([], 0))
    fixture[2]({'stage': 'awaiting_passphrase', 'passphrase_needed': True})
    assert fixture[3]()['state'] == 'idle'
    assert fixture[3]()['input_needed'] is False


def test_terminal_failure_and_exited_worker_do_not_prompt(fixture, monkeypatch):
    worker, record, write, read, log = fixture
    monkeypatch.setattr(status, '_processes', lambda: ([], 0))
    assert read()['state'] == 'worker_exited'
    record.update(state='failed', error_category='VaultIntegrityError',
                  failure_stage='validating_pre_backup', passphrase_needed=False)
    write(record)
    result = read()
    assert result['state'] == 'previous_run_failed'
    assert result['failure_stage'] == 'validating_pre_backup'
    assert result['input_needed'] is False
    record['runtime_binding'] = 'wrong'
    write(record)
    assert read()['state'] == 'unknown'


def test_no_worker_but_inaccessible_inventory_is_unknown(fixture, monkeypatch):
    monkeypatch.setattr(status, '_processes', lambda: ([], 1))
    assert fixture[3]()['state'] == 'unknown'
    assert fixture[3]()['input_needed'] is False


def test_duplicate_workers_and_bounded_inventory(fixture, monkeypatch):
    worker, record, write, read, log = fixture
    monkeypatch.setattr(status, '_processes', lambda: ([worker, worker], 0))
    assert read()['state'] == 'unknown'
    monkeypatch.setattr(status, '_processes', lambda: ([], 0))
    for index in range(64):
        create_private_file(log.parent / f'{index}.progress.jsonl')
    assert read()['state'] == 'unknown'


def test_linked_or_public_progress_is_unknown(fixture):
    log = fixture[4]
    # Parent traversal cannot redirect even a currently unlinked path.
    fixture[0]['cmdline'][-1] = str(log.parent / '..' / 'triage-progress' / log.name)
    assert fixture[3]()['state'] == 'unknown'


def test_file_identity_change_during_read(fixture, monkeypatch):
    log = fixture[4]
    original = Path.open
    class AppendDuringRead:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def read(self, size):
            data = self.stream.read(size)
            with original(log, 'ab') as writer:
                writer.write(b'{}\n')
            return data
    def changing_open(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return AppendDuringRead(stream) if path == log and args == ('rb',) else stream
    monkeypatch.setattr(Path, 'open', changing_open)
    assert fixture[3]()['state'] == 'unknown'
    assert fixture[3]()['input_needed'] is False


def test_hardlinked_progress_is_not_read(fixture):
    import os
    log = fixture[4]
    os.link(log, log.parent / 'alias.progress.jsonl')
    assert fixture[3]()['state'] == 'unknown'
    assert fixture[3]()['input_needed'] is False


def test_foreign_interpreter_is_unknown_not_idle(fixture):
    fixture[0]['exe'] = str(fixture[4].parent / 'other-python.exe')
    assert fixture[3]()['state'] == 'unknown'
    assert fixture[3]()['input_needed'] is False


def test_unknown_format_terminal_is_not_legacy_idle(fixture, monkeypatch):
    monkeypatch.setattr(status, '_processes', lambda: ([], 0))
    fixture[1]['format'] = True
    fixture[2](fixture[1])
    assert fixture[3]()['state'] == 'unknown'
