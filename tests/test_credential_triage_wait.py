"""Local admission waiting uses no secrets, models or paid retries."""
import io
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import triage_credential_ambiguity as runner
from muninn.history import auto_routing, remote_accounting


class TTY(io.StringIO):
    def isatty(self):
        return True


def harness(monkeypatch, *, states, enabled=lambda: True, provider=None):
    now = [0.0]
    sleeps = []
    remaining = iter(states)
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(runner.time, "sleep", sleep)
    monkeypatch.setattr(auto_routing, "remote_policy_snapshot",
                        lambda root: SimpleNamespace(enabled=enabled()))
    monkeypatch.setattr(remote_accounting, "status",
                        lambda root: {"unresolved": next(remaining)})
    monkeypatch.setattr(auto_routing, "openrouter_key_status", provider or
                        (lambda **kw: {"admission_ready": True, "state": "ready"}))
    return now, sleeps


def test_waits_for_owned_admission_without_provider_calls_while_busy(monkeypatch):
    calls, progress = [], []
    now, sleeps = harness(monkeypatch, states=[1, 1, 0], provider=lambda **kw:
                          calls.append("key-status") or {"admission_ready": True})
    result = runner.wait_remote_readiness("unused", 180, on_progress=progress.append)
    assert result["state"] == "ready" and now[0] == 120
    assert sleeps == [60, 60] and calls == ["key-status"]
    assert [row["remaining_seconds"] for row in progress] == [180, 120]
    assert all(row["passphrase_needed"] is False for row in progress)


def test_wait_deadline_is_anchored_and_never_resets(monkeypatch):
    _, sleeps = harness(monkeypatch, states=[1, 1, 1], provider=lambda **kw:
                        pytest.fail("busy admission queried provider"))
    result = runner.wait_remote_readiness("unused", 65)
    assert result["state"] == "remote_readiness_timeout"
    assert result["passphrase_needed"] is False and sleeps == [60, 5]


def test_revocation_during_wait_stops_before_provider_or_unlock(monkeypatch):
    now, sleeps = harness(monkeypatch, states=[1, 1], enabled=lambda: now[0] < 60,
                          provider=lambda **kw: pytest.fail("revoked route queried provider"))
    result = runner.wait_remote_readiness("unused", 180)
    assert result["state"] == "remote_consent_revoked" and sleeps == [60]


@pytest.mark.parametrize("state", ["key_missing", "budget_exhausted", "provider_unavailable"])
def test_only_busy_admission_is_waitable(monkeypatch, state):
    _, sleeps = harness(monkeypatch, states=[0], provider=lambda **kw:
                        {"admission_ready": False, "state": state})
    assert runner.wait_remote_readiness("unused", 180)["state"] == state
    assert sleeps == []


def test_slow_ready_check_cannot_prompt_after_deadline(monkeypatch):
    def provider(**kw):
        now[0] = 61
        return {"admission_ready": True}
    now, _ = harness(monkeypatch, states=[0], provider=provider)
    assert runner.wait_remote_readiness("unused", 60)["state"] == "remote_readiness_timeout"


def test_readiness_exceptions_never_echo_private_text(monkeypatch):
    harness(monkeypatch, states=[0], provider=lambda **kw:
            (_ for _ in ()).throw(RuntimeError("synthetic-private-never-log")))
    result = runner.wait_remote_readiness("unused", 180)
    assert result["state"] == "readiness_check_failed"
    assert "synthetic-private-never-log" not in json.dumps(result)


def test_keyboard_interrupt_stops_wait_without_unlock(monkeypatch):
    harness(monkeypatch, states=[1])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds:
                        (_ for _ in ()).throw(KeyboardInterrupt()))
    assert runner.wait_remote_readiness("unused", 180)["state"] == "readiness_wait_cancelled"


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_malformed_provider_readiness_never_unlocks(monkeypatch, flag):
    _, sleeps = harness(monkeypatch, states=[0], provider=lambda **kw:
                        {"admission_ready": flag})
    assert runner.wait_remote_readiness("unused", 180)["state"] == "readiness_check_failed"
    assert not sleeps


@pytest.mark.parametrize("count", [True, "0", -1, 2])
def test_malformed_accounting_never_queries_provider(monkeypatch, count):
    _, sleeps = harness(monkeypatch, states=[count], provider=lambda **kw:
                        pytest.fail("malformed accounting queried provider"))
    assert runner.wait_remote_readiness("unused", 180)["state"] == "readiness_check_failed"
    assert not sleeps


def test_default_zero_preserves_one_shot_readiness(monkeypatch):
    _, sleeps = harness(monkeypatch, states=[1])
    assert runner.wait_remote_readiness("unused")["state"] == "remote_admission_busy"
    assert not sleeps


def test_denied_provider_cannot_claim_ready_with_contradictory_state(monkeypatch):
    harness(monkeypatch, states=[0], provider=lambda **kw:
            {"admission_ready": False, "state": "ready"})
    assert runner.wait_remote_readiness("unused", 180)["state"] == "readiness_check_failed"


def test_noninteractive_cli_refuses_wait_without_reading_policy_or_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["triage", "--root", str(tmp_path),
        "--archive-root", str(tmp_path), "--policy-root", str(tmp_path),
        "--provider", "openrouter", "--wait-for-readiness", "180"])
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    output = TTY()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(runner, "wait_remote_readiness", lambda *a, **kw:
                        pytest.fail("invisible console waited"))
    monkeypatch.setattr(runner.getpass, "getpass", lambda *a: pytest.fail("unlock requested"))
    assert runner.main() == 2
    assert json.loads(output.getvalue())["state"] == "interactive_terminal_required"


def test_interactive_wait_reaches_hidden_prompt_only_once_ready(tmp_path, monkeypatch):
    harness(monkeypatch, states=[1, 0])
    output, prompts = TTY(), []
    monkeypatch.setattr(sys, "stdin", TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["triage", "--root", str(tmp_path),
        "--archive-root", str(tmp_path), "--policy-root", str(tmp_path),
        "--provider", "openrouter", "--wait-for-readiness", "180"])
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt:
                        prompts.append(prompt) or "synthetic-local-only-phrase")
    monkeypatch.setattr(runner, "CredentialReviewSource", lambda *a: None)
    monkeypatch.setattr(runner, "run", lambda **kw: {
        "groups_seen": 0, "next_cursor": None, "model_calls": 0,
        "queue_counts": {"pending": 0}})
    assert runner.main() == 0 and len(prompts) == 1
    assert "synthetic-local-only-phrase" not in output.getvalue()
    assert "waiting_for_remote_admission" in output.getvalue()
    assert "awaiting_passphrase" in output.getvalue()


@pytest.mark.parametrize("extra", [
    ["--provider", "ollama"], ["--model-limit", "0"], ["--check-readiness"],
    ["--wait-for-readiness", "86401"], ["--wait-for-readiness", "-1"],
])
def test_wait_rejects_inapplicable_or_unbounded_requests(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["triage", "--root", str(tmp_path),
        "--archive-root", str(tmp_path), "--policy-root", str(tmp_path),
        "--provider", "openrouter", "--wait-for-readiness", "180", *extra])
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2


def test_progress_file_is_live_and_excludes_text_values_and_cursors(tmp_path, capsys):
    from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
    create_private_directory(tmp_path / 'progress')
    log = tmp_path / 'progress' / 'progress.jsonl'
    create_private_file(log)
    runner.emit_progress({'stage': 'review_page', 'rows': 3,
                         'state': 'synthetic_secret_code', 'error_category': 'synthetic_secret_code',
                         'candidate': 'synthetic-secret-never-persist',
                         'next_cursor': {'id': 'synthetic-secret-cursor'},
                         'queue_counts': {'pending': 2, 'private-name': 5}}, log)
    assert json.loads(log.read_text()) == {
        'stage': 'review_page', 'rows': 3, 'queue_counts': {'pending': 2}}
    runner.emit_progress({'stage': 'awaiting_passphrase', 'passphrase_needed': True}, log)
    assert len(log.read_text().splitlines()) == 2
    assert 'synthetic-secret' not in log.read_text()
    verify_private(log)


def test_cli_progress_log_timeout_is_visible_without_unlock(tmp_path, monkeypatch):
    harness(monkeypatch, states=[1, 1])
    log = tmp_path / 'progress' / 'progress.jsonl'
    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr(sys, 'stdout', TTY())
    monkeypatch.setattr(runner.getpass, 'getpass', lambda *a: pytest.fail('timeout requested unlock'))
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(tmp_path / 'vault'),
        '--archive-root', str(tmp_path / 'archive'), '--policy-root', str(tmp_path),
        '--provider', 'openrouter', '--wait-for-readiness', '1', '--progress-log', str(log)])
    (tmp_path / 'vault').mkdir()
    (tmp_path / 'archive').mkdir()
    assert runner.main() == 2
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['stage'] == 'waiting_for_remote_admission'
    assert rows[-1]['state'] == 'remote_readiness_timeout'


@pytest.mark.parametrize('destination', ['existing', 'vault', 'archive'])
def test_progress_log_cannot_overwrite_or_pollute_private_stores(tmp_path, monkeypatch, destination):
    (tmp_path / 'vault').mkdir()
    (tmp_path / 'archive').mkdir()
    log = tmp_path / 'progress.jsonl' if destination == 'existing' else tmp_path / destination / 'progress.jsonl'
    if destination == 'existing':
        log.write_text('preserve this original', encoding='utf-8')
    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr(sys, 'stdout', TTY())
    monkeypatch.setattr(runner.getpass, 'getpass', lambda *a: pytest.fail('invalid log requested unlock'))
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(tmp_path / 'vault'),
        '--archive-root', str(tmp_path / 'archive'), '--progress-log', str(log), '--model-limit', '0'])
    if destination == 'existing':
        assert runner.main() == 2
        assert log.read_text() == 'preserve this original'
    else:
        with pytest.raises(SystemExit) as caught:
            runner.main()
        assert caught.value.code == 2 and not log.exists()


def test_existing_nonprivate_progress_parent_is_not_repermissioned(tmp_path, monkeypatch):
    parent = tmp_path / 'public'
    parent.mkdir()
    (tmp_path / 'vault').mkdir()
    (tmp_path / 'archive').mkdir()
    monkeypatch.setattr(sys, 'stdin', TTY())
    output = TTY()
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(runner, 'wait_remote_readiness', lambda *a, **kw:
                        pytest.fail('unsafe logging started admission wait'))
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(tmp_path / 'vault'),
        '--archive-root', str(tmp_path / 'archive'), '--provider', 'openrouter',
        '--policy-root', str(tmp_path), '--wait-for-readiness', '1',
        '--progress-log', str(parent / 'progress.jsonl')])
    from muninn.history.private_acl import VaultPermissionError, verify_private
    if sys.platform != 'win32':
        parent.chmod(0o755)
    assert runner.main() == 2
    assert json.loads(output.getvalue())['state'] == 'progress_log_unavailable'
    assert not (parent / 'progress.jsonl').exists()
    with pytest.raises(VaultPermissionError):
        verify_private(parent)


def test_progress_rejects_real_linked_ancestor_before_creating_anything(tmp_path):
    import os
    import subprocess
    target = tmp_path / 'target'
    target.mkdir()
    link = tmp_path / 'linked'
    if os.name == 'nt':
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                                capture_output=True, check=False)
        assert result.returncode == 0
    else:
        link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='linked components'):
        runner.validate_progress_path(link / 'private-child' / 'progress.jsonl')
    assert not (target / 'private-child').exists()


def test_progress_rejects_parent_traversal_before_normalizing(tmp_path):
    with pytest.raises(ValueError, match='parent traversal'):
        runner.validate_progress_path(tmp_path / 'ignored' / '..' / 'progress.jsonl')


@pytest.mark.parametrize('error_name', ['EOFError', 'VaultIntegrityError'])
def test_prompt_failure_replaces_stale_input_status_without_private_text(tmp_path, monkeypatch, error_name):
    from muninn.history.credential_crypto import VaultIntegrityError
    errors = {'EOFError': EOFError, 'VaultIntegrityError': VaultIntegrityError}
    vault = tmp_path / 'vault'
    vault.mkdir()
    log = tmp_path / 'progress' / 'progress.jsonl'
    output = TTY()
    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(vault),
        '--model-limit', '0', '--progress-log', str(log)])
    def fail_prompt(*args):
        raise errors[error_name]('synthetic-private-never-log')
    monkeypatch.setattr(runner.getpass, 'getpass', fail_prompt)
    assert runner.main() == 1
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['passphrase_needed'] is True
    assert rows[-1] == {'state': 'failed', 'error_category': error_name,
                        'failure_stage': 'awaiting_passphrase',
                        'backup_state': 'not_started', 'passphrase_needed': False,
                        'post_backup_unavailable': False}
    assert 'synthetic-private-never-log' not in log.read_text() + output.getvalue()


def test_failed_progress_destination_is_not_retried_when_reporting_failure(tmp_path, monkeypatch):
    vault = tmp_path / 'vault'
    vault.mkdir()
    log = tmp_path / 'progress' / 'progress.jsonl'
    output, writes = TTY(), []
    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(vault),
        '--model-limit', '0', '--progress-log', str(log)])
    monkeypatch.setattr(runner.getpass, 'getpass', lambda *args: 'synthetic-local-only-phrase')
    original = runner.emit_progress
    def emit(report, path=None):
        if path is not None:
            writes.append(report.get('stage') or report.get('state'))
            if report.get('stage') == 'review_page':
                raise OSError('synthetic-private-never-log')
        return original(report, path)
    def run(**kwargs):
        kwargs['on_progress']({'stage': 'review_page', 'rows': 1})
        pytest.fail('failed progress write did not stop review')
    monkeypatch.setattr(runner, 'emit_progress', emit)
    monkeypatch.setattr(runner, 'run', run)
    assert runner.main() == 1
    assert writes == ['awaiting_passphrase', 'passphrase_received', 'starting_review', 'review_page']
    final = json.loads(output.getvalue().splitlines()[-1])
    assert final['state'] == 'failed' and final['passphrase_needed'] is False
    assert final['failure_stage'] == 'review_page'
    assert 'synthetic-private-never-log' not in output.getvalue()


@pytest.mark.parametrize('operation,expected', [
    ('constructor', 'opening_pre_backup_vault'),
    ('backup', 'validating_pre_backup'),
    ('source', 'starting_review'),
    ('review', 'starting_review'),
    ('post_backup', 'validating_post_backup'),
])
def test_failure_records_last_entered_operation_without_unlock_material(tmp_path, monkeypatch, operation, expected):
    from muninn.history.credential_crypto import VaultIntegrityError
    vault = tmp_path / 'vault'
    vault.mkdir()
    log = tmp_path / 'progress' / 'progress.jsonl'
    output = TTY()
    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(sys, 'argv', ['triage', '--root', str(vault),
        '--archive-root', str(vault), '--apply', '--backup-before', str(tmp_path / 'before'),
        '--backup-after', str(tmp_path / 'after'), '--progress-log', str(log)])
    monkeypatch.setattr(runner.getpass, 'getpass', lambda *args: 'synthetic-passphrase-never-log')
    def fail():
        raise VaultIntegrityError('synthetic-private-never-log')
    class Store:
        def __init__(self, root):
            if operation == 'constructor':
                fail()
        def backup(self, destination, **kwargs):
            if operation == 'backup' or (operation == 'post_backup' and destination.name == 'after'):
                fail()
            return 1
        def ambiguity_status(self):
            return {'pending': 0}
    def source(*args):
        if operation == 'source':
            fail()
        return None
    def review(**kwargs):
        if operation == 'review':
            fail()
        return {'groups_seen': 0, 'next_cursor': None, 'model_calls': 0,
                'queue_counts': {'pending': 0}}
    monkeypatch.setattr(runner, 'CredentialStore', Store)
    monkeypatch.setattr(runner, 'CredentialReviewSource', source)
    monkeypatch.setattr(runner, 'run', review)
    assert runner.main() == 1
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[-1]['failure_stage'] == expected
    assert rows[-1]['passphrase_needed'] is False
    assert rows[1] == {'stage': 'passphrase_received', 'passphrase_needed': False}
    assert all(row.get('passphrase_needed') is not True for row in rows[1:])
    assert rows[-1]['backup_state'] == ('validated_pre_triage_backup' if operation in {
        'source', 'review', 'post_backup'} else 'not_started')
    assert 'synthetic-private-never-log' not in log.read_text() + output.getvalue()
    assert 'synthetic-passphrase-never-log' not in log.read_text() + output.getvalue()


def test_failure_stage_progress_field_rejects_arbitrary_text(tmp_path):
    from muninn.history.private_acl import create_private_directory, create_private_file
    create_private_directory(tmp_path / 'progress')
    log = tmp_path / 'progress' / 'progress.jsonl'
    create_private_file(log)
    runner.emit_progress({'state': 'failed', 'failure_stage': 'synthetic-private-never-log'}, log)
    runner.emit_progress({'state': 'failed', 'failure_stage': 'validating_pre_backup'}, log)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [
        {'state': 'failed'}, {'state': 'failed', 'failure_stage': 'validating_pre_backup'}]
