"""Explicit native Gemini hook/capture probe; no inference or settings writes.

Invokes only the matching installed Muninn commands against one existing,
already archived Gemini transcript. Receipts are operator-triggered, not proof
that a new full Gemini chat/compaction cycle happened. Never prints source paths,
hook output, credentials, exception messages or transcript contents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from muninn.history.auto_routing import _local_setting
from muninn.history.hook_install import gemini_plan, installed
from muninn.history.secure_archive import SecureHistoryArchive
from scripts.enroll_history_backlog import ReadOnlyJournal


def select_source(manifest, home):
    base = (home / '.gemini').resolve()
    choices = []
    for locator, entries in manifest['files'].items():
        if not entries or entries[-1]['provider'] != 'gemini_cli' or entries[-1]['kind'] != 'transcript':
            continue
        path = Path(locator).resolve()
        if path.is_relative_to(base) and path.is_file():
            choices.append((path.stat().st_size, str(path), path))
    if not choices:
        raise RuntimeError('probe_source_unavailable')
    return min(choices)[2]


def receipts(client):
    reply = client.get('http://127.0.0.1:42069/history/status')
    reply.raise_for_status()
    return {(r['provider'], r['event']): r for r in reply.json()['data']['hook_receipts']}


def accepted(before, after, event):
    key = ('gemini_cli', event)
    old, new = before.get(key, {}), after.get(key, {})
    return (type(new.get('accepted_invocations')) is int
            and new['accepted_invocations'] > old.get('accepted_invocations', 0)
            and new.get('last_outcome') == 'capture_intent')


def source_state(journal, key):
    with journal._connect() as db:
        row = db.execute('SELECT state,revision,updated_at FROM jobs WHERE source_key=?', (key,)).fetchone()
    return dict(row) if row else None


def verify_capture(archive, source):
    """Authenticate the committed ciphertext and compare bytes, never print them."""
    before = source.stat()
    original = hashlib.sha256()
    with source.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            original.update(chunk)
    entry = archive._load_manifest()['files'][str(source.resolve())][-1]
    stored = hashlib.sha256()
    for chunk in archive._iter_verified_entry(entry):
        stored.update(chunk)
    after = source.stat()
    return ((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            and entry['provider'] == 'gemini_cli' and entry['kind'] == 'transcript'
            and stored.hexdigest() == entry['sha256'] == original.hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--execute-existing-capture', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = gemini_plan('http://127.0.0.1:42069')
        if plan.changed or not {'AfterAgent', 'PreCompress'} <= set(installed(plan)):
            raise RuntimeError('installed_settings_mismatch')
        package = Path(os.environ['APPDATA']) / 'npm/node_modules/@google/gemini-cli'
        info = json.loads((package / 'package.json').read_text(encoding='utf-8'))
        candidates = sorted((package / 'bundle').glob('core-*.js'))
        if not candidates or not shutil.which('node'):
            raise RuntimeError('installed_runner_unavailable')
        core = candidates[0]
        print(json.dumps({'stage': 'settings_verified', 'gemini_version': info['version'],
                          'core_sha256': hashlib.sha256(core.read_bytes()).hexdigest(),
                          'events': ['AfterAgent', 'PreCompress']}), flush=True)
        if not args.execute_existing_capture:
            return 0
        archive = SecureHistoryArchive(args.archive_root.resolve())
        source = select_source(archive._load_manifest(), Path.home())
        journal = ReadOnlyJournal(archive)
        key = journal.source_key(source, 'gemini_cli')
        prior = source_state(journal, key)
        token = _local_setting('MUNINN_AUTH_TOKEN')
        if not token:
            raise RuntimeError('local_auth_unavailable')
        with httpx.Client(headers={'Authorization': 'Bearer ' + token}, trust_env=False, timeout=15) as client:
            before = receipts(client)
            hooks = {event: next(h for g in plan.before['hooks'][event] for h in g['hooks']
                                  if h.get('name') == 'muninn-local-memory')
                     for event in ('AfterAgent', 'PreCompress')}
            # No path, source text or key in argv, command echo, or persisted input.
            probe = {'core': str(core), 'cwd': str(Path(__file__).resolve().parents[1]),
                     'transcript': str(source), 'hooks': hooks}
            started = time.time()
            result = subprocess.run([shutil.which('node'), str(Path(__file__).with_suffix('.mjs'))],
                                    input=json.dumps(probe), text=True, capture_output=True, timeout=50)
            rows = [json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"stage": "native_runner"') or line.startswith('{"stage":"native_runner"')]
            print(json.dumps({'stage': 'native_runner_results', 'events': rows,
                              'return_code': result.returncode}), flush=True)
            if result.returncode or len(rows) != 2 or any(r.get('success') is not True for r in rows):
                raise RuntimeError('native_runner_failed')
            after = receipts(client)
            if not all(accepted(before, after, event) for event in hooks):
                raise RuntimeError('durable_hook_receipt_missing')
            print(json.dumps({'stage': 'native_handoff_verified', 'operator_triggered': True,
                              'natural_chat_cycle': False, 'events': rows}), flush=True)
        deadline = time.monotonic() + 30
        state = source_state(journal, key)
        while state and state['state'] in {'pending', 'retry', 'capturing'} and time.monotonic() < deadline:
            time.sleep(0.5)
            state = source_state(journal, key)
        complete = bool(state and state['state'] == 'archived' and state['updated_at'] >= started
                        and (prior is None or state['revision'] > prior['revision']))
        authenticated = verify_capture(archive, source) if complete else False
        print(json.dumps({'stage': 'capture_result', 'archived_after_probe': complete,
                          'encrypted_content_matches_source': authenticated,
                          'state': state['state'] if state else 'missing', 'model_calls_launched_by_probe': 0}), flush=True)
        return 0 if complete and authenticated else 2
    except Exception as exc:
        allowed = {'probe_source_unavailable', 'installed_settings_mismatch', 'installed_runner_unavailable',
                   'local_auth_unavailable', 'native_runner_failed', 'durable_hook_receipt_missing'}
        code = str(exc) if type(exc) is RuntimeError and str(exc) in allowed else 'probe_unavailable'
        print(json.dumps({'stage': 'probe_failed', 'error_code': code}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
