"""Existing backup section checks are nonmutating and never dispatch a model."""
import hashlib
import sqlite3

import pytest

from scripts.verify_history_backup_sections import verify_sections
from muninn.history.credential_crypto import VaultIntegrityError
from tests.test_paid_history_recovery import paid_history


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).digest()
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.asyncio
async def test_selected_backup_sections_keep_all_bytes_and_abandoned_stages(tmp_path, monkeypatch):
    from muninn.history.capture_journal import CaptureJournal

    journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    source = tmp_path / 'synthetic-capture.jsonl'
    source.write_text('isolated capture locator fixture', encoding='utf-8')
    assert journal.enqueue(source, 'codex', force=True) == 'queued'
    bundle = tmp_path / 'bundle'
    archive.backup_to(bundle)
    database = bundle / 'history_secure_archive' / 'source-evidence' / 'projections.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO attempts SELECT ?,vault,blob,sha,size,version,'building',count,digest,completion "
                   "FROM attempts LIMIT 1", ('f' * 32,))
    before = fingerprint(bundle)
    events = []

    def writer_forbidden(*args, **kwargs):
        raise AssertionError('A retained backup must not initialize a writer')

    monkeypatch.setattr(CaptureJournal, '__init__', writer_forbidden)
    report = verify_sections(bundle, ('archive', 'accounting', 'batches', 'journal', 'ledger', 'windows'),
                             emit=events.append)
    assert report['journal']['publications'] == 2
    assert report['journal']['captures'] == 1
    assert report['batches']['batches'] == 1
    assert report['windows']['windows'] == 3
    assert fingerprint(bundle) == before
    assert len(events) == 12 and all('text' not in event for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize('defect', ['ciphertext', 'provider', 'source_key'])
async def test_backup_capture_locators_still_authenticate_without_writer_init(tmp_path, monkeypatch, defect):
    from muninn.history.capture_journal import CaptureJournal

    journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    source = tmp_path / 'synthetic-capture.jsonl'
    source.write_text('isolated capture locator fixture', encoding='utf-8')
    journal.enqueue(source, 'codex', force=True)
    bundle = tmp_path / 'bundle'
    archive.backup_to(bundle)
    database = bundle / 'history_secure_archive' / 'capture-jobs.db'
    with sqlite3.connect(database) as db:
        if defect == 'ciphertext':
            sealed = bytearray(db.execute('SELECT sealed_locator FROM jobs').fetchone()[0])
            sealed[-1] ^= 1
            db.execute('UPDATE jobs SET sealed_locator=?', (bytes(sealed),))
        elif defect == 'provider':
            db.execute("UPDATE jobs SET provider='claude_code'")
        else:
            db.execute("UPDATE jobs SET source_key=?", ('f' * 64,))
    before = fingerprint(bundle)

    def writer_forbidden(*args, **kwargs):
        raise AssertionError('A retained backup must not initialize a writer')

    monkeypatch.setattr(CaptureJournal, '__init__', writer_forbidden)
    with pytest.raises(VaultIntegrityError):
        verify_sections(bundle, ('journal',))
    assert fingerprint(bundle) == before


@pytest.mark.asyncio
async def test_invalid_section_has_no_backup_effects(tmp_path):
    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    bundle = tmp_path / 'bundle'
    archive.backup_to(bundle)
    before = fingerprint(bundle)
    with pytest.raises(ValueError):
        verify_sections(bundle, ('unknown',))
    assert fingerprint(bundle) == before
