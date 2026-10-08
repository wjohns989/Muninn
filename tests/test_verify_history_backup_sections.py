"""Existing backup section checks are nonmutating and never dispatch a model."""
import hashlib
import sqlite3

import pytest

from scripts.verify_history_backup_sections import verify_sections
from tests.test_paid_history_recovery import paid_history


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).digest()
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.asyncio
async def test_selected_backup_sections_keep_all_bytes_and_abandoned_stages(tmp_path):
    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    bundle = tmp_path / 'bundle'
    archive.backup_to(bundle)
    database = bundle / 'history_secure_archive' / 'source-evidence' / 'projections.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO attempts SELECT ?,vault,blob,sha,size,version,'building',count,digest,completion "
                   "FROM attempts LIMIT 1", ('f' * 32,))
    before = fingerprint(bundle)
    events = []
    report = verify_sections(bundle, ('archive', 'accounting', 'batches', 'journal', 'ledger', 'windows'),
                             emit=events.append)
    assert report['journal']['publications'] == 2
    assert report['batches']['batches'] == 1
    assert report['windows']['windows'] == 3
    assert fingerprint(bundle) == before
    assert len(events) == 12 and all('text' not in event for event in events)


@pytest.mark.asyncio
async def test_invalid_section_has_no_backup_effects(tmp_path):
    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    bundle = tmp_path / 'bundle'
    archive.backup_to(bundle)
    before = fingerprint(bundle)
    with pytest.raises(ValueError):
        verify_sections(bundle, ('unknown',))
    assert fingerprint(bundle) == before
