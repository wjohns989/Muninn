"""Existing backup section checks are nonmutating and never dispatch a model."""
import hashlib
import sqlite3

import pytest

from scripts.verify_history_backup_sections import ReadOnlyContext, verify_sections
from muninn.history.credential_crypto import VaultIntegrityError
from tests.test_paid_history_recovery import paid_history


@pytest.fixture
def explicit_fixture_unlock(recovery_copy, monkeypatch):
    """Portable library proof uses a real phrase; Windows CLI proof is unchanged."""
    def unlock(phrase):
        if recovery_copy.backend == 'portable_snapshot':
            from scripts import verify_history_backup_sections as verifier
            from muninn.history.secure_archive import SecureHistoryArchive

            # Only this standalone verifier's constructor gets the fixture's
            # explicit synthetic phrase. No DPAPI, key or verifier is mocked.
            monkeypatch.setattr(verifier, 'SecureHistoryArchive',
                                lambda path: SecureHistoryArchive(path, phrase))
    return unlock


def legacy_context_fixture(tmp_path, recovery_copy):
    from tests.test_credential_context import _fixture
    from muninn.history.credential_context import CredentialContextStore
    from muninn.history.secure_archive import SecureHistoryArchive

    archive, entry = _fixture(tmp_path)
    store = CredentialContextStore(archive)
    store._parser_revision = 1
    attempt = store.build_snapshot(entry, 0)
    store.record_review(entry, 0, attempt, 0, 'a' * 64, 'deferred')
    store._parser_revision = 2
    attempt = store.build_snapshot(entry, 0)
    store.record_review(entry, 0, attempt, 0, 'b' * 64, 'rejected')
    bundle = tmp_path / 'bundle'
    from muninn.history.private_acl import create_private_directory
    create_private_directory(bundle)
    recovery_copy(archive, bundle / 'history_secure_archive', 'synthetic portable recovery phrase')
    backup = SecureHistoryArchive(bundle / 'history_secure_archive', 'synthetic portable recovery phrase')
    for root in (archive.root, backup.root):
        with sqlite3.connect(root / 'credential-context' / 'projections.sqlite3') as db:
            db.execute('DROP TABLE context_remote_calls')
    return archive, backup, bundle


def test_legacy_context_witness_authenticates_without_changing_either_copy(tmp_path, monkeypatch,
                                                                       recovery_copy, explicit_fixture_unlock):
    archive, backup, bundle = legacy_context_fixture(tmp_path, recovery_copy)
    explicit_fixture_unlock('synthetic portable recovery phrase')
    from muninn.history.credential_context import CredentialContextStore

    def writer_forbidden(*args, **kwargs):
        raise AssertionError('The witness must not initialize a disk writer')

    monkeypatch.setattr(CredentialContextStore, '__init__', writer_forbidden)
    before = fingerprint(archive.root), fingerprint(bundle)
    report = ReadOnlyContext(backup).verify_with_source_witness(archive)
    assert report == {'snapshots': 2, 'contexts': 4,
                      'source_witness': 'matching_original_contents',
                      'remote_receipts': 'schema_absent_unknown'}
    assert (fingerprint(archive.root), fingerprint(bundle)) == before
    # Generic verification must still reject an unexplained missing table.
    with pytest.raises(sqlite3.OperationalError):
        ReadOnlyContext(backup).verify_all()
    report = verify_sections(bundle, ('credential_context',), legacy_context_source=archive.root)
    assert report['credential_context']['contexts'] == 4
    assert (fingerprint(archive.root), fingerprint(bundle)) == before


@pytest.mark.parametrize('defect', ['same_database', 'schema', 'row', 'equal_corruption',
                                  'equal_review_corruption', 'different_vault', 'different_key', 'modern_schema'])
def test_legacy_context_witness_rejects_missing_or_corrupt_proof(tmp_path, defect, recovery_copy):
    from muninn.history.secure_archive import SecureHistoryArchive
    from muninn.history.secure_projection_store import ProjectionIntegrityError

    archive, backup, bundle = legacy_context_fixture(tmp_path, recovery_copy)
    if defect == 'same_database':
        archive = backup
    elif defect == 'different_vault':
        archive = SecureHistoryArchive.create(tmp_path / 'other', 'synthetic recovery phrase')
    elif defect == 'different_key':
        archive._key = bytes(value ^ 1 for value in archive._key)
    elif defect == 'schema':
        with sqlite3.connect(backup.root / 'credential-context' / 'projections.sqlite3') as db:
            db.execute('CREATE INDEX extra_review_index ON context_reviews(page)')
    elif defect == 'modern_schema':
        for root in (archive.root, backup.root):
            with sqlite3.connect(root / 'credential-context' / 'projections.sqlite3') as db:
                db.execute('CREATE TABLE context_remote_calls(attempt TEXT, page INTEGER, '
                           'model_identity TEXT, ciphertext BLOB)')
    else:
        roots = (backup.root, archive.root) if defect.startswith('equal_') else (backup.root,)
        table, column = ('context_reviews', 'page') if defect == 'equal_review_corruption' else ('pages', 'ordinal')
        for root in roots:
            with sqlite3.connect(root / 'credential-context' / 'projections.sqlite3') as db:
                sealed = bytearray(db.execute(f'SELECT ciphertext FROM {table} WHERE {column}=0').fetchone()[0])
                sealed[-1] ^= 1
                db.execute(f'UPDATE {table} SET ciphertext=? WHERE {column}=0', (bytes(sealed),))
    before = fingerprint(archive.root), fingerprint(bundle)
    with pytest.raises(ProjectionIntegrityError):
        ReadOnlyContext(backup).verify_with_source_witness(archive)
    assert (fingerprint(archive.root), fingerprint(bundle)) == before


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).digest()
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.asyncio
async def test_selected_backup_sections_keep_all_bytes_and_abandoned_stages(tmp_path, monkeypatch,
                                                                        recovery_copy, explicit_fixture_unlock):
    from muninn.history.capture_journal import CaptureJournal

    journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    source = tmp_path / 'synthetic-capture.jsonl'
    source.write_text('isolated capture locator fixture', encoding='utf-8')
    assert journal.enqueue(source, 'codex', force=True) == 'queued'
    bundle = tmp_path / 'bundle'
    recovery_copy(archive, bundle, 'test-only portable passphrase')
    explicit_fixture_unlock('test-only portable passphrase')
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
async def test_backup_capture_locators_still_authenticate_without_writer_init(tmp_path, monkeypatch, defect,
                                                                           recovery_copy, explicit_fixture_unlock):
    from muninn.history.capture_journal import CaptureJournal

    journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    source = tmp_path / 'synthetic-capture.jsonl'
    source.write_text('isolated capture locator fixture', encoding='utf-8')
    journal.enqueue(source, 'codex', force=True)
    bundle = tmp_path / 'bundle'
    recovery_copy(archive, bundle, 'test-only portable passphrase')
    explicit_fixture_unlock('test-only portable passphrase')
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
async def test_invalid_section_has_no_backup_effects(tmp_path, recovery_copy, explicit_fixture_unlock):
    _journal, archive, _outbox, _ident, _bindings = await paid_history(tmp_path)
    bundle = tmp_path / 'bundle'
    recovery_copy(archive, bundle, 'test-only portable passphrase')
    explicit_fixture_unlock('test-only portable passphrase')
    before = fingerprint(bundle)
    with pytest.raises(ValueError):
        verify_sections(bundle, ('unknown',))
    assert fingerprint(bundle) == before
