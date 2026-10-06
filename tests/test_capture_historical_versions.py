"""All-version backfill on isolated encrypted archives; no provider dispatch."""
import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.credential_crypto import VaultIntegrityError
from scripts.enroll_history_backlog import ReadOnlyJournal
from tests.test_capture_historical_enrollment import legacy_fixture


def seals(journal):
    with journal._connect() as db:
        latest = journal._historical_progress(db)[0]
        grants = dict(db.execute('SELECT work_id,sealed_grant FROM capture_historical_receipts'))
    return latest, grants


def test_all_versions_enrolls_lost_older_content_without_rewriting_latest(tmp_path):
    journal, archive = legacy_fixture(tmp_path, count=2)
    latest = journal.enroll_historical_latest()
    old_seal, old_grants = seals(journal)
    first = journal.enroll_historical_versions(limit=1)
    assert first['visited_versions'] == 1 and first['queued'] == 1
    assert not first['complete']
    reopened = CaptureJournal(archive, recover=False)
    while not first['complete']:
        first = reopened.enroll_historical_versions(limit=1)
    assert first['total_versions'] == first['visited_versions'] == 4
    assert first['queued'] == first['existing'] == 2
    assert {r['version'] for r in reopened.pending_enrichment()} == {0, 1}
    assert len(reopened.pending_enrichment()) == 4
    current_seal, current_grants = seals(reopened)
    assert current_seal == old_seal
    assert all(current_grants[k] == value for k, value in old_grants.items())
    assert reopened.historical_enrollment_status() == latest
    with reopened._connect() as db:
        completed_seal = reopened._historical_versions_progress(db)[0]
    assert reopened.enroll_historical_versions(limit=128) == first
    with reopened._connect() as db:
        assert reopened._historical_versions_progress(db)[0] == completed_seal
    assert reopened.verify_all() == 0


def test_versions_cursor_and_receipts_roll_back_together(tmp_path, monkeypatch):
    journal, _ = legacy_fixture(tmp_path, count=2)
    journal.enroll_historical_latest()
    original = journal._store_enrichment_receipt
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError('isolated enrollment interruption')
        return result
    monkeypatch.setattr(journal, '_store_enrichment_receipt', interrupted)
    with pytest.raises(RuntimeError):
        journal.enroll_historical_versions(limit=4)
    assert len(journal.pending_enrichment()) == 2
    monkeypatch.setattr(journal, '_store_enrichment_receipt', original)
    assert journal.enroll_historical_versions(limit=4)['queued'] == 2
    assert journal.verify_all() == 0


def test_all_versions_preview_does_not_create_schema_or_mutate(tmp_path):
    journal, archive = legacy_fixture(tmp_path, count=2)
    journal.enroll_historical_latest()
    with journal._connect() as db:
        db.execute('DROP TABLE IF EXISTS capture_historical_versions')
    before = journal.path.read_bytes()
    preview = ReadOnlyJournal(archive).preview_historical_versions(limit=3)
    assert preview['batch_versions'] == 3
    assert preview['would_queue'] == 2 and preview['would_existing'] == 1
    assert before == journal.path.read_bytes()


def test_all_versions_partial_cursor_survives_portable_restore(tmp_path):
    journal, archive = legacy_fixture(tmp_path, count=2)
    journal.enroll_historical_latest()
    state = journal.enroll_historical_versions(limit=1)
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path/'restored', 'test-only portable passphrase')
    other = CaptureJournal(restored, recover=False)
    assert other.historical_versions_status() == state
    assert other.enroll_historical_versions(limit=128)['queued'] == 2
    assert other.verify_all() == 0


def test_versions_require_completed_latest_authority(tmp_path):
    journal, _ = legacy_fixture(tmp_path, count=2)
    with pytest.raises(VaultIntegrityError):
        journal.enroll_historical_versions()
    journal.enroll_historical_latest(limit=1)
    with pytest.raises(VaultIntegrityError):
        journal.preview_historical_versions()
    assert journal.historical_versions_status() is None
    assert len(journal.pending_enrichment()) == 1


def test_stale_version_writer_cannot_regress_cursor(tmp_path, monkeypatch):
    journal, _ = legacy_fixture(tmp_path, count=2)
    journal.enroll_historical_latest()
    original = journal._historical_versions_selection
    raced = False
    def race(latest, cursor):
        nonlocal raced
        result = original(latest, cursor)
        if not raced:
            raced = True
            journal.enroll_historical_versions(limit=1)
        return result
    monkeypatch.setattr(journal, '_historical_versions_selection', race)
    assert journal.enroll_historical_versions(limit=1)['visited_versions'] == 1
    assert journal.enroll_historical_versions(limit=1)['visited_versions'] == 2
    assert journal.verify_all() == 0


@pytest.mark.parametrize('damage', ['position', 'count', 'pin', 'lost_cursor'])
def test_version_evidence_damage_fails_verification(tmp_path, damage):
    from muninn.history.capture_historical_versions import _ID, _PURPOSE
    journal, _ = legacy_fixture(tmp_path, count=2)
    journal.enroll_historical_latest()
    journal.enroll_historical_versions(limit=1)
    with journal._connect() as db:
        if damage == 'lost_cursor':
            db.execute('DELETE FROM capture_historical_versions')
        else:
            state = journal._historical_versions_progress(db)[1]
            if damage == 'position': state['version_index'] = 999
            elif damage == 'count':
                state['visited_versions'] += 1
                state['existing'] += 1
            else: state['manifest_sha'] = 'a' * 64
            db.execute('UPDATE capture_historical_versions SET sealed_cursor=?',
                       (journal._seal_search(state, _ID, _PURPOSE),))
    with pytest.raises(VaultIntegrityError):
        journal.verify_all()


def test_older_version_is_selected_by_actual_batch_preparation(tmp_path):
    from muninn.history.batch_activation import configure_batch, prepare_next_batch
    from muninn.history.historical_batch import BatchOutbox
    from muninn.history.remote_policy import write_policy
    journal, archive = legacy_fixture(tmp_path, count=1)
    journal.enroll_historical_latest()
    journal.enroll_historical_versions()
    older = next(r for r in journal.pending_enrichment() if r['version'] == 0)
    write_policy(journal.policy_root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 1, 30, False))
    journal.queue_capture_windows(older, limit=4, remote_policy_generation=1)
    configure_batch(journal.policy_root, enabled=True)
    ident = prepare_next_batch(journal)
    assert ident is not None
    record = BatchOutbox(archive).read(ident)
    assert record['items'] and all(item['window']['version'] == 0 for item in record['items'])
    assert record['state'] == 'prepared'  # Local preparation only; no provider POST.
    owner = journal.historical_batch_owner()
    seals_before = seals(journal)
    assert journal.enroll_historical_versions() == journal.historical_versions_status()
    assert journal.historical_batch_owner() == owner
    assert BatchOutbox(archive).read(ident)['items'] == record['items']
    assert seals(journal) == seals_before


def test_cli_versions_preview_and_apply_are_bounded_metadata_only(tmp_path, monkeypatch, capsys):
    import json
    from scripts import enroll_history_backlog
    journal, archive = legacy_fixture(tmp_path, count=2)
    journal.enroll_historical_latest()
    monkeypatch.setattr(enroll_history_backlog, 'SecureHistoryArchive', lambda *_: archive)
    args = ['--archive-root', str(archive.root), '--all-versions', '--limit', '1']
    before = journal.path.read_bytes()
    assert enroll_history_backlog.main(args) == 0
    assert json.loads(capsys.readouterr().out)['would_queue'] == 1
    assert before == journal.path.read_bytes()
    assert enroll_history_backlog.main([*args, '--apply', '--max-batches', '2']) == 0
    output = capsys.readouterr().out
    assert json.loads(output.splitlines()[-1])['processed_by_models'] is False
    assert str(tmp_path) not in output and 'old observation' not in output
    assert journal.historical_versions_status()['visited_versions'] == 2
    assert not journal.historical_versions_status()['complete']


def test_all_versions_excludes_unsupported_sources_without_grants(tmp_path):
    journal, archive = legacy_fixture(tmp_path, count=1)
    path = tmp_path/'unsupported.txt'
    path.write_text('isolated non-history bytes', encoding='utf-8')
    archive.archive_file(path, 'export')
    journal.enroll_historical_latest()
    state = journal.enroll_historical_versions()
    assert state['complete'] and state['total_versions'] == 3
    assert state['queued'] == state['existing'] == state['excluded'] == 1
    assert len(journal.pending_enrichment()) == 2
    assert journal.verify_all() == 0


def test_empty_source_consumes_bounded_traversal_step(tmp_path):
    journal, _ = legacy_fixture(tmp_path, count=1)
    cursor = {'source_index': 0, 'version_index': 0, 'visited_versions': 0, 'complete': False}
    manifest = {'files': {'empty': [], 'nonempty': [{'fixture': True}]}}
    advanced, batch = journal._historical_version_batch(cursor, manifest, 1)
    assert advanced['source_index'] == 1 and advanced['visited_versions'] == 0
    assert batch == [] and not advanced['complete']
    advanced, batch = journal._historical_version_batch(advanced, manifest, 1)
    assert advanced['complete'] and advanced['visited_versions'] == 1
    assert len(batch) == 1
