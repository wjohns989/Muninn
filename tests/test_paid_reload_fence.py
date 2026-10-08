"""Owned stop cannot interrupt an unsaved paid POST; isolated stores only."""
import sqlite3
import os
from contextlib import contextmanager
from copy import deepcopy
from uuid import uuid4

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from scripts.reload_shared_local import paid_stop_fence, preimage_databases
from tests.test_historical_batch_jobs import admission, fixture
from tests.test_historical_batch_worker import responses


@pytest.fixture(params=['portable_unlock', 'windows_unattended'])
def fence(request, tmp_path, monkeypatch):
    """Same real paid fence; only isolated archive unlock differs by platform."""
    if request.param == 'windows_unattended' and os.name != 'nt':
        pytest.skip('Actual unattended archive unlock requires Windows user protection')
    from muninn.history import secure_archive
    original = secure_archive.SecureHistoryArchive

    @contextmanager
    def scoped(data, archive, journal):
        assert archive.resolve() == (tmp_path / 'encrypted').resolve()
        with monkeypatch.context() as patch:
            if request.param == 'portable_unlock':
                def unlock(path):
                    assert path.resolve() == archive.resolve()
                    return original(path, 'test-only portable passphrase')
                patch.setattr(secure_archive, 'SecureHistoryArchive', unlock)
            with paid_stop_fence(data, archive, journal) as result:
                yield result
    return scoped


def submitted(tmp_path):
    journal, archive, outbox, ident, _ = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    paid = admission(journal, ident)
    outbox.begin_submission(ident, 0)
    journal.mark_historical_batch_dispatched(ident, paid.identifier)
    accepted, terminal = responses(archive, outbox, ident)
    outbox.save_submission(ident, 1, accepted)
    return journal, archive, outbox, ident, paid, terminal


def test_known_submission_fences_policy_outbox_and_journal_until_exit(tmp_path, fence):
    journal, archive, outbox, _, _, _ = submitted(tmp_path)
    paths = [journal.policy_root / 'remote_policy' / 'policy.sqlite3', outbox.path, journal.path]
    before = [path.read_bytes() for path in paths]
    with fence(journal.policy_root, archive.root, journal.path):
        for path in paths:
            with sqlite3.connect(path, timeout=0.01) as contender:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    contender.execute('BEGIN IMMEDIATE')
    assert [path.read_bytes() for path in paths] == before
    for path in paths:
        with sqlite3.connect(path, timeout=0.01) as contender:
            contender.execute('BEGIN IMMEDIATE')
            contender.rollback()


@pytest.mark.parametrize('begin_submission', [False, True])
def test_unknown_fence_before_or_after_outbox_marker_refuses_stop(tmp_path, begin_submission, fence):
    journal, archive, outbox, ident, _ = fixture(tmp_path)
    journal.reserve_historical_batch(ident)
    admission(journal, ident)  # committed before outbox marker/POST
    if begin_submission:
        outbox.begin_submission(ident, 0)
    with pytest.raises(RuntimeError, match='Paid submission identity is uncertain'):
        with fence(journal.policy_root, archive.root, journal.path):
            pytest.fail('uncertain submission must not reach process stop')


def test_unknown_synchronous_request_cannot_be_mistaken_for_known_batch(tmp_path, fence):
    journal, archive, _outbox, _ident, _ = fixture(tmp_path)
    admission(journal)  # no batch owner
    with pytest.raises(RuntimeError, match='Paid submission identity is uncertain'):
        with fence(journal.policy_root, archive.root, journal.path):
            pytest.fail('ordinary unknown POST must not reach process stop')


def test_wrong_admission_generation_refuses_stop(tmp_path, fence):
    journal, archive, _, _, _, _ = submitted(tmp_path)
    with sqlite3.connect(journal.policy_root / 'remote_policy' / 'policy.sqlite3') as db:
        db.execute('UPDATE remote_admissions SET generation=2')
    with pytest.raises(RuntimeError, match='Paid batch admission binding differs'):
        with fence(journal.policy_root, archive.root, journal.path):
            pytest.fail('wrong paid binding must not reach process stop')


def test_submitted_child_requires_exact_parent_and_admission(tmp_path, fence):
    journal, archive, outbox, parent, paid, terminal = submitted(tmp_path)
    outbox.save_terminal(parent, outbox.read(parent)['revision'], terminal)
    assert paid.settle_response(terminal)
    item = deepcopy(outbox.read(parent)['items'][0])
    item['custom_id'] = uuid4().hex
    child = outbox.prepare_repair(parent, [item])
    child_paid = admission(journal, child)
    outbox.bind_repair_admission(child, 0, child_paid.identifier)
    outbox.begin_submission(child, 1)
    accepted, _ = responses(archive, outbox, child)
    accepted['id'] = 'batch_child_fixture'
    outbox.save_submission(child, 2, accepted)
    with fence(journal.policy_root, archive.root, journal.path):
        pass
    with sqlite3.connect(journal.policy_root / 'remote_policy' / 'policy.sqlite3') as db:
        db.execute('UPDATE remote_admissions SET id=? WHERE id=?', (uuid4().hex, child_paid.identifier))
    with pytest.raises(RuntimeError, match='Paid batch admission binding differs'):
        with fence(journal.policy_root, archive.root, journal.path):
            pytest.fail('wrong repair admission must not reach process stop')


def test_tampered_outbox_is_not_trusted_from_plaintext_state(tmp_path, fence):
    journal, archive, outbox, _, _, _ = submitted(tmp_path)
    with sqlite3.connect(outbox.path) as db:
        db.execute("UPDATE batches SET sealed=x'00'")
    with pytest.raises(VaultIntegrityError):
        with fence(journal.policy_root, archive.root, journal.path):
            pytest.fail('unauthenticated submitted state must not reach process stop')


def test_stop_proof_only_decrypts_current_paid_batch(tmp_path, monkeypatch, fence):
    journal, archive, outbox, ident, _, _ = submitted(tmp_path)
    retained = deepcopy(outbox.read(ident))
    with sqlite3.connect(outbox.path) as db:
        for _ in range(40):
            retained['id'] = uuid4().hex
            retained['state'] = 'terminal_saved'
            db.execute('INSERT INTO batches VALUES(?,?,?,?)', (
                retained['id'], retained['revision'], retained['state'], outbox._seal(retained)))
    from muninn.history.historical_batch import BatchOutbox
    decrypted = []
    original = BatchOutbox._read

    def tracked(self, row):
        decrypted.append(row[0])
        return original(self, row)

    monkeypatch.setattr(BatchOutbox, '_read', tracked)
    with fence(journal.policy_root, archive.root, journal.path):
        pass
    assert decrypted == [ident]


def test_reload_preimage_includes_the_retained_paid_outbox_under_fence(tmp_path, fence):
    journal, archive, outbox, ident, _, _ = submitted(tmp_path)
    assert outbox.path in preimage_databases(archive.root, journal.path)
    before = outbox.read(ident)
    target = tmp_path / "isolated-encrypted-outbox-backup.db"
    with fence(journal.policy_root, archive.root, journal.path):
        with sqlite3.connect(outbox.path.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(target) as destination:
                source.backup(destination)
                assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with sqlite3.connect(outbox.path) as source, sqlite3.connect(target) as copy:
        assert source.execute("SELECT * FROM batches").fetchall() == copy.execute("SELECT * FROM batches").fetchall()
    assert outbox.read(ident) == before


def test_wrong_portable_phrase_never_yields_paid_stop_fence(tmp_path, monkeypatch):
    from muninn.history import secure_archive
    original = secure_archive.SecureHistoryArchive
    journal, archive, outbox, _, _, _ = submitted(tmp_path)
    paths = [journal.policy_root / 'remote_policy' / 'policy.sqlite3', outbox.path, journal.path]
    before = [path.read_bytes() for path in paths]
    def wrong_phrase(path):
        assert path.resolve() == archive.root.resolve()
        return original(path, 'wrong synthetic phrase')
    with monkeypatch.context() as patch:
        patch.setattr(secure_archive, 'SecureHistoryArchive', wrong_phrase)
        with pytest.raises(VaultIntegrityError):
            with paid_stop_fence(journal.policy_root, archive.root, journal.path):
                pytest.fail('Bad unlock must not reach an owned stop')
    assert [path.read_bytes() for path in paths] == before
