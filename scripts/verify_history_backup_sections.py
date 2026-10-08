"""Verify selected existing encrypted backup sections; no copy, recovery or publication."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
import time
from unittest.mock import patch

from muninn.history.credential_context import CredentialContextStore
from muninn.history.historical_batch import BatchOutbox, _MARKER
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.portable_accounting import verify_snapshot
from muninn.history.private_acl import verify_private
from muninn.history.recovery_pool import unlinked
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.secure_projection_store import ProjectionIntegrityError
from scripts.enroll_history_backlog import ReadOnlyJournal

SECTIONS = ('archive', 'accounting', 'batches', 'journal', 'credential_context', 'ledger', 'windows')


class ReadOnlyCitedSource(CitedAnalysisSource):
    def __init__(self, archive, *, read_only=True):
        super().__init__(archive, read_only=True)


class ReadOnlyPlans(CitedWindowPlanStore):
    def __init__(self, archive, root=None, *, read_only=True):
        super().__init__(archive, root, read_only=True)


class ReadOnlyVerificationJournal(ReadOnlyJournal):
    @contextmanager
    def _connect(self, **kwargs):
        # Some inherited checks explicitly BEGIN. The standalone bundle is
        # frozen; leave transaction control to those checks, never to a writer.
        verify_private(self.path)
        db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA query_only=ON')
            yield db
        finally:
            db.close()


class ReadOnlyContext(CredentialContextStore):
    def __init__(self, archive):
        self._parser_revision = 2
        SourceEvidenceStore.__init__(self, archive, archive.root / 'credential-context', read_only=True)

    @contextmanager
    def _connect(self):
        if hasattr(self, '_memory_db'):
            try:
                yield self._memory_db
            finally:
                # Inherited readers BEGIN a logical snapshot. Disk adapters
                # close their connection; this immutable memory copy is reused.
                self._memory_db.rollback()
        else:
            with super()._connect() as db:
                yield db

    def verify_with_source_witness(self, source_archive):
        """Verify matching original contents, not absent historical remote receipts.

        Older unopened stores may lack the optional remote-call table. Generic
        verification still rejects that absence. Only an exact original witness
        permits the normal empty-table migration, entirely inside memory.
        """
        if (source_archive.vault_id != self.archive.vault_id or
                not hmac.compare_digest(source_archive._key, self.archive._key)):
            raise ProjectionIntegrityError('Credential context witness vault differs')
        source = ReadOnlyContext(source_archive)
        target_path, source_path = unlinked(self.db_path), unlinked(source.db_path)
        if target_path.samefile(source_path):
            raise ProjectionIntegrityError('Credential context witness is not distinct')
        with self._connect() as target_db, source._connect() as source_db:
            target_db.execute('BEGIN')
            source_db.execute('BEGIN')
            inventory = _context_inventory(target_db)
            tables = {row[1] for row in inventory['schema'] if row[0] == 'table'}
            if tables != {'attempts', 'pages', 'context_reviews', 'unit_screens'}:
                raise ProjectionIntegrityError('Credential context legacy schema differs')
            if inventory != _context_inventory(source_db):
                raise ProjectionIntegrityError('Credential context witness contents differ')
            memory = sqlite3.connect(':memory:')
            try:
                target_db.backup(memory)
                if _context_inventory(memory) != inventory:
                    raise ProjectionIntegrityError('Credential context memory copy differs')
                memory.execute('CREATE TABLE context_remote_calls(attempt TEXT NOT NULL, '
                               'page INTEGER NOT NULL, model_identity TEXT NOT NULL, ciphertext BLOB NOT NULL, '
                               'PRIMARY KEY(attempt,page,model_identity))')
                memory.execute('PRAGMA query_only=ON')
                reader = copy.copy(self)
                reader._memory_db = memory
                report = reader.verify_all()
            finally:
                memory.close()
        return {**report, 'source_witness': 'matching_original_contents',
                'remote_receipts': 'schema_absent_unknown'}


def _context_inventory(db):
    """Compare complete SQLite schema and type-bound logical rows, never print them."""
    schema = db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()
    rows = {}
    for kind, name, _table, _sql in schema:
        if kind != 'table':
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        columns = len(db.execute('SELECT * FROM ' + quoted + ' LIMIT 0').description)
        order = ','.join(str(index + 1) for index in range(columns))
        digest, count = hashlib.sha256(), 0
        for row in db.execute('SELECT * FROM ' + quoted + ' ORDER BY ' + order):
            for value in row:
                encoded = b'bytes:' + value if isinstance(value, bytes) else (
                    type(value).__name__ + ':' + json.dumps(value, ensure_ascii=False, allow_nan=False)).encode()
                digest.update(len(encoded).to_bytes(8, 'big'))
                digest.update(encoded)
            count += 1
        rows[name] = (count, digest.digest())
    return {'schema': schema, 'rows': rows}


class ReadOnlyOutbox(BatchOutbox):
    def __init__(self, archive):
        # Refuse constructor initialization of a missing recovery pair.
        for name in ('historical-batches.db', 'historical-batches-managed'):
            verify_private(unlinked(archive.root / name))
        super().__init__(archive)

    @contextmanager
    def _db(self, *, check_schema=True):
        verify_private(self.path)
        verify_private(self.marker)
        if self.marker.read_bytes() != _MARKER:
            raise ValueError('Batch marker differs')
        db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=5)
        try:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            if check_schema:
                if db.execute('SELECT version FROM sentinel').fetchall() != [(1,)]:
                    raise ValueError('Batch schema differs')
                db.execute('SELECT id,revision,state,sealed FROM batches LIMIT 0')
            yield db
        finally:
            db.close()


def verify_sections(root, sections=SECTIONS, *, emit=lambda row: None, legacy_context_source=None):
    root = unlinked(root)
    verify_private(root)
    archive = SecureHistoryArchive(root / 'history_secure_archive')
    journal = ReadOnlyVerificationJournal(archive)
    report = {}
    # Journal verification imports this class inside several methods. Force
    # read-only derived stores in THIS standalone verifier process only; do not
    # run constructor recovery or schema initialization on a retained backup.
    with patch('muninn.history.cited_analysis_source.CitedAnalysisSource', ReadOnlyCitedSource), \
            patch('muninn.history.historical_batch_jobs.BatchOutbox', ReadOnlyOutbox), \
            patch('muninn.history.cited_windows.CitedWindowPlanStore', ReadOnlyPlans):
        for section in sections:
            if section not in SECTIONS:
                raise ValueError('Unknown backup section')
            emit({'section': section, 'stage': 'verifying'})
            started = time.monotonic()
            if section == 'archive':
                result = archive.verify_all()
            elif section == 'accounting':
                verify_snapshot(archive, root)
                result = {'accounting_verified': True}
            elif section == 'batches':
                result = ReadOnlyOutbox(archive).verify_all()
            elif section == 'journal':
                result = {'captures': journal.verify_all(),
                          'publications': journal.verify_publications(),
                          'classifications': journal.verify_classifications()}
            elif section == 'credential_context':
                context = ReadOnlyContext(archive)
                result = (context.verify_all() if legacy_context_source is None else
                          context.verify_with_source_witness(SecureHistoryArchive(unlinked(legacy_context_source))))
            elif section == 'ledger':
                result = MemoryLedger(archive, read_only=True).verify_all()
            else:
                result = CitedWindowPlanStore(archive, read_only=True).verify_all()
            report[section] = result
            emit({'section': section, 'stage': 'verified', 'counts': result,
                  'elapsed_seconds': round(time.monotonic() - started, 1)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sections', nargs='+', choices=SECTIONS, default=SECTIONS)
    parser.add_argument('--legacy-context-source', type=Path,
                        help='Original archive witness for legacy context contents; no historical remote-receipt proof')
    args = parser.parse_args()
    try:
        result = verify_sections(args.root, args.sections,
                                 emit=lambda row: print(json.dumps(row), flush=True),
                                 legacy_context_source=args.legacy_context_source)
    except Exception as exc:
        print(json.dumps({'stage': 'failed', 'error_type': type(exc).__name__}), flush=True)
        return 1
    print(json.dumps({'stage': 'selected_sections_verified', 'sections': list(result),
                      'source_evidence_verified_here': False, 'published': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
