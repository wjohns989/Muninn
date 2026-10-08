"""Verify selected existing encrypted backup sections; no copy, recovery or publication."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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


def verify_sections(root, sections=SECTIONS, *, emit=lambda row: None):
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
                result = ReadOnlyContext(archive).verify_all()
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
    args = parser.parse_args()
    try:
        result = verify_sections(args.root, args.sections,
                                 emit=lambda row: print(json.dumps(row), flush=True))
    except Exception as exc:
        print(json.dumps({'stage': 'failed', 'error_type': type(exc).__name__}), flush=True)
        return 1
    print(json.dumps({'stage': 'selected_sections_verified', 'sections': list(result),
                      'source_evidence_verified_here': False, 'published': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
