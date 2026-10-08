"""Additive same-pin version enrollment; no inference or paid-batch authority."""
import re

from muninn.history.credential_crypto import VaultIntegrityError

_ID = '0' * 32
_PURPOSE = 'capture-historical-versions-cursor-v1'
_FIELDS = {'format', 'selection', 'generation', 'manifest_sha', 'total_sources',
           'total_versions', 'source_index', 'version_index', 'visited_versions',
           'queued', 'existing', 'excluded', 'complete'}
_PROVIDERS = {'codex', 'claude_code', 'gemini_cli'}


class HistoricalVersionsMixin:
    def _init_historical_versions(self, db):
        db.execute('CREATE TABLE IF NOT EXISTS capture_historical_versions '
                   '(id INTEGER PRIMARY KEY CHECK(id=1), sealed_cursor BLOB NOT NULL)')

    def _historical_versions_progress(self, db):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                          "AND name='capture_historical_versions'").fetchone():
            return None, None
        row = db.execute('SELECT sealed_cursor FROM capture_historical_versions WHERE id=1').fetchone()
        if row is None:
            return None, None
        value = self._open_search(row[0], _ID, _PURPOSE)
        numbers = _FIELDS - {'format', 'selection', 'manifest_sha', 'complete'}
        if (not isinstance(value, dict) or set(value) != _FIELDS
                or type(value['format']) is not int or value['format'] != 1
                or value['selection'] != 'all-versions-v1'
                or any(type(value[k]) is not int or not 0 <= value[k] < 2**63 for k in numbers)
                or value['generation'] < 1
                or not isinstance(value['manifest_sha'], str)
                or re.fullmatch(r'[0-9a-f]{64}', value['manifest_sha']) is None
                or type(value['complete']) is not bool
                or value['source_index'] > value['total_sources']
                or value['visited_versions'] > value['total_versions']
                or value['visited_versions'] != value['queued'] + value['existing'] + value['excluded']
                or value['complete'] != (value['source_index'] == value['total_sources'])
                or value['complete'] and (value['version_index'] or
                                         value['visited_versions'] != value['total_versions'])):
            raise VaultIntegrityError('Historical version enrollment cursor is invalid')
        return row[0], value

    def historical_versions_status(self):
        with self._connect() as db:
            return self._historical_versions_progress(db)[1]

    def _historical_versions_selection(self, latest, cursor):
        if latest is None or not latest['complete']:
            raise VaultIntegrityError('Historical version enrollment requires completed latest selection')
        manifest = self._historical_manifest(latest)
        lengths = [len(entries) for entries in manifest['files'].values()]
        if cursor is None:
            cursor = {'format': 1, 'selection': 'all-versions-v1',
                      'generation': latest['generation'], 'manifest_sha': latest['manifest_sha'],
                      'total_sources': latest['total_sources'], 'total_versions': sum(lengths),
                      'source_index': 0, 'version_index': 0, 'visited_versions': 0,
                      'queued': 0, 'existing': 0, 'excluded': 0, 'complete': not lengths}
        if (any(cursor[k] != latest[k] for k in ('generation', 'manifest_sha', 'total_sources'))
                or cursor['total_versions'] != sum(lengths)):
            raise VaultIntegrityError('Historical version enrollment pin differs')
        index, version = cursor['source_index'], cursor['version_index']
        if (index > len(lengths) or index == len(lengths) and version
                or index < len(lengths) and (version >= lengths[index] if lengths[index] else version != 0)
                or cursor['visited_versions'] != sum(lengths[:index]) + version):
            raise VaultIntegrityError('Historical version enrollment position differs')
        return cursor, manifest

    def _historical_version_batch(self, cursor, manifest, limit):
        cursor = dict(cursor)
        sources = list(manifest['files'].values())
        batch, examined = [], 0
        while cursor['source_index'] < len(sources) and examined < limit:
            entries = sources[cursor['source_index']]
            examined += 1  # Empty sources also consume the bounded CPU budget.
            if entries:
                version = cursor['version_index']
                entry = entries[version]
                batch.append((entry, version))
                cursor['visited_versions'] += 1
                cursor['version_index'] += 1
            if not entries or cursor['version_index'] == len(entries):
                cursor['source_index'] += 1
                cursor['version_index'] = 0
        cursor['complete'] = cursor['source_index'] == len(sources)
        return cursor, batch

    def preview_historical_versions(self, *, limit=128):
        self._enrichment_limit(limit)
        with self._connect() as db:
            if self._enrichment_baseline(db) is None:
                raise VaultIntegrityError('Historical version enrollment requires configured capture')
            _, latest = self._historical_progress(db)
            _, cursor = self._historical_versions_progress(db)
        cursor, manifest = self._historical_versions_selection(latest, cursor)
        _, batch = self._historical_version_batch(cursor, manifest, limit)
        counts = {'would_queue': 0, 'would_existing': 0, 'would_exclude': 0}
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            for entry, version in batch:
                if entry['provider'] not in _PROVIDERS or entry['kind'] != 'transcript':
                    counts['would_exclude'] += 1
                    continue
                receipt = self.archive._snapshot_receipt(entry, version)
                ident = self._enrichment_id(receipt)
                row = db.execute('SELECT work_id,sealed_receipt FROM capture_enrichment_sources WHERE work_id=?',
                                 (ident,)).fetchone()
                if row is not None and self._read_enrichment_receipt(row, baseline, db=db) != receipt:
                    raise VaultIntegrityError('Existing enrichment receipt differs')
                counts['would_existing' if row is not None else 'would_queue'] += 1
        return {'stage': 'preview', 'batch_versions': len(batch), 'enrollment': cursor, **counts}

    def enroll_historical_versions(self, *, limit=128):
        self._enrichment_limit(limit)
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            latest_seal, latest = self._historical_progress(db)
            previous, cursor = self._historical_versions_progress(db)
        if baseline is None:
            raise VaultIntegrityError('Historical version enrollment requires configured capture')
        cursor, manifest = self._historical_versions_selection(latest, cursor)
        if baseline > latest['generation']:
            raise VaultIntegrityError('Historical version baseline exceeds pin')
        if cursor['complete']:
            return cursor
        cursor, batch = self._historical_version_batch(cursor, manifest, limit)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if self._enrichment_baseline(db) != baseline or self._historical_progress(db)[0] != latest_seal:
                raise VaultIntegrityError('Historical version enrollment authority changed')
            current_seal, current = self._historical_versions_progress(db)
            if current_seal != previous:
                return current
            self._capture_schedule(db)
            for entry, version in batch:
                if entry['provider'] not in _PROVIDERS or entry['kind'] != 'transcript':
                    cursor['excluded'] += 1
                    continue
                receipt = self.archive._snapshot_receipt(entry, version)
                outcome = self._store_enrichment_receipt(db, receipt, baseline, historical=latest)
                cursor['queued' if outcome == 'queued' else 'existing'] += 1
            db.execute('INSERT INTO capture_historical_versions VALUES(1,?) '
                       'ON CONFLICT(id) DO UPDATE SET sealed_cursor=excluded.sealed_cursor',
                       (self._seal_search(cursor, _ID, _PURPOSE),))
        return cursor

    def _historical_versions_selected(self, db, latest, manifest):
        _, cursor = self._historical_versions_progress(db)
        if cursor is None:
            return {}
        cursor, _ = self._historical_versions_selection(latest, cursor)
        selected = {}
        sources = list(manifest['files'].values())
        for index, entries in enumerate(sources[:cursor['source_index'] + 1]):
            end = len(entries) if index < cursor['source_index'] else cursor['version_index']
            for version, entry in enumerate(entries[:end]):
                if entry['provider'] in _PROVIDERS and entry['kind'] == 'transcript':
                    receipt = self.archive._snapshot_receipt(entry, version)
                    selected[self._enrichment_id(receipt)] = receipt
        if len(selected) != cursor['queued'] + cursor['existing']:
            raise VaultIntegrityError('Historical version selection differs from cursor')
        return selected
