"""Durable typed source observations, NOT a consolidated verified truth store.

Sensitive payloads and citations remain encrypted under the portable archive
envelope. This component does not dispatch models, schedule jobs or publish
ordinary indexes. Only exact, scoped user observations auto-file; typed model
interpretations remain provisional. Whole-database rollback is not detected.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.safe_span import sanitize_agent_span
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.streaming_redaction import redacted_fragments
from muninn.history.transcript_units import PARSER_VERSION, SourceUnit

POLICY = "source-observation-v1"
TYPES = {"observation", "fact", "preference", "decision", "task", "procedure",
         "project_attribution", "duplicate", "conflict", "possible_credential"}
_ZERO = "0" * 64
_LIMIT = 65536
_USER_HOME = re.compile(r"(?i)(?:\b[a-z]:[\\/]+users[\\/]+|(?<!\w)/home/)[^\\/\s\"']+")
_REVIEW_STATES = {"filed", "rejected", "needs_user"}
_REVIEWABLE_STATES = _REVIEW_STATES | {"provisional"}
_REVIEW_REASONS = {
    "filed": {"user_confirmed", "source_supported"},
    "rejected": {"user_rejected", "not_reliable"},
    "needs_user": {"insufficient_context", "possible_contradiction"},
}


class MemoryLedgerIntegrityError(RuntimeError):
    """No private payload or source path is included in integrity failures."""


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class MemoryLedger:
    def __init__(self, archive, *, read_only=False):
        self.archive = archive
        self.read_only = read_only
        self.root = Path(archive.root) / "memory-ledger"
        if not self.root.exists() and not read_only:
            create_private_directory(self.root)
        verify_private(self.root)
        self.db_path = self.root / "ledger.sqlite3"
        if not self.db_path.exists() and not read_only:
            create_private_file(self.db_path)
        verify_private(self.db_path)
        self._key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                         info=b"muninn memory ledger key v1").derive(archive._key)
        self.units = SourceEvidenceStore(archive, read_only=read_only)
        self._screen_cache = OrderedDict()
        self._entries = {(e["blob"], version): e
                         for versions in archive._load_manifest()["files"].values()
                         for version, e in enumerate(versions)}
        with self._connect() as db:
            if read_only:
                try:
                    db.execute("SELECT seq,ref,ciphertext FROM events LIMIT 0")
                    self._head(db)
                except sqlite3.DatabaseError as exc:
                    raise MemoryLedgerIntegrityError("Ledger schema is unavailable") from exc
                return
            db.executescript("CREATE TABLE IF NOT EXISTS events("
                             "seq INTEGER PRIMARY KEY,ref TEXT NOT NULL,ciphertext BLOB NOT NULL);"
                             "CREATE INDEX IF NOT EXISTS event_ref ON events(ref,seq);"
                             "CREATE TABLE IF NOT EXISTS head("
                             "id INTEGER PRIMARY KEY CHECK(id=1),ciphertext BLOB NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM head").fetchone() is None:
                if db.execute("SELECT 1 FROM events LIMIT 1").fetchone() is not None:
                    raise MemoryLedgerIntegrityError("Ledger head is missing")
                self._set_head(db, 0, _ZERO)

    @contextmanager
    def _connect(self):
        verify_private(self.db_path)
        db = (sqlite3.connect(self.db_path.absolute().as_uri() + "?mode=ro", uri=True, timeout=30)
              if self.read_only else sqlite3.connect(self.db_path, timeout=30))
        try:
            if self.read_only:
                db.execute("PRAGMA query_only=ON")
                yield db
                return
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA secure_delete=ON")
            with db:
                yield db
        finally:
            db.close()

    def _aad(self, purpose, seq, ref):
        return _json({"domain": "memory-ledger-v1", "vault": self.archive.vault_id,
                      "purpose": purpose, "seq": seq, "ref": ref})

    def _seal(self, value, purpose, seq, ref):
        raw = _json(value)
        if len(raw) > _LIMIT:
            raise ValueError("Ledger payload is not bounded")
        nonce = os.urandom(12)
        return nonce + AESGCM(self._key).encrypt(nonce, raw, self._aad(purpose, seq, ref))

    def _open(self, ciphertext, purpose, seq, ref):
        try:
            if not isinstance(ciphertext, bytes) or not 28 <= len(ciphertext) <= _LIMIT + 28:
                raise ValueError
            raw = AESGCM(self._key).decrypt(ciphertext[:12], ciphertext[12:], self._aad(purpose, seq, ref))
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (InvalidTag, ValueError, TypeError, UnicodeError) as exc:
            raise MemoryLedgerIntegrityError("Ledger authentication failed") from exc

    def _set_head(self, db, seq, digest):
        db.execute("INSERT OR REPLACE INTO head VALUES(1,?)",
                   (self._seal({"seq": seq, "digest": digest}, "head", 0, "head"),))

    def _head(self, db):
        row = db.execute("SELECT CASE WHEN length(ciphertext)<=? THEN ciphertext ELSE NULL END "
                         "FROM head WHERE id=1", (_LIMIT + 28,)).fetchone()
        head = self._open(row[0] if row else None, "head", 0, "head")
        if (set(head) != {"seq", "digest"} or type(head["seq"]) is not int
                or not 0 <= head["seq"] < 2**63 or not self._hex(head["digest"])):
            raise MemoryLedgerIntegrityError("Ledger head is invalid")
        count, first, last = db.execute("SELECT COUNT(*),MIN(seq),MAX(seq) FROM events").fetchone()
        if count != head["seq"] or (count and (first != 1 or last != count)):
            raise MemoryLedgerIntegrityError("Ledger event sequence is incomplete")
        return head

    @staticmethod
    def _hex(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    @staticmethod
    def _digest(seq, ref, sealed):
        return hashlib.sha256(seq.to_bytes(8, "big") + ref.encode("ascii") + sealed).hexdigest()

    def _walk(self, db):
        head, previous, expected = self._head(db), _ZERO, 1
        for seq, ref, sealed in db.execute(
                "SELECT seq,ref,CASE WHEN length(ciphertext)<=? THEN ciphertext ELSE NULL END "
                "FROM events ORDER BY seq", (_LIMIT + 28,)):
            if seq != expected or not self._hex(ref):
                raise MemoryLedgerIntegrityError("Ledger event reference is invalid")
            event = self._open(sealed, "event", seq, ref)
            if (set(event) != {"previous", "payload"} or event["previous"] != previous
                    or not isinstance(event["payload"], dict)):
                raise MemoryLedgerIntegrityError("Ledger event chain is invalid")
            previous = self._digest(seq, ref, sealed)
            expected += 1
            yield ref, event["payload"]
        if head["digest"] != previous or head["seq"] != expected - 1:
            raise MemoryLedgerIntegrityError("Ledger committed head does not match events")

    def _append(self, ref, payload, *, idempotent=False):
        self._append_batch([(ref, payload)], idempotent=idempotent)

    def _append_batch(self, events, *, idempotent=False):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Never extend a broken prefix merely because its tail/head survived.
            # Authenticate the old chain once for this bounded atomic batch.
            # Bulk backfill still requires measured scaling before activation.
            for _ in self._walk(db):
                pass
            head = self._head(db)
            seq, previous = head["seq"], head["digest"]
            seen = set()
            for ref, payload in events:
                if idempotent and (ref in seen or db.execute(
                        "SELECT 1 FROM events WHERE ref=? LIMIT 1", (ref,)).fetchone()):
                    # The whole chain was validated before accepting retries.
                    continue
                seq += 1
                sealed = self._seal({"previous": previous, "payload": payload}, "event", seq, ref)
                db.execute("INSERT INTO events VALUES(?,?,?)", (seq, ref, sealed))
                previous = self._digest(seq, ref, sealed)
                seen.add(ref)
            if seq != head["seq"]:
                self._set_head(db, seq, previous)

    def _source(self, entry, version, attempt, page):
        try:
            if type(version) is not int or self._entries.get((entry.get("blob"), version)) != entry:
                raise ValueError
            data = json.loads(self.units.get_page(entry, version, attempt, page))
            if (set(data) != {"unit", "fragment", "text", "final"}
                    or type(data["fragment"]) is not int or data["fragment"] < 0
                    or data["final"] is not False or not isinstance(data["text"], str)
                    or not 1 <= len(data["text"]) <= 4096):
                raise ValueError
            unit = SourceUnit(**data["unit"])
            if type(unit.ordinal) is not int or unit.ordinal < 0:
                raise ValueError
            return unit, data
        except (ProjectionIntegrityError, ValueError, TypeError, KeyError) as exc:
            raise MemoryLedgerIntegrityError("Ledger citation is not authenticated") from exc

    def _unit_info(self, entry, version, attempt, unit, *, use_persisted=True, persist_screen=True):
        """Stream the WHOLE source unit so labels/quotes cannot hide in other chunks.

        Only bounded raw/redacted digest state is retained. Within one worker,
        immutable completed-unit results use a bounded 128-entry cache and an
        encrypted cross-worker attestation bound to source and screen policy. No
        transaction remains open when the model is called or a ledger is written.
        """
        cache_key = self.units._screen_ref(self.units._screen_binding(entry, version, attempt, unit))
        if cache_key in self._screen_cache:
            self._screen_cache.move_to_end(cache_key)
            return self._screen_cache[cache_key]
        try:
            cached = self.units.screen_info(entry, version, attempt, unit) if use_persisted else None
        except ProjectionIntegrityError as exc:
            raise MemoryLedgerIntegrityError("Ledger context is not authenticated") from exc
        if cached is not None:
            self._screen_cache[cache_key] = cached
            if len(self._screen_cache) > 128:
                self._screen_cache.popitem(last=False)
            return cached
        raw, screened, body = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        body_length = 0

        def chunks():
            nonlocal body_length
            first = True
            for part in self.units.unit_fragments(entry, version, attempt, unit.ordinal):
                if part.unit != unit:
                    raise ProjectionIntegrityError("source unit identity changed")
                raw.update(part.text.encode("utf-8"))
                if not (first and part.text == f"\n\n{unit.role}: "):
                    encoded = part.text.encode("utf-8")
                    body.update(encoded)
                    body_length += len(encoded)
                first = False
                yield part.text

        try:
            for text in redacted_fragments(chunks()):
                screened.update(text.encode("utf-8"))
        except ProjectionIntegrityError as exc:
            raise MemoryLedgerIntegrityError("Ledger context is not authenticated") from exc
        info = raw.digest() == screened.digest(), body_length, body.digest()
        if not persist_screen:
            self._screen_cache[cache_key] = info
            if len(self._screen_cache) > 128:
                self._screen_cache.popitem(last=False)
            return info
        try:
            self.units._store_screen_info(entry, version, attempt, unit,
                raw_sha=raw.hexdigest(), screened_sha=screened.hexdigest(),
                body_length=body_length, body_sha=body.hexdigest())
        except ProjectionIntegrityError as exc:
            raise MemoryLedgerIntegrityError("Ledger context is not authenticated") from exc
        except sqlite3.OperationalError as exc:
            # The full immutable unit was authenticated and screened above.
            # Persistence is only a cross-worker optimization: a competing
            # reader may prevent its DELETE-journal commit. Keep this proof in
            # memory and rescan next time; never soften source/integrity errors.
            code = getattr(exc, "sqlite_errorcode", None)
            if type(code) is not int or code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
        self._screen_cache[cache_key] = info
        if len(self._screen_cache) > 128:
            self._screen_cache.popitem(last=False)
        return info

    @staticmethod
    def _screen(value):
        # Compare the entire serialized input, not merely a proposed claim.
        serialized = _json(value).decode("utf-8")
        if "".join(redacted_fragments([serialized])) != serialized:
            return False
        # Inspect actual strings too: JSON escaping cannot hide a private home
        # path or sensitive label from the normal agent privacy boundary.
        for text in value.values():
            if not isinstance(text, str):
                continue
            if _USER_HOME.search(text):
                return False
            for offset in range(0, len(text), 3500):
                span = text[max(0, offset - 256):offset + 3500]
                if sanitize_agent_span(span, max_chars=4000) != span:
                    return False
        return True

    def remote_input(self, entry, version, attempt, page):
        unit, data = self._source(entry, version, attempt, page)
        body = {"text": data["text"], "provider": unit.provider,
                "role": unit.role.casefold() if unit.role else None,
                "event_at": unit.event_at, "time_basis": unit.time_basis,
                "project_basis": unit.project_basis}
        # Deliberately omit cwd, path and native IDs. Unknown/redacted screening
        # denies egress rather than silently changing model evidence.
        return body if (self._unit_info(entry, version, attempt, unit)[0]
                        and self._screen(body)) else None

    def record(self, entry, version, attempt, page, proposal, *, model_identity):
        """Trusted direct-source rule API; model jobs must use record_batch."""
        ident, payload = self._prepare_record(entry, version, attempt, page, proposal,
                                             model_identity=model_identity, proposal_origin="source_rule")
        self._append(ident, payload, idempotent=True)
        return ident

    def record_batch(self, entry, version, attempt, items, *, model_identity, source_view=None):
        """Authenticate MODEL proposals before one bounded, all-or-none commit.

        This API does not dispatch inference or activate automatic backfill.
        No source-read transaction survives into the ledger write transaction.
        Callers cannot opt into source-rule filing authority through this API.
        """
        if (not isinstance(items, list) or not 1 <= len(items) <= 64
                or any(not isinstance(item, dict) or set(item) != {"page", "proposal"}
                       or type(item["page"]) is not int or item["page"] < 0 for item in items)):
            raise ValueError("Invalid bounded memory batch")
        events = self._prepare_batch(entry, version, attempt, items,
                                    model_identity=model_identity, source_view=source_view)
        self._append_batch(events, idempotent=True)
        return [ref for ref, _payload in events]

    def _range_projection(self, view):
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        from muninn.history.cited_zdr_projection import CitedZDRProjection
        # Reuse this ledger's authenticated source reader, not a new writer or
        # a persisted screening attestation. Full-unit EOF precedes admission.
        source = object.__new__(CitedAnalysisSource)
        source.archive, source.ledger = self.archive, self
        descriptor = view.get('window') if isinstance(view, dict) else None
        return CitedZDRProjection.from_source_view(source, descriptor, view)

    def _prepare_batch(self, entry, version, attempt, items, *, model_identity, source_view=None):
        projection = self._range_projection(source_view) if source_view is not None else None
        if projection is not None:
            descriptor = projection.source_view()['window']
            if (descriptor['blob'] != entry['blob'] or descriptor['sha256'] != entry['sha256']
                    or descriptor['version'] != version or descriptor['attempt'] != attempt):
                raise ValueError('Source view is not bound to publication source')
        return [self._prepare_record(entry, version, attempt, item['page'], item['proposal'],
                model_identity=model_identity, proposal_origin='model', _projection=projection)
                for item in items]

    def _prepare_record(self, entry, version, attempt, page, proposal, *, model_identity,
                        proposal_origin, _projection=None):
        if proposal_origin not in ("source_rule", "model"):
            raise ValueError("Invalid memory proposal origin")
        if (not isinstance(proposal, dict) or set(proposal) != {"type", "text", "quote", "start"}
                or proposal["type"] not in TYPES or not self._hex(model_identity)
                or any(not isinstance(proposal[k], str) or not 1 <= len(proposal[k]) <= 2048
                       for k in ("text", "quote")) or type(proposal["start"]) is not int
                or proposal["start"] < 0):
            raise ValueError("Invalid bounded memory proposal")
        unit, data = self._source(entry, version, attempt, page)
        start, quote = proposal["start"], proposal["quote"]
        if data["text"][start:start + len(quote)] != quote:
            raise ValueError("Memory quote does not match its authenticated source")
        if _projection is None:
            safe = self._screen({"window": data["text"], "claim": proposal["text"], "quote": quote})
            unit_safe, body_length, body_digest = self._unit_info(entry, version, attempt, unit)
            credential_risk = not safe or not unit_safe or proposal["type"] == "possible_credential"
        else:
            if proposal_origin != 'model':
                raise ValueError('Projected evidence does not grant source-rule authority')
            _projection.validate_page_proposal(page, proposal)
            # The canonical projection already drained the entire unit. It
            # cannot grant whole-unit observation authority and needs no second
            # drain or persisted raw-unit screening attestation.
            body_length, body_digest = 0, b''
            credential_risk = (proposal['type'] == 'possible_credential'
                               or not self._screen({'claim': proposal['text'], 'quote': quote}))
        project_ref = (hmac.new(self._key, b"project\0" + unit.cwd.encode("utf-8"), hashlib.sha256).hexdigest()
                       if unit.cwd else None)
        observation = bool(unit.role and unit.role.casefold() == "user" and proposal["text"] == quote
                           and body_length == len(quote.encode("utf-8"))
                           and body_digest == hashlib.sha256(quote.encode("utf-8")).digest())
        excerpt = bool(not observation and unit.role and unit.role.casefold() == "user"
                       and proposal["text"] == quote)
        filed = (proposal_origin == "source_rule" and observation
                 and proposal["type"] == "observation" and not credential_risk
                 and project_ref is not None and unit.project_basis != "unknown"
                 and unit.event_at is not None and unit.time_basis == "provider_record")
        identity = {
            "blob": entry["blob"], "sha": entry["sha256"], "version": version,
            "unit": unit.ordinal, "fragment": data["fragment"], "proposal": proposal,
            "policy": POLICY, "model": model_identity}
        # Preserve existing trusted source-rule IDs. Model candidates use a
        # separate domain so they cannot alias a prior filed observation.
        if proposal_origin == "model":
            identity["proposal_origin"] = proposal_origin
        if _projection is not None:
            identity['source_view'] = _projection.source_view()
        ident = hmac.new(self._key, b"candidate\0" + _json(identity), hashlib.sha256).hexdigest()
        payload = {"event": "candidate", "policy": POLICY, "model_identity": model_identity,
                   "proposal_origin": proposal_origin,
                   "type": "possible_credential" if credential_risk else proposal["type"],
                   "state": "pending" if credential_risk else "filed" if filed else "provisional",
                   "epistemic_kind": "source_observation" if observation else "source_excerpt" if excerpt else "model_interpretation",
                   "truth_status": "unverified_assertion" if observation else "unverified_excerpt" if excerpt else "model_inferred",
                   "text": proposal["text"], "quote": quote, "credential_risk": credential_risk,
                   "screening": "complete_unit",
                   "event_at": unit.event_at, "time_basis": unit.time_basis,
                   "project_ref": project_ref, "project_basis": unit.project_basis,
                   "citation": {"blob": entry["blob"], "sha": entry["sha256"], "version": version,
                                "attempt": attempt, "page": page, "unit": unit.ordinal,
                                "fragment": data["fragment"], "start": start, "length": len(quote),
                                "parser_version": PARSER_VERSION}}
        if _projection is not None:
            payload.update(screening='original_ranges', source_view=_projection.source_view(),
                           epistemic_kind='model_interpretation', truth_status='model_inferred')
        return ident, payload

    def _check_candidate(self, payload):
        try:
            cite = payload["citation"]
            if cite["parser_version"] != PARSER_VERSION:
                raise ValueError
            entry = self._entries[(cite["blob"], cite["version"])]
            unit, data = self._source(entry, cite["version"], cite["attempt"], cite["page"])
            if (entry["sha256"] != cite["sha"] or unit.ordinal != cite["unit"]
                    or data["fragment"] != cite["fragment"]
                    or data["text"][cite["start"]:cite["start"] + cite["length"]] != payload["quote"]):
                raise ValueError
        except (KeyError, ValueError, TypeError) as exc:
            raise MemoryLedgerIntegrityError("Ledger citation is not authenticated") from exc

    def _read_candidate(self, ident):
        if not self._hex(ident):
            raise ValueError("Invalid memory reference")
        candidate, state = None, None
        with self._connect() as db:
            db.execute("BEGIN")
            for ref, payload in self._walk(db):
                if ref != ident:
                    continue
                if payload.get("event") == "candidate":
                    if candidate is not None:
                        raise MemoryLedgerIntegrityError("Duplicate memory candidate")
                    self._check_candidate(payload)
                    candidate, state = payload, payload["state"]
                elif candidate is not None and payload.get("event") in ("decision", "human_review"):
                    state = self._apply_review_event(ident, candidate, state, payload)
                else:
                    raise MemoryLedgerIntegrityError("Memory review has no candidate")
        return candidate, state

    def _apply_review_event(self, ident, candidate, current_state, payload):
        """Validate a review transition while retaining legacy policy events."""
        if payload.get("event") == "decision":
            if payload.get("state") != "needs_user":
                raise MemoryLedgerIntegrityError("Invalid memory review decision")
            return "needs_user"
        required = {"event", "state", "expected_state", "reason", "actor",
                    "candidate_sha256", "citation_sha256"}
        state = payload.get("state")
        if (set(payload) != required or payload.get("event") != "human_review"
                or type(state) is not str
                or state not in _REVIEW_STATES
                or payload.get("expected_state") != current_state
                or current_state not in _REVIEWABLE_STATES
                or type(payload.get("reason")) is not str
                or payload.get("reason") not in _REVIEW_REASONS[state]
                or payload.get("actor") != "local-user"
                or candidate.get("credential_risk") is not False
                or candidate.get("screening") not in {"complete_unit", "original_ranges"}
                or not isinstance(candidate.get("citation"), dict)
                or payload.get("candidate_sha256") != hashlib.sha256(_json(candidate)).hexdigest()
                or payload.get("citation_sha256") != hashlib.sha256(_json(candidate["citation"])).hexdigest()):
            raise MemoryLedgerIntegrityError("Memory review decision authentication failed")
        return state

    def _public_candidate(self, ident, candidate, state, *, persist_screen=True, include_text=True):
        if candidate is None: return None
        public = {k: candidate[k] for k in ("type", "epistemic_kind", "truth_status",
                  "event_at", "time_basis", "project_ref", "project_basis")}
        public.update(id=ident, state=state, source_ref=hmac.new(
            self._key, b"citation\0" + _json(candidate["citation"]), hashlib.sha256).hexdigest())
        public["proposal_origin"] = candidate.get("proposal_origin", "legacy_unrecorded")
        if include_text and self._public_text_safe(candidate, persist_screen=persist_screen):
            public.update(text=sanitize_agent_span(candidate["text"], max_chars=2048),
                          quote=sanitize_agent_span(candidate["quote"], max_chars=2048))
        return public

    def _public_text_safe(self, candidate, *, persist_screen=True):
        if candidate["credential_risk"]:
            return False
        if candidate['screening'] == 'original_ranges':
            return self._public_projection(candidate) is not None
        if candidate['screening'] != 'complete_unit':
            return False
        cite = candidate["citation"]
        entry = self._entries[(cite["blob"], cite["version"])]
        unit, data = self._source(entry, cite["version"], cite["attempt"], cite["page"])
        # Ordinary agent reads retain the stronger pre-existing requirement:
        # freshly authenticate the whole cited unit, including unselected pages.
        # Cross-worker reuse only attests the original immutable unit for worker
        # preparation; it must not silently relax this public-read guarantee.
        return (self._unit_info(entry, cite["version"], cite["attempt"], unit,
                               use_persisted=False, persist_screen=persist_screen)[0]
                and self._screen({"window": data["text"], "claim": candidate["text"],
                                  "quote": candidate["quote"]}))

    def _public_projection(self, candidate):
        """One fresh canonical proof shared by public text/context and review."""
        try:
            if (candidate.get('credential_risk') is not False
                    or candidate.get('screening') != 'original_ranges'
                    or candidate.get('proposal_origin') != 'model'
                    or candidate.get('type') == 'possible_credential'
                    or not self._screen({'claim': candidate['text'], 'quote': candidate['quote']})):
                return None
            self._check_candidate(candidate)
            projection = self._range_projection(candidate['source_view'])
            desc, cite = projection.source_view()['window'], candidate['citation']
            if (desc['blob'] != cite['blob'] or desc['sha256'] != cite['sha']
                    or desc['version'] != cite['version'] or desc['attempt'] != cite['attempt']):
                return None
            if projection.remote_input(desc) is None:
                return None
            projection.validate_page_proposal(cite['page'], {'type': candidate['type'],
                'text': candidate['text'], 'quote': candidate['quote'], 'start': cite['start']})
            return projection
        except (ValueError, RuntimeError, KeyError, TypeError):
            return None

    def get(self, ident):
        self._screen_cache.clear()  # no stale source-safety proof across public reads
        candidate, state = self._read_candidate(ident)
        return self._public_candidate(ident, candidate, state)

    def search(self, query, *, limit=10):
        """One authenticated chain scan; no plaintext index or inference."""
        from muninn.history.blind_index import _terms
        self._screen_cache.clear()
        if (not isinstance(query, str) or not 1 <= len(query.encode('utf-8')) <= 512
                or type(limit) is not int or not 1 <= limit <= 20
                or not self._screen({"query": query})):
            raise ValueError("Invalid cited memory query")
        terms = list(dict.fromkeys(_terms(query)))
        if not 1 <= len(terms) <= 8:
            raise ValueError("Invalid cited memory query")
        candidates, current_states, matching = {}, {}, OrderedDict()
        with self._connect() as db:
            db.execute("BEGIN")
            for ref, payload in self._walk(db):
                if payload.get("event") == "candidate":
                    if ref in candidates:
                        raise MemoryLedgerIntegrityError("Duplicate memory candidate")
                    self._check_candidate(payload)
                    candidates[ref] = payload
                    current_states[ref] = payload["state"]
                    # Never match private credential text, even if the caller
                    # happens to know it. Exact refs use metadata-only get.
                    if payload["credential_risk"] or payload["screening"] not in {"complete_unit", "original_ranges"}:
                        continue
                    safe_text = sanitize_agent_span(payload["text"], max_chars=2048)
                    searchable = (safe_text + ' ' + payload["type"]).casefold()
                    if not all(term in searchable for term in terms): continue
                    public = self._public_candidate(ref, payload, current_states[ref])
                    if "text" not in public: continue
                    matching[ref] = public
                elif payload.get("event") in ("decision", "human_review") and ref in candidates:
                    current_states[ref] = self._apply_review_event(
                        ref, candidates[ref], current_states[ref], payload)
                    if ref in matching:
                        matching[ref]["state"] = current_states[ref]
                else:
                    raise MemoryLedgerIntegrityError("Invalid memory event sequence")
        eligible = [item for item in matching.values() if item["state"] != "rejected"]
        total = len(eligible)
        return {"matches": list(reversed(eligible[-limit:])), "total_matches": total,
                "truncated": total > limit, "ordering": "newest_publication_first"}

    def source(self, ident, *, max_chars=3000):
        """Follow an exact citation; unsafe units have metadata, not raw context.

        The returned expiring bearer only grants the existing redacted transcript
        projection, never a raw original or credential reveal. Its readable
        payload contains a fixed public term rather than any source quote.
        """
        from muninn.history.blind_index import SecureHistoryBlindIndex
        self._screen_cache.clear()
        if type(max_chars) is not int or not 1 <= max_chars <= 4000:
            raise ValueError("Invalid cited source context bound")
        candidate, state = self._read_candidate(ident)
        if candidate is None: return None
        cite = candidate["citation"]
        entry = self._entries[(cite["blob"], cite["version"])]
        unit, data = self._source(entry, cite["version"], cite["attempt"], cite["page"])
        projected = candidate['screening'] == 'original_ranges'
        projection = self._public_projection(candidate) if projected else None
        public = self._public_candidate(ident, candidate, state, include_text=not projected)
        if projection is not None:
            public.update(text=sanitize_agent_span(candidate['text'], max_chars=2048),
                          quote=sanitize_agent_span(candidate['quote'], max_chars=2048))
        result = {"memory": public, "provider": unit.provider,
                  "context_state": "withheld", "redaction": "strict-best-effort",
                  "citation": {"version": cite["version"], "unit": cite["unit"],
                      "fragment": cite["fragment"], "quote_start": cite["start"],
                      "quote_length": cite["length"], "parser_version": cite["parser_version"]},
                  "transcript_capability": SecureHistoryBlindIndex(self.archive)._capability(
                      entry, cite["version"], "transcript"),
                  "transcript_tool": "start_secure_history_transcript"}
        if "text" in public:
            if candidate['screening'] == 'original_ranges':
                descriptor = candidate['source_view']['window']
                window = projection.reopen(descriptor)
                quote_start = projection.validate_page_proposal(cite['page'], {
                    'type': candidate['type'], 'text': candidate['text'],
                    'quote': candidate['quote'], 'start': cite['start']})
                start = max(0, quote_start - min(500, max_chars // 4))
                result.update(context=window['text'][start:start + max_chars], context_state='available',
                              context_coordinate='cited_window', context_start=start,
                              context_truncated=start > 0 or start + max_chars < len(window['text']),
                              partial_visible_ranges=True)
                return result
            start = max(0, cite["start"] - min(500, max_chars // 4))
            result.update(context=sanitize_agent_span(data["text"][start:start+max_chars],
                                                     max_chars=max_chars),
                          context_state="available", context_fragment_start=start,
                          context_truncated=start > 0 or start+max_chars < len(data["text"]))
        return result

    def mark_needs_user(self, ident, *, reason):
        if reason not in {"possible_contradiction", "ambiguous_scope", "missing_evidence"}:
            raise ValueError("Invalid memory review reason")
        if self.get(ident) is None:
            raise ValueError("Memory candidate does not exist")
        self._append(ident, {"event": "decision", "state": "needs_user",
                             "reason": reason, "actor": "local-evidence-policy"})

    def resolve_review(self, ident, *, state, expected_state, reason):
        """Append an operator decision; this API is for an explicitly unlocked local CLI."""
        if not getattr(self.archive, "_unlocked_with_passphrase", False):
            raise PermissionError("Memory review requires a portable passphrase unlock")
        if (not self._hex(ident) or type(state) is not str or state not in _REVIEW_STATES
                or type(expected_state) is not str
                or expected_state not in _REVIEWABLE_STATES
                or type(reason) is not str
                or reason not in _REVIEW_REASONS[state]):
            raise ValueError("Invalid bounded memory review decision")
        # Do the expensive full-unit privacy/source check before taking the
        # ledger writer. The immutable candidate/citation binding is rechecked
        # again inside the CAS transaction below.
        candidate, observed_state = self._read_candidate(ident)
        if candidate is None or observed_state != expected_state:
            raise ValueError("Memory review state changed")
        if (candidate.get("credential_risk") is not False
                or candidate.get("screening") not in {"complete_unit", "original_ranges"}
                or not self._public_text_safe(candidate)):
            raise ValueError("Only safe noncredential cited memories can be reviewed")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current, current_state = None, None
            for ref, payload in self._walk(db):
                if ref != ident:
                    continue
                if payload.get("event") == "candidate":
                    if current is not None:
                        raise MemoryLedgerIntegrityError("Duplicate memory candidate")
                    self._check_candidate(payload)
                    current, current_state = payload, payload["state"]
                elif current is not None and payload.get("event") in ("decision", "human_review"):
                    current_state = self._apply_review_event(ref, current, current_state, payload)
                else:
                    raise MemoryLedgerIntegrityError("Memory review has no candidate")
            if (current is None or current_state != expected_state
                    or _json(current) != _json(candidate)
                    or current.get("credential_risk") is not False):
                raise ValueError("Memory review state changed")
            event = {"event": "human_review", "state": state,
                     "expected_state": expected_state, "reason": reason,
                     "actor": "local-user",
                     "candidate_sha256": hashlib.sha256(_json(current)).hexdigest(),
                     "citation_sha256": hashlib.sha256(_json(current["citation"])).hexdigest()}
            head = self._head(db)
            seq = head["seq"] + 1
            sealed = self._seal({"previous": head["digest"], "payload": event}, "event", seq, ident)
            db.execute("INSERT INTO events VALUES(?,?,?)", (seq, ident, sealed))
            self._set_head(db, seq, self._digest(seq, ident, sealed))
        self._screen_cache.clear()
        return ident

    def review_page(self, *, limit=20, cursor=None):
        """Browse a stable candidate prefix using current decisions, without writes.

        Every call verifies the full current chain, including its tail. Only the
        returned page and a lookahead receive fresh whole-unit privacy checks.
        This is bounded output, not constant-time or deadline-bounded retrieval.
        """
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Invalid review page bound")
        states = ["provisional", "needs_user"]
        anchor = None
        after = 0
        if cursor is not None:
            if type(cursor) is not str or not 1 <= len(cursor) <= 2048:
                raise ValueError("Invalid review cursor")
            try:
                sealed = base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
                anchor = self._open(sealed, "review-cursor", 0, "queue")
            except (ValueError, UnicodeError, MemoryLedgerIntegrityError) as exc:
                raise ValueError("Invalid review cursor") from exc
            if (set(anchor) != {"format", "seq", "digest", "after", "limit", "states"}
                    or type(anchor["format"]) is not int or anchor["format"] != 1
                    or type(anchor["seq"]) is not int or not 0 <= anchor["seq"] < 2**63
                    or not self._hex(anchor["digest"])
                    or type(anchor["after"]) is not int or not 0 <= anchor["after"] <= anchor["seq"]
                    or type(anchor["limit"]) is not int or anchor["limit"] != limit
                    or anchor["states"] != states):
                raise ValueError("Invalid review cursor")
            after = anchor["after"]
        self._screen_cache.clear()
        matches, has_more = [], False
        with self._connect() as db:
            db.execute("BEGIN")
            head = self._head(db)
            if anchor is None:
                anchor = {"format": 1, **head, "after": 0, "limit": limit, "states": states}
            _report, candidates, current_states = self._review_snapshot(db)
            row = db.execute("SELECT ref,ciphertext FROM events WHERE seq=?", (anchor["seq"],)).fetchone()
            digest = _ZERO if anchor["seq"] == 0 else (self._digest(anchor["seq"], *row) if row else None)
            if anchor["seq"] > head["seq"] or digest != anchor["digest"]:
                raise ValueError("Review cursor prefix changed")
            positions = db.execute("SELECT MIN(seq),ref FROM events WHERE seq<=? GROUP BY ref "
                                   "HAVING MIN(seq)>? ORDER BY MIN(seq)", (anchor["seq"], after))
            for seq, ref in positions:
                candidate = candidates[ref]
                if (current_states[ref] not in states or candidate["credential_risk"]
                        or candidate["screening"] not in {"complete_unit", "original_ranges"}):
                    continue
                public = self._public_candidate(ref, candidate, current_states[ref], persist_screen=False)
                if "text" not in public:
                    continue
                if len(matches) == limit:
                    has_more = True
                    break
                matches.append(public)
                after = seq
        next_cursor = None
        if has_more:
            next_cursor = base64.urlsafe_b64encode(self._seal(
                {**anchor, "after": after}, "review-cursor", 0, "queue")).decode("ascii")
        return {"matches": matches, "next_cursor": next_cursor, "has_more": has_more,
                "snapshot_events": anchor["seq"], "current_events": head["seq"],
                "limit": limit, "states": states, "credential_or_withheld_excluded": True}

    def review_status(self):
        """Return aggregate state counts, excluding credential-risk candidates."""
        with self._connect() as db:
            db.execute("BEGIN")
            _report, candidates, states = self._review_snapshot(db)
        counts = {state: 0 for state in ("provisional", "filed", "needs_user", "rejected")}
        for ref, candidate in candidates.items():
            if (candidate.get("credential_risk") is False
                    and candidate.get("screening") in {"complete_unit", "original_ranges"}
                    and (candidate['screening'] == 'complete_unit' or self._public_text_safe(candidate))):
                counts[states[ref]] += 1
        return counts

    def backup_review_preimage(self, destination):
        """Create a verified, encrypted ledger-only rollback preimage in a new private directory.

        This file depends on the original archive key and cited source archive;
        it is deliberately not a standalone portable archive backup.
        """
        if not getattr(self.archive, "_unlocked_with_passphrase", False):
            raise PermissionError("Review preimage requires a portable passphrase unlock")
        destination = Path(os.path.abspath(destination))
        if not destination.parent.is_dir() or destination.exists() or destination.is_symlink():
            raise ValueError("Review preimage destination must be a new directory")
        archive_root = Path(self.archive.root).resolve()
        try:
            if os.path.commonpath((os.path.normcase(str(destination)),
                                   os.path.normcase(str(archive_root)))) == os.path.normcase(str(archive_root)):
                raise ValueError("Review preimage must be outside the archive")
        except ValueError as exc:
            if str(exc) == "Review preimage must be outside the archive":
                raise
            raise ValueError("Invalid review preimage destination") from exc
        current = Path(destination.anchor)
        for part in destination.parts[1:]:
            current = current / part
            if current.exists():
                details = current.lstat()
                reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & reparse:
                    raise ValueError("Review preimage path contains a reparse point")
        verify_private(destination.parent)
        create_private_directory(destination)
        snapshot_path = destination / "memory-ledger.sqlite3"
        create_private_file(snapshot_path)
        with self._connect() as source:
            source.execute("BEGIN")
            before = self._verify_snapshot(source)
            source_head = self._head(source)
            copied = sqlite3.connect(snapshot_path)
            try:
                source.backup(copied)
                if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise MemoryLedgerIntegrityError("Review preimage database integrity failed")
                after = self._verify_snapshot(copied)
                copied_head = self._head(copied)
                if before != after or source_head != copied_head:
                    raise MemoryLedgerIntegrityError("Review preimage does not match the verified ledger")
            finally:
                copied.close()
        verify_private(snapshot_path)
        return {"path": str(snapshot_path), **before}

    def _review_snapshot(self, db):
        report = {"events": 0, "candidates": 0, "decisions": 0}
        candidates, states = {}, {}
        for ref, payload in self._walk(db):
            report["events"] += 1
            if payload.get("event") == "candidate" and ref not in candidates:
                self._check_candidate(payload)
                candidates[ref] = payload
                states[ref] = payload["state"]
                report["candidates"] += 1
            elif payload.get("event") in ("decision", "human_review") and ref in candidates:
                states[ref] = self._apply_review_event(ref, candidates[ref], states[ref], payload)
                report["decisions"] += 1
            else:
                raise MemoryLedgerIntegrityError("Invalid memory event sequence")
        return report, candidates, states

    def _verify_snapshot(self, db):
        report, _candidates, _states = self._review_snapshot(db)
        return report

    def verify_all(self):
        with self._connect() as db:
            db.execute("BEGIN")
            return self._verify_snapshot(db)

    @contextmanager
    def verified_reference_reader(self):
        """Authenticate once; check bounded refs in that same pinned snapshot.

        Construct all dependent stores before entering this reader. It is for
        offline backup validation, not a lock across live publication writers.
        """
        with self._connect() as db:
            db.execute("BEGIN")
            report = self._verify_snapshot(db)

            def contains(refs):
                if not isinstance(refs, list) or len(refs) > 64 or any(not self._hex(ref) for ref in refs):
                    raise ValueError("Invalid bounded memory references")
                # Full-chain validation established that the first event for
                # every indexed ref is an authenticated, cited candidate.
                return all(db.execute("SELECT 1 FROM events WHERE ref=? LIMIT 1", (ref,)).fetchone()
                           is not None for ref in refs)

            yield contains, report

    def verify_refs(self, refs):
        """Authenticate one full chain and every requested candidate citation."""
        if not isinstance(refs, list) or len(refs) > 64 or any(not self._hex(ref) for ref in refs):
            raise ValueError("Invalid bounded memory references")
        wanted, found = set(refs), set()
        with self._connect() as db:
            db.execute("BEGIN")
            for ref, payload in self._walk(db):
                if ref in wanted and payload.get("event") == "candidate":
                    if ref in found:
                        raise MemoryLedgerIntegrityError("Duplicate memory candidate")
                    self._check_candidate(payload)
                    found.add(ref)
        return found == wanted
