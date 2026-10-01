"""Durable typed source observations, NOT a consolidated verified truth store.

Sensitive payloads and citations remain encrypted under the portable archive
envelope. This component does not dispatch models, schedule jobs or publish
ordinary indexes. Only exact, scoped user observations auto-file; typed model
interpretations remain provisional. Whole-database rollback is not detected.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.safe_span import sanitize_agent_span
from muninn.history.streaming_redaction import redacted_fragments
from muninn.history.transcript_units import PARSER_VERSION, SourceUnit

POLICY = "source-observation-v1"
TYPES = {"observation", "fact", "preference", "decision", "task", "procedure",
         "project_attribution", "duplicate", "conflict", "possible_credential"}
_ZERO = "0" * 64
_LIMIT = 65536
_USER_HOME = re.compile(r"(?i)(?:\b[a-z]:[\\/]+users[\\/]+|(?<!\w)/home/)[^\\/\s\"']+")


class MemoryLedgerIntegrityError(RuntimeError):
    """No private payload or source path is included in integrity failures."""


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class MemoryLedger:
    def __init__(self, archive):
        self.archive = archive
        self.root = Path(archive.root) / "memory-ledger"
        if not self.root.exists():
            create_private_directory(self.root)
        verify_private(self.root)
        self.db_path = self.root / "ledger.sqlite3"
        if not self.db_path.exists():
            create_private_file(self.db_path)
        verify_private(self.db_path)
        self._key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                         info=b"muninn memory ledger key v1").derive(archive._key)
        self.units = SourceEvidenceStore(archive)
        self._screen_cache = OrderedDict()
        self._entries = {(e["blob"], version): e
                         for versions in archive._load_manifest()["files"].values()
                         for version, e in enumerate(versions)}
        with self._connect() as db:
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
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
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

    def _unit_info(self, entry, version, attempt, unit):
        """Stream the WHOLE source unit so labels/quotes cannot hide in other chunks.

        Only bounded raw/redacted digest state is retained. Within one worker,
        immutable completed-unit results use a bounded 128-entry cache. No
        transaction remains open when the model is called or a ledger is written.
        """
        cache_key = (attempt, unit.ordinal)
        if cache_key in self._screen_cache:
            self._screen_cache.move_to_end(cache_key)
            return self._screen_cache[cache_key]
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

    def record_batch(self, entry, version, attempt, items, *, model_identity):
        """Authenticate MODEL proposals before one bounded, all-or-none commit.

        This API does not dispatch inference or activate automatic backfill.
        No source-read transaction survives into the ledger write transaction.
        Callers cannot opt into source-rule filing authority through this API.
        """
        if (not isinstance(items, list) or not 1 <= len(items) <= 64
                or any(not isinstance(item, dict) or set(item) != {"page", "proposal"}
                       or type(item["page"]) is not int or item["page"] < 0 for item in items)):
            raise ValueError("Invalid bounded memory batch")
        events = [self._prepare_record(entry, version, attempt, item["page"], item["proposal"],
                                      model_identity=model_identity, proposal_origin="model") for item in items]
        self._append_batch(events, idempotent=True)
        return [ref for ref, _payload in events]

    def _prepare_record(self, entry, version, attempt, page, proposal, *, model_identity, proposal_origin):
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
        safe = self._screen({"window": data["text"], "claim": proposal["text"], "quote": quote})
        unit_safe, body_length, body_digest = self._unit_info(entry, version, attempt, unit)
        credential_risk = not safe or not unit_safe or proposal["type"] == "possible_credential"
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

    def get(self, ident):
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
                elif payload.get("event") == "decision" and candidate is not None:
                    if payload.get("state") != "needs_user":
                        raise MemoryLedgerIntegrityError("Invalid memory review decision")
                    state = payload["state"]
                else:
                    raise MemoryLedgerIntegrityError("Memory review has no candidate")
        if candidate is None:
            return None
        public = {k: candidate[k] for k in ("type", "epistemic_kind", "truth_status",
                  "event_at", "time_basis", "project_ref", "project_basis")}
        public.update(id=ident, state=state, source_ref=hmac.new(
            self._key, b"citation\0" + _json(candidate["citation"]), hashlib.sha256).hexdigest())
        public["proposal_origin"] = candidate.get("proposal_origin", "legacy_unrecorded")
        if not candidate["credential_risk"] and candidate["screening"] == "complete_unit":
            public.update(text=sanitize_agent_span(candidate["text"], max_chars=2048),
                          quote=sanitize_agent_span(candidate["quote"], max_chars=2048))
        return public

    def mark_needs_user(self, ident, *, reason):
        if reason not in {"possible_contradiction", "ambiguous_scope", "missing_evidence"}:
            raise ValueError("Invalid memory review reason")
        if self.get(ident) is None:
            raise ValueError("Memory candidate does not exist")
        self._append(ident, {"event": "decision", "state": "needs_user",
                             "reason": reason, "actor": "local-evidence-policy"})

    def verify_all(self):
        report = {"events": 0, "candidates": 0, "decisions": 0}
        candidates = set()
        with self._connect() as db:
            db.execute("BEGIN")
            for ref, payload in self._walk(db):
                report["events"] += 1
                if payload.get("event") == "candidate" and ref not in candidates:
                    self._check_candidate(payload)
                    candidates.add(ref)
                    report["candidates"] += 1
                elif (payload.get("event") == "decision" and ref in candidates
                      and payload.get("state") == "needs_user"):
                    report["decisions"] += 1
                else:
                    raise MemoryLedgerIntegrityError("Invalid memory event sequence")
        return report

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
