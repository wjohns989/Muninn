"""Private durable strict-ZDR admission and cost floors, NOT invoice hard caps.

One unresolved call blocks another across processes, restarts and UTC rollovers.
No transcript, provider request identifier, credential or PID is persisted.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path

from muninn.history.private_acl import VaultPermissionError, create_private_file, verify_private
from muninn.history.remote_policy import _paths

_MARKER = b"muninn-managed-remote-admission-v1\n"
_MAX = 2**63 - 1
_RESERVED_TIMEOUT = 900.0


class AdmissionError(RuntimeError):
    def __init__(self, code="remote_accounting_unavailable"):
        self.code = code
        super().__init__(code)


def _micros(value, *, ceiling=True):
    if type(value) not in (int, float, Decimal):
        raise AdmissionError("remote_accounting_invalid_cost")
    try:
        number = Decimal(str(value)) * 1_000_000
        if not number.is_finite() or not 0 <= number <= _MAX:
            raise ValueError
        return int(number.to_integral_value(rounding=ROUND_CEILING if ceiling else ROUND_FLOOR))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise AdmissionError("remote_accounting_invalid_cost") from exc


def _periods(now):
    stamp = datetime.fromtimestamp(now, timezone.utc)
    return stamp.strftime("%Y-%m-%d"), stamp.strftime("%Y-%m")


def _check_id(identifier):
    if (not isinstance(identifier, str) or len(identifier) != 32
            or any(c not in "0123456789abcdef" for c in identifier)):
        raise AdmissionError("remote_accounting_invalid_reference")


@contextmanager
def _db(root, *, initialize=False, generation=None):
    """Additive schema; a persistent marker/sentinel loss never resets spend."""
    if root is None:
        raise AdmissionError("remote_accounting_unconfigured")
    directory, policy_marker, database = _paths(Path(root))
    marker = directory / "admission-managed"
    db = None
    try:
        for path in (directory, policy_marker, database):
            verify_private(path)
        with policy_marker.open("rb") as stream:
            if stream.read(64) != b"muninn-managed-remote-policy-v1\n" or stream.read(1):
                raise AdmissionError()
        db = sqlite3.connect(f"{database.as_uri()}?mode=rw", uri=True, timeout=0.5)
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE" if initialize else "BEGIN")
        if initialize:
            _policy(db, generation)  # Reject revocation BEFORE writing the schema marker.
        columns = {row[1] for row in db.execute("PRAGMA table_info(policy)")}
        managed = (db.execute("SELECT accounting_version FROM policy WHERE id=1").fetchone()
                   if "accounting_version" in columns else None)
        has_tables = bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='remote_admissions'").fetchone())
        exists = marker.exists() or marker.is_symlink()
        if not exists:
            if managed not in (None, (0,)) or has_tables:
                raise AdmissionError()
            if not initialize:
                yield db, False
                return
            # Marker precedes schema commit: interrupted initialization fails closed.
            create_private_file(marker)
            with marker.open("wb") as stream:
                stream.write(_MARKER)
                stream.flush()
                os.fsync(stream.fileno())
            if "accounting_version" not in columns:
                db.execute("ALTER TABLE policy ADD COLUMN accounting_version INTEGER NOT NULL DEFAULT 0")
            db.execute("CREATE TABLE remote_admissions("
                       "id TEXT PRIMARY KEY, generation INTEGER NOT NULL, state TEXT NOT NULL "
                       "CHECK(state IN ('reserved','unknown','settled','released')), "
                       "started REAL NOT NULL, start_day TEXT NOT NULL, start_month TEXT NOT NULL, "
                       "finished REAL, end_day TEXT, end_month TEXT, cost_micro INTEGER, "
                       "resolution TEXT CHECK(resolution IN ('response','unsent','operator')))")
            db.execute("CREATE UNIQUE INDEX one_remote_admission ON remote_admissions((1)) "
                       "WHERE state IN ('reserved','unknown')")
            db.execute("UPDATE policy SET accounting_version=1 WHERE id=1")
            managed = (1,)
            # A refused reservation must not roll back a successful bootstrap
            # while leaving its durable marker behind.
            db.commit()
            db.execute("BEGIN IMMEDIATE")
        verify_private(marker)
        with marker.open("rb") as stream:
            if stream.read(64) != _MARKER or stream.read(1):
                raise AdmissionError()
        if managed != (1,):
            raise AdmissionError()
        if initialize and "batch_owner" not in {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}:
            db.execute("ALTER TABLE remote_admissions ADD COLUMN batch_owner TEXT")
        if initialize:
            for column in ("classification_job", "classification_input"):
                if column not in {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}:
                    db.execute(f"ALTER TABLE remote_admissions ADD COLUMN {column} TEXT")
        # Querying required columns detects missing/incompatible ledger tables.
        db.execute("SELECT id,generation,state,started,start_day,start_month,finished,"
                   "end_day,end_month,cost_micro,resolution FROM remote_admissions LIMIT 0")
        yield db, True
        db.commit()
    except (OSError, sqlite3.Error, VaultPermissionError, ValueError) as exc:
        if db is not None:
            db.rollback()
        raise AdmissionError() from exc
    finally:
        if db is not None:
            db.close()


def _policy(db, generation):
    if type(generation) is not int or generation < 1:
        raise AdmissionError("remote_consent_revoked")
    row = db.execute("SELECT enabled,generation,daily_usd,monthly_usd,override_ceiling "
                     "FROM policy WHERE id=1").fetchone()
    if not row or row[0] != 1 or row[1] != generation or row[4] not in (0, 1):
        raise AdmissionError("remote_consent_revoked")
    daily, monthly = (_micros(row[2], ceiling=False), _micros(row[3], ceiling=False))
    if not daily or not monthly or not row[4] and (daily > 10_000_000 or monthly > 100_000_000):
        raise AdmissionError("remote_accounting_invalid_policy")
    return daily, monthly


def _spent(db, day, month):
    daily, monthly = 0, 0
    for state, start_day, start_month, end_day, end_month, cost in db.execute(
            "SELECT state,start_day,start_month,end_day,end_month,cost_micro FROM remote_admissions "
            "WHERE state='settled' AND (start_day=? OR end_day=? OR start_month=? OR end_month=?)",
            (day, day, month, month)):
        if type(cost) is not int or not 0 <= cost <= _MAX:
            raise AdmissionError()
        daily += cost if day in (start_day, end_day) else 0
        monthly += cost if month in (start_month, end_month) else 0
    return daily, monthly


def _reported_usage(provider_status):
    if not isinstance(provider_status, dict) or provider_status.get("admission_ready") is not True:
        raise AdmissionError("daily_zdr_cap_unverified")
    return tuple(_micros(provider_status.get(k)) for k in ("usage_daily_usd", "usage_monthly_usd"))


def _require_headroom(db, caps, reported, day, month, ceiling):
    spent = _spent(db, day, month)
    if any(max(local, remote) >= cap for local, remote, cap in zip(spent, reported, caps)):
        raise AdmissionError("remote_admission_threshold_reached")
    if ceiling is not None and any(max(local, remote) + ceiling > cap
                                  for local, remote, cap in zip(spent, reported, caps)):
        raise AdmissionError("remote_admission_headroom_insufficient")


def reserve(root, generation, provider_status, *, now=None, batch_owner=None,
            classification_job=None, classification_input=None, classification_limit=None,
            classification_once=False, cost_ceiling_usd=None):
    """Reserve all remaining admission capacity, allowing one paid call at a time."""
    reported = _reported_usage(provider_status)
    ceiling = None if cost_ceiling_usd is None else _micros(cost_ceiling_usd)
    if ceiling == 0:
        raise AdmissionError("remote_accounting_invalid_cost")
    if type(classification_once) is not bool or classification_once and classification_job is None:
        raise AdmissionError("remote_accounting_invalid_reference")
    if batch_owner is not None:
        _check_id(batch_owner)
    if classification_limit is not None and (classification_job is None
            or type(classification_limit) is not int or not 1 <= classification_limit <= 10000):
        raise AdmissionError("remote_accounting_invalid_reference")
    if classification_job is not None or classification_input is not None:
        _check_id(classification_job)
        if (batch_owner is not None or not isinstance(classification_input, str)
                or len(classification_input) != 64 or any(c not in "0123456789abcdef" for c in classification_input)):
            raise AdmissionError("remote_accounting_invalid_reference")
    when = time.time() if now is None else now
    day, month = _periods(when)
    with _db(root, initialize=True, generation=generation) as (db, _):
        caps = _policy(db, generation)
        if (classification_limit is not None and db.execute(
                "SELECT COUNT(*) FROM remote_admissions WHERE classification_job IS NOT NULL").fetchone()[0]
                >= classification_limit):
            raise AdmissionError("classification_pilot_complete")
        # reserved is strictly pre-POST: mark_unknown must win its CAS before
        # transport may send. Expiry fences a stalled old worker, never guesses
        # that an unknown request was free. Serialize with the next reservation.
        db.execute("UPDATE remote_admissions SET state='released',finished=?,"
                   "end_day=?,end_month=?,resolution='unsent' "
                   "WHERE state='reserved' AND started<=?",
                   (when, day, month, when - _RESERVED_TIMEOUT))
        if classification_once and db.execute(
                "SELECT 1 FROM remote_admissions WHERE classification_job=? "
                "AND NOT(state='released' AND resolution='unsent' AND cost_micro IS NULL) LIMIT 1",
                (classification_job,)).fetchone():
            raise AdmissionError("classification_already_admitted")
        if db.execute("SELECT 1 FROM remote_admissions WHERE state IN ('reserved','unknown') LIMIT 1").fetchone():
            raise AdmissionError("remote_admission_busy")
        _require_headroom(db, caps, reported, day, month, ceiling)
        identifier = uuid.uuid4().hex
        db.execute("INSERT INTO remote_admissions(id,generation,state,started,start_day,start_month,batch_owner,"
                   "classification_job,classification_input) VALUES(?,?,'reserved',?,?,?,?,?,?)",
                   (identifier, generation, when, day, month, batch_owner, classification_job, classification_input))
    return Admission(Path(root), identifier, generation)


@dataclass(frozen=True)
class Admission:
    root: Path
    identifier: str
    generation: int

    def check_headroom(self, provider_status, *, cost_ceiling_usd, now=None):
        """Recheck this existing global hold immediately before transport.

        The caller must separately enforce the provider price/token contract.
        Unmanaged concurrent key usage is not made atomic by our local hold.
        """
        reported = _reported_usage(provider_status)
        ceiling = _micros(cost_ceiling_usd)
        if not ceiling:
            raise AdmissionError("remote_accounting_invalid_cost")
        day, month = _periods(time.time() if now is None else now)
        with _db(self.root) as (db, _):
            caps = _policy(db, self.generation)
            row = db.execute("SELECT state,start_day,start_month FROM remote_admissions "
                             "WHERE id=? AND generation=?", (self.identifier, self.generation)).fetchone()
            if not row or row[0] not in ("reserved", "unknown") or row[1:] != (day, month):
                raise AdmissionError("remote_accounting_conflict")
            _require_headroom(db, caps, reported, day, month, ceiling)

    def mark_unknown(self, *, policy_guard=None):
        with _db(self.root) as (db, _):
            # Upgrade before any reads so concurrent policy edits/admissions serialize.
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            _policy(db, self.generation)
            if policy_guard is not None and not policy_guard(db):
                raise AdmissionError("remote_consent_revoked")
            changed = db.execute("UPDATE remote_admissions SET state='unknown' "
                                 "WHERE id=? AND generation=? AND state='reserved'",
                                 (self.identifier, self.generation)).rowcount
            if changed != 1:
                raise AdmissionError("remote_accounting_conflict")

    def release_unsent(self):
        _finish(self.root, self.identifier, None, "unsent")

    def release_reserved(self):
        """Release only a pre-send slot; unknown/settled charges stay intact."""
        when = time.time()
        day, month = _periods(when)
        with _db(self.root) as (db, _):
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("UPDATE remote_admissions SET state='released',"
                "finished=?,end_day=?,end_month=?,resolution='unsent' "
                "WHERE id=? AND generation=? AND state='reserved'",
                (when, day, month, self.identifier, self.generation)).rowcount
            if changed:
                return True
            row = db.execute("SELECT state,resolution,cost_micro FROM remote_admissions "
                             "WHERE id=? AND generation=?",
                             (self.identifier, self.generation)).fetchone()
            return row == ("released", "unsent", None)

    def settle_response(self, data):
        usage = data.get("usage") if isinstance(data, dict) else None
        try:
            cost = _micros(usage.get("cost") if isinstance(usage, dict) else None)
        except AdmissionError:
            return False  # Keep the blocking unknown row; absence is never zero.
        _finish(self.root, self.identifier, cost, "response")
        return True


def _finish(root, identifier, cost, resolution, *, now=None):
    _check_id(identifier)
    when = time.time() if now is None else now
    day, month = _periods(when)
    state = "released" if resolution == "unsent" else "settled"
    with _db(root) as (db, _):
        db.rollback()
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT state,cost_micro FROM remote_admissions WHERE id=?", (identifier,)).fetchone()
        if not prior:
            raise AdmissionError("remote_accounting_invalid_reference")
        if prior == (state, cost):
            return
        if prior[0] not in ({"unknown"} if resolution == "response" else {"reserved", "unknown"}):
            raise AdmissionError("remote_accounting_conflict")
        db.execute("UPDATE remote_admissions SET state=?,finished=?,end_day=?,end_month=?,"
                   "cost_micro=?,resolution=? WHERE id=?", (state, when, day, month, cost, resolution, identifier))


def status(root, *, now=None, local_details=False):
    day, month = _periods(time.time() if now is None else now)
    with _db(root) as (db, managed):
        if not managed:
            return {"state": "uninitialized", "unresolved": 0, "daily_cost_usd": 0.0, "monthly_cost_usd": 0.0}
        rows = db.execute("SELECT id,state FROM remote_admissions WHERE state IN ('reserved','unknown')").fetchall()
        daily, monthly = _spent(db, day, month)
        result = {"state": "blocked" if rows else "ready", "unresolved": len(rows),
                  "daily_cost_usd": daily / 1_000_000, "monthly_cost_usd": monthly / 1_000_000}
        if local_details:
            result["admissions"] = [{"id": row[0], "state": row[1]} for row in rows]
        return result


def unknown_response(root, identifier, generation, *, batch_owner=None):
    """Check a durable pre-send uncertainty fence without releasing capacity."""
    _check_id(identifier)
    if batch_owner is not None:
        _check_id(batch_owner)
    if type(generation) is not int or generation < 1:
        return False
    with _db(root) as (db, managed):
        if not managed:
            return False
        if batch_owner is not None:
            if "batch_owner" not in {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}:
                return False
            bound = db.execute("SELECT batch_owner FROM remote_admissions WHERE id=?", (identifier,)).fetchone()
            if bound != (batch_owner,):
                return False
        row = db.execute("SELECT state,resolution,cost_micro FROM remote_admissions "
                         "WHERE id=? AND generation=?", (identifier, generation)).fetchone()
        return row == ("unknown", None, None)


def settled_response(root, identifier, generation, *, batch_owner=None, require_unowned=False,
                     classification_job=None, classification_input=None):
    """Authenticate one settled provider response for durable publication."""
    _check_id(identifier)
    if batch_owner is not None:
        _check_id(batch_owner)
    if type(generation) is not int or generation < 1:
        return False
    with _db(root) as (db, managed):
        if not managed:
            return False
        if not _classification_binding(db, identifier, classification_job, classification_input):
            return False
        if require_unowned:
            columns = {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}
            if "batch_owner" in columns:
                if db.execute("SELECT batch_owner FROM remote_admissions WHERE id=?", (identifier,)).fetchone() != (None,):
                    return False
        if batch_owner is not None:
            if "batch_owner" not in {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}:
                return False
            bound = db.execute("SELECT batch_owner FROM remote_admissions WHERE id=?", (identifier,)).fetchone()
            if bound != (batch_owner,):
                return False
        row = db.execute(
            "SELECT state,resolution,cost_micro FROM remote_admissions WHERE id=? AND generation=?",
            (identifier, generation),
        ).fetchone()
        return bool(row and row[0] == "settled" and row[1] == "response"
                    and type(row[2]) is int and 0 <= row[2] <= _MAX)


def _classification_binding(db, identifier, job, input_sha):
    columns = {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}
    if job is None and input_sha is None:
        if not {"classification_job", "classification_input"} <= columns:
            return True  # Legacy nonclassification accounting.
        return db.execute("SELECT classification_job,classification_input FROM remote_admissions WHERE id=?",
                          (identifier,)).fetchone() == (None, None)
    _check_id(job)
    if not isinstance(input_sha, str) or len(input_sha) != 64 or any(c not in "0123456789abcdef" for c in input_sha):
        raise AdmissionError("remote_accounting_invalid_reference")
    if not {"classification_job", "classification_input"} <= columns:
        return False
    return db.execute("SELECT classification_job,classification_input FROM remote_admissions WHERE id=?",
                      (identifier,)).fetchone() == (job, input_sha)


def unowned_unknown_response(root, identifier, generation, *, classification_job=None, classification_input=None):
    """Only an unsatisfied synchronous admission may acquire a new job binding."""
    _check_id(identifier)
    if type(generation) is not int or generation < 1:
        return False
    with _db(root) as (db, managed):
        if not managed:
            return False
        if not _classification_binding(db, identifier, classification_job, classification_input):
            return False
        columns = {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}
        if "batch_owner" in columns:
            if db.execute("SELECT batch_owner FROM remote_admissions WHERE id=?", (identifier,)).fetchone() != (None,):
                return False
        row = db.execute("SELECT state,resolution,cost_micro FROM remote_admissions WHERE id=? AND generation=?",
                         (identifier, generation)).fetchone()
        return row == ("unknown", None, None)


def classification_admission_state(root, identifier, generation, job, input_sha):
    """Authenticate owned bookkeeping without pretending unknown work settled."""
    _check_id(identifier)
    if type(generation) is not int or generation < 1:
        return None
    with _db(root) as (db, managed):
        if not managed or not _classification_binding(db, identifier, job, input_sha):
            return None
        if db.execute("SELECT batch_owner FROM remote_admissions WHERE id=?", (identifier,)).fetchone() != (None,):
            return None
        row = db.execute("SELECT state,resolution,cost_micro FROM remote_admissions WHERE id=? AND generation=?",
                         (identifier, generation)).fetchone()
        if row == ("unknown", None, None) or row == ("released", "unsent", None):
            return row[0]
        # Operator reconciliation establishes bookkeeping, not a provider reply.
        # settled_response remains response-only for staging/publication authority.
        if row and row[0] == "settled" and row[1] in {"response", "operator"} and type(row[2]) is int and 0 <= row[2] <= _MAX:
            return "settled"
        return None


def main(argv=None):
    def decimal_argument(text):
        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise argparse.ArgumentTypeError("A numeric cost is required") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "reconcile"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission")
    parser.add_argument("--cost-usd", type=decimal_argument)
    parser.add_argument("--confirm-outcome-reviewed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "reconcile":
            if not args.confirm_outcome_reviewed or args.cost_usd is None or not args.admission:
                parser.error("Reconciliation requires an admission, verified cost, and explicit outcome confirmation")
            _finish(args.root, args.admission, _micros(args.cost_usd), "operator")
        print(json.dumps(status(args.root, local_details=True), sort_keys=True))
        return 0
    except AdmissionError as exc:
        print(json.dumps({"state": "unavailable", "reason": exc.code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
