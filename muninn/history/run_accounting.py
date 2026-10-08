"""Read-only admission totals; no provider call, initialization or reconciliation.

The interval includes ALL managed activity started since the supplied cutoff,
not just backlog jobs. A batch bill is one admission, never multiplied by its
window count. Provider response charges are rounded up per admission to a
microdollar by the existing ledger; operator settlements remain distinguishable.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

from muninn.history.remote_accounting import AdmissionError, _MAX, _db

SCOPE = "all_managed_admissions_since_start"
ROUNDING = "up_per_admission_micro_usd"


def _timestamp(value):
    return type(value) in (int, float) and 0 <= value <= 253402300799 and math.isfinite(value)


def run_status(root, *, since, now=None):
    sampled = time.time() if now is None else now
    if not _timestamp(since) or not _timestamp(sampled) or since > sampled:
        raise ValueError("Invalid run interval")
    result = {"scope": SCOPE, "rounding": ROUNDING, "since": since,
        "since_utc": datetime.fromtimestamp(since, timezone.utc).isoformat(timespec="microseconds"),
        "sampled_at": sampled, "state": "uninitialized", "settled_cost_usd": None,
        "admission_states": None, "settled_resolutions": None, "global_unresolved": None,
        "settled_resolution_cost_usd": None,
        "batch_owned": {"state": "unknown_legacy_schema", "settled_admissions": None,
                        "settled_cost_usd": None}}
    with _db(root, initialize=False) as (db, managed):
        # _db establishes one SQLite read snapshot before yielding. Timestamp
        # that snapshot, not invocation: an admission may finish during setup.
        sampled = time.time() if now is None else now
        if not _timestamp(sampled) or sampled < since:
            raise AdmissionError()
        result["sampled_at"] = sampled
        if not managed:
            return result
        columns = {row[1] for row in db.execute("PRAGMA table_info(remote_admissions)")}
        diagnostics = "diagnostic_parent" in columns
        attribution = "batch_owner" in columns
        rows = db.execute("SELECT state,started,finished,cost_micro,resolution," +
                          ("batch_owner" if attribution else "NULL") + " FROM remote_admissions")
        counts = {state: 0 for state in ("settled", "released", "reserved", "unknown")}
        resolutions = {"response": 0, "operator": 0}
        resolution_cost = {"response": 0, "operator": 0}
        cost_total = batch_cost = batch_settled = unresolved = 0
        for state, started, finished, cost, resolution, batch in rows:
            # Validate before interval filtering: malformed earlier rows must not
            # silently disappear from an apparently healthy accounting reader.
            if state not in counts or not _timestamp(started) or started > sampled:
                raise AdmissionError()
            if attribution and batch is not None and (not isinstance(batch, str) or
                    len(batch) != 32 or any(c not in "0123456789abcdef" for c in batch)):
                raise AdmissionError()
            if state in {"reserved", "unknown"}:
                if finished is not None or cost is not None or resolution is not None:
                    raise AdmissionError()
                unresolved += 1
            else:
                if not _timestamp(finished) or not started <= finished <= sampled:
                    raise AdmissionError()
                if state == "released":
                    if cost is not None or resolution != "unsent":
                        raise AdmissionError()
                elif (resolution not in resolutions or type(cost) is not int or not 0 <= cost <= _MAX):
                    raise AdmissionError()
            if started < since:
                continue
            counts[state] += 1
            if state == "settled":
                cost_total += cost
                resolutions[resolution] += 1
                resolution_cost[resolution] += cost
                if batch is not None:
                    batch_cost += cost
                    batch_settled += 1
        result.update(state="observed", settled_cost_usd=cost_total / 1_000_000,
                      admission_states=counts, settled_resolutions=resolutions,
                      settled_resolution_cost_usd={name: cost / 1_000_000
                                                   for name, cost in resolution_cost.items()},
                      global_unresolved=unresolved)
        if attribution:
            result["batch_owned"] = {"state": "known", "settled_admissions": batch_settled,
                                     "settled_cost_usd": batch_cost / 1_000_000}
        if diagnostics:
            diagnostic_counts = dict(db.execute("SELECT state,COUNT(*) FROM remote_admissions "
                "WHERE diagnostic_parent IS NOT NULL AND started>=? GROUP BY state", (since,)))
            diagnostic_cost = db.execute("SELECT COALESCE(SUM(cost_micro),0) FROM remote_admissions "
                "WHERE diagnostic_parent IS NOT NULL AND started>=? AND state='settled'", (since,)).fetchone()[0]
            result["diagnostics"] = {"admission_states": diagnostic_counts,
                "settled_cost_usd": diagnostic_cost / 1_000_000,
                "included_in_total": True, "backlog_publications": 0}
            if "diagnostic_kind" in columns:
                by_kind = {}
                for kind, state, count, cost in db.execute("SELECT diagnostic_kind,state,COUNT(*),COALESCE(SUM(cost_micro),0) "
                        "FROM remote_admissions WHERE diagnostic_parent IS NOT NULL AND started>=? GROUP BY diagnostic_kind,state", (since,)):
                    if kind not in {"batch", "streaming"}:
                        raise AdmissionError()
                    entry = by_kind.setdefault(kind, {"admission_states": {}, "settled_cost_usd": 0})
                    entry["admission_states"][state] = count
                    if state == "settled":
                        entry["settled_cost_usd"] += cost / 1_000_000
                result["diagnostics"]["by_kind"] = by_kind
    return result
