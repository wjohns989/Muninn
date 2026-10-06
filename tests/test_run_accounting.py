"""Run totals from isolated real admissions, never a provider or live store."""
import sqlite3
from datetime import datetime, timezone

import pytest

from muninn.history.remote_accounting import AdmissionError, _finish, reserve
from muninn.history.remote_policy import write_policy
from muninn.history.run_accounting import run_status

READY = {"admission_ready": True, "usage_daily_usd": 0, "usage_monthly_usd": 0}


def stamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def setup(root):
    write_policy(root, enabled=True, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 5, 50, False))


def settled(root, start, cost, *, finish=None, resolution="response", batch=None):
    admission = reserve(root, 1, READY, now=start, batch_owner=batch)
    admission.mark_unknown()
    _finish(root, admission.identifier, cost, resolution, now=finish or start + 1)
    return admission


def test_run_total_spans_rollover_and_counts_aggregate_once(tmp_path):
    setup(tmp_path)
    since = stamp("2026-10-31T23:59:58") + .000905
    settled(tmp_path, since - 1, 90000, finish=since + .5)
    settled(tmp_path, since, 125000, finish=since + 3, batch="a" * 32)
    settled(tmp_path, since + 4, 25000, batch="b" * 32)  # Failed-only repair charge.
    settled(tmp_path, since + 6, 12000)  # Charged response, independent of job success.
    settled(tmp_path, since + 8, 7000, resolution="operator")
    unsent = reserve(tmp_path, 1, READY, now=since + 10)
    _finish(tmp_path, unsent.identifier, None, "unsent", now=since + 11)
    pending = reserve(tmp_path, 1, READY, now=since + 12)
    pending.mark_unknown()
    path = tmp_path / "remote_policy" / "policy.sqlite3"
    before = path.read_bytes()
    report = run_status(tmp_path, since=since, now=since + 20)
    assert report["since"] == since
    assert report["settled_cost_usd"] == .169
    assert report["admission_states"] == {"settled": 4, "released": 1, "reserved": 0, "unknown": 1}
    assert report["settled_resolutions"] == {"response": 3, "operator": 1}
    assert report["batch_owned"] == {"state": "known", "settled_admissions": 2, "settled_cost_usd": .15}
    assert report["global_unresolved"] == 1
    assert path.read_bytes() == before
    assert report["scope"] == "all_managed_admissions_since_start"


def test_uninitialized_is_unknown_and_does_not_initialize(tmp_path):
    setup(tmp_path)
    report = run_status(tmp_path, since=0, now=10)
    assert report["state"] == "uninitialized"
    assert report["settled_cost_usd"] is None and report["admission_states"] is None
    assert not (tmp_path / "remote_policy" / "admission-managed").exists()


@pytest.mark.parametrize("since", [True, None, "1", -1, float("nan"), float("inf"), 11, 10**500])
def test_invalid_cutoff_never_opens_or_creates_policy(tmp_path, since):
    with pytest.raises(ValueError):
        run_status(tmp_path, since=since, now=10)
    assert not (tmp_path / "remote_policy").exists()


@pytest.mark.parametrize("assignment", ["started=NULL", "started='invalid'", "started=-1",
    "finished=999", "cost_micro=NULL", "cost_micro=-1", "resolution=NULL"])
def test_malformed_rows_fail_even_before_cutoff(tmp_path, assignment):
    setup(tmp_path)
    settled(tmp_path, 1, 1000, finish=2)
    with sqlite3.connect(tmp_path / "remote_policy" / "policy.sqlite3") as db:
        if assignment == "started=NULL":
            # SQLite NOT NULL protects this case already; a missing timestamp is
            # represented by invalid text to exercise the consumer's own check.
            assignment = "started='missing'"
        db.execute("UPDATE remote_admissions SET " + assignment)
    with pytest.raises(AdmissionError):
        run_status(tmp_path, since=5, now=10)


def test_legacy_batch_attribution_is_unknown_and_revocation_does_not_hide_cost(tmp_path):
    setup(tmp_path)
    settled(tmp_path, 1, 1000)
    with sqlite3.connect(tmp_path / "remote_policy" / "policy.sqlite3") as db:
        db.execute("ALTER TABLE remote_admissions DROP COLUMN batch_owner")
    write_policy(tmp_path, enabled=False, daily_usd=5, monthly_usd=50,
                 override_ceiling=False, fallback=lambda: (False, 5, 50, False))
    report = run_status(tmp_path, since=0, now=10)
    assert report["settled_cost_usd"] == .001
    assert report["batch_owned"] == {"state": "unknown_legacy_schema",
        "settled_admissions": None, "settled_cost_usd": None}


def test_older_unknown_admission_is_not_misreported_as_a_free_complete_run(tmp_path):
    setup(tmp_path)
    reserve(tmp_path, 1, READY, now=1).mark_unknown()
    report = run_status(tmp_path, since=5, now=10)
    assert report["admission_states"]["unknown"] == 0
    assert report["global_unresolved"] == 1
    assert report["settled_cost_usd"] == 0


def test_snapshot_time_is_sampled_after_entering_the_read_transaction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from muninn.history import run_accounting
    setup(tmp_path)
    settled(tmp_path, 5, 1000, finish=6)
    ticks = iter((1, 10))  # Admission completes after invocation, before snapshot.
    monkeypatch.setattr(run_accounting, "time", SimpleNamespace(time=lambda: next(ticks)))
    report = run_status(tmp_path, since=0)
    assert report["sampled_at"] == 10 and report["settled_cost_usd"] == .001
