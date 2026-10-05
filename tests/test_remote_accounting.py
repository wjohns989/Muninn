"""Budget durability on owner-private temporary SQLite, never user accounts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import sqlite3

import pytest

from muninn.history.remote_accounting import AdmissionError, reserve, status, _finish, main
from muninn.history.remote_policy import write_policy, read_policy

READY = {"admission_ready": True, "usage_daily_usd": 0, "usage_monthly_usd": 0}
FALLBACK = lambda: (False, 5, 50, False)


def policy(root, *, enabled=True):
    return write_policy(root, enabled=enabled, daily_usd=5, monthly_usd=50,
                        override_ceiling=False, fallback=FALLBACK)


def stamp(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def test_settled_cost_is_up_rounded_and_blocks_stale_provider_usage(tmp_path):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    assert admission.settle_response({"usage": {"cost": 5.00000001}})
    assert status(tmp_path)["daily_cost_usd"] == 5.000001
    with pytest.raises(AdmissionError, match="threshold_reached"):
        reserve(tmp_path, 1, READY)


def test_parallel_process_equivalent_connections_allow_only_one_admission(tmp_path):
    policy(tmp_path)
    def attempt(_):
        try:
            return reserve(tmp_path, 1, READY)
        except AdmissionError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(attempt, range(6)))
    assert sum(not isinstance(row, str) for row in results) == 1
    assert status(tmp_path)["unresolved"] == 1


def test_unresolved_row_survives_reopen_and_next_day_month(tmp_path):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY, now=stamp("2026-10-31T23:59:59"))
    admission.mark_unknown()
    assert status(tmp_path, now=stamp("2026-11-01T00:00:01"))["state"] == "blocked"
    with pytest.raises(AdmissionError, match="admission_busy"):
        reserve(tmp_path, 1, READY, now=stamp("2026-11-01T00:00:01"))


def test_cross_midnight_month_cost_is_conservatively_in_both_periods(tmp_path):
    policy(tmp_path)
    before, after = stamp("2026-10-31T23:59:59"), stamp("2026-11-01T00:00:01")
    admission = reserve(tmp_path, 1, READY, now=before)
    admission.mark_unknown()
    _finish(tmp_path, admission.identifier, 750000, "response", now=after)
    for now in (before, after):
        assert status(tmp_path, now=now)["daily_cost_usd"] == 0.75
        assert status(tmp_path, now=now)["monthly_cost_usd"] == 0.75


def test_settle_and_proven_unsent_release_still_work_after_revocation(tmp_path):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    policy(tmp_path, enabled=False)
    assert admission.settle_response({"usage": {"cost": 0.1}})
    policy(tmp_path)
    other = reserve(tmp_path, 3, READY)
    policy(tmp_path, enabled=False)
    other.release_unsent()
    assert status(tmp_path)["unresolved"] == 0
    assert status(tmp_path)["daily_cost_usd"] == 0.1


def test_revoked_before_first_reserve_does_not_strand_bootstrap(tmp_path):
    policy(tmp_path, enabled=False)
    with pytest.raises(AdmissionError, match="consent_revoked"):
        reserve(tmp_path, 1, READY)
    assert not (tmp_path / "remote_policy" / "admission-managed").exists()
    assert status(tmp_path)["state"] == "uninitialized"


def test_threshold_denial_leaves_completed_bootstrap_usable(tmp_path):
    policy(tmp_path)
    with pytest.raises(AdmissionError, match="threshold_reached"):
        reserve(tmp_path, 1, {**READY, "usage_daily_usd": 5})
    assert status(tmp_path)["state"] == "ready"
    reserve(tmp_path, 1, READY).release_unsent()


@pytest.mark.parametrize("damage", ["marker", "table", "sentinel"])
def test_accounting_loss_does_not_reset_spend_or_revive_admission(tmp_path, damage):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    admission.settle_response({"usage": {"cost": 1}})
    if damage == "marker":
        (tmp_path / "remote_policy" / "admission-managed").unlink()
    else:
        with sqlite3.connect(tmp_path / "remote_policy" / "policy.sqlite3") as db:
            db.execute("DROP TABLE remote_admissions" if damage == "table" else
                       "UPDATE policy SET accounting_version=0")
    with pytest.raises(AdmissionError):
        reserve(tmp_path, 1, READY)
    with pytest.raises(AdmissionError):
        status(tmp_path)


@pytest.mark.parametrize("cost", [None, True, -1, float("nan"), float("inf"), "0.1", 10**100])
def test_missing_or_invalid_cost_remains_unknown_not_zero(tmp_path, cost):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    assert admission.settle_response({"usage": {"cost": cost}}) is False
    assert status(tmp_path)["state"] == "blocked"
    with pytest.raises(AdmissionError, match="admission_busy"):
        reserve(tmp_path, 1, READY)


def test_zero_cost_and_same_receipt_are_valid_but_conflict_is_not(tmp_path):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    assert admission.settle_response({"usage": {"cost": 0}})
    assert admission.settle_response({"usage": {"cost": 0}})
    with pytest.raises(AdmissionError, match="conflict"):
        admission.settle_response({"usage": {"cost": 1}})
    assert status(tmp_path)["state"] == "ready"


def test_operator_reconciliation_requires_explicit_local_confirmation(tmp_path, capsys):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    args = ["reconcile", "--root", str(tmp_path), "--admission", admission.identifier, "--cost-usd", "0.2"]
    with pytest.raises(SystemExit):
        main(args)
    assert status(tmp_path)["state"] == "blocked"
    assert main([*args, "--confirm-outcome-reviewed"]) == 0
    assert status(tmp_path)["daily_cost_usd"] == 0.2
    assert read_policy(tmp_path, FALLBACK).generation == 1
    assert "fixture-key" not in capsys.readouterr().out


def test_missing_policy_or_root_does_not_create_database(tmp_path):
    with pytest.raises(AdmissionError):
        reserve(tmp_path, 1, READY)
    assert not (tmp_path / "remote_policy").exists()
    with pytest.raises(AdmissionError):
        reserve(None, 1, READY)


def test_operator_cli_preserves_decimal_cost_lexeme(tmp_path, capsys):
    policy(tmp_path)
    admission = reserve(tmp_path, 1, READY)
    admission.mark_unknown()
    assert main(["reconcile", "--root", str(tmp_path), "--admission", admission.identifier,
                 "--cost-usd", "4.9999990000000001", "--confirm-outcome-reviewed"]) == 0
    assert status(tmp_path)["daily_cost_usd"] == 5
    with pytest.raises(AdmissionError, match="threshold_reached"):
        reserve(tmp_path, 1, READY)


def test_stale_reserved_is_fenced_before_new_admission(tmp_path):
    policy(tmp_path)
    old = reserve(tmp_path, 1, READY, now=1000)
    with pytest.raises(AdmissionError, match="admission_busy"):
        reserve(tmp_path, 1, READY, now=1899.999)
    new = reserve(tmp_path, 1, READY, now=1900)
    with pytest.raises(AdmissionError, match="conflict"):
        old.mark_unknown()
    new.mark_unknown()
    assert status(tmp_path, now=1900)["unresolved"] == 1


def test_unknown_never_expires_even_after_reserved_timeout(tmp_path):
    policy(tmp_path)
    old = reserve(tmp_path, 1, READY, now=1000)
    old.mark_unknown()
    with pytest.raises(AdmissionError, match="admission_busy"):
        reserve(tmp_path, 1, READY, now=100000)
    assert old.release_reserved() is False
    assert status(tmp_path)["unresolved"] == 1


def test_reserved_release_is_idempotent_and_preserves_billed_cost(tmp_path):
    policy(tmp_path)
    old = reserve(tmp_path, 1, READY)
    assert old.release_reserved() is True
    assert old.release_reserved() is True
    with pytest.raises(AdmissionError, match="conflict"):
        old.mark_unknown()
    paid = reserve(tmp_path, 1, READY)
    paid.mark_unknown()
    paid.settle_response({"usage": {"cost": 0.2}})
    assert paid.release_reserved() is False
    assert status(tmp_path)["daily_cost_usd"] == 0.2
