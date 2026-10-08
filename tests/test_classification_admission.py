"""Continuation and headroom on private synthetic accounting, no paid calls."""
import pytest

from muninn.history.remote_accounting import AdmissionError, reserve, status
from tests.test_remote_accounting import READY, policy


def review(root, generation, job, **kwargs):
    return reserve(root, generation, READY, classification_job=job * 32,
                   classification_input="a" * 64, classification_once=True,
                   cost_ceiling_usd=0.27, **kwargs)


def test_distinct_reviews_continue_but_paid_review_never_repeats(tmp_path):
    generation = policy(tmp_path).generation
    first = review(tmp_path, generation, "b")
    first.mark_unknown()
    first.settle_response({"usage": {"cost": 0.003}})
    with pytest.raises(AdmissionError, match="classification_already_admitted"):
        review(tmp_path, generation, "b")
    other = review(tmp_path, generation, "c")
    other.release_reserved()
    assert status(tmp_path)["unresolved"] == 0


def test_proven_unsent_review_can_retry_but_unknown_cannot(tmp_path):
    generation = policy(tmp_path).generation
    first = review(tmp_path, generation, "b")
    first.mark_unknown()
    first.release_unsent()
    retry = review(tmp_path, generation, "b")
    retry.mark_unknown()
    with pytest.raises(AdmissionError):
        review(tmp_path, generation, "b")
    with pytest.raises(AdmissionError, match="remote_admission_busy"):
        review(tmp_path, generation, "c")


@pytest.mark.parametrize("field,value", [("usage_daily_usd", 4.74), ("usage_monthly_usd", 49.74)])
def test_headroom_rejects_before_reservation(tmp_path, field, value):
    generation = policy(tmp_path).generation
    with pytest.raises(AdmissionError, match="headroom"):
        reserve(tmp_path, generation, {**READY, field: value}, cost_ceiling_usd=0.27)
    assert status(tmp_path)["unresolved"] == 0


def test_fresh_usage_blocks_presend_hold_without_losing_it(tmp_path):
    generation = policy(tmp_path).generation
    hold = review(tmp_path, generation, "b")
    hold.check_headroom(READY, cost_ceiling_usd=0.27)
    hold.mark_unknown()
    with pytest.raises(AdmissionError, match="headroom"):
        hold.check_headroom({**READY, "usage_daily_usd": 4.74}, cost_ceiling_usd=0.27)
    assert status(tmp_path)["unresolved"] == 1
    hold.release_unsent()
    with pytest.raises(AdmissionError, match="conflict"):
        hold.check_headroom(READY, cost_ceiling_usd=0.27)


def test_settled_local_cost_blocks_even_with_stale_provider_usage(tmp_path):
    generation = policy(tmp_path).generation
    first = review(tmp_path, generation, "b")
    first.mark_unknown()
    first.settle_response({"usage": {"cost": 4.74}})
    with pytest.raises(AdmissionError, match="headroom"):
        review(tmp_path, generation, "c")


def test_exact_headroom_fits_but_fractional_micro_rounds_up(tmp_path):
    from decimal import Decimal
    generation = policy(tmp_path).generation
    usage = {**READY, "usage_daily_usd": Decimal("4.73")}
    hold = reserve(tmp_path, generation, usage, cost_ceiling_usd=Decimal("0.27"))
    hold.release_reserved()
    with pytest.raises(AdmissionError, match="headroom"):
        reserve(tmp_path, generation, usage, cost_ceiling_usd=Decimal("0.27000001"))


def test_current_hold_cannot_recheck_after_utc_rollover(tmp_path):
    from tests.test_remote_accounting import stamp
    generation = policy(tmp_path).generation
    start = stamp("2026-10-31T23:50:00")
    hold = reserve(tmp_path, generation, READY, now=start, cost_ceiling_usd=0.27)
    with pytest.raises(AdmissionError, match="conflict"):
        hold.check_headroom(READY, cost_ceiling_usd=0.27, now=start + 601)
    hold.release_reserved()
