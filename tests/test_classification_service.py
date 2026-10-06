"""Classification turns share the existing paid checkpoint consumer."""
import pytest

from tests.test_private_zdr_service import setup


@pytest.mark.asyncio
async def test_one_related_turn_per_passed_checkpoint_preserves_clean_progress(monkeypatch):
    service, owner, preparations, steps, _calls = setup(monkeypatch)
    service._private_zdr_checkpoint = owner["id"]
    service._private_zdr_opportunities = 4
    calls = []
    async def classify(**kwargs):
        calls.append(kwargs)
        return True
    service._process_secure_classification_once = classify
    assert await service._process_secure_batch_once()
    assert calls == [{}] and preparations == []
    assert await service._process_secure_batch_once()
    assert len(calls) == 1 and len(preparations) == 1 and len(steps) == 1
    owner["id"] = "b" * 32
    service._private_zdr_checkpoint = owner["id"]
    service._private_zdr_opportunities = 4
    assert await service._process_secure_batch_once()
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("quota", [10, 0])
async def test_classification_tail_continues_without_new_clean_checkpoint(monkeypatch, quota):
    from types import SimpleNamespace
    from muninn.history import batch_activation
    service, _owner, _preparations, _steps, _calls = setup(monkeypatch)
    service._require_capture_journal = lambda: SimpleNamespace(policy_root=None, historical_batch_owner=lambda: None)
    monkeypatch.setattr(batch_activation, "prepare_next_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_activation, "read_batch_policy", lambda _: {"enabled": True, "remaining_batches": quota})
    service._historical_batch_gather_deadline = 0
    calls = []
    async def classify(**kwargs):
        calls.append(kwargs)
        return True
    service._process_secure_classification_once = classify
    for _ in range(3):
        assert await service._process_secure_batch_once()
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_cooldown_does_not_consume_classification_checkpoint(monkeypatch):
    service, owner, preparations, _steps, _calls = setup(monkeypatch)
    service._private_zdr_checkpoint = owner["id"]
    service._private_zdr_opportunities = 4
    ready = [False]
    service._capture_cadence.attempt_ready = lambda: ready[0]
    calls = []
    async def classify(**kwargs):
        calls.append(kwargs)
        return True
    service._process_secure_classification_once = classify
    assert await service._process_secure_batch_once()
    assert calls == [] and preparations == []
    ready[0] = True
    assert await service._process_secure_batch_once()
    assert calls == [{}]
