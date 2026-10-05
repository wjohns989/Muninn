"""Bounded gathering in the existing serial consumer; no provider traffic."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_partial_polls_do_not_reset_deadline_or_launch_another_route(monkeypatch):
    from muninn.history import auto_routing, batch_activation, historical_batch_worker
    from muninn.history import service as module
    monkeypatch.setattr(auto_routing, "remote_policy_snapshot", lambda _: SimpleNamespace(enabled=True))
    clock = [0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    journal = SimpleNamespace(policy_root=None, historical_batch_owner=lambda: None)
    policy = {"enabled": True,"remaining_batches": 10}
    monkeypatch.setattr(batch_activation,"read_batch_policy",lambda _:policy)
    thresholds = []
    def prepare(_journal, *, min_items):
        thresholds.append(min_items)
        return "ready" if min_items == 1 else None
    monkeypatch.setattr(batch_activation,"prepare_next_batch",prepare)
    steps = []
    class Worker:
        def __init__(self,*args,**kwargs):
            self.status = {}
        async def step(self):
            steps.append(1)
    monkeypatch.setattr(historical_batch_worker,"HistoricalBatchWorker",Worker)
    service = module.HistoryService.__new__(module.HistoryService)
    service._require_capture_journal = lambda:journal
    service._historical_batch_worker = None
    service._historical_batch_gather_deadline = None
    for moment in [0,50,89]:
        clock[0] = moment
        assert await service._process_secure_batch_once()
        assert not steps
        assert service._historical_batch_worker.status["state"] == "gathering_batch"
    clock[0] = 90
    assert await service._process_secure_batch_once()
    assert thresholds == [128,128,128,1] and steps == [1]
    assert service._historical_batch_gather_deadline is None


@pytest.mark.asyncio
async def test_remote_revoke_and_reenable_start_a_new_gathering_cycle(monkeypatch):
    from muninn.history import auto_routing, batch_activation, historical_batch_worker
    from muninn.history import service as module
    clock = [0]
    remote = SimpleNamespace(enabled=True)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(auto_routing, "remote_policy_snapshot", lambda _: remote)
    monkeypatch.setattr(batch_activation, "read_batch_policy",
                        lambda _: {"enabled": True, "remaining_batches": 10})
    thresholds = []
    def prepare(_journal, *, min_items):
        thresholds.append(min_items)
        return None
    monkeypatch.setattr(batch_activation, "prepare_next_batch", prepare)
    class Worker:
        def __init__(self, *args, **kwargs):
            self.status = {}
    monkeypatch.setattr(historical_batch_worker, "HistoricalBatchWorker", Worker)
    service = module.HistoryService.__new__(module.HistoryService)
    service._require_capture_journal = lambda: SimpleNamespace(
        policy_root=None, historical_batch_owner=lambda: None)
    service._historical_batch_worker = None
    service._historical_batch_gather_deadline = None
    await service._process_secure_batch_once()
    assert service._historical_batch_gather_deadline == 90
    remote.enabled = False
    clock[0] = 120
    assert await service._process_secure_batch_once()
    assert service._historical_batch_gather_deadline is None
    remote.enabled = True
    await service._process_secure_batch_once()
    assert service._historical_batch_gather_deadline == 210 and thresholds == [128, 128]


@pytest.mark.asyncio
async def test_revocation_resets_only_unowned_gathering(monkeypatch):
    from muninn.history import batch_activation
    from muninn.history import service as module
    policy = {"enabled": False,"remaining_batches":10}
    monkeypatch.setattr(batch_activation,"read_batch_policy",lambda _:policy)
    service = module.HistoryService.__new__(module.HistoryService)
    service._require_capture_journal = lambda:SimpleNamespace(policy_root=None,historical_batch_owner=lambda:None)
    service._historical_batch_gather_deadline = 12
    service._historical_batch_worker = None
    assert not await service._process_secure_batch_once()
    assert service._historical_batch_gather_deadline is None
