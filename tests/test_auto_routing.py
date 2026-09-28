"""Resource policy tests never load models or make network requests."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from muninn.history.auto_routing import GpuState, choose_route, model_hints_for_thread


def _gpu(free=13_600, used=1, loaded=()):
    return GpuState(free, 16_376, used, 100.0, loaded)


MODELS = [
    {"name": "muninn-qwen35-defiant-q8-test:latest", "size": 10_000 * 1024 * 1024},
    {"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024},
]


def test_normal_thread_uses_small_model_and_complex_thread_prefers_q8():
    for turns in (0, 59):
        assert choose_route(_gpu(), MODELS, model_hints=model_hints_for_thread(turns),
                            now=100.0).model == "qwen2.5:7b"
    for turns in (60, 61):
        assert choose_route(_gpu(), MODELS, model_hints=model_hints_for_thread(turns),
                            now=100.0).model == MODELS[0]["name"]
    assert model_hints_for_thread(60, configured=("custom",)) == ("custom",)


def test_complex_thread_falls_back_to_smaller_model_when_headroom_falls():
    smaller = choose_route(_gpu(free=7_500), MODELS,
                           model_hints=model_hints_for_thread(60), now=100.0)
    assert (smaller.provider, smaller.model) == ("ollama", "qwen2.5:7b")


def test_busy_or_occupied_gpu_never_loads_another_local_model():
    for gpu in (_gpu(used=70), _gpu(loaded=("other-model",))):
        assert choose_route(gpu, MODELS, now=100.0).provider == "deferred"
        assert choose_route(gpu, MODELS, cloud_allowed=True, cloud_available=True,
                            now=100.0).provider == "openrouter"


def test_missing_stale_or_insufficient_gpu_fails_closed():
    assert choose_route(None, MODELS, now=100.0).provider == "deferred"
    assert choose_route(_gpu(free=5_000), MODELS, now=100.0).provider == "deferred"
    stale = GpuState(13_600, 16_376, 0, 80.0)
    assert choose_route(stale, MODELS, now=100.0).provider == "deferred"


def test_cloud_requires_both_explicit_policy_and_available_key():
    assert choose_route(None, MODELS, cloud_allowed=False, cloud_available=True,
                        now=100.0).provider == "deferred"
    assert choose_route(None, MODELS, cloud_allowed=True, cloud_available=False,
                        now=100.0).provider == "deferred"


def test_auto_activation_excludes_existing_history(monkeypatch):
    from muninn.history.service import HistoryService

    values = {}
    service = HistoryService.__new__(HistoryService)
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        get_meta=lambda key: values.get(key),
        set_meta=lambda key, value: values.__setitem__(key, value),
    ))
    monkeypatch.setenv("MUNINN_INSIGHTS_AUTO", "1")
    monkeypatch.setattr("muninn.history.service.time.time", lambda: 1234.0)
    assert service._auto_since() == 1234.0
    assert service._auto_since() == 1234.0
    assert values == {"history_auto_insights_since": "1234.0"}


def test_auto_analysis_is_off_by_default_without_touching_provider(monkeypatch):
    from muninn.history.service import HistoryService

    monkeypatch.delenv("MUNINN_INSIGHTS_AUTO", raising=False)
    service = HistoryService.__new__(HistoryService)
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        get_meta=lambda _key: pytest.fail("auto-off must not read analysis state"),
    ))
    assert service._auto_since() is None


@pytest.mark.asyncio
async def test_auto_analysis_routes_only_new_threads_to_idle_local_model(monkeypatch):
    from muninn.history import auto_routing
    from muninn.history.service import HistoryService

    service = HistoryService.__new__(HistoryService)
    service._auto_since = lambda: 100.0
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        list_history_threads=lambda *_args, **_kwargs: [
            {"thread_key": "test-thread", "turns_imported": 12}],
    ))
    service.run_analysis = AsyncMock(return_value={})
    service.last_auto_route = None
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: GpuState(13_600, 16_376, 1, time.time()))
    monkeypatch.setattr(auto_routing, "probe_ollama", lambda _base: (MODELS, ()))
    monkeypatch.setattr(auto_routing, "guarded_openrouter_available",
                        lambda _cap: pytest.fail("local route must not query OpenRouter"))
    await service._auto_analyze()
    assert service.run_analysis.await_args.kwargs == {
        "apply": True, "provider": "ollama", "model": "qwen2.5:7b",
        "since": 100.0, "limit": 1, "concurrency": 1,
        "thread_key": "test-thread",
    }


@pytest.mark.asyncio
async def test_auto_analysis_defers_when_gpu_busy(monkeypatch):
    from muninn.history import auto_routing
    from muninn.history.service import HistoryService

    service = HistoryService.__new__(HistoryService)
    service._auto_since = lambda: 100.0
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        list_history_threads=lambda *_args, **_kwargs: [
            {"thread_key": "test-thread", "turns_imported": 12}],
    ))
    service.run_analysis = AsyncMock()
    service.last_auto_route = None
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: GpuState(13_600, 16_376, 90, time.time()))
    monkeypatch.setattr(auto_routing, "probe_ollama", lambda _base: (MODELS, ()))
    monkeypatch.setattr(auto_routing, "guarded_openrouter_available", lambda _cap: False)
    await service._auto_analyze()
    service.run_analysis.assert_not_awaited()
    assert service.last_auto_route["reason"] == "gpu_busy"


@pytest.mark.asyncio
async def test_auto_analysis_uses_budgeted_cloud_when_gpu_busy(monkeypatch):
    from muninn.history import auto_routing
    from muninn.history.service import HistoryService

    service = HistoryService.__new__(HistoryService)
    service._auto_since = lambda: 100.0
    service.memory = SimpleNamespace(_metadata=SimpleNamespace(
        list_history_threads=lambda *_args, **_kwargs: [
            {"thread_key": "test-thread", "turns_imported": 12}],
    ))
    service.run_analysis = AsyncMock(return_value={})
    service.last_auto_route = None
    monkeypatch.setattr(auto_routing, "probe_gpu", lambda: GpuState(13_600, 16_376, 90, time.time()))
    monkeypatch.setattr(auto_routing, "probe_ollama", lambda _base: (MODELS, ()))
    monkeypatch.setattr(auto_routing, "guarded_openrouter_available", lambda cap: cap == 1.0)
    await service._auto_analyze()
    assert service.run_analysis.await_args.kwargs == {
        "apply": True, "provider": "openrouter", "since": 100.0,
        "limit": 1, "concurrency": 1, "thread_key": "test-thread",
    }
