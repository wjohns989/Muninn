import asyncio
import os

import pytest

from muninn.ingestion.periodic import PeriodicIngestionScheduler, PeriodicIngestionSettings


class _MemoryStub:
    def __init__(self) -> None:
        self.calls = []

    async def ingest_sources(self, **kwargs):
        self.calls.append(kwargs)
        return {"event": "INGEST_COMPLETED", "added_memories": 1}


@pytest.mark.asyncio
async def test_periodic_trigger_once_updates_status():
    memory = _MemoryStub()
    settings = PeriodicIngestionSettings(
        sources=["/tmp/a.txt"],
        project="muninn",
        namespace="global",
        user_id="global_user",
    )
    scheduler = PeriodicIngestionScheduler(memory=memory, settings=settings)

    result = await scheduler.trigger_once(reason="manual")

    assert result["success"] is True
    assert result["event"] == "PERIODIC_INGESTION_COMPLETED"
    assert len(memory.calls) == 1
    assert memory.calls[0]["sources"] == ["/tmp/a.txt"]
    status = scheduler.status
    assert status["runtime"]["run_count"] == 1
    assert status["runtime"]["success_count"] == 1
    assert status["runtime"]["failure_count"] == 0
    assert status["runtime"]["last_run_status"] == "completed"


@pytest.mark.asyncio
async def test_periodic_trigger_injects_operator_model_profile():
    memory = _MemoryStub()
    settings = PeriodicIngestionSettings(
        sources=["/tmp/a.txt"],
        model_profile="low_latency",
    )
    scheduler = PeriodicIngestionScheduler(memory=memory, settings=settings)

    result = await scheduler.trigger_once(reason="manual")

    assert result["success"] is True
    assert memory.calls[0]["metadata"]["operator_model_profile"] == "low_latency"


@pytest.mark.asyncio
async def test_periodic_trigger_can_inject_skip_extraction():
    memory = _MemoryStub()
    settings = PeriodicIngestionSettings(
        sources=["/tmp/a.txt"],
        skip_extraction=True,
    )
    scheduler = PeriodicIngestionScheduler(memory=memory, settings=settings)

    result = await scheduler.trigger_once(reason="manual")

    assert result["success"] is True
    assert memory.calls[0]["metadata"]["muninn_skip_extraction"] is True


@pytest.mark.asyncio
async def test_periodic_trigger_can_inject_extraction_timeout():
    memory = _MemoryStub()
    settings = PeriodicIngestionSettings(
        sources=["/tmp/a.txt"],
        extraction_timeout_seconds=2.5,
    )
    scheduler = PeriodicIngestionScheduler(memory=memory, settings=settings)

    result = await scheduler.trigger_once(reason="manual")

    assert result["success"] is True
    assert memory.calls[0]["metadata"]["muninn_extraction_timeout_seconds"] == 2.5


@pytest.mark.asyncio
async def test_periodic_trigger_skips_when_already_running():
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockingMemory:
        async def ingest_sources(self, **kwargs):
            started.set()
            await release.wait()
            return {"event": "INGEST_COMPLETED"}

    settings = PeriodicIngestionSettings(sources=["/tmp/a.txt"])
    scheduler = PeriodicIngestionScheduler(memory=_BlockingMemory(), settings=settings)

    first = asyncio.create_task(scheduler.trigger_once(reason="manual"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    second = await scheduler.trigger_once(reason="manual")
    release.set()
    await first

    assert second["success"] is False
    assert second["reason"] == "already_running"


@pytest.mark.asyncio
async def test_periodic_loop_runs_on_interval():
    event = asyncio.Event()

    class _CountingMemory:
        def __init__(self) -> None:
            self.count = 0

        async def ingest_sources(self, **kwargs):
            self.count += 1
            if self.count >= 2:
                event.set()
            return {"event": "INGEST_COMPLETED", "added_memories": 1}

    memory = _CountingMemory()
    settings = PeriodicIngestionSettings(
        interval_seconds=0.05,
        sources=["/tmp/a.txt"],
        run_on_start=False,
    )
    scheduler = PeriodicIngestionScheduler(memory=memory, settings=settings)
    started = await scheduler.start()
    assert started is True

    await asyncio.wait_for(event.wait(), timeout=1.0)
    stopped = await scheduler.stop()
    assert stopped is True
    assert memory.count >= 2
    status = scheduler.status
    assert status["runtime"]["running"] is False
    assert status["runtime"]["success_count"] >= 2


@pytest.mark.asyncio
async def test_periodic_consecutive_failures_increase_backoff_and_reset_on_success():
    class _FlakyMemory:
        def __init__(self) -> None:
            self.calls = 0

        async def ingest_sources(self, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("transient failure")
            return {"event": "INGEST_COMPLETED", "added_memories": 1}

    settings = PeriodicIngestionSettings(
        interval_seconds=10.0,
        failure_backoff_multiplier=2.0,
        max_backoff_seconds=60.0,
        jitter_ratio=0.0,
        sources=["/tmp/a.txt"],
    )
    scheduler = PeriodicIngestionScheduler(memory=_FlakyMemory(), settings=settings)

    first = await scheduler.trigger_once(reason="manual")
    assert first["success"] is False
    assert scheduler.status["runtime"]["consecutive_failures"] == 1
    assert scheduler._compute_next_sleep_seconds() == pytest.approx(20.0)

    second = await scheduler.trigger_once(reason="manual")
    assert second["success"] is False
    assert scheduler.status["runtime"]["consecutive_failures"] == 2
    assert scheduler._compute_next_sleep_seconds() == pytest.approx(40.0)

    third = await scheduler.trigger_once(reason="manual")
    assert third["success"] is True
    assert scheduler.status["runtime"]["consecutive_failures"] == 0
    assert scheduler._compute_next_sleep_seconds() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_periodic_trigger_run_timeout_marks_failure():
    class _VerySlowMemory:
        async def ingest_sources(self, **kwargs):
            await asyncio.sleep(1.0)
            return {"event": "INGEST_COMPLETED", "added_memories": 1}

    settings = PeriodicIngestionSettings(
        sources=["/tmp/a.txt"],
        run_timeout_seconds=0.05,
    )
    scheduler = PeriodicIngestionScheduler(memory=_VerySlowMemory(), settings=settings)

    result = await scheduler.trigger_once(reason="manual")

    assert result["success"] is False
    assert result["error"] == "periodic run timeout"
    runtime = scheduler.status["runtime"]
    assert runtime["failure_count"] == 1
    assert runtime["consecutive_failures"] == 1
    assert runtime["last_run_status"] == "failed"
    assert runtime["last_run_timeout_enforced"] is True
    assert runtime["last_run_elapsed_seconds"] is not None


@pytest.mark.asyncio
async def test_periodic_trigger_run_timeout_can_skip_warmup_runs():
    class _SlowMemory:
        async def ingest_sources(self, **kwargs):
            await asyncio.sleep(0.05)
            return {"event": "INGEST_COMPLETED", "added_memories": 1}

    settings = PeriodicIngestionSettings(
        sources=["/tmp/a.txt"],
        run_timeout_seconds=0.01,
        run_timeout_skip_warmup_runs=1,
    )
    scheduler = PeriodicIngestionScheduler(memory=_SlowMemory(), settings=settings)

    first = await scheduler.trigger_once(reason="manual")
    assert first["success"] is True
    assert first["timeout_enforced"] is False
    assert scheduler.status["runtime"]["last_run_timeout_enforced"] is False

    second = await scheduler.trigger_once(reason="manual")
    assert second["success"] is False
    assert second["error"] == "periodic run timeout"
    assert second["timeout_enforced"] is True
    assert scheduler.status["runtime"]["last_run_timeout_enforced"] is True


def test_periodic_settings_from_env(monkeypatch):
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_ENABLED", "1")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_RUN_ON_START", "1")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_SOURCES", f"/tmp/a.txt{os.pathsep}/tmp/b.txt")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_METADATA_JSON", "{\"source\":\"periodic\"}")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_RECURSIVE", "true")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_CHRONOLOGICAL_ORDER", "oldest_first")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_CHUNK_OVERLAP_CHARS", "0")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_MODEL_PROFILE", "low_latency")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_SKIP_EXTRACTION", "1")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_EXTRACT_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SKIP_WARMUP_RUNS", "2")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_FAILURE_BACKOFF_MULTIPLIER", "3")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_MAX_BACKOFF_SECONDS", "120")
    monkeypatch.setenv("MUNINN_PERIODIC_INGESTION_JITTER_RATIO", "0.25")

    settings = PeriodicIngestionSettings.from_env()

    assert settings.enabled_on_startup is True
    assert settings.run_on_start is True
    assert settings.interval_seconds == 30.0
    assert settings.sources == ["/tmp/a.txt", "/tmp/b.txt"]
    assert settings.metadata == {"source": "periodic"}
    assert settings.recursive is True
    assert settings.chronological_order == "oldest_first"
    assert settings.chunk_overlap_chars == 0
    assert settings.model_profile == "low_latency"
    assert settings.skip_extraction is True
    assert settings.extraction_timeout_seconds == 2.5
    assert settings.run_timeout_seconds == 20.0
    assert settings.run_timeout_skip_warmup_runs == 2
    assert settings.failure_backoff_multiplier == 3.0
    assert settings.max_backoff_seconds == 120.0
    assert settings.jitter_ratio == 0.25
