"""Embedded Qdrant add operations must not overlap in one Muninn process."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from muninn.core.memory import MuninnMemory


@pytest.mark.asyncio
async def test_concurrent_adds_are_serialized_before_vector_access():
    memory = MuninnMemory.__new__(MuninnMemory)
    memory._add_lock = asyncio.Lock()
    memory._check_initialized = lambda: None
    memory._otel = SimpleNamespace(span=lambda *_args, **_kwargs: nullcontext())
    active = 0
    peak = 0

    async def process_add(**_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return {"id": None, "event": "DEDUP_SKIP"}
        finally:
            active -= 1

    memory._ingestion_manager = SimpleNamespace(process_add=process_add)
    await asyncio.gather(memory.add("first"), memory.add("second"))
    assert peak == 1
