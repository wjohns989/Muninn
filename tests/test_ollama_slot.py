"""Local Ollama GPU requests serialize without blocking unrelated async work."""

import asyncio
import subprocess
import sys
import threading
import time

import pytest

from muninn.extraction import ollama_slot as slots


def test_sync_slot_is_exclusive_and_released(tmp_path, monkeypatch):
    monkeypatch.setattr(slots, "_lock_path", tmp_path / "gpu.lock")
    monkeypatch.setenv("MUNINN_OLLAMA_MAX_CONCURRENT", "1")
    monkeypatch.setenv("MUNINN_OLLAMA_SLOT_WAIT_SEC", "0.1")
    entered = threading.Event()
    leave = threading.Event()

    def hold():
        with slots.ollama_slot():
            entered.set()
            leave.wait(timeout=3)

    worker = threading.Thread(target=hold)
    worker.start()
    assert entered.wait(timeout=3)
    try:
        with pytest.raises(TimeoutError, match="Ollama slot"):
            with slots.ollama_slot():
                pytest.fail("second request overlapped the first")
    finally:
        leave.set()
        worker.join(timeout=3)
    assert not worker.is_alive()
    with slots.ollama_slot():
        pass


def test_async_slot_wait_does_not_block_event_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(slots, "_lock_path", tmp_path / "gpu.lock")
    monkeypatch.setenv("MUNINN_OLLAMA_MAX_CONCURRENT", "1")
    monkeypatch.setenv("MUNINN_OLLAMA_SLOT_WAIT_SEC", "0.1")

    async def run():
        async with slots.async_ollama_slot():
            ticked = asyncio.Event()

            async def tick():
                await asyncio.sleep(0)
                ticked.set()

            tick_task = asyncio.create_task(tick())
            with pytest.raises(TimeoutError):
                async with slots.async_ollama_slot():
                    pytest.fail("second request overlapped the first")
            await tick_task
            assert ticked.is_set()

    asyncio.run(run())


def test_slot_is_exclusive_across_processes(tmp_path, monkeypatch):
    lock_path = tmp_path / "gpu.lock"
    ready_path = tmp_path / "ready"
    monkeypatch.setattr(slots, "_lock_path", lock_path)
    monkeypatch.setenv("MUNINN_OLLAMA_MAX_CONCURRENT", "1")
    monkeypatch.setenv("MUNINN_OLLAMA_SLOT_WAIT_SEC", "0.2")
    child_code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from muninn.extraction import ollama_slot as slots\n"
        "slots._lock_path = Path(sys.argv[1])\n"
        "with slots.ollama_slot():\n"
        "    Path(sys.argv[2]).write_text('ready')\n"
        "    time.sleep(5)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code, str(lock_path), str(ready_path)])
    try:
        deadline = time.monotonic() + 3
        while not ready_path.exists() and time.monotonic() < deadline:
            assert child.poll() is None, "lock-holder exited before acquiring slot"
            time.sleep(0.02)
        assert ready_path.exists(), "lock-holder did not acquire slot"
        with pytest.raises(TimeoutError, match="Ollama slot"):
            with slots.ollama_slot():
                pytest.fail("cross-process request overlapped")
    finally:
        child.terminate()
        child.wait(timeout=3)
