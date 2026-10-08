"""Bound concurrent Muninn requests to a shared local Ollama GPU.

The lock is process-wide and cross-process for Muninn instances under the same
user. It does not control other GPU applications; request-scoped keep_alive is
still required to release Ollama's model after each request.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

_semaphores: dict[int, threading.BoundedSemaphore] = {}
_semaphores_guard = threading.Lock()
_lock_path = Path(tempfile.gettempdir()) / "muninn-ollama-gpu.lock"


def _settings() -> tuple[int, float]:
    slots = max(1, min(8, int(os.environ.get("MUNINN_OLLAMA_MAX_CONCURRENT", "1"))))
    wait = max(0.0, float(os.environ.get("MUNINN_OLLAMA_SLOT_WAIT_SEC", "300")))
    return slots, wait


def _semaphore(slots: int) -> threading.BoundedSemaphore:
    with _semaphores_guard:
        return _semaphores.setdefault(slots, threading.BoundedSemaphore(slots))


class _Slot:
    def __init__(self) -> None:
        self.slots, self.wait = _settings()
        self.semaphore = _semaphore(self.slots)
        self.stream = None

    def try_acquire(self) -> bool:
        if not self.semaphore.acquire(blocking=False):
            return False
        try:
            stream = _lock_path.open("a+b")
            if stream.seek(0, os.SEEK_END) < self.slots:
                stream.write(b"\0" * self.slots)
                stream.flush()
            for offset in range(self.slots):
                stream.seek(offset)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
                    self.stream = stream
                    self.offset = offset
                    return True
                except OSError:
                    continue
            stream.close()
            self.semaphore.release()
            return False
        except BaseException:
            self.semaphore.release()
            raise

    def release(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(self.offset)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.lockf(self.stream.fileno(), fcntl.LOCK_UN, 1, self.offset)
        finally:
            self.stream.close()
            self.stream = None
            self.semaphore.release()


@contextmanager
def ollama_slot():
    slot = _Slot()
    deadline = time.monotonic() + slot.wait
    while not slot.try_acquire():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for a local Ollama slot")
        time.sleep(0.2)
    try:
        yield
    finally:
        slot.release()


@asynccontextmanager
async def async_ollama_slot():
    slot = _Slot()
    deadline = time.monotonic() + slot.wait
    while not slot.try_acquire():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for a local Ollama slot")
        await asyncio.sleep(0.2)
    try:
        yield
    finally:
        slot.release()
