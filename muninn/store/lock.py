"""
Muninn Store Lock
-----------------
Cross-process advisory file locking for storage backends.
Ensures single-writer integrity for Kuzu, SQLite, and Qdrant.
"""

import logging
import os
import contextlib
from pathlib import Path
from typing import Optional

import portalocker

logger = logging.getLogger("Muninn.StoreLock")

class StoreLock:
    """
    Manages cross-process advisory locks using portalocker.
    Typically placed in the data directory (e.g., .muninn.lock).
    """
    def __init__(self, lock_file_path: Path, timeout: float = 10.0):
        self.lock_file_path = lock_file_path
        self.timeout = timeout
        self._lock = None

    @contextlib.contextmanager
    def acquire(self, shared: bool = False):
        """
        Acquire the lock. 
        'shared=True' allows multiple readers if supported by the backend,
        but Muninn primarily uses this for serializing writers.
        """
        # Ensure parent directory exists
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        flags = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
        flags |= portalocker.LOCK_NB
        
        try:
            with portalocker.Lock(
                str(self.lock_file_path),
                mode='a',
                timeout=self.timeout,
                flags=flags,
                fail_when_locked=False
            ) as lock:
                yield lock
        except portalocker.exceptions.LockException as e:
            logger.error(f"Failed to acquire lock on {self.lock_file_path} after {self.timeout}s: {e}")
            raise RuntimeError(f"Database lock contention: {e}")

def get_store_lock(data_path: Path) -> StoreLock:
    """Helper to get a standard lock for a given data directory."""
    lock_file = data_path / ".muninn.lock"
    return StoreLock(lock_file)
