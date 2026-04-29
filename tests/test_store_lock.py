import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import portalocker

from muninn.store.lock import StoreLock, get_store_lock


def test_get_store_lock(tmp_path: Path):
    """Verify get_store_lock returns a StoreLock with the correct .muninn.lock path."""
    lock = get_store_lock(tmp_path)
    assert isinstance(lock, StoreLock)
    assert lock.lock_file_path == tmp_path / ".muninn.lock"


def test_store_lock_acquire_happy_path(tmp_path: Path):
    """Verify StoreLock.acquire creates parent directories and yields correctly without throwing."""
    # We use a path inside tmp_path that doesn't exist yet to verify directory creation
    lock_file = tmp_path / "subdir" / ".test.lock"
    lock = StoreLock(lock_file)

    mock_lock_instance = MagicMock()
    # Need to simulate __enter__ for context manager
    mock_lock_instance.__enter__.return_value = mock_lock_instance

    with patch("muninn.store.lock.portalocker.Lock", return_value=mock_lock_instance) as mock_portalocker_lock:
        with lock.acquire():
            pass

    # Verify directory was created
    assert lock_file.parent.exists()

    # Verify portalocker.Lock was called with expected arguments
    mock_portalocker_lock.assert_called_once_with(
        str(lock_file),
        mode='a',
        timeout=10.0,
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        fail_when_locked=False
    )


def test_store_lock_acquire_shared_flags(tmp_path: Path):
    """Verify shared=True uses portalocker.LOCK_SH and shared=False uses portalocker.LOCK_EX."""
    lock_file = tmp_path / ".test.lock"
    lock = StoreLock(lock_file)

    mock_lock_instance = MagicMock()
    mock_lock_instance.__enter__.return_value = mock_lock_instance

    # Test shared=False (default)
    with patch("muninn.store.lock.portalocker.Lock", return_value=mock_lock_instance) as mock_portalocker_lock:
        with lock.acquire(shared=False):
            pass
        mock_portalocker_lock.assert_called_once_with(
            str(lock_file),
            mode='a',
            timeout=10.0,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            fail_when_locked=False
        )

    # Test shared=True
    with patch("muninn.store.lock.portalocker.Lock", return_value=mock_lock_instance) as mock_portalocker_lock:
        with lock.acquire(shared=True):
            pass
        mock_portalocker_lock.assert_called_once_with(
            str(lock_file),
            mode='a',
            timeout=10.0,
            flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
            fail_when_locked=False
        )


def test_store_lock_acquire_lock_contention(tmp_path: Path):
    """Verify lock contention correctly raises RuntimeError."""
    lock_file = tmp_path / ".test.lock"
    lock = StoreLock(lock_file)

    # We patch portalocker.Lock to raise a LockException when instantiated
    with patch("muninn.store.lock.portalocker.Lock", side_effect=portalocker.exceptions.LockException("Already locked")):
        with pytest.raises(RuntimeError, match="Database lock contention: Already locked"):
            with lock.acquire():
                pass
