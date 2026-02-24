import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import portalocker
import pytest

import server


@pytest.fixture(autouse=True)
def _reset_server_lock_state():
    server._SERVER_INSTANCE_LOCK_HANDLE = None
    server._SERVER_INSTANCE_LOCK_PATH = None
    yield
    server._SERVER_INSTANCE_LOCK_HANDLE = None
    server._SERVER_INSTANCE_LOCK_PATH = None


def _config_with_data_dir(path: Path):
    return types.SimpleNamespace(data_dir=str(path))


def test_acquire_server_instance_lock_success(tmp_path: Path):
    config = _config_with_data_dir(tmp_path)
    lock_handle = MagicMock()

    with patch("server.portalocker.Lock", return_value=lock_handle) as lock_ctor:
        server._acquire_server_instance_lock(config)

    expected_path = tmp_path / ".muninn_server.instance.lock"
    lock_ctor.assert_called_once()
    lock_handle.acquire.assert_called_once()
    assert server._SERVER_INSTANCE_LOCK_HANDLE is lock_handle
    assert server._SERVER_INSTANCE_LOCK_PATH == expected_path


def test_acquire_server_instance_lock_raises_on_contention(tmp_path: Path):
    config = _config_with_data_dir(tmp_path)
    lock_handle = MagicMock()
    lock_handle.acquire.side_effect = portalocker.exceptions.LockException("locked")

    with patch("server.portalocker.Lock", return_value=lock_handle):
        with pytest.raises(RuntimeError, match="already held"):
            server._acquire_server_instance_lock(config)

    assert server._SERVER_INSTANCE_LOCK_HANDLE is None
    assert server._SERVER_INSTANCE_LOCK_PATH is None


def test_release_server_instance_lock_is_idempotent():
    lock_handle = MagicMock()
    server._SERVER_INSTANCE_LOCK_HANDLE = lock_handle
    server._SERVER_INSTANCE_LOCK_PATH = Path("dummy.lock")

    server._release_server_instance_lock()
    server._release_server_instance_lock()

    lock_handle.release.assert_called_once()
    lock_handle.close.assert_called_once()
    assert server._SERVER_INSTANCE_LOCK_HANDLE is None
    assert server._SERVER_INSTANCE_LOCK_PATH is None
