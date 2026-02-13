"""Tests for muninn.platform — Cross-platform abstraction layer."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from muninn.platform import (
    IS_WINDOWS,
    IS_MACOS,
    IS_LINUX,
    get_data_dir,
    get_config_dir,
    get_log_dir,
    get_legacy_data_dir,
    get_process_creation_flags,
    find_python_executable,
    find_ollama_executable,
    is_running_in_docker,
    get_platform_info,
    ensure_directories,
    spawn_detached_process,
    log_platform_summary,
)


class TestPlatformDetection:
    """Test OS detection flags."""

    def test_exactly_one_platform_true(self):
        """Exactly one platform flag should be True (or none on exotic OS)."""
        platforms = [IS_WINDOWS, IS_MACOS, IS_LINUX]
        assert sum(platforms) <= 1

    def test_current_platform_matches_sys(self):
        if sys.platform == "win32":
            assert IS_WINDOWS is True
        elif sys.platform == "darwin":
            assert IS_MACOS is True
        elif sys.platform.startswith("linux"):
            assert IS_LINUX is True


class TestDockerDetection:
    """Test Docker environment detection."""

    def test_docker_env_var(self):
        with patch.dict(os.environ, {"MUNINN_DOCKER": "1"}):
            assert is_running_in_docker() is True

    def test_not_docker_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            # May or may not be in Docker depending on CI, just test it runs
            result = is_running_in_docker()
            assert isinstance(result, bool)


class TestDataDir:
    """Test data directory resolution."""

    def test_env_override(self):
        with patch.dict(os.environ, {"MUNINN_DATA_DIR": "/custom/data"}):
            result = get_data_dir()
            assert result == Path("/custom/data")

    def test_default_is_path(self, monkeypatch):
        monkeypatch.delenv("MUNINN_DATA_DIR", raising=False)
        with patch("muninn.platform.is_running_in_docker", return_value=False):
            result = get_data_dir()
            assert isinstance(result, Path)
            assert "muninn" in str(result).lower()

    def test_docker_default(self, monkeypatch):
        monkeypatch.delenv("MUNINN_DATA_DIR", raising=False)
        with patch("muninn.platform.is_running_in_docker", return_value=True):
            result = get_data_dir()
            assert result == Path("/data")


class TestConfigDir:
    """Test config directory resolution."""

    def test_env_override(self):
        with patch.dict(os.environ, {"MUNINN_CONFIG_DIR": "/custom/config"}):
            result = get_config_dir()
            assert result == Path("/custom/config")

    def test_default_is_path(self):
        os.environ.pop("MUNINN_CONFIG_DIR", None)
        result = get_config_dir()
        assert isinstance(result, Path)


class TestLogDir:
    """Test log directory resolution."""

    def test_env_override(self):
        with patch.dict(os.environ, {"MUNINN_LOG_DIR": "/custom/logs"}):
            result = get_log_dir()
            assert result == Path("/custom/logs")

    def test_default_is_path(self):
        os.environ.pop("MUNINN_LOG_DIR", None)
        result = get_log_dir()
        assert isinstance(result, Path)


class TestLegacyDataDir:
    """Test legacy path detection for migration."""

    def test_returns_home_based_path(self):
        result = get_legacy_data_dir()
        assert result == Path.home() / ".muninn" / "data"


class TestProcessCreationFlags:
    """Test cross-platform process flags."""

    def test_returns_int(self):
        flags = get_process_creation_flags()
        assert isinstance(flags, int)

    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only test")
    def test_windows_flags_nonzero(self):
        flags = get_process_creation_flags()
        assert flags > 0
        # Should include CREATE_NO_WINDOW (0x08000000)
        assert flags & 0x08000000 != 0

    @pytest.mark.skipif(IS_WINDOWS, reason="Unix-only test")
    def test_unix_flags_zero(self):
        flags = get_process_creation_flags()
        assert flags == 0


class TestFindPythonExecutable:
    """Test Python executable discovery."""

    def test_returns_string(self):
        result = find_python_executable()
        assert isinstance(result, str)

    def test_executable_exists(self):
        result = find_python_executable()
        assert Path(result).exists()


class TestFindOllamaExecutable:
    """Test Ollama executable discovery."""

    def test_returns_string_or_none(self):
        result = find_ollama_executable()
        assert result is None or isinstance(result, str)


class TestPlatformInfo:
    """Test platform diagnostic info."""

    def test_returns_dict(self):
        info = get_platform_info()
        assert isinstance(info, dict)

    def test_required_keys(self):
        info = get_platform_info()
        required = ["os", "python", "is_windows", "is_macos", "is_linux",
                     "is_docker", "data_dir", "config_dir", "log_dir"]
        for key in required:
            assert key in info, f"Missing key: {key}"

    def test_os_matches_sys(self):
        info = get_platform_info()
        assert info["os"] == sys.platform


class TestEnsureDirectories:
    """Test directory creation."""

    def test_creates_directories(self, tmp_path):
        with patch.dict(os.environ, {"MUNINN_DATA_DIR": str(tmp_path / "data")}):
            dirs = ensure_directories()
            assert dirs["data"].exists()
            assert dirs["vectors"].exists()
            assert dirs["graph"].exists()

    def test_returns_dict_of_paths(self, tmp_path):
        with patch.dict(os.environ, {"MUNINN_DATA_DIR": str(tmp_path / "data")}):
            dirs = ensure_directories()
            assert isinstance(dirs, dict)
            for key, path in dirs.items():
                assert isinstance(path, Path)


class TestSpawnDetachedProcess:
    """Test cross-platform detached process spawning."""

    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only test")
    def test_spawn_on_windows(self):
        """Verify spawn works on Windows with echo."""
        import subprocess
        proc = spawn_detached_process(["cmd.exe", "/c", "echo", "test"])
        assert isinstance(proc, subprocess.Popen)
        proc.wait(timeout=5)

    @pytest.mark.skipif(IS_WINDOWS, reason="Unix-only test")
    def test_spawn_on_unix(self):
        """Verify spawn works on Unix with echo."""
        import subprocess
        proc = spawn_detached_process(["echo", "test"])
        assert isinstance(proc, subprocess.Popen)
        proc.wait(timeout=5)


class TestLogPlatformSummary:
    """Test platform summary logging."""

    def test_does_not_raise(self):
        """Summary logging should never fail."""
        log_platform_summary()
