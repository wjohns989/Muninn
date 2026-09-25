from __future__ import annotations

import os
from pathlib import Path

from muninn.core.env_loader import load_project_env


def test_load_project_env_supports_local_windows_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MUNINN_DATA_DIR", raising=False)
    monkeypatch.delenv("MUNINN_NO_AUTH", raising=False)
    monkeypatch.delenv("MUNINN_CONSOLIDATION_ENABLED", raising=False)
    (tmp_path / ".env").write_text(
        "# private local configuration\n"
        'MUNINN_DATA_DIR="D:\\MuninnTest\\.muninn_runtime"\n'
        "export MUNINN_NO_AUTH=1\n"
        "MUNINN_CONSOLIDATION_ENABLED=false  # keep production idle\n",
        encoding="utf-8",
    )

    assert load_project_env(tmp_path) is True
    assert os.environ["MUNINN_DATA_DIR"] == r"D:\MuninnTest\.muninn_runtime"
    assert os.environ["MUNINN_NO_AUTH"] == "1"
    assert os.environ["MUNINN_CONSOLIDATION_ENABLED"] == "false"


def test_load_project_env_preserves_explicit_process_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MUNINN_DATA_DIR", "explicit-process-value")
    (tmp_path / ".env").write_text(
        "MUNINN_DATA_DIR=local-file-value\n",
        encoding="utf-8",
    )

    assert load_project_env(tmp_path) is True
    assert os.environ["MUNINN_DATA_DIR"] == "explicit-process-value"


def test_load_project_env_ignores_invalid_names_and_missing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("VALID_NAME", raising=False)
    (tmp_path / ".env").write_text(
        "INVALID-NAME=ignored\n"
        "VALID_NAME=loaded\n",
        encoding="utf-8",
    )

    assert load_project_env(tmp_path) is True
    assert "INVALID-NAME" not in os.environ
    assert os.environ["VALID_NAME"] == "loaded"
    assert load_project_env(tmp_path / "missing") is False
