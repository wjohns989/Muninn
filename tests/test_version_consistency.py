"""Ensure runtime/server/package versions stay in sync."""

from pathlib import Path
import tomllib

import mcp_wrapper
import server
from muninn.version import __version__


def test_version_single_source_of_truth():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    assert pyproject["project"]["version"] == __version__
    assert server.app.version == __version__
    assert mcp_wrapper.__version__ == __version__
