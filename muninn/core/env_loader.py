"""Load private project-local environment settings without extra dependencies."""

from __future__ import annotations

import os
import re
from pathlib import Path


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    for index, character in enumerate(value):
        if character == "#" and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value


def load_project_env(project_root: Path, *, override: bool = False) -> bool:
    """Load ``project_root/.env`` while never logging names or values.

    The parser intentionally supports the conservative subset Muninn's local
    configuration uses: comments, optional ``export``, quoted values, and
    whitespace-delimited inline comments. Existing process values win unless
    ``override`` is explicitly requested.
    """

    env_path = Path(project_root) / ".env"
    if not env_path.is_file():
        return False

    try:
        with env_path.open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                name, separator, raw_value = line.partition("=")
                name = name.strip()
                if not separator or not _ENV_NAME.fullmatch(name):
                    continue
                if override or name not in os.environ:
                    os.environ[name] = _parse_value(raw_value)
    except OSError:
        return False

    return True
