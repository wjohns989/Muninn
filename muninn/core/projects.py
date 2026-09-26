"""One project name for a directory, shared by live agents and imported history.

A project is named after its git remote (``Muninn`` for ``.../Muninn.git``),
else the main repository folder (so every git worktree of a repo shares its
name), else the folder itself. Worktree folders created by Claude Code
(``<repo>/.claude/worktrees/<name>``) and the Codex app
(``~/.codex/worktrees/<id>/<repo>``) resolve to their repository even after the
worktree has been deleted.
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path, PurePath, PureWindowsPath
from typing import Optional

_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)|\\")


def name_from_remote(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split(":")[-1].removesuffix(".git")


def _git(directory: Path, *args: str) -> str:
    kwargs = {"stderr": subprocess.DEVNULL, "text": True, "timeout": 5}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        return subprocess.check_output(["git", "-C", str(directory), *args], **kwargs).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _as_path(directory: str) -> PurePath:
    """The directory as a path; Windows paths stay Windows paths when read on another OS."""
    if os.name != "nt" and _WINDOWS_PATH.search(directory):
        return PureWindowsPath(directory)
    return Path(directory)


def repository_hint(directory: str) -> PurePath:
    """The repository a worktree folder belongs to, when the path shows it."""
    path = _as_path(directory)
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index] == ".claude" and parts[index + 1] == "worktrees":
            return type(path)(*parts[:index])
        if parts[index] == ".codex" and parts[index + 1] == "worktrees" and len(parts) > index + 3:
            return type(path)(*parts[: index + 4])
    return path


def _is_home(path: PurePath, home: Path) -> bool:
    if path == home:
        return True
    if isinstance(path, Path) and path.exists():
        return False
    # A home folder from another machine or account (C:\Users\<name>, /home/<name>, /Users/<name>).
    parent = path.parent
    return parent.parent == parent.parent.parent and parent.name.lower() in ("users", "home")


@lru_cache(maxsize=4096)
def project_for_directory(directory: Optional[str], home: Optional[str] = None) -> Optional[str]:
    """Project name for a working directory, or None when it is not a project (home, root, empty)."""
    if not directory:
        return None
    path = repository_hint(directory)
    home_path = Path(home or Path.home())
    if str(path) in ("", ".", "/", "\\") or _is_home(path, home_path) or path.parent == path:
        return None
    if isinstance(path, Path) and path.is_dir():
        remote = _git(path, "config", "--get", "remote.origin.url")
        if remote:
            return name_from_remote(remote) or None
        common = _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if common:
            common_path = Path(common)
            root = common_path.parent if common_path.name == ".git" else common_path
            return root.name or None
    return path.name or None
