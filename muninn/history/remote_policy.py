"""Owner-local, nonsecret ZDR consent and admission-threshold policy.

The marker is created before the database. Losing or corrupting a managed policy
must not silently revive an older environment opt-in. SQLite commits the current
policy and its audit row together; no transcript, bearer, or API key is stored.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from muninn.history.private_acl import (
    VaultPermissionError,
    create_private_directory,
    create_private_file,
    verify_private,
)

_LOCK = threading.RLock()


class PolicyError(RuntimeError):
    """A managed policy cannot be authenticated or read; remote use must stop."""


@dataclass(frozen=True)
class RemotePolicy:
    enabled: bool
    daily_usd: float
    monthly_usd: float
    override_ceiling: bool
    generation: int
    source: str


def _paths(root: Path) -> tuple[Path, Path, Path]:
    directory = Path(root).absolute() / "remote_policy"
    return directory, directory / "managed", directory / "policy.sqlite3"


def _valid_budget(value: float) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def read_policy(root: Path, fallback: Callable[[], tuple[bool, float, float, bool]]) -> RemotePolicy:
    """Read each dispatch afresh; never cache an enabled state across revocation."""
    directory, marker, database = _paths(root)
    with _LOCK:
        if not directory.exists() and not directory.is_symlink():
            _legacy_enabled, daily, monthly, override = fallback()
            if not (_valid_budget(daily) and _valid_budget(monthly)):
                daily = monthly = 0.0
            # A whole-directory loss is indistinguishable from first launch.
            # Never let an older environment opt-in revive remote use.
            return RemotePolicy(False, float(daily), float(monthly),
                                bool(override), 0, "unconfigured")
        try:
            verify_private(directory)
            verify_private(marker)
            verify_private(database)
            with marker.open("rb") as stream:
                marker_bytes = stream.read(64)
                marker_has_extra = bool(stream.read(1))
            if marker_has_extra or marker_bytes != b"muninn-managed-remote-policy-v1\n":
                raise PolicyError("Managed remote policy marker is invalid")
            with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=2)) as db:
                row = db.execute(
                    "SELECT version,enabled,daily_usd,monthly_usd,override_ceiling,generation "
                    "FROM policy WHERE id=1"
                ).fetchone()
            if row is None or row[0] != 1 or row[1] not in (0, 1) or row[4] not in (0, 1):
                raise PolicyError("Managed remote policy is invalid")
            if not (_valid_budget(row[2]) and _valid_budget(row[3])) or type(row[5]) is not int or row[5] < 1:
                raise PolicyError("Managed remote policy is invalid")
            if not row[4] and (row[2] > 10 or row[3] > 100):
                raise PolicyError("Managed remote policy exceeds default ceilings")
            return RemotePolicy(bool(row[1]), float(row[2]), float(row[3]),
                                bool(row[4]), row[5], "managed")
        except (OSError, sqlite3.Error, VaultPermissionError, ValueError) as exc:
            raise PolicyError("Managed remote policy is unavailable") from exc


def write_policy(root: Path, *, enabled: bool, daily_usd: float, monthly_usd: float,
                 override_ceiling: bool,
                 fallback: Callable[[], tuple[bool, float, float, bool]]) -> RemotePolicy:
    """Record one explicit local consent/budget choice and its audit event."""
    if type(enabled) is not bool or type(override_ceiling) is not bool:
        raise ValueError("Invalid remote policy flags")
    if not (_valid_budget(daily_usd) and _valid_budget(monthly_usd)):
        raise ValueError("Remote admission thresholds must be finite positive values")
    if not override_ceiling and (daily_usd > 10 or monthly_usd > 100):
        raise ValueError("An explicit override is required above $10/day or $100/month")
    directory, marker, database = _paths(root)
    with _LOCK:
        try:
            if not directory.exists() and not directory.is_symlink():
                create_private_directory(directory)
            else:
                verify_private(directory)
            if not marker.exists() and not marker.is_symlink():
                create_private_file(marker)
                with marker.open("wb") as stream:
                    stream.write(b"muninn-managed-remote-policy-v1\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                verify_private(marker)
            if not database.exists() and not database.is_symlink():
                create_private_file(database)
            else:
                verify_private(database)
            with closing(sqlite3.connect(database, timeout=5)) as db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), "
                    "version INTEGER NOT NULL, enabled INTEGER NOT NULL, daily_usd REAL NOT NULL, "
                    "monthly_usd REAL NOT NULL, override_ceiling INTEGER NOT NULL, generation INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS audit (generation INTEGER PRIMARY KEY, "
                    "changed_at REAL NOT NULL, enabled INTEGER NOT NULL, daily_usd REAL NOT NULL, "
                    "monthly_usd REAL NOT NULL, override_ceiling INTEGER NOT NULL)"
                )
                prior = db.execute("SELECT generation FROM policy WHERE id=1").fetchone()
                generation = (prior[0] if prior else 0) + 1
                values = (int(enabled), float(daily_usd), float(monthly_usd),
                          int(override_ceiling), generation)
                db.execute(
                    "INSERT INTO policy(id,version,enabled,daily_usd,monthly_usd,override_ceiling,generation) "
                    "VALUES(1,1,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "enabled=excluded.enabled,daily_usd=excluded.daily_usd,"
                    "monthly_usd=excluded.monthly_usd,override_ceiling=excluded.override_ceiling,"
                    "generation=excluded.generation", values,
                )
                db.execute(
                    "INSERT INTO audit(generation,changed_at,enabled,daily_usd,monthly_usd,override_ceiling) "
                    "VALUES(?,?,?,?,?,?)", (generation, time.time(), *values[:4]),
                )
                db.commit()
            return read_policy(root, fallback)
        except (OSError, sqlite3.Error, VaultPermissionError) as exc:
            raise PolicyError("Managed remote policy could not be saved") from exc
