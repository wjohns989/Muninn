"""Make a verified owner-only preimage before managed remote-accounting migration.

This is evidence for recovery, not an installable policy restore. Replacing live
policy with an older enabled copy could revive consent or reset spend tracking.
No credential or transcript is read or printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.remote_accounting import _MARKER as _ADMISSION_MARKER
from muninn.history.remote_policy import _paths, read_policy


def _same_database(source: sqlite3.Connection, copy: sqlite3.Connection) -> None:
    sentinel = object()
    queries = ["SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name",
               "PRAGMA user_version"]
    names = [row[0] for row in source.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    if set(names) - {"policy", "audit", "remote_admissions"} or not {"policy", "audit"} <= set(names):
        raise ValueError("Remote policy backup schema is unsupported")
    queries.extend(f"SELECT * FROM {name} ORDER BY 1" for name in names)
    for query in queries:
        for before, after in zip_longest(source.execute(query), copy.execute(query), fillvalue=sentinel):
            if before is sentinel or after is sentinel or tuple(before) != tuple(after):
                raise ValueError("Remote policy backup differs from source")


def backup(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(strict=True), destination.absolute()
    directory, marker, database = _paths(root)
    if (destination == root or root in destination.parents or destination in root.parents
            or not destination.parent.is_dir() or destination.parent.resolve(strict=True) != destination.parent):
        raise ValueError("Unsafe remote policy preimage destination")
    verify_private(destination.parent)
    for path in (directory, marker, database):
        verify_private(path)
    policy = read_policy(root, lambda: (False, 1.0, 30.0, False))
    admission_marker = directory / "admission-managed"
    create_private_directory(destination)
    for name, source_path in (("managed", marker), ("policy.sqlite3", database)):
        target = destination / name
        create_private_file(target)
        if name == "managed":
            shutil.copyfile(source_path, target)
            with target.open("r+b") as stream:
                os.fsync(stream.fileno())
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as source:
        source.execute("BEGIN")
        snapshot_generation = source.execute(
            "SELECT generation FROM policy WHERE id=1").fetchone()[0]
        if snapshot_generation != policy.generation:
            raise ValueError("Remote policy changed during preimage")
        columns = {row[1] for row in source.execute("PRAGMA table_info(policy)")}
        version = (source.execute("SELECT accounting_version FROM policy WHERE id=1").fetchone()
                   if "accounting_version" in columns else None)
        admissions_table = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='remote_admissions'").fetchone()
        if version == (1,) and admissions_table:
            initialized = True
        elif version in (None, (0,)) and not admissions_table:
            initialized = False
        else:
            raise ValueError("Remote accounting snapshot is inconsistent")

        def check_admission_marker() -> None:
            present = admission_marker.exists() or admission_marker.is_symlink()
            if present != initialized:
                raise ValueError("Remote accounting marker and snapshot disagree")
            if present:
                verify_private(admission_marker)
                if admission_marker.read_bytes() != _ADMISSION_MARKER:
                    raise ValueError("Remote accounting marker is invalid")

        check_admission_marker()
        with closing(sqlite3.connect(destination / "policy.sqlite3")) as copy:
            source.backup(copy)
            if copy.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Remote policy backup integrity check failed")
            _same_database(source, copy)
        check_admission_marker()
        if initialized:
            target = destination / "admission-managed"
            create_private_file(target)
            shutil.copyfile(admission_marker, target)
            with target.open("r+b") as stream:
                os.fsync(stream.fileno())
            check_admission_marker()
    with (destination / "policy.sqlite3").open("r+b") as stream:
        os.fsync(stream.fileno())
    verify_private(destination / "managed")
    verify_private(destination / "policy.sqlite3")
    if initialized:
        verify_private(destination / "admission-managed")
        if (destination / "admission-managed").read_bytes() != _ADMISSION_MARKER:
            raise ValueError("Copied remote accounting marker is invalid")
    with closing(sqlite3.connect(f"{(destination / 'policy.sqlite3').as_uri()}?mode=ro", uri=True)) as copy:
        unresolved = (copy.execute("SELECT count(*) FROM remote_admissions "
                                   "WHERE state IN ('reserved','unknown')").fetchone()[0]
                      if initialized else 0)
    accounting_state = "blocked" if unresolved else "ready" if initialized else "uninitialized"
    return {"state": "validated_preimage", "policy_generation": policy.generation,
            "policy_enabled": policy.enabled, "accounting_state": accounting_state,
            "unresolved_admissions": unresolved,
            "restorable_as_active_policy": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(backup(args.root, args.destination), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"state": "preimage_failed", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
