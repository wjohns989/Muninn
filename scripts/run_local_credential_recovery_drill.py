"""Validate a portable credential backup and separate restore with one local prompt.

The passphrase is read from the controlling terminal, never an argument, environment
variable, status record, or printed output. All destinations must be new paths.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sqlite3
import stat
import sys
import time
from contextlib import closing
from pathlib import Path

from muninn.history.credential_store import CredentialStore
from muninn.history.private_acl import (
    _windows_identity, create_private_directory, create_private_file, verify_private,
)


def _ace_may_replace_child(ace, security, trusted: set[str], dangerous: int,
                           inherit_only: int) -> bool:
    (ace_type, flags), mask, sid = ace[:3]
    if flags & inherit_only or ace_type == security.ACCESS_DENIED_ACE_TYPE:
        return False
    if ace_type != security.ACCESS_ALLOWED_ACE_TYPE:
        # Callback/conditional and object ACEs need effective-access evaluation.
        # Reject them rather than assuming a foreign grant is harmless.
        raise ValueError("Recovery drill ancestor has unsupported ACL entry")
    return bool(mask & dangerous and security.ConvertSidToStringSid(sid) not in trusted)


def _verify_nonreplaceable_ancestors(parent: Path) -> None:
    """Reject locations another nonadministrative principal can replace."""
    parent = parent.resolve(strict=True)
    if os.name == "nt":
        import ntsecuritycon
        import win32con

        security, user = _windows_identity()
        trusted = {security.ConvertSidToStringSid(user), "S-1-5-18", "S-1-5-32-544"}
        dangerous = (ntsecuritycon.FILE_DELETE_CHILD | ntsecuritycon.DELETE
                     | ntsecuritycon.WRITE_DAC | ntsecuritycon.WRITE_OWNER
                     | win32con.GENERIC_ALL)
    while True:
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("Recovery drill ancestor is linked")
        if os.name == "nt":
            descriptor = security.GetNamedSecurityInfo(
                str(parent), security.SE_FILE_OBJECT, security.DACL_SECURITY_INFORMATION)
            acl = descriptor.GetSecurityDescriptorDacl()
            if acl is None:
                raise ValueError("Recovery drill ancestor has unrestricted ACL")
            for index in range(acl.GetAceCount()):
                if _ace_may_replace_child(acl.GetAce(index), security, trusted,
                                          dangerous, win32con.INHERIT_ONLY_ACE):
                    raise ValueError("Recovery drill ancestor permits foreign replacement")
        elif parent.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("Recovery drill ancestor is group/world writable")
        if parent == parent.parent:
            return
        parent = parent.parent


def _read_only_db(path: Path):
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def run(vault_root: Path, backup_destination: Path, restore_destination: Path,
        status_file: Path, *, prompt=None) -> int:
    vault_root = vault_root.resolve(strict=True)
    backup_destination = backup_destination.absolute()
    restore_destination = restore_destination.absolute()
    status_file = status_file.absolute()
    if (backup_destination == restore_destination
            or status_file in {backup_destination, restore_destination}
            or backup_destination in restore_destination.parents
            or restore_destination in backup_destination.parents
            or backup_destination in status_file.parents
            or restore_destination in status_file.parents
            or any(destination == vault_root or vault_root in destination.parents
                   for destination in (backup_destination, restore_destination, status_file))):
        raise ValueError("Recovery drill paths overlap the live vault")
    for destination in (backup_destination, restore_destination, status_file):
        if (not destination.parent.is_dir() or destination.exists() or destination.is_symlink()
                or destination.parent.resolve(strict=True) != destination.parent):
            raise ValueError("Recovery drill destinations must be new paths with existing parents")
    verify_private(status_file.parent)
    create_private_file(status_file)

    def report(stage: str, **fields: object) -> None:
        item = {"stage": stage, **fields}
        with status_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
            stream.flush()
            if stage in {"complete", "failed"}:
                os.fsync(stream.fileno())
        print(json.dumps(item, sort_keys=True), flush=True)

    started = time.monotonic()
    report("awaiting_passphrase")
    try:
        passphrase = (prompt or getpass.getpass)("Credential vault recovery passphrase (hidden): ")
        store = CredentialStore(vault_root)
        report("backup_started")
        count = store.backup(backup_destination, passphrase=passphrase)
        report("backup_validated", credential_records=count)
        report("restore_started")
        restored = CredentialStore.restore(backup_destination, restore_destination,
                                           passphrase=passphrase)
        del passphrase
        with closing(_read_only_db(backup_destination / "records.db")) as source:
            with closing(_read_only_db(restored.db_path)) as copy:
                CredentialStore._compare_recovery_snapshot(source, copy)
                copied_count = copy.execute("SELECT COUNT(*) FROM credentials").fetchone()[0]
                ambiguity_count = copy.execute("SELECT COUNT(*) FROM ambiguity_queue").fetchone()[0]
        if copied_count != count:
            raise ValueError("Credential recovery count mismatch")
        report("restore_validated", credential_records=copied_count,
               ambiguity_records=ambiguity_count)
        report("complete", elapsed_seconds=round(time.monotonic() - started, 1))
        return 0
    except Exception as exc:
        report("failed", error_category=type(exc).__name__)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-root", type=Path, required=True)
    parser.add_argument("--drill-root", type=Path, required=True,
                        help="New owner-only directory for backup, restore, and monitor status")
    args = parser.parse_args(argv)
    try:
        root = args.drill_root.absolute()
        live = args.vault_root.resolve(strict=True)
        if (root == live or live in root.parents or root in live.parents
                or not root.parent.is_dir() or root.parent.resolve(strict=True) != root.parent):
            raise ValueError("Recovery drill root overlaps live vault or has linked parent")
        _verify_nonreplaceable_ancestors(root.parent)
        create_private_directory(root)
        return run(live, root / "backup", root / "restore", root / "status.jsonl")
    except Exception as exc:
        print(json.dumps({"stage": "preflight_failed", "error_category": type(exc).__name__}),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
