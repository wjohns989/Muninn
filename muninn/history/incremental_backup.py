"""Local incremental storage around closed Muninn backup bundles.

No live-directory backup, retention deletion, provider dispatch or repository
repair. Muninn's application-level cold restore remains the acceptance gate.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from pathlib import Path

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_directory, verify_private
from muninn.history.recovery_pool import private_file, unlinked
from muninn.history.secure_archive import SecureHistoryArchive, _write_private

VERSION = "0.19.1"
BINARY_SHA = "b0dd1fd21eea5d8fe1325f55f7118213c21f36de8a261e04c0624a5ab9fd7830"
PASSWORD_DOMAIN = b"muninn-restic-full-backup-password-v1"


def _drive_type(path):
    if os.name != "nt":
        return 3
    import ctypes
    from ctypes import wintypes
    drive = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
    drive.argtypes, drive.restype = (wintypes.LPCWSTR,), wintypes.UINT
    return drive(Path(path).anchor)


def local_path(path):
    # Reject network/device paths before even lstat can contact their backend.
    raw = os.fspath(path)
    if raw.startswith(("\\\\", "//")):
        raise VaultIntegrityError("Incremental backup requires a local disk path")
    absolute = os.path.abspath(raw)
    if absolute.startswith(("\\\\", "//")) or _drive_type(absolute) not in {2, 3}:
        raise VaultIntegrityError("Incremental backup requires a local disk path")
    return unlinked(absolute)


class IncrementalBackup:
    def __init__(self, root, binary, *, archive=None, passphrase=None):
        self.root, self.binary = local_path(root), local_path(binary)
        private_file(self.binary)
        with self.binary.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != BINARY_SHA:
                raise VaultIntegrityError("Backup utility differs from the pinned binary")
        if not self.root.exists():
            if archive is None:
                raise VaultIntegrityError("Incremental backup repository is missing")
            create_private_directory(self.root)
            anchor = self.root / "key-anchor"
            create_private_directory(anchor)
            create_private_directory(anchor / "blobs")
            _write_private(anchor / "header.json", archive._header_path.read_bytes())
            _write_private(anchor / "archive.lock", b"\0")
        verify_private(self.root)
        anchor = unlinked(self.root / "key-anchor")
        self.anchor = SecureHistoryArchive(anchor, passphrase, _initializing=True)
        if archive is not None and (archive.vault_id != self.anchor.vault_id
                or not hmac.compare_digest(archive._key, self.anchor._key)):
            raise VaultIntegrityError("Incremental repository belongs to another archive")
        self.store = unlinked(self.root / "repository")
        self._password = hmac.digest(self.anchor._key, PASSWORD_DOMAIN, "sha256").hex()
        if not self.store.exists():
            if archive is None:
                raise VaultIntegrityError("Incremental repository data is missing")
            create_private_directory(self.store)
            self._run("init")
        verify_private(self.store)

    def _run(self, *args, cwd=None):
        # Password stays in the parent/child pipe: never argv, env, a file or log.
        environment = {name: value for name, value in os.environ.items()
                       if name.upper() in {"SYSTEMROOT", "WINDIR", "USERPROFILE",
                                           "LOCALAPPDATA", "APPDATA", "TEMP", "TMP"}}
        command = [str(self.binary), "--repo", str(self.store), "--no-cache", "--json", *args]
        password = self._password.encode("ascii")
        records, tail, echo, skipping = [], b"", False, False
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, env=environment, cwd=cwd,
                              shell=False) as process:
            process.stdin.write(password + b"\n")
            process.stdin.close()
            while line := process.stdout.readline(65536):
                echo = echo or password in tail + line
                tail = (tail + line)[-len(password):]
                if len(line) == 65536 and not line.endswith(b"\n"):
                    skipping = True
                    continue
                if skipping:
                    skipping = False
                    continue  # Drain unusually long progress records, never buffer them.
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(row, dict) and row.get("message_type") == "summary":
                    if len(records) < 2:
                        records.append(row)  # Duplicate summaries fail the acknowledgement gate.
            code = process.wait()
        if code:
            # Incomplete snapshots (exit 3) are never accepted as success. Raw
            # stderr may contain filenames and is intentionally not published.
            raise VaultIntegrityError(f"Incremental backup command failed (exit {code})")
        if echo:
            raise VaultIntegrityError("Backup utility unexpectedly echoed its password")
        return records

    def backup_archive(self, archive, destination):
        """Create and verify an application bundle before incremental enrollment."""
        if (not isinstance(archive, SecureHistoryArchive) or archive.vault_id != self.anchor.vault_id
                or not hmac.compare_digest(archive._key, self.anchor._key)):
            raise VaultIntegrityError("Authenticated source archive is required")
        destination = local_path(destination)
        if destination == self.root or destination in self.root.parents or self.root in destination.parents:
            raise VaultIntegrityError("Backup bundle overlaps its repository")
        application = archive.backup_to(destination)
        return {"application_backup": application, "incremental_snapshot": self._backup_bundle(destination)}

    def _backup_bundle(self, bundle):
        # Internal only: the public write path owns backup_to()'s acceptance.
        bundle = local_path(bundle)
        verify_private(bundle)
        if not bundle.is_dir() or ".incomplete-" in bundle.name:
            raise VaultIntegrityError("Only a closed, published backup bundle may be enrolled")
        if bundle == self.root or bundle in self.root.parents or self.root in bundle.parents:
            raise VaultIntegrityError("Backup bundle overlaps its repository")
        records = self._run("backup", "--force", "--tag", "muninn-closed-bundle", "--", ".", cwd=bundle)
        summaries = [row for row in records if isinstance(row, dict) and row.get("message_type") == "summary"]
        if len(summaries) != 1 or not isinstance(summaries[0].get("snapshot_id"), str):
            raise VaultIntegrityError("Incremental snapshot acknowledgement is missing")
        row = summaries[0]
        return {key: row[key] for key in ("snapshot_id", "data_added", "data_added_packed",
                                          "total_files_processed", "total_bytes_processed") if key in row}

    def check(self):
        self._run("check", "--read-data")
        return {"repository_data_checked": True}

    def restore(self, snapshot_id, destination):
        if (not isinstance(snapshot_id, str) or len(snapshot_id) != 64
                or any(c not in "0123456789abcdef" for c in snapshot_id)):
            raise VaultIntegrityError("Exact incremental snapshot identity is required")
        destination = local_path(destination)
        if destination.exists():
            raise FileExistsError("Incremental restore destination must be new")
        if destination == self.root or self.root in destination.parents:
            raise VaultIntegrityError("Restore cannot write inside its backup repository")
        create_private_directory(destination)
        self._run("restore", snapshot_id, "--target", str(destination), "--verify")
        return {"snapshot_id": snapshot_id, "restored_and_verified": True}
