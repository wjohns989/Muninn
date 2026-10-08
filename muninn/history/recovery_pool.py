"""Private, portable, incremental recovery for one expensive restart preimage.

Never opens a live database for writing, never collects chunks, and never packs
batch records. Full runtime backups and the credential vault remain separate.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
import uuid
from contextlib import closing
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_directory, verify_private
from muninn.history.secure_archive import SecureHistoryArchive, _write_private, _rename_noreplace

CHUNK = 1024 * 1024
RELATIVE = "source-evidence/projections.sqlite3"
MARKER = "source-evidence-recovery.enc"
_SNAPSHOT = re.compile(r"restart-\d{8}-\d{6}-[0-9a-f]{32}")
_HEX = re.compile(r"[0-9a-f]{64}")


def durable_publish(source, target):
    """No overwrite, with a metadata durability barrier before retirement."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        # No REPLACE_EXISTING or COPY_ALLOWED: same-volume atomic publication.
        # WRITE_THROUGH waits for the move to reach disk (Microsoft API contract).
        if not move(str(source), str(target), 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        _rename_noreplace(source, target)
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durability_barrier(directory):
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def unlinked(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise VaultIntegrityError("Recovery path is linked")
    return path


def private_file(path):
    unlinked(path)
    verify_private(path)
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise VaultIntegrityError("Recovery file is not independent")
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def integrity(path):
    # Immutable read-only connection avoids creating sidecars in recovery files.
    private_file(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise VaultIntegrityError("Recovered preimage SQLite integrity failed")


class RecoveryPool:
    def __init__(self, root, *, archive=None, passphrase=None):
        self.root = unlinked(root)
        if not self.root.exists():
            if archive is None:
                raise VaultIntegrityError("Recovery pool is missing")
            create_private_directory(self.root)
            create_private_directory(self.root / "key-anchor")
            create_private_directory(self.root / "key-anchor" / "blobs")
            create_private_directory(self.root / "chunks")
            # Retain the original portable wrapping as well as Windows unlock.
            self._publish(self.root / "key-anchor" / "header.json", archive._header_path.read_bytes())
            self._publish(self.root / "key-anchor" / "archive.lock", b"\0")
        verify_private(self.root)
        unlinked(self.root / "key-anchor")
        unlinked(self.root / "chunks")
        verify_private(self.root / "chunks")
        self.anchor = SecureHistoryArchive(self.root / "key-anchor", passphrase, _initializing=True)
        if archive is not None and (self.anchor.vault_id != archive.vault_id
                                   or not hmac.compare_digest(self.anchor._key, archive._key)):
            raise VaultIntegrityError("Recovery pool belongs to another archive")
        self._key = hmac.digest(self.anchor._key, b"muninn-recovery-pool-aead-v1", "sha256")
        self._ids = hmac.digest(self.anchor._key, b"muninn-recovery-pool-address-v1", "sha256")
        self.snapshots = unlinked(self.root / "snapshots")
        if not self.snapshots.exists():
            create_private_directory(self.snapshots)
        verify_private(self.snapshots)

    def _aad(self, purpose, identity):
        return ("muninn-recovery-pool-v1\0" + self.anchor.vault_id + "\0" + purpose + "\0" + identity).encode()

    def _seal(self, purpose, identity, data):
        nonce = os.urandom(12)
        return nonce + AESGCM(self._key).encrypt(nonce, data, self._aad(purpose, identity))

    def _open(self, purpose, identity, data):
        try:
            return AESGCM(self._key).decrypt(data[:12], data[12:], self._aad(purpose, identity))
        except (InvalidTag, ValueError) as exc:
            raise VaultIntegrityError("Recovery ciphertext authentication failed") from exc

    def _publish(self, target, data):
        temp = target.parent / (".incomplete-" + uuid.uuid4().hex)
        _write_private(temp, data)
        durable_publish(temp, target)

    def _chunk(self, identity, length):
        if not isinstance(identity, str) or not _HEX.fullmatch(identity) or not 1 <= length <= CHUNK:
            raise VaultIntegrityError("Invalid recovery chunk reference")
        path = self.root / "chunks" / (identity + ".enc")
        details = private_file(path)
        if details[2] != length + 28:
            raise VaultIntegrityError("Recovery chunk size differs")
        data = self._open("chunk", identity, path.read_bytes())
        if not hmac.compare_digest(hmac.digest(self._ids, data, "sha256").hex(), identity):
            raise VaultIntegrityError("Recovery chunk identity differs")
        return data

    def _read_manifest(self, marker, snapshot):
        if not _SNAPSHOT.fullmatch(snapshot):
            raise VaultIntegrityError("Invalid restart snapshot identity")
        details = private_file(marker)
        if details[2] > 8 * 1024 * 1024:
            raise VaultIntegrityError("Recovery manifest is too large")
        try:
            value = json.loads(self._open("manifest", snapshot, marker.read_bytes()))
            if (set(value) != {"format", "snapshot", "path", "size", "sha256", "chunks"}
                    or type(value["format"]) is not int or value["format"] != 1
                    or value["snapshot"] != snapshot or value["path"] != RELATIVE
                    or type(value["size"]) is not int or value["size"] < 0
                    or not isinstance(value["sha256"], str) or not _HEX.fullmatch(value["sha256"])
                    or not isinstance(value["chunks"], list)
                    or len(value["chunks"]) != (value["size"] + CHUNK - 1) // CHUNK):
                raise ValueError
            for index, part in enumerate(value["chunks"]):
                if (not isinstance(part, list) or len(part) != 2
                        or not isinstance(part[0], str) or not _HEX.fullmatch(part[0])
                        or type(part[1]) is not int
                        or part[1] != min(CHUNK, value["size"] - index * CHUNK)):
                    raise ValueError
            return value
        except (ValueError, TypeError, KeyError) as exc:
            raise VaultIntegrityError("Recovery manifest is invalid") from exc

    def _reconstruct(self, manifest, target):
        from muninn.history.private_acl import create_private_file

        create_private_file(target)
        digest = hashlib.sha256()
        with target.open("wb") as output:
            for identity, length in manifest["chunks"]:
                raw = self._chunk(identity, length)
                digest.update(raw)
                output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != manifest["sha256"] or target.stat().st_size != manifest["size"]:
            raise VaultIntegrityError("Recovered preimage bytes differ")
        integrity(target)

    def _central_manifest(self, marker, snapshot):
        manifest = self._read_manifest(marker, snapshot)
        central = self.snapshots / (snapshot + ".enc")
        raw = marker.read_bytes()
        if central.exists():
            private_file(central)
            if central.read_bytes() != raw:
                raise VaultIntegrityError("Central recovery manifest conflicts")
        else:
            self._publish(central, raw)
        if self._read_manifest(central, snapshot) != manifest:
            raise VaultIntegrityError("Central recovery manifest differs")
        return manifest

    def backfill_manifests(self, archive):
        """Close portable references for already retired, verified old copies.

        Validate each unique ciphertext chunk once, without rewriting chunks or
        replaying SQLite validation for byte-identical retained snapshot data.
        Call only after the previous compaction process has completed.
        """
        parent = unlinked(archive.root / "operator-preimages")
        seen = set()
        count = 0
        with self.anchor._write_lock():
            for snapshot in sorted(parent.iterdir()):
                if not _SNAPSHOT.fullmatch(snapshot.name):
                    continue
                snapshot = self._snapshot(archive, snapshot)
                marker = snapshot / MARKER
                # Never grant a completed-copy claim to a prepared/incomplete
                # marker whose original DB is still present.
                if not marker.exists() or (snapshot / RELATIVE).exists():
                    continue
                manifest = self._read_manifest(marker, snapshot.name)
                for identity, length in manifest["chunks"]:
                    if (identity, length) not in seen:
                        self._chunk(identity, length)
                        seen.add((identity, length))
                self._central_manifest(marker, snapshot.name)
                count += 1
            for directory in (self.root, self.root / "chunks", self.snapshots):
                durability_barrier(directory)
        return {"central_manifests": count, "unique_chunks_authenticated": len(seen)}

    def _snapshot(self, archive, snapshot):
        path = unlinked(snapshot)
        expected = unlinked(archive.root / "operator-preimages")
        if path.parent != expected or not _SNAPSHOT.fullmatch(path.name):
            raise VaultIntegrityError("Snapshot is outside the restart recovery namespace")
        for directory in (expected, path, path / "source-evidence"):
            verify_private(directory)
        return path

    def pack(self, archive, snapshot, *, retire=False, on_stage=None):
        """Pack an isolated, closed preimage; retire only after recovery read-back.

        No age-based deletion: chunks and all snapshot manifests are retained.
        A retry either proves the existing manifest or leaves originals intact.
        """
        if archive.vault_id != self.anchor.vault_id:
            raise VaultIntegrityError("Recovery pool identity differs")
        snapshot = self._snapshot(archive, snapshot)
        source = snapshot / RELATIVE
        marker = snapshot / MARKER
        with self.anchor._write_lock():
            if marker.exists():
                manifest = self._read_manifest(marker, snapshot.name)
                fingerprint = private_file(source) if source.exists() else None
                new_bytes = 0
            else:
                fingerprint = private_file(source)
                digest = hashlib.sha256()
                refs = []
                new_bytes = 0
                with source.open("rb") as handle:
                    while raw := handle.read(CHUNK):
                        digest.update(raw)
                        identity = hmac.digest(self._ids, raw, "sha256").hex()
                        chunk = self.root / "chunks" / (identity + ".enc")
                        if chunk.exists():
                            self._chunk(identity, len(raw))
                        else:
                            self._publish(chunk, self._seal("chunk", identity, raw))
                            new_bytes += len(raw) + 28
                        refs.append([identity, len(raw)])
                if private_file(source) != fingerprint:
                    raise VaultIntegrityError("Preimage changed during packing")
                manifest = {"format": 1, "snapshot": snapshot.name, "path": RELATIVE,
                            "size": fingerprint[2], "sha256": digest.hexdigest(), "chunks": refs}
                self._publish(marker, self._seal("manifest", snapshot.name,
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()))
                manifest = self._read_manifest(marker, snapshot.name)
            manifest = self._central_manifest(marker, snapshot.name)
            if on_stage:
                on_stage("manifest_published")
            proof = snapshot / (".recovery-proof-" + uuid.uuid4().hex + ".sqlite3")
            try:
                self._reconstruct(manifest, proof)
            finally:
                # Only this invocation's exact disposable validation file.
                if proof.exists():
                    private_file(proof)
                    proof.unlink()
            removed = 0
            if retire and fingerprint is not None:
                # Also cover a Linux retry after rename succeeded but fsync
                # failed: existing markers/chunks must not bypass durability.
                for directory in (self.root / "key-anchor", self.root, self.root / "chunks",
                                  self.snapshots, snapshot):
                    durability_barrier(directory)
                if private_file(source) != fingerprint:
                    raise VaultIntegrityError("Preimage changed before retirement")
                # Existing marker retries also prove the ORIGINAL matches the
                # manifest: a valid old manifest must not retire new user bytes.
                with source.open("rb") as handle:
                    original_sha = hashlib.file_digest(handle, "sha256").hexdigest()
                if original_sha != manifest["sha256"] or private_file(source) != fingerprint:
                    raise VaultIntegrityError("Original preimage differs from recovery proof")
                source.unlink()
                removed = fingerprint[2]
                if on_stage:
                    on_stage("original_retired")
            return {"snapshot": snapshot.name, "original_bytes": manifest["size"],
                    "new_chunk_bytes": new_bytes, "retired_bytes": removed,
                    "recovery_verified": True}

    def restore(self, snapshot, destination):
        """Restore only the packed DB to a NEW private directory, not a live DB."""
        snapshot = unlinked(snapshot)
        verify_private(snapshot)
        manifest = self._read_manifest(snapshot / MARKER, snapshot.name)
        return self._restore_manifest(manifest, destination)

    def restore_id(self, snapshot_id, destination):
        if not isinstance(snapshot_id, str) or not _SNAPSHOT.fullmatch(snapshot_id):
            raise VaultIntegrityError("Invalid restart snapshot identity")
        manifest = self._read_manifest(self.snapshots / (snapshot_id + ".enc"), snapshot_id)
        return self._restore_manifest(manifest, destination)

    def _restore_manifest(self, manifest, destination):
        destination = unlinked(destination)
        if destination.exists():
            raise FileExistsError("Recovery destination must be new")
        staging = destination.parent / ("." + destination.name + ".incomplete-" + uuid.uuid4().hex)
        create_private_directory(staging)
        create_private_directory(staging / "source-evidence")
        self._reconstruct(manifest, staging / RELATIVE)
        durable_publish(staging, destination)
        return {"snapshot": manifest["snapshot"], "bytes": manifest["size"], "recovery_verified": True}
