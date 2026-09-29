"""Owner-only, portable, encrypted transcript snapshots.

This archive does not index content or call a model. Legacy gzip history copies
remain separate and are never silently migrated or removed.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import os
import shutil
import struct
import sys
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.credential_crypto import VaultHeader, VaultIntegrityError, derive_key
from muninn.history.private_acl import _is_link, create_private_directory, create_private_file, verify_private

_MAGIC = b"MUNINNH1\0"
_CHUNK = 1024 * 1024
_MAX_FRAME = _CHUNK + 65536
_WRAP_AAD = b"muninn-history-archive-key-v1"
_thread_lock = threading.RLock()


@dataclass(frozen=True, slots=True)
class SafeHistoryMetadata:
    """The complete allowlist for ordinary history catalog results."""

    ref: str
    provider: str
    kind: str
    captured_day_utc: str
    size_bucket_kib: int
    versions: int

    def as_dict(self) -> dict[str, str | int]:
        return {"ref": self.ref, "provider": self.provider, "kind": self.kind,
                "captured_day_utc": self.captured_day_utc,
                "size_bucket_kib": self.size_bucket_kib, "versions": self.versions}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_private(path: Path, data: bytes) -> None:
    create_private_file(path)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _dpapi_wrap(value: bytes) -> bytes:
    if os.name != "nt":
        return b""
    try:
        import win32crypt

        return win32crypt.CryptProtectData(value, "Muninn history archive", None, None, None, 0)
    except (ImportError, OSError) as exc:
        raise VaultIntegrityError("Windows user key protection unavailable") from exc


def _dpapi_unwrap(value: bytes) -> bytes:
    try:
        import win32crypt

        return win32crypt.CryptUnprotectData(value, None, None, None, 0)[1]
    except (ImportError, OSError) as exc:
        raise VaultIntegrityError("History archive cannot unlock for this Windows user") from exc


class SecureHistoryArchive:
    """Immutable encrypted blobs plus versioned encrypted manifests.

    The newest complete manifest is authoritative. An unfinished ``.tmp`` file
    is ignored and preserved for forensic recovery. Existing versions are kept.
    """

    def __init__(self, root: Path, passphrase: str | None = None, *, _initializing: bool = False):
        self.root = Path(root)
        self._header_path = self.root / "header.json"
        self._lock_path = self.root / "archive.lock"
        self._blobs = self.root / "blobs"
        self._unlocked_with_passphrase = passphrase is not None
        for path in (self.root, self._header_path, self._lock_path, self._blobs):
            verify_private(path)
        try:
            header = json.loads(self._header_path.read_text("utf-8"))
            if (set(header) != {"format", "kdf_header", "portable_nonce", "portable_ciphertext", "dpapi"}
                    or header["format"] != 1):
                raise ValueError
            self.kdf_header = VaultHeader.from_json(_json_bytes(header["kdf_header"]).decode("utf-8"))
            self.vault_id = self.kdf_header.vault_id
            if passphrase is not None:
                wrapping_key = derive_key(passphrase, self.kdf_header)
                key = AESGCM(wrapping_key).decrypt(
                    bytes.fromhex(header["portable_nonce"]),
                    base64.b64decode(header["portable_ciphertext"], validate=True),
                    _WRAP_AAD + bytes.fromhex(self.vault_id),
                )
            elif os.name == "nt" and header["dpapi"]:
                key = _dpapi_unwrap(base64.b64decode(header["dpapi"], validate=True))
            else:
                raise VaultIntegrityError("History archive passphrase required")
            if len(key) != 32:
                raise ValueError
            self._key = key
            self._load_manifest(allow_empty=_initializing)
        except VaultIntegrityError:
            raise
        except (InvalidTag, KeyError, TypeError, ValueError) as exc:
            raise VaultIntegrityError("History archive unlock or integrity check failed") from exc

    @classmethod
    def create(cls, root: Path, passphrase: str) -> "SecureHistoryArchive":
        root = Path(root)
        if not root.parent.is_dir() or root.exists() or _is_link(root):
            raise FileExistsError("History archive needs a new directory under an existing parent")
        kdf_header = VaultHeader.new()
        wrapping_key = derive_key(passphrase, kdf_header)
        key = os.urandom(32)
        nonce = os.urandom(12)
        portable = AESGCM(wrapping_key).encrypt(
            nonce, key, _WRAP_AAD + bytes.fromhex(kdf_header.vault_id),
        )
        dpapi = _dpapi_wrap(key)
        header = {
            "format": 1,
            "kdf_header": json.loads(kdf_header.to_json()),
            "portable_nonce": nonce.hex(),
            "portable_ciphertext": base64.b64encode(portable).decode("ascii"),
            "dpapi": base64.b64encode(dpapi).decode("ascii") if dpapi else "",
        }
        create_private_directory(root)
        create_private_directory(root / "blobs")
        _write_private(root / "header.json", _json_bytes(header))
        _write_private(root / "archive.lock", b"\0")
        archive = cls(root, passphrase, _initializing=True)
        archive._save_manifest({"format": 1, "vault_id": archive.vault_id, "generation": 1, "files": {}})
        return archive

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        verify_private(self._lock_path)
        with _thread_lock, self._lock_path.open("r+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _manifest_aad(self, generation: int) -> bytes:
        return _json_bytes({"format": 1, "vault_id": self.vault_id, "generation": generation})

    def _load_manifest(self, *, allow_empty: bool = False) -> dict[str, Any]:
        verify_private(self.root)
        candidates = []
        for path in self.root.glob("manifest-*.enc"):
            try:
                generation = int(path.stem.split("-")[1])
                candidates.append((generation, path))
            except (IndexError, ValueError):
                raise VaultIntegrityError("Invalid history archive manifest name")
        if not candidates:
            if not allow_empty:
                raise VaultIntegrityError("History archive has no authenticated manifest")
            self._manifest = {"format": 1, "vault_id": self.vault_id, "generation": 0, "files": {}}
            return self._manifest
        generation, path = max(candidates)
        verify_private(path)
        raw = path.read_bytes()
        if len(raw) < 28:
            raise VaultIntegrityError("History archive manifest is truncated")
        try:
            manifest = json.loads(AESGCM(self._key).decrypt(raw[:12], raw[12:], self._manifest_aad(generation)))
            if (not isinstance(manifest, dict) or manifest.get("format") != 1
                    or manifest.get("vault_id") != self.vault_id
                    or manifest.get("generation") != generation
                    or not isinstance(manifest.get("files"), dict)):
                raise ValueError
        except (InvalidTag, ValueError, TypeError) as exc:
            raise VaultIntegrityError("History archive manifest authentication failed") from exc
        self._manifest = manifest
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        generation = manifest["generation"]
        nonce = os.urandom(12)
        sealed = nonce + AESGCM(self._key).encrypt(nonce, _json_bytes(manifest), self._manifest_aad(generation))
        target = self.root / f"manifest-{generation:012d}.enc"
        staging = self.root / f"manifest-{generation:012d}.{uuid.uuid4().hex}.tmp"
        _write_private(staging, sealed)
        if target.exists():
            raise VaultIntegrityError("History archive manifest generation already exists")
        os.replace(staging, target)
        verify_private(target)
        self._manifest = manifest

    def _chunk_aad(self, blob_id: str, index: int, final: bool) -> bytes:
        return _json_bytes({"format": 1, "vault_id": self.vault_id, "blob": blob_id,
                            "index": index, "final": final})

    def _write_chunk(self, handle: BinaryIO, nonce_prefix: bytes, blob_id: str,
                     index: int, raw: bytes, final: bool) -> None:
        nonce = nonce_prefix + struct.pack(">I", index)
        payload = raw if final else zlib.compress(raw, 6)
        sealed = AESGCM(self._key).encrypt(nonce, payload, self._chunk_aad(blob_id, index, final))
        handle.write(struct.pack(">I", len(sealed)))
        handle.write(sealed)

    def archive_file(self, source: Path, provider: str, kind: str = "transcript", *,
                     expected_source: Path | None = None) -> dict[str, Any]:
        with self._write_lock():
            manifest = self._load_manifest()
            result = self._archive_one(manifest, source, provider, kind, expected_source=expected_source)
            if result["status"] == "captured":
                manifest["generation"] += 1
                self._save_manifest(manifest)
            return result

    def archive_many(self, items: list[tuple[Path, str, str]], *, commit_every: int = 100) -> dict[str, Any]:
        """Batch encrypted captures with bounded manifest rewrite cost."""
        if not 1 <= commit_every <= 1000:
            raise ValueError("Invalid archive commit interval")
        report: dict[str, Any] = {"captured": 0, "unchanged": 0, "errors": [], "commits": 0}
        with self._write_lock():
            manifest = self._load_manifest()
            pending = 0
            for source, provider, kind in items:
                try:
                    result = self._archive_one(manifest, source, provider, kind, expected_source=Path(source))
                    report[result["status"]] += 1
                    if result["status"] == "captured":
                        pending += 1
                except (OSError, RuntimeError, ValueError) as exc:
                    # Paths are private data; report a bounded class and not the source.
                    report["errors"].append({"provider": provider, "error": type(exc).__name__})
                if pending >= commit_every:
                    manifest["generation"] += 1
                    self._save_manifest(manifest)
                    report["commits"] += 1
                    pending = 0
            if pending:
                manifest["generation"] += 1
                self._save_manifest(manifest)
                report["commits"] += 1
        return report

    def _archive_one(self, manifest: dict[str, Any], source: Path,
                     provider: str, kind: str, *, expected_source: Path | None = None) -> dict[str, Any]:
        source = Path(source).resolve(strict=True)
        if expected_source is not None and source != Path(expected_source):
            raise ValueError("History source identity changed after authorization")
        if provider not in {"claude_code", "claude_desktop", "codex", "gemini_cli", "export", "cursor", "opencode"}:
            raise ValueError("Unsupported history provider")
        if kind not in {"transcript", "prompt_history", "desktop_session", "export", "state_db"}:
            raise ValueError("Unsupported history kind")
        prior = manifest["files"].get(str(source), [])
        before = source.stat()

        def assert_open_identity(handle: BinaryIO) -> None:
            if expected_source is None:
                return
            opened = os.fstat(handle.fileno())
            if (source.resolve(strict=True) != Path(expected_source)
                    or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError("History source identity changed after authorization")

        if prior and prior[-1]["size"] == before.st_size and prior[-1]["mtime_ns"] == before.st_mtime_ns:
            unchanged_digest = hashlib.sha256()
            with source.open("rb") as current:
                assert_open_identity(current)
                for block in iter(lambda: current.read(_CHUNK), b""):
                    unchanged_digest.update(block)
            after_check = source.stat()
            if (expected_source is not None and (source.resolve(strict=True) != Path(expected_source)
                    or (after_check.st_dev, after_check.st_ino) != (before.st_dev, before.st_ino))):
                raise ValueError("History source identity changed after authorization")
            if ((before.st_size, before.st_mtime_ns) == (after_check.st_size, after_check.st_mtime_ns)
                    and unchanged_digest.hexdigest() == prior[-1]["sha256"]):
                return {"status": "unchanged", "versions": len(prior)}
        blob_id = uuid.uuid4().hex
        nonce_prefix = os.urandom(8)
        blob = self._blobs / f"{blob_id}.enc"
        staging = self._blobs / f"{blob_id}.tmp"
        create_private_file(staging)
        digest = hashlib.sha256()
        count = 0
        size = 0
        with source.open("rb") as src, staging.open("wb") as dst:
            assert_open_identity(src)
            dst.write(_MAGIC + nonce_prefix)
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                self._write_chunk(dst, nonce_prefix, blob_id, count, chunk, False)
                count += 1
                if count >= 0xFFFFFFFF:
                    raise VaultIntegrityError("History source exceeds archive chunk limit")
            self._write_chunk(dst, nonce_prefix, blob_id, count,
                              _json_bytes({"size": size, "sha256": digest.hexdigest(), "chunks": count}), True)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (expected_source is not None and (source.resolve(strict=True) != Path(expected_source)
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino))):
            raise ValueError("History source identity changed after authorization")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
            raise RuntimeError("History source changed during encrypted capture; retry later")
        os.replace(staging, blob)
        verify_private(blob)
        entry = {"blob": blob_id, "provider": provider, "kind": kind,
                 "size": size, "mtime_ns": before.st_mtime_ns,
                 "sha256": digest.hexdigest(), "chunks": count, "captured_at": time.time()}
        manifest["files"][str(source)] = [*prior, entry]
        return {"status": "captured", "versions": len(manifest["files"][str(source)]), "size": size}

    def read_file(self, source: Path, version: int = -1) -> bytes:
        """Explicit local decrypt; never pass this result to ordinary indexes or models."""
        manifest = self._load_manifest()
        try:
            entry = manifest["files"][str(Path(source).resolve())][version]
        except (KeyError, IndexError, TypeError) as exc:
            raise FileNotFoundError("History snapshot not found") from exc
        result = self._verify_entry(entry, collect=True)
        assert result is not None
        return result

    def _verify_entry(self, entry: dict[str, Any], *, collect: bool,
                      on_chunk: Callable[[bytes], None] | None = None) -> bytes | None:
        blob_id = entry["blob"]
        if not isinstance(blob_id, str) or len(blob_id) != 32 or any(c not in "0123456789abcdef" for c in blob_id):
            raise VaultIntegrityError("History archive blob identity is invalid")
        blob = self._blobs / f"{blob_id}.enc"
        verify_private(blob)
        output = bytearray() if collect else None
        digest = hashlib.sha256()
        size = 0
        with blob.open("rb") as handle:
            prefix = handle.read(len(_MAGIC) + 8)
            if len(prefix) != len(_MAGIC) + 8 or not prefix.startswith(_MAGIC):
                raise VaultIntegrityError("History blob header is invalid")
            nonce_prefix = prefix[len(_MAGIC):]
            for index in range(entry["chunks"] + 1):
                length_raw = handle.read(4)
                if len(length_raw) != 4:
                    raise VaultIntegrityError("History blob is truncated")
                length = struct.unpack(">I", length_raw)[0]
                if not 16 <= length <= _MAX_FRAME:
                    raise VaultIntegrityError("History blob frame is invalid")
                sealed = handle.read(length)
                if len(sealed) != length:
                    raise VaultIntegrityError("History blob is truncated")
                final = index == entry["chunks"]
                try:
                    payload = AESGCM(self._key).decrypt(
                        nonce_prefix + struct.pack(">I", index), sealed,
                        self._chunk_aad(blob_id, index, final),
                    )
                    if final:
                        trailer = json.loads(payload)
                    else:
                        chunk = zlib.decompress(payload)
                        if len(chunk) > _CHUNK:
                            raise ValueError
                        size += len(chunk)
                        if output is not None:
                            output.extend(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)
                        digest.update(chunk)
                except (InvalidTag, ValueError, zlib.error) as exc:
                    raise VaultIntegrityError("History blob authentication failed") from exc
            if handle.read(1):
                raise VaultIntegrityError("History blob has trailing data")
        if (size != entry["size"] or digest.hexdigest() != entry["sha256"]
                or trailer != {"size": entry["size"], "sha256": entry["sha256"],
                               "chunks": entry["chunks"]}):
            raise VaultIntegrityError("History blob does not match its authenticated manifest")
        return bytes(output) if output is not None else None

    def verify_all(self) -> dict[str, int]:
        """Stream every archived snapshot through authentication without retaining plaintext."""
        manifest = self._load_manifest()
        snapshots = 0
        total_bytes = 0
        for entries in manifest["files"].values():
            for entry in entries:
                self._verify_entry(entry, collect=False)
                snapshots += 1
                total_bytes += entry["size"]
        return {"snapshots_verified": snapshots, "plaintext_bytes_authenticated": total_bytes,
                "generation": manifest["generation"]}

    def status(self) -> dict[str, int]:
        manifest = self._load_manifest()
        return {"sources": len(manifest["files"]),
                "snapshots": sum(len(items) for items in manifest["files"].values()),
                "generation": manifest["generation"]}

    def latest_source_signatures(self) -> dict[str, tuple[int, int]]:
        """Authenticated in-process size/mtime projection for cheap source discovery.

        Paths are private manifest material and must not be logged or returned
        through a status/API response. A signature match is not a content proof;
        the archive writer hashes again before declaring a capture unchanged.
        """
        manifest = self._load_manifest()
        return {path: (entries[-1]["size"], entries[-1]["mtime_ns"])
                for path, entries in manifest["files"].items() if entries}

    def rebind_windows_user(self) -> None:
        """After portable restore, attach a new CurrentUser DPAPI wrapper."""
        if os.name != "nt" or not self._unlocked_with_passphrase:
            raise VaultIntegrityError("Rebinding requires a local passphrase unlock on Windows")
        old = self._header_path.read_bytes()
        header = json.loads(old)
        header["dpapi"] = base64.b64encode(_dpapi_wrap(self._key)).decode("ascii")
        previous = self.root / f"header.previous-{uuid.uuid4().hex}.json"
        staging = self.root / f"header.{uuid.uuid4().hex}.tmp"
        with self._write_lock():
            _write_private(previous, old)
            _write_private(staging, _json_bytes(header))
            os.replace(staging, self._header_path)
            verify_private(self._header_path)

    @classmethod
    def _copy_archive_files(cls, source_root: Path, destination: Path, *, copy_journal: bool = True) -> None:
        """Copy ciphertext only into a new owner-only root; never mutate source."""
        source_root = Path(source_root)
        destination = Path(destination)
        if (_is_link(source_root) or not source_root.is_dir() or destination.exists()
                or _is_link(destination) or not destination.parent.is_dir()):
            raise ValueError("Invalid history archive restore locations")
        create_private_directory(destination)
        create_private_directory(destination / "blobs")

        def copy_sealed(source: Path, target: Path) -> None:
            if _is_link(source) or not source.is_file():
                raise VaultIntegrityError("History backup contains a missing or linked archive file")
            create_private_file(target)
            with source.open("rb") as read, target.open("wb") as write:
                shutil.copyfileobj(read, write, _CHUNK)
                write.flush()
                os.fsync(write.fileno())

        copy_sealed(source_root / "header.json", destination / "header.json")
        _write_private(destination / "archive.lock", b"\0")
        if _is_link(source_root / "blobs") or not (source_root / "blobs").is_dir():
            raise VaultIntegrityError("History backup blob directory is missing or linked")
        manifests = list(source_root.glob("manifest-*.enc"))
        if not manifests:
            raise VaultIntegrityError("History backup has no authenticated manifest")
        for source in manifests:
            copy_sealed(source, destination / source.name)
        for source in (source_root / "blobs").glob("*.enc"):
            copy_sealed(source, destination / "blobs" / source.name)
        journal = source_root / "capture-jobs.db"
        if copy_journal and (journal.exists() or _is_link(journal)):
            copy_sealed(journal, destination / journal.name)

    @classmethod
    def restore_from_backup(cls, backup_root: Path, destination: Path,
                            passphrase: str) -> "SecureHistoryArchive":
        """Portable restore to a new private root; verify all snapshots."""
        cls._copy_archive_files(backup_root, destination)
        restored = cls(destination, passphrase)
        restored.verify_all()
        if (destination / "capture-jobs.db").exists():
            from muninn.history.capture_journal import CaptureJournal

            CaptureJournal(restored, recover=False).verify_all()
        return restored

    def backup_to(self, destination: Path) -> dict[str, int]:
        """Take a consistent owner-only ciphertext backup under this Windows user."""
        if os.name != "nt":
            raise VaultIntegrityError("Local unattended backup requires Windows user protection")
        with self._write_lock():
            self._load_manifest()
            self._copy_archive_files(self.root, destination, copy_journal=False)
            from muninn.history.capture_journal import CaptureJournal

            CaptureJournal(self, recover=False).backup_to(destination / "capture-jobs.db")
            backup = SecureHistoryArchive(destination)
            if backup.vault_id != self.vault_id:
                raise VaultIntegrityError("History backup identity mismatch")
            report = backup.verify_all()
            CaptureJournal(backup, recover=False).verify_all()
            return report

    def metadata_catalog(self, *, provider: str | None = None, offset: int = 0,
                         limit: int = 100) -> list[SafeHistoryMetadata]:
        """Return typed metadata only; source paths and transcript text remain encrypted."""
        manifest = self._load_manifest()
        if provider is not None and provider not in {
            "claude_code", "claude_desktop", "codex", "gemini_cli", "export", "cursor", "opencode"
        }:
            raise ValueError("Unsupported history provider")
        selected = []
        for source_path, entries in manifest["files"].items():
            latest = entries[-1]
            if provider is not None and latest["provider"] != provider:
                continue
            ref = hmac.new(self._key, b"history-metadata-ref-v1\0" + source_path.encode("utf-8"),
                           hashlib.sha256).hexdigest()
            selected.append(SafeHistoryMetadata(
                ref=ref, provider=latest["provider"], kind=latest["kind"],
                captured_day_utc=time.strftime("%Y-%m-%d", time.gmtime(latest["captured_at"])),
                size_bucket_kib=(latest["size"] + 1023) // 1024,
                versions=len(entries),
            ))
        selected.sort(key=lambda item: (item.captured_day_utc, item.ref), reverse=True)
        return selected[max(0, offset): max(0, offset) + max(1, min(limit, 200))]


def main() -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Local encrypted Muninn history archive")
    parser.add_argument(
        "action",
        choices=("init", "status", "plan", "sync", "catalog", "verify", "backup", "restore", "rebind"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--backup-root", type=Path,
        help="New backup destination, or existing backup source for restore",
    )
    parser.add_argument("--home", type=Path, help="Home whose configured agent history is scanned for sync")
    parser.add_argument("--portable", action="store_true", help="Prompt for recovery passphrase for status")
    args = parser.parse_args()
    passphrase = None
    if args.action == "init":
        first = getpass.getpass("New history archive recovery passphrase (12+ chars): ")
        second = getpass.getpass("Confirm recovery passphrase: ")
        if first != second:
            raise SystemExit("Passphrases did not match")
        if len(first) < 12:
            raise SystemExit("Recovery passphrase must be at least 12 characters; archive not created")
        archive = SecureHistoryArchive.create(args.root, first)
    elif args.action == "restore":
        if args.backup_root is None:
            parser.error("restore requires --backup-root")
        passphrase = getpass.getpass("Recovery passphrase: ")
        archive = SecureHistoryArchive.restore_from_backup(
            args.backup_root, args.root, passphrase)
    else:
        portable = args.portable or args.action == "rebind"
        passphrase = getpass.getpass("Recovery passphrase: ") if portable else None
        archive = SecureHistoryArchive(args.root, passphrase)
    if args.action == "rebind":
        archive.rebind_windows_user()
    if args.action == "backup":
        if args.backup_root is None:
            parser.error("backup requires --backup-root as a new destination")
        report = archive.backup_to(args.backup_root)
    elif args.action in ("plan", "sync"):
        from muninn.history.service import HistoryService

        service = HistoryService(None, args.root.parent / "history_vault", home=args.home,
                                 secure_archive_root=args.root, archive_passphrase=passphrase)
        report = asyncio.run(service.secure_sync(dry_run=args.action == "plan"))
    elif args.action == "catalog":
        report = [item.as_dict() for item in archive.metadata_catalog()]
    elif args.action == "verify":
        report = archive.verify_all()
    else:
        report = archive.status()
    print(json.dumps(report, sort_keys=True))
    if args.action == "sync" and (report["errors"] or report["skipped_live_state_db"]):
        return 2  # Partial capture is never reported as a clean complete import.
    return 0


if __name__ == "__main__":
    sys.exit(main())
