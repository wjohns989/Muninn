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

from muninn.history.archive_lineage import (
    canonical_source, capture_anchor, record_alias, source_signatures, validate_lineage,
)
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


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish a sibling directory atomically without replacing another owner."""
    if os.name == "nt":
        # Windows os.rename fails even when the existing target is empty.
        os.rename(source, destination)
        return
    if sys.platform != "linux":
        raise VaultIntegrityError("Atomic no-replace archive publication is unavailable")
    import ctypes
    import errno

    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise VaultIntegrityError("Atomic no-replace archive publication is unavailable") from exc
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # AT_FDCWD, RENAME_NOREPLACE. Unsupported kernels/filesystems fail closed.
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError("History archive destination already exists")
        raise VaultIntegrityError("Atomic no-replace archive publication failed")


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

    def _load_manifest(self, *, allow_empty: bool = False,
                       generation: int | None = None) -> dict[str, Any]:
        verify_private(self.root)
        pinned = generation is not None
        candidates = []
        if pinned:
            if type(generation) is not int or generation < 1:
                raise VaultIntegrityError("Invalid pinned history archive generation")
            path = self.root / f"manifest-{generation:012d}.enc"
            if not path.is_file():
                raise VaultIntegrityError("Pinned history archive generation is unavailable")
            candidates.append((generation, path))
        else:
            for path in self.root.glob("manifest-*.enc"):
                try:
                    candidate_generation = int(path.stem.split("-")[1])
                    candidates.append((candidate_generation, path))
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
            if (not isinstance(manifest, dict)
                    or manifest.get("vault_id") != self.vault_id
                    or manifest.get("generation") != generation
                    or not isinstance(manifest.get("files"), dict)):
                raise ValueError
            validate_lineage(manifest)
        except VaultIntegrityError:
            raise
        except (InvalidTag, ValueError, TypeError) as exc:
            raise VaultIntegrityError("History archive manifest authentication failed") from exc
        if not pinned:
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
                     expected_source: Path | None = None,
                     include_snapshot_receipt: bool = False) -> dict[str, Any]:
        with self._write_lock():
            source_key = str(Path(source).resolve(strict=True))
            manifest = self._load_manifest()
            aliases_before = dict(manifest.get("aliases", {}))
            result = self._archive_one(manifest, Path(source_key), provider, kind, expected_source=expected_source)
            if result["status"] == "captured" or aliases_before != manifest.get("aliases", {}):
                manifest["generation"] += 1
                self._save_manifest(manifest)
            if include_snapshot_receipt:
                version = result["versions"] - 1
                anchor = canonical_source(manifest, source_key)
                result["snapshot_receipt"] = self._snapshot_receipt(manifest["files"][anchor][version], version)
            return result

    def _snapshot_receipt(self, entry: dict, version: int) -> dict[str, Any]:
        """Internal path-free commit identity; unchanged entries keep their epoch."""
        return {"vault_id": self.vault_id, "blob": entry["blob"], "sha256": entry["sha256"],
                "version": version, "commit_generation": entry.get("commit_generation"),
                "provider": entry["provider"], "kind": entry["kind"]}

    def iter_committed_receipts(self, *, after_generation: int):
        """Read one authenticated manifest snapshot, never resolve latest by path.

        Legacy entries without a commit epoch are not post-enable work. The
        snapshot is stable for this traversal; subsequent commits are found on
        the next pass without retaining the archive writer lock.
        """
        manifest = self._load_manifest()
        for entries in manifest["files"].values():
            for version, entry in enumerate(entries):
                generation = entry.get("commit_generation")
                if type(generation) is int and after_generation < generation <= manifest["generation"]:
                    yield self._snapshot_receipt(entry, version)

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
                    aliases_before = dict(manifest.get("aliases", {}))
                    result = self._archive_one(manifest, source, provider, kind, expected_source=Path(source))
                    report[result["status"]] += 1
                    if result["status"] == "captured" or aliases_before != manifest.get("aliases", {}):
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
        source_key = str(source)
        anchor = capture_anchor(manifest, source_key, provider, kind)
        relocated = source_key != anchor
        continuity_required = relocated or any(
            alias["anchor"] == anchor for alias in manifest.get("aliases", {}).values())
        prior = manifest["files"].get(anchor, [])
        before = source.stat()
        if continuity_required and before.st_size < prior[-1]["size"]:
            raise ValueError("History relocation does not preserve the latest snapshot")

        def assert_open_identity(handle: BinaryIO) -> None:
            opened = os.fstat(handle.fileno())
            if (source.resolve(strict=True) != source
                    or (expected_source is not None and source != Path(expected_source))
                    or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError("History source identity changed after authorization")

        def assert_path_identity(current: os.stat_result) -> None:
            if (source.resolve(strict=True) != source
                    or (expected_source is not None and source != Path(expected_source))
                    or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError("History source identity changed after authorization")

        if prior and prior[-1]["size"] == before.st_size:
            unchanged_digest = hashlib.sha256()
            with source.open("rb") as current:
                assert_open_identity(current)
                for block in iter(lambda: current.read(_CHUNK), b""):
                    unchanged_digest.update(block)
                assert_open_identity(current)
            after_check = source.stat()
            assert_path_identity(after_check)
            if (prior[-1]["provider"] == provider and prior[-1]["kind"] == kind
                    and (before.st_size, before.st_mtime_ns) == (after_check.st_size, after_check.st_mtime_ns)
                    and unchanged_digest.hexdigest() == prior[-1]["sha256"]):
                record_alias(manifest, source_key, anchor, prior[-1], before.st_mtime_ns)
                return {"status": "unchanged", "versions": len(prior)}
            if continuity_required:
                raise ValueError("History relocation does not preserve the latest snapshot")
        blob_id = uuid.uuid4().hex
        nonce_prefix = os.urandom(8)
        blob = self._blobs / f"{blob_id}.enc"
        staging = self._blobs / f"{blob_id}.tmp"
        create_private_file(staging)
        digest = hashlib.sha256()
        # This second digest shares the encrypted capture pass. It proves only
        # byte preservation at the same origin, not prior analysis coverage.
        previous = prior[-1] if prior else None
        prefix_size = (previous["size"] if previous is not None
                       and previous["provider"] == provider and previous["kind"] == kind
                       and type(previous["size"]) is int and 0 <= previous["size"] < before.st_size
                       else None)
        prefix_digest = hashlib.sha256()
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
                if prefix_size is not None and size < prefix_size:
                    prefix_digest.update(chunk[:min(len(chunk), prefix_size - size)])
                size += len(chunk)
                self._write_chunk(dst, nonce_prefix, blob_id, count, chunk, False)
                count += 1
                if count >= 0xFFFFFFFF:
                    raise VaultIntegrityError("History source exceeds archive chunk limit")
            self._write_chunk(dst, nonce_prefix, blob_id, count,
                              _json_bytes({"size": size, "sha256": digest.hexdigest(), "chunks": count}), True)
            dst.flush()
            os.fsync(dst.fileno())
            assert_open_identity(src)
        after = source.stat()
        assert_path_identity(after)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
            raise RuntimeError("History source changed during encrypted capture; retry later")
        if continuity_required and (prefix_size is None or prefix_digest.hexdigest() != previous["sha256"]):
            # Preserve encrypted staging for recovery, but acknowledge no snapshot.
            raise ValueError("History relocation does not preserve the latest snapshot")
        os.replace(staging, blob)
        verify_private(blob)
        entry = {"blob": blob_id, "provider": provider, "kind": kind,
                 "size": size, "mtime_ns": before.st_mtime_ns,
                 "sha256": digest.hexdigest(), "chunks": count, "captured_at": time.time(),
                 "commit_generation": manifest["generation"] + 1}
        if prefix_size is not None and prefix_digest.hexdigest() == previous["sha256"]:
            entry["prefix_of"] = {"blob": previous["blob"], "sha256": previous["sha256"],
                                  "size": prefix_size, "version": len(prior) - 1}
        if relocated:
            entry["observed_source"] = source_key
        manifest["files"][anchor] = [*prior, entry]
        record_alias(manifest, source_key, anchor, entry, before.st_mtime_ns)
        return {"status": "captured", "versions": len(manifest["files"][anchor]), "size": size}

    def read_file(self, source: Path, version: int = -1) -> bytes:
        """Explicit local decrypt; never pass this result to ordinary indexes or models."""
        manifest = self._load_manifest()
        try:
            anchor = canonical_source(manifest, str(Path(source).resolve()))
            entry = manifest["files"][anchor][version]
        except (KeyError, IndexError, TypeError) as exc:
            raise FileNotFoundError("History snapshot not found") from exc
        result = self._verify_entry(entry, collect=True)
        assert result is not None
        return result

    def _verify_entry(self, entry: dict[str, Any], *, collect: bool,
                      on_chunk: Callable[[bytes], None] | None = None) -> bytes | None:
        """Authenticate the complete entry before returning to a caller.

        The private iterator yields per-frame plaintext before final trailer
        verification. This wrapper always exhausts it; a future projection
        writer must do the same before publishing its encrypted staging data.
        """
        output = bytearray() if collect else None
        for chunk in self._iter_verified_entry(entry):
            if output is not None:
                output.extend(chunk)
            if on_chunk is not None:
                try:
                    on_chunk(chunk)
                except (InvalidTag, ValueError, zlib.error) as exc:
                    # Preserve the old callback contract: a derived reader's
                    # parse/integrity failure cannot appear as a successful
                    # archive verification with a harmless caller error.
                    raise VaultIntegrityError("History blob authentication failed") from exc
        return bytes(output) if output is not None else None

    @staticmethod
    def _prefix_certificate(entry: dict[str, Any]) -> dict[str, Any] | None:
        """Validate a certificate's shape; absent legacy evidence stays absent."""
        if "prefix_of" not in entry:
            return None
        certificate = entry["prefix_of"]
        if (not isinstance(certificate, dict)
                or set(certificate) != {"blob", "sha256", "size", "version"}
                or type(certificate["size"]) is not int
                or type(certificate["version"]) is not int
                or certificate["version"] < 0
                or type(entry.get("size")) is not int
                or not 0 <= certificate["size"] < entry["size"]
                or not isinstance(certificate["blob"], str)
                or len(certificate["blob"]) != 32
                or any(c not in "0123456789abcdef" for c in certificate["blob"])
                or not isinstance(certificate["sha256"], str)
                or len(certificate["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in certificate["sha256"])):
            raise VaultIntegrityError("History prefix certificate is invalid")
        return certificate

    @classmethod
    def _prefix_parent(cls, entries: list[dict[str, Any]], version: int) -> dict[str, Any] | None:
        """Resolve an authenticated immediate same-origin parent, never text dedup.

        Callers supply one source's authenticated manifest version list. This
        relationship is not a substitute for full source-byte authentication.
        """
        if type(version) is not int or not 0 <= version < len(entries):
            raise VaultIntegrityError("History prefix version is invalid")
        entry = entries[version]
        certificate = cls._prefix_certificate(entry)
        if certificate is None:
            return None
        if version == 0 or certificate["version"] != version - 1:
            raise VaultIntegrityError("History prefix parent version is invalid")
        parent = entries[version - 1]
        if (any(certificate[key] != parent.get(key) for key in ("blob", "sha256", "size"))
                or type(parent.get("size")) is not int
                or any(entry.get(key) != parent.get(key) for key in ("provider", "kind"))):
            raise VaultIntegrityError("History prefix parent origin is invalid")
        return parent

    def _iter_verified_entry(self, entry: dict[str, Any]) -> Iterator[bytes]:
        """Yield bounded decrypted chunks; normal exhaustion is the integrity gate.

        No caller may publish derived data after breaking, closing, or failing
        this iterator early. The final manifest digest, size, and trailer are
        checked only after its final yielded chunk has been consumed.
        """
        blob_id = entry["blob"]
        if not isinstance(blob_id, str) or len(blob_id) != 32 or any(c not in "0123456789abcdef" for c in blob_id):
            raise VaultIntegrityError("History archive blob identity is invalid")
        blob = self._blobs / f"{blob_id}.enc"
        verify_private(blob)
        digest = hashlib.sha256()
        certificate = self._prefix_certificate(entry)
        prefix_digest = hashlib.sha256()
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
                        inflater = zlib.decompressobj()
                        chunk = inflater.decompress(payload, _CHUNK + 1)
                        if (len(chunk) > _CHUNK or not inflater.eof
                                or inflater.unused_data or inflater.unconsumed_tail
                                or inflater.flush(1)):
                            raise ValueError
                        if certificate is not None and size < certificate["size"]:
                            prefix_digest.update(chunk[:min(len(chunk), certificate["size"] - size)])
                        size += len(chunk)
                        digest.update(chunk)
                        yield chunk
                except (InvalidTag, ValueError, zlib.error) as exc:
                    raise VaultIntegrityError("History blob authentication failed") from exc
            if handle.read(1):
                raise VaultIntegrityError("History blob has trailing data")
        if (size != entry["size"] or digest.hexdigest() != entry["sha256"]
                or trailer != {"size": entry["size"], "sha256": entry["sha256"],
                               "chunks": entry["chunks"]}):
            raise VaultIntegrityError("History blob does not match its authenticated manifest")
        if certificate is not None and prefix_digest.hexdigest() != certificate["sha256"]:
            raise VaultIntegrityError("History prefix bytes do not match their authenticated certificate")

    def verify_all(self) -> dict[str, int]:
        """Stream every archived snapshot through authentication without retaining plaintext."""
        manifest = self._load_manifest()
        snapshots = 0
        total_bytes = 0
        for entries in manifest["files"].values():
            for version, entry in enumerate(entries):
                self._prefix_parent(entries, version)
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
        return source_signatures(manifest)

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
    def _copy_archive_files(cls, source_root: Path, destination: Path, *, copy_journal: bool = True,
                            copy_accounting: bool = True) -> None:
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
        from muninn.history.portable_accounting import SNAPSHOT
        accounting = source_root / SNAPSHOT
        if copy_accounting and (accounting.exists() or _is_link(accounting)):
            verify_private(accounting)
            copy_sealed(accounting, destination / SNAPSHOT)
        journal = source_root / "capture-jobs.db"
        if copy_journal and (journal.exists() or _is_link(journal)):
            import sqlite3

            verify_private(journal)
            target = destination / journal.name
            create_private_file(target)
            source_db = sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)
            target_db = sqlite3.connect(target)
            try:
                source_db.backup(target_db)
            finally:
                source_db.close()
                target_db.close()
        batch_marker = source_root / "historical-batches-managed"
        batch_db = source_root / "historical-batches.db"
        has_batch_marker = batch_marker.exists() or _is_link(batch_marker)
        has_batch_db = batch_db.exists() or _is_link(batch_db)
        if has_batch_marker != has_batch_db:
            raise VaultIntegrityError("History batch recovery pair is incomplete")
        if has_batch_marker:
            import sqlite3

            verify_private(batch_marker)
            verify_private(batch_db)
            copy_sealed(batch_marker, destination / batch_marker.name)
            target = destination / batch_db.name
            create_private_file(target)
            source_db = sqlite3.connect(batch_db.resolve().as_uri() + "?mode=ro", uri=True)
            target_db = sqlite3.connect(target)
            try:
                source_db.backup(target_db)
            finally:
                source_db.close()
                target_db.close()
        for evidence_name, db_name in (("source-evidence", "projections.sqlite3"),
                                      ("credential-context", "projections.sqlite3"),
                                      ("cited-windows", "projections.sqlite3"),
                                      ("memory-ledger", "ledger.sqlite3")):
            evidence = source_root / evidence_name
            if not (evidence.exists() or _is_link(evidence)):
                continue
            # Snapshot SQLite transactionally: copying a live DB file alone
            # could split a committed evidence seal from its encrypted pages.
            import sqlite3

            verify_private(evidence)
            original = evidence / db_name
            verify_private(original)
            create_private_directory(destination / evidence_name)
            target = destination / evidence_name / db_name
            create_private_file(target)
            source_db = sqlite3.connect(original.resolve().as_uri() + "?mode=ro", uri=True)
            target_db = sqlite3.connect(target)
            try:
                source_db.backup(target_db)
            finally:
                source_db.close()
                target_db.close()

    @staticmethod
    def _staging_destination(destination: Path) -> Path:
        destination = Path(destination)
        if (destination.exists() or _is_link(destination) or not destination.parent.is_dir()
                or _is_link(destination.parent)):
            raise ValueError("Invalid history archive restore locations")
        return destination.parent / f".{destination.name}.incomplete-{uuid.uuid4().hex}"

    @staticmethod
    def _publish_staging(staging: Path, destination: Path) -> None:
        # Failed stages are deliberately retained, private and visibly incomplete.
        if destination.exists() or _is_link(destination):
            raise ValueError("History archive destination already exists")
        verify_private(staging)
        _rename_noreplace(staging, destination)

    @classmethod
    def restore_from_backup(cls, backup_root: Path, destination: Path,
                            passphrase: str, *, on_staging=None) -> "SecureHistoryArchive":
        """Restore a legacy archive or self-contained runtime to a new root.

        Runtime bundles return the archive inside destination/history_secure_archive.
        Their accounting is reconstructed from authenticated ciphertext with all
        provider policies disabled. Never write to the destination's parent.
        """
        from muninn.history.portable_accounting import (
            MAGIC,
            MARKER,
            SNAPSHOT,
            _marker,
            require_snapshot_support,
            restore_into,
        )
        backup_root = Path(backup_root)
        runtime = ((backup_root / MARKER).exists() or _is_link(backup_root / MARKER)
                   or (backup_root / "history_secure_archive").exists())
        source = backup_root
        if runtime:
            require_snapshot_support()
            _marker(backup_root / MARKER, MAGIC)
            source = backup_root / "history_secure_archive"
            verify_private(source / SNAPSHOT)
            if (backup_root / "header.json").exists():
                raise VaultIntegrityError("History backup format is ambiguous")
        elif (source / SNAPSHOT).exists():
            raise VaultIntegrityError("Accounting recovery requires the full runtime bundle")
        destination = Path(destination)
        staging = cls._staging_destination(destination)
        if on_staging is not None:
            on_staging(staging)
        if runtime:
            create_private_directory(staging)
        archive_staging = staging / "history_secure_archive" if runtime else staging
        cls._copy_archive_files(source, archive_staging)
        restored = cls(archive_staging, passphrase)
        if runtime:
            restore_into(restored, staging)
        restored.verify_all()
        if (archive_staging / "historical-batches-managed").exists():
            from muninn.history.historical_batch import BatchOutbox

            BatchOutbox(restored).verify_all()
        if (archive_staging / "capture-jobs.db").exists():
            from muninn.history.capture_journal import CaptureJournal

            journal = CaptureJournal(restored, recover=False)
            journal.verify_all()
            journal.verify_publications()
            journal.verify_classifications()
        if (archive_staging / "source-evidence").exists():
            from muninn.history.source_evidence import SourceEvidenceStore

            SourceEvidenceStore(restored).verify_all()
        if (archive_staging / "credential-context").exists():
            from muninn.history.credential_context import CredentialContextStore

            CredentialContextStore(restored).verify_all()
        if (archive_staging / "memory-ledger").exists():
            from muninn.history.memory_ledger import MemoryLedger

            MemoryLedger(restored).verify_all()
        if (archive_staging / "cited-windows").exists():
            from muninn.history.cited_windows import CitedWindowPlanStore

            CitedWindowPlanStore(restored).verify_all()
        # Rebase the already authenticated object before the final publication.
        # No second unlock or fallible filesystem operation follows success.
        restored.root = destination / "history_secure_archive" if runtime else destination
        restored._header_path = restored.root / "header.json"
        restored._lock_path = restored.root / "archive.lock"
        restored._blobs = restored.root / "blobs"
        cls._publish_staging(staging, destination)
        return restored

    def backup_to(self, destination: Path, *, policy_root=None, on_staging=None) -> dict[str, int]:
        """Copy ciphertext and cross-check references, then publish a private backup.

        Managed accounting produces a self-contained runtime bundle. Its copied
        remote/batch permissions are disabled and its billing snapshot encrypted
        under this archive's portable key. The separate credential vault and
        environment/model configuration are deliberately not included.
        Independent stores may have different snapshot times; cross-checks reject
        inconsistent references instead of claiming a globally atomic snapshot.
        """
        if os.name != "nt":
            raise VaultIntegrityError("Local unattended backup requires Windows user protection")
        destination = Path(destination)
        staging = self._staging_destination(destination)
        from muninn.history.portable_accounting import (
            has_accounting,
            require_snapshot_support,
            snapshot_into,
            verify_snapshot,
        )
        policy_root = Path(policy_root) if policy_root is not None else self.root.parent
        runtime = has_accounting(policy_root)
        if runtime:
            require_snapshot_support()
        if on_staging is not None:
            on_staging(staging)
        if runtime:
            create_private_directory(staging)
        archive_staging = staging / "history_secure_archive" if runtime else staging
        with self._write_lock():
            self._load_manifest()
            self._copy_archive_files(self.root, archive_staging, copy_journal=False, copy_accounting=False)
            from muninn.history.capture_journal import CaptureJournal

            CaptureJournal(self, recover=False, policy_root=policy_root).backup_to(archive_staging / "capture-jobs.db")
            backup = SecureHistoryArchive(archive_staging)
            if runtime:
                snapshot_into(backup, policy_root, staging)
                verify_snapshot(backup, staging)
            elif has_accounting(policy_root):
                raise VaultIntegrityError("History accounting changed during backup; retry is required")
            if backup.vault_id != self.vault_id:
                raise VaultIntegrityError("History backup identity mismatch")
            report = backup.verify_all()
            report["runtime_bundle"] = int(runtime)
            if (archive_staging / "historical-batches-managed").exists():
                from muninn.history.historical_batch import BatchOutbox

                report["historical_batches_verified"] = BatchOutbox(backup).verify_all()["batches"]
            journal = CaptureJournal(backup, recover=False)
            journal.verify_all()
            report["publication_receipts_verified"] = journal.verify_publications()
            report["classification_jobs_verified"] = journal.verify_classifications()
            if (archive_staging / "source-evidence").exists():
                from muninn.history.source_evidence import SourceEvidenceStore

                report["evidence_snapshots_verified"] = SourceEvidenceStore(backup).verify_all()["snapshots"]
            if (archive_staging / "credential-context").exists():
                from muninn.history.credential_context import CredentialContextStore

                report["credential_context_snapshots_verified"] = (
                    CredentialContextStore(backup).verify_all()["snapshots"])
            if (archive_staging / "memory-ledger").exists():
                from muninn.history.memory_ledger import MemoryLedger

                report["memory_candidates_verified"] = MemoryLedger(backup).verify_all()["candidates"]
            if (archive_staging / "cited-windows").exists():
                from muninn.history.cited_windows import CitedWindowPlanStore

                report["cited_windows_verified"] = CitedWindowPlanStore(backup).verify_all()["windows"]
            self._publish_staging(staging, destination)
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
