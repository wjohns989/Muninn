"""Portable encrypted credential records (no persistence or API exposure here).

This module is a building block, not an enabled credential index. Callers must
enforce vault ACLs, authenticated reveal, backup consistency, and no-egress rules
before storing real credentials.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id


class VaultIntegrityError(ValueError):
    """The envelope, unlock material, or record could not be authenticated."""


_VERSION = 1
_SCHEMA_VERSION = 1
_KDF = "argon2id"
_CIPHER = "aes-256-gcm"
_ITERATIONS = 3
_LANES = 1
_MEMORY_COST_KIB = 64 * 1024


@dataclass(frozen=True)
class VaultHeader:
    version: int
    schema_version: int
    vault_id: str
    kdf: str
    salt: str
    iterations: int
    lanes: int
    memory_cost_kib: int
    cipher: str

    @classmethod
    def new(cls) -> VaultHeader:
        return cls(
            version=_VERSION,
            schema_version=_SCHEMA_VERSION,
            vault_id=os.urandom(16).hex(),
            kdf=_KDF,
            salt=os.urandom(16).hex(),
            iterations=_ITERATIONS,
            lanes=_LANES,
            memory_cost_kib=_MEMORY_COST_KIB,
            cipher=_CIPHER,
        )

    @classmethod
    def from_json(cls, raw: str) -> VaultHeader:
        try:
            if len(raw) > 4096:
                raise ValueError
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
                raise ValueError
            header = cls(**data)
            header.validate()
            return header
        except (TypeError, ValueError) as exc:
            raise VaultIntegrityError("Unsupported or invalid credential vault header") from exc

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def validate(self) -> None:
        if (
            type(self.version) is not int or self.version != _VERSION
            or type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION
            or self.kdf != _KDF or self.cipher != _CIPHER
            or type(self.iterations) is not int or self.iterations != _ITERATIONS
            or type(self.lanes) is not int or self.lanes != _LANES
            or type(self.memory_cost_kib) is not int or self.memory_cost_kib != _MEMORY_COST_KIB
        ):
            raise VaultIntegrityError("Unsupported or invalid credential vault header")
        try:
            if len(bytes.fromhex(self.vault_id)) != 16 or len(bytes.fromhex(self.salt)) != 16:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise VaultIntegrityError("Unsupported or invalid credential vault header") from exc


@dataclass(frozen=True)
class EncryptedValue:
    version: int
    nonce: str
    ciphertext: str

    @classmethod
    def from_json(cls, raw: str) -> EncryptedValue:
        try:
            if len(raw) > 150_000:
                raise ValueError
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
                raise ValueError
            record = cls(**data)
            record._decode()
            return record
        except (TypeError, ValueError, binascii.Error) as exc:
            raise VaultIntegrityError("Invalid encrypted credential record") from exc

    def to_json(self) -> str:
        self._decode()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def _decode(self) -> tuple[bytes, bytes]:
        try:
            if type(self.version) is not int or self.version != _VERSION:
                raise ValueError
            nonce = bytes.fromhex(self.nonce)
            ciphertext = base64.b64decode(self.ciphertext, validate=True)
            if len(nonce) != 12 or not 16 <= len(ciphertext) <= 131_088:
                raise ValueError
            return nonce, ciphertext
        except (TypeError, ValueError, binascii.Error) as exc:
            raise VaultIntegrityError("Invalid encrypted credential record") from exc


def derive_key(passphrase: str, header: VaultHeader) -> bytes:
    """Derive a session-only key; never persist the passphrase or returned bytes."""
    header.validate()
    if not isinstance(passphrase, str) or len(passphrase) < 12:
        raise VaultIntegrityError("Credential vault passphrase is missing or too short")
    try:
        kdf = Argon2id(
            salt=bytes.fromhex(header.salt),
            length=32,
            iterations=header.iterations,
            lanes=header.lanes,
            memory_cost=header.memory_cost_kib,
        )
        return kdf.derive(unicodedata.normalize("NFC", passphrase).encode("utf-8"))
    except (UnsupportedAlgorithm, ValueError) as exc:
        raise VaultIntegrityError("Credential vault key derivation unavailable") from exc


def _aad(header: VaultHeader, record_id: str, metadata: dict[str, Any]) -> bytes:
    header.validate()
    if not isinstance(record_id, str) or not 1 <= len(record_id) <= 256 or not isinstance(metadata, dict):
        raise VaultIntegrityError("Invalid credential record identity")
    _validate_portable(metadata)
    try:
        encoded = json.dumps(
            {"format": _VERSION, "vault_id": header.vault_id,
             "record_id": record_id, "metadata": metadata},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > 16_384:
            raise ValueError
        return encoded
    except (TypeError, ValueError) as exc:
        raise VaultIntegrityError("Invalid credential record metadata") from exc


def _validate_portable(value: Any, depth: int = 0) -> None:
    if depth > 8:
        raise VaultIntegrityError("Invalid credential record metadata")
    if isinstance(value, dict):
        if len(value) > 256 or any(type(k) is not str or len(k) > 256 for k in value):
            raise VaultIntegrityError("Invalid credential record metadata")
        for item in value.values():
            _validate_portable(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > 256:
            raise VaultIntegrityError("Invalid credential record metadata")
        for item in value:
            _validate_portable(item, depth + 1)
    elif type(value) is str:
        if len(value) > 4096:
            raise VaultIntegrityError("Invalid credential record metadata")
    elif value is not None and type(value) not in (bool, int):
        raise VaultIntegrityError("Invalid credential record metadata")


def _require_key(key: bytes) -> None:
    if type(key) is not bytes or len(key) != 32:
        raise VaultIntegrityError("Invalid credential vault key")


def encrypt_record(key: bytes, header: VaultHeader, record_id: str,
                   metadata: dict[str, Any], value: str) -> EncryptedValue:
    _require_key(key)
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 131_072:
        raise VaultIntegrityError("Invalid credential value")
    nonce = os.urandom(12)
    try:
        ciphertext = AESGCM(key).encrypt(nonce, value.encode("utf-8"), _aad(header, record_id, metadata))
    except (TypeError, ValueError) as exc:
        raise VaultIntegrityError("Credential encryption failed") from exc
    return EncryptedValue(
        version=_VERSION, nonce=nonce.hex(), ciphertext=base64.b64encode(ciphertext).decode("ascii"),
    )


def decrypt_record(key: bytes, header: VaultHeader, record_id: str,
                   metadata: dict[str, Any], encrypted: EncryptedValue) -> str:
    _require_key(key)
    try:
        nonce, ciphertext = encrypted._decode()
        return AESGCM(key).decrypt(nonce, ciphertext, _aad(header, record_id, metadata)).decode("utf-8")
    except (InvalidTag, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise VaultIntegrityError("Credential record authentication failed") from exc
