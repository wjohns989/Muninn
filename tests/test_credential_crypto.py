"""Portable credential-vault envelope tests; synthetic values only."""

import json
from dataclasses import replace

import pytest

from muninn.history.credential_crypto import (
    EncryptedValue,
    VaultHeader,
    VaultIntegrityError,
    decrypt_record,
    derive_key,
    encrypt_record,
)


@pytest.fixture(scope="module")
def envelope():
    header = VaultHeader.new()
    key = derive_key("local-test-passphrase", header)
    metadata = {"service": "example", "project": "sample", "kind": "api_key"}
    encrypted = encrypt_record(key, header, "record-1", metadata, "synthetic-value-only")
    return header, key, metadata, encrypted


def test_portable_header_and_ciphertext_round_trip(envelope):
    header, _, metadata, encrypted = envelope
    restored_header = VaultHeader.from_json(header.to_json())
    restored_record = EncryptedValue.from_json(encrypted.to_json())
    restored_key = derive_key("local-test-passphrase", restored_header)
    assert decrypt_record(restored_key, restored_header, "record-1", metadata, restored_record) == (
        "synthetic-value-only"
    )
    assert "synthetic-value-only" not in header.to_json() + encrypted.to_json()
    assert "local-test-passphrase" not in header.to_json() + encrypted.to_json()


def test_wrong_passphrase_and_tampering_fail_without_leaking(envelope):
    header, key, metadata, encrypted = envelope
    wrong_key = derive_key("different-test-passphrase", header)
    cases = [
        (wrong_key, header, "record-1", metadata, encrypted),
        (key, header, "record-2", metadata, encrypted),
        (key, header, "record-1", {**metadata, "project": "other"}, encrypted),
        (key, replace(header, vault_id="0" * 32), "record-1", metadata, encrypted),
        (key, header, "record-1", metadata,
         replace(encrypted, ciphertext="AAAA" + encrypted.ciphertext[4:])),
    ]
    for args in cases:
        with pytest.raises(VaultIntegrityError) as error:
            decrypt_record(*args)
        assert "synthetic-value-only" not in str(error.value)


def test_header_rejects_unsupported_or_weakened_parameters(envelope):
    header = envelope[0]
    for changes in ({"version": 2}, {"iterations": 1}, {"memory_cost_kib": 1024},
                    {"cipher": "none"}, {"salt": "00"}):
        payload = json.loads(header.to_json())
        payload.update(changes)
        with pytest.raises(VaultIntegrityError):
            VaultHeader.from_json(json.dumps(payload))
    with pytest.raises(VaultIntegrityError):
        derive_key("", header)


@pytest.mark.parametrize("length", [16, 24])
def test_non_256_bit_keys_are_rejected(envelope, length):
    header, _, metadata, encrypted = envelope
    with pytest.raises(VaultIntegrityError):
        encrypt_record(b"x" * length, header, "record-1", metadata, "synthetic-value-only")
    with pytest.raises(VaultIntegrityError):
        decrypt_record(b"x" * length, header, "record-1", metadata, encrypted)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 1.5])
def test_non_portable_metadata_is_rejected(envelope, bad_value):
    header, key, metadata, encrypted = envelope
    bad_metadata = {**metadata, "score": bad_value}
    with pytest.raises(VaultIntegrityError):
        encrypt_record(key, header, "record-1", bad_metadata, "synthetic-value-only")
    with pytest.raises(VaultIntegrityError):
        decrypt_record(key, header, "record-1", bad_metadata, encrypted)
