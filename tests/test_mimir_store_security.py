import base64
import hashlib

import pytest

from muninn.mimir.store import _derive_key_from_env


def test_derive_key_from_env_uses_explicit_passphrase(monkeypatch):
    monkeypatch.setenv("MUNINN_MIMIR_ENCRYPTION_KEY", "my-test-passphrase")
    monkeypatch.delenv("MUNINN_DEV_MODE", raising=False)

    key = _derive_key_from_env()

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(b"my-test-passphrase").digest()
    )
    assert key == expected


def test_derive_key_from_env_uses_deterministic_dev_key(monkeypatch):
    monkeypatch.delenv("MUNINN_MIMIR_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("MUNINN_DEV_MODE", "true")

    key = _derive_key_from_env()

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(b"muninn-mimir-dev-mode-only").digest()
    )
    assert key == expected


def test_derive_key_from_env_fails_closed_without_key_in_prod(monkeypatch):
    monkeypatch.delenv("MUNINN_MIMIR_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("MUNINN_DEV_MODE", raising=False)

    with pytest.raises(RuntimeError, match="MUNINN_MIMIR_ENCRYPTION_KEY must be set"):
        _derive_key_from_env()
