"""Explicit fixture for legacy transport tests; real accounting has its own tests."""
import os

import pytest


@pytest.fixture(params=["portable_snapshot", "windows_unattended"])
def recovery_copy(request, tmp_path):
    """Keep portable recovery and actual Windows unattended backup distinct."""
    backend = request.param
    if backend == "windows_unattended" and os.name != "nt":
        pytest.skip("Actual unattended backup requires Windows user protection")

    def copy(archive, destination, passphrase, *, policy_root=None, on_staging=None):
        if backend == "windows_unattended":
            return archive.backup_to(destination, policy_root=policy_root, on_staging=on_staging)
        if on_staging is not None:
            raise ValueError("Portable recovery inputs do not prove unattended staging callbacks")
        from tests.recovery_fixture import recovery_input

        return recovery_input(archive, destination, passphrase,
                              temporary_root=tmp_path, policy_root=policy_root)

    copy.backend = backend
    return copy


@pytest.fixture
def fake_strict_remote_admission(monkeypatch):
    """Extend those tests' existing fake budget boundary; never probe user keys.

    Only modules explicitly requesting this fixture bypass admission persistence.
    test_remote_admission_transport uses actual temporary policy/accounting instead.
    """
    from muninn.history import secure_analysis

    class Permit:
        identifier = "a" * 32
        def mark_unknown(self):
            pass
        def release_unsent(self):
            pass
        def release_reserved(self):
            return True
        def settle_response(self, data):
            return True

    async def reserve(*args, **kwargs):
        return Permit()

    monkeypatch.setattr(secure_analysis, "_reserve_remote_admission", reserve)
