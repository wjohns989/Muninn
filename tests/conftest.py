"""Explicit fixture for legacy transport tests; real accounting has its own tests."""
import pytest


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
        def settle_response(self, data):
            return True

    async def reserve(*args, **kwargs):
        return Permit()

    monkeypatch.setattr(secure_analysis, "_reserve_remote_admission", reserve)
