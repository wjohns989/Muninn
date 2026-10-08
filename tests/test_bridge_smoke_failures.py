"""Diagnostics cannot echo credentials, private context, or provider errors."""
from scripts.smoke_user_bridge_live import _failure_code


def test_known_static_failure_is_actionable():
    assert _failure_code(RuntimeError('context_call_failed')) == 'context_call_failed'


def test_untrusted_exception_text_is_not_released():
    assert _failure_code(RuntimeError('synthetic-private-context')) == 'RuntimeError'
