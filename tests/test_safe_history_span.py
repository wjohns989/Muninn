"""Credential metadata can be discussed without exposing credential values."""

from muninn.history.safe_span import sanitize_agent_span


def test_preserves_nonsecret_context_about_credentials() -> None:
    text = "The OpenRouter API key is stored in the user environment; use Ollama locally."
    assert sanitize_agent_span(text) == text


def test_suppresses_unresolved_credential_value() -> None:
    text = "Use the credential CANARY-SECRET-91919 to test the route."
    result = sanitize_agent_span(text)
    assert "CANARY-SECRET-91919" not in result


def test_redacts_assignment_value_but_keeps_adjacent_context() -> None:
    text = "Ollama is local. API_KEY=abcdefghijklmnopqrstuvwxyz1234567890. Keep the GPU idle."
    result = sanitize_agent_span(text)
    assert "abcdefghijklmnopqrstuvwxyz1234567890" not in result
    assert "Ollama is local" in result
    assert "Keep the GPU idle" in result


def test_suppresses_short_unrecognized_secret_assignment() -> None:
    assert "abc" not in sanitize_agent_span("API_KEY=abc")
    assert "abc" not in sanitize_agent_span("The password is abc")
