"""The anonymous dashboard must not hand local clients the API bearer."""

from fastapi.testclient import TestClient

import server
from muninn.core import security


def test_anonymous_dashboard_never_contains_active_bearer(monkeypatch):
    sentinel = "dashboard-test-bearer-not-for-anonymous-readers"
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", sentinel)
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)

    response = TestClient(server.app).get("/")

    assert response.status_code == 200
    assert "Muninn Hub" in response.text
    assert sentinel not in response.text
    assert "{{MUNINN_TOKEN}}" not in response.text


def test_dashboard_token_check_rejects_wrong_bearer_without_disclosing_right_one(monkeypatch):
    sentinel = "dashboard-test-valid-bearer"
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "core_verify_token", lambda token: token == sentinel)
    client = TestClient(server.app)

    missing = client.get("/auth/check")
    wrong = client.get("/auth/check", headers={"Authorization": "Bearer wrong"})
    valid = client.get("/auth/check", headers={"Authorization": f"Bearer {sentinel}"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert valid.status_code == 200
    assert valid.json() == {"authenticated": True}
    assert sentinel not in valid.text


def test_development_no_auth_mode_still_has_usable_dashboard(monkeypatch):
    monkeypatch.setattr(server, "is_security_enabled", lambda: False)
    client = TestClient(server.app)

    page = client.get("/")
    check = client.get("/auth/check")

    assert page.status_code == 200
    assert "let SECURITY_ENABLED = false;" in page.text
    assert check.status_code == 200


def test_generated_fallback_token_is_required_when_security_is_enabled(monkeypatch):
    for name in ("MUNINN_API_KEY", "MUNINN_AUTH_TOKEN", "MUNINN_SERVER_AUTH_TOKEN",
                 "MUNINN_NO_AUTH", "MUNINN_DEV_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", "generated-test-bearer")
    client = TestClient(server.app)

    assert client.get("/auth/check").status_code == 401
    assert client.get("/auth/check", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/auth/check", headers={"Authorization": "Bearer generated-test-bearer"}).status_code == 200


def test_generated_fallback_token_is_never_written_to_logs(monkeypatch, caplog):
    for name in ("MUNINN_API_KEY", "MUNINN_AUTH_TOKEN", "MUNINN_SERVER_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", None)
    with caplog.at_level("WARNING", logger="Muninn.security"):
        generated = security.initialize_security()

    assert generated not in caplog.text


def test_mimir_api_auth_does_not_bypass_generated_bearer(monkeypatch):
    for name in ("MUNINN_API_KEY", "MUNINN_AUTH_TOKEN", "MUNINN_SERVER_AUTH_TOKEN",
                 "MUNINN_NO_AUTH", "MUNINN_DEV_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", "generated-test-bearer")

    assert security.verify_api_token(None) is False
    assert security.verify_api_token("wrong") is False
    assert security.verify_api_token("generated-test-bearer") is True
