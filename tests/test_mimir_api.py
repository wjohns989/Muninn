"""
Tests for muninn/mimir/api.py — Mimir Interop Relay HTTP API.

Uses httpx.AsyncClient with ASGITransport to exercise the FastAPI router
without running a real server.  Module-level relay / store singletons are
injected directly into muninn.mimir.api before each test and reset to None
after to ensure test isolation.

Coverage targets:
  - Helper functions: _ok, _fail, _serialise_run_status, _build_probe_envelope
  - init_mimir singleton binding
  - _verify_token authentication middleware
  - 503 guards when singletons are not initialised
  - POST /mimir/relay          — success, FAILED result, relay crash, 422
  - GET  /mimir/providers      — success, score_all error
  - GET  /mimir/runs           — pagination, status filter, store error
  - GET  /mimir/runs/{run_id} — found, 404, store error
  - GET  /mimir/runs/{run_id}/audit — success, with events, store error
  - GET  /mimir/settings       — success, user_id forwarding, store error
  - POST /mimir/settings       — success, settings object validation, store error
  - GET  /mimir/connections    — success, with data, store error
  - POST /mimir/audit/purge    — success, default, boundary validation, store error
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi import FastAPI
from fastapi.exceptions import HTTPException

import muninn.mimir.api as api_module
from muninn.mimir.api import (
    _build_probe_envelope,
    _fail,
    _ok,
    _serialise_run_status,
    init_mimir,
    mimir_router,
)
from muninn.mimir.models import (
    InteropSettings,
    IRPMode,
    ProviderName,
    ReconciliationResult,
    RelayResult,
    RoutingScore,
    RunStatus,
)

# ---------------------------------------------------------------------------
# Shared FastAPI test app (one instance, reused across all tests)
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(mimir_router)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_relay_result(
    status: RunStatus = RunStatus.SUCCESS,
    output: str = "Advisory response from provider.",
    mode: IRPMode = IRPMode.ADVISORY,
) -> RelayResult:
    return RelayResult(
        run_id="TEST-RUN-001",
        irp_id="IRP-001",
        mode=mode,
        provider=ProviderName.CLAUDE_CODE,
        status=status,
        output=output,
    )


def _make_routing_score(
    provider: ProviderName = ProviderName.CLAUDE_CODE,
    availability_score: float = 1.0,
    composite: float = 0.80,
) -> RoutingScore:
    return RoutingScore(
        provider=provider,
        capability_score=0.90,
        availability_score=availability_score,
        cost_score=0.60,
        safety_score=0.92,
        history_score=0.50,
        composite=composite,
    )


def _make_mock_relay(result: RelayResult | None = None) -> MagicMock:
    relay = MagicMock()
    relay.run = AsyncMock(return_value=result or _make_relay_result())
    relay._router = MagicMock()
    relay._router.score_all = AsyncMock(
        return_value={
            ProviderName.CLAUDE_CODE: _make_routing_score(ProviderName.CLAUDE_CODE),
            ProviderName.CODEX_CLI: _make_routing_score(
                ProviderName.CODEX_CLI, composite=0.72
            ),
            ProviderName.GEMINI_CLI: _make_routing_score(
                ProviderName.GEMINI_CLI, composite=0.68
            ),
        }
    )
    return relay


def _make_mock_store() -> MagicMock:
    store = MagicMock()
    store.list_runs = MagicMock(return_value=[])
    store.get_run = MagicMock(return_value={"run_id": "TEST-RUN-001"})
    store.get_audit_events = MagicMock(return_value=[])
    store.get_settings = MagicMock(return_value=InteropSettings())
    store.update_settings = MagicMock(return_value=None)
    store.list_connections = MagicMock(return_value=[])
    store.purge_old_audit_events = MagicMock(return_value=5)
    return store


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app),
        base_url="http://test",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singletons():
    """Reset module-level singletons before and after every test."""
    api_module._relay = None
    api_module._store = None
    yield
    api_module._relay = None
    api_module._store = None


@pytest.fixture(autouse=True)
def _clear_api_key(monkeypatch):
    """Ensure tests run in dev mode by default (MUNINN_DEV_MODE=true)."""
    monkeypatch.setenv("MUNINN_DEV_MODE", "true")
    monkeypatch.delenv("MUNINN_API_KEY", raising=False)


@pytest.fixture
def relay():
    return _make_mock_relay()


@pytest.fixture
def store():
    return _make_mock_store()


@pytest.fixture
def with_singletons(relay, store):
    """Inject both singletons into the API module and return them."""
    api_module._relay = relay
    api_module._store = store
    return relay, store


# ---------------------------------------------------------------------------
# Tests: pure helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for module-level helper functions (no HTTP overhead)."""

    def test_ok_wraps_payload_in_success_envelope(self):
        result = _ok({"count": 42})
        assert result == {"success": True, "data": {"count": 42}}

    def test_ok_accepts_none_data(self):
        assert _ok(None) == {"success": True, "data": None}

    def test_ok_accepts_list_data(self):
        result = _ok([1, 2, 3])
        assert result == {"success": True, "data": [1, 2, 3]}

    def test_fail_returns_http_exception(self):
        exc = _fail("Something went wrong", status_code=400)
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 400
        assert exc.detail == {"success": False, "error": "Something went wrong"}

    def test_fail_default_status_code_is_500(self):
        exc = _fail("Internal error")
        assert exc.status_code == 500

    def test_serialise_run_status_none_returns_none(self):
        assert _serialise_run_status(None) is None

    def test_serialise_run_status_valid_values_pass_through(self):
        for valid in ("pending", "running", "success", "failed", "cancelled", "policy_blocked"):
            assert _serialise_run_status(valid) == valid

    def test_serialise_run_status_invalid_raises_http_422(self):
        with pytest.raises(HTTPException) as exc_info:
            _serialise_run_status("not_a_real_status")
        assert exc_info.value.status_code == 422

    def test_build_probe_envelope_is_valid_irp_envelope(self):
        from muninn.mimir.models import IRPEnvelope
        envelope = _build_probe_envelope()
        assert isinstance(envelope, IRPEnvelope)
        assert envelope.from_agent == "muninn"
        assert envelope.to == "auto"
        assert envelope.mode == IRPMode.ADVISORY
        assert envelope.id == "MIMIR_PROVIDER_PROBE"

    def test_init_mimir_binds_both_singletons(self):
        mock_relay = _make_mock_relay()
        mock_store = _make_mock_store()
        init_mimir(mock_relay, mock_store)
        assert api_module._relay is mock_relay
        assert api_module._store is mock_store

    def test_init_mimir_overwrites_existing_singletons(self):
        first_relay = _make_mock_relay()
        second_relay = _make_mock_relay()
        init_mimir(first_relay, _make_mock_store())
        init_mimir(second_relay, _make_mock_store())
        assert api_module._relay is second_relay


# ---------------------------------------------------------------------------
# Tests: authentication middleware
# ---------------------------------------------------------------------------


class TestAuth:
    """Verify _verify_token behaviour via live HTTP requests."""

    pytestmark = pytest.mark.asyncio

    async def test_no_api_key_env_var_allows_all_requests(self, with_singletons):
        """MUNINN_API_KEY unset → open-access dev mode."""
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        assert resp.status_code == 200

    async def test_correct_bearer_token_is_accepted(self, monkeypatch, with_singletons):
        monkeypatch.setenv("MUNINN_API_KEY", "s3cr3t-k3y")
        async with _make_client() as client:
            resp = await client.get(
                "/mimir/runs",
                headers={"Authorization": "Bearer s3cr3t-k3y"},
            )
        assert resp.status_code == 200

    async def test_wrong_bearer_token_returns_401(self, monkeypatch, with_singletons):
        monkeypatch.setenv("MUNINN_API_KEY", "s3cr3t-k3y")
        async with _make_client() as client:
            resp = await client.get(
                "/mimir/runs",
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert resp.status_code == 401

    async def test_missing_credentials_with_key_configured_returns_401(
        self, monkeypatch, with_singletons
    ):
        monkeypatch.setenv("MUNINN_API_KEY", "s3cr3t-k3y")
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        assert resp.status_code == 401

    async def test_empty_string_api_key_treats_as_dev_mode(
        self, monkeypatch, with_singletons
    ):
        """An empty string value for MUNINN_API_KEY is equivalent to unset."""
        monkeypatch.setenv("MUNINN_API_KEY", "")
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: 503 guards (uninitialised singletons)
# ---------------------------------------------------------------------------


class TestSingletonGuards:
    """503 is returned when relay or store singletons are not yet set."""

    pytestmark = pytest.mark.asyncio

    async def test_relay_uninit_post_relay_returns_503(self):
        api_module._store = _make_mock_store()  # store set, relay not set
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/relay",
                json={"instruction": "Hello", "target": "auto", "mode": "A"},
            )
        assert resp.status_code == 503

    async def test_relay_uninit_get_providers_returns_503(self):
        async with _make_client() as client:
            resp = await client.get("/mimir/providers")
        assert resp.status_code == 503

    async def test_store_uninit_get_runs_returns_503(self):
        api_module._relay = _make_mock_relay()  # relay set, store not set
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        assert resp.status_code == 503

    async def test_store_uninit_get_run_returns_503(self):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/some-id")
        assert resp.status_code == 503

    async def test_store_uninit_get_run_audit_returns_503(self):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/some-id/audit")
        assert resp.status_code == 503

    async def test_store_uninit_get_settings_returns_503(self):
        async with _make_client() as client:
            resp = await client.get("/mimir/settings")
        assert resp.status_code == 503

    async def test_store_uninit_post_settings_returns_503(self):
        async with _make_client() as client:
            resp = await client.post("/mimir/settings", json={"user_id": "u"})
        assert resp.status_code == 503

    async def test_store_uninit_get_connections_returns_503(self):
        async with _make_client() as client:
            resp = await client.get("/mimir/connections")
        assert resp.status_code == 503

    async def test_store_uninit_post_purge_returns_503(self):
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/audit/purge", json={"retention_days": 30}
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Tests: POST /mimir/relay
# ---------------------------------------------------------------------------


class TestRelayEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_successful_relay_returns_200_with_result(self, with_singletons):
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/relay",
                json={"instruction": "Summarize this code.", "target": "auto", "mode": "A"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["status"] == "success"
        assert data["run_id"] == "TEST-RUN-001"
        assert data["output"] == "Advisory response from provider."

    async def test_relay_failed_status_still_returns_http_200(self, relay, store):
        """Relay errors surface as FAILED in the result, not as HTTP 4xx/5xx."""
        relay.run = AsyncMock(
            return_value=_make_relay_result(status=RunStatus.FAILED, output="")
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/relay",
                json={"instruction": "Test", "target": "auto", "mode": "A"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "failed"

    async def test_relay_policy_blocked_still_returns_http_200(self, relay, store):
        relay.run = AsyncMock(
            return_value=_make_relay_result(status=RunStatus.POLICY_BLOCKED, output="")
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/relay",
                json={"instruction": "Test", "target": "auto", "mode": "A"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "policy_blocked"

    async def test_relay_run_raises_returns_500(self, relay, store):
        """An unexpected exception from relay.run is surfaced as HTTP 500."""
        relay.run = AsyncMock(side_effect=RuntimeError("Unexpected crash"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/relay",
                json={"instruction": "Test", "target": "auto", "mode": "A"},
            )
        assert resp.status_code == 500

    async def test_relay_missing_instruction_field_returns_422(self, with_singletons):
        """Pydantic validation failure for missing required 'instruction' field."""
        async with _make_client() as client:
            resp = await client.post("/mimir/relay", json={"mode": "A"})
        assert resp.status_code == 422

    async def test_relay_mode_c_result_serialises_reconciliation(self, relay, store):
        result = RelayResult(
            run_id="RUN-C",
            irp_id="IRP-C",
            mode=IRPMode.RECONCILE,
            status=RunStatus.SUCCESS,
            output="Consensus output.",
            reconciliation=ReconciliationResult(
                synthesis="## Consensus\n- All providers agreed."
            ),
        )
        relay.run = AsyncMock(return_value=result)
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/relay",
                json={"instruction": "Reconcile providers.", "mode": "C"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["mode"] == "C"
        assert data["reconciliation"] is not None
        assert "synthesis" in data["reconciliation"]

    async def test_relay_relay_request_object_passed_to_run(self, relay, store):
        """relay.run receives a MimirRelayRequest (not the raw dict)."""
        from muninn.mimir.models import MimirRelayRequest
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.post(
                "/mimir/relay",
                json={
                    "instruction": "Do something.",
                    "target": "codex_cli",
                    "mode": "B",
                    "user_id": "alice",
                },
            )
        relay.run.assert_called_once()
        call_arg = relay.run.call_args[0][0]
        assert isinstance(call_arg, MimirRelayRequest)
        assert call_arg.target == "codex_cli"
        assert call_arg.mode == "B"
        assert call_arg.user_id == "alice"


# ---------------------------------------------------------------------------
# Tests: GET /mimir/providers
# ---------------------------------------------------------------------------


class TestProvidersEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_providers_returns_all_three_providers(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/providers")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert set(data.keys()) == {"claude_code", "codex_cli", "gemini_cli"}

    async def test_providers_each_entry_has_all_score_fields(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/providers")
        score = resp.json()["data"]["claude_code"]
        expected_fields = {
            "provider",
            "capability_score",
            "availability_score",
            "cost_score",
            "safety_score",
            "history_score",
            "composite",
        }
        assert expected_fields <= set(score.keys())

    async def test_providers_score_all_raises_returns_500(self, relay, store):
        relay._router.score_all = AsyncMock(side_effect=RuntimeError("Router down"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/providers")
        assert resp.status_code == 500

    async def test_providers_user_id_query_param_is_accepted(self, with_singletons):
        """user_id is a documented query param; the endpoint does not reject it."""
        async with _make_client() as client:
            resp = await client.get("/mimir/providers?user_id=alice")
        assert resp.status_code == 200

    async def test_providers_response_wrapped_in_success_envelope(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/providers")
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], dict)


# ---------------------------------------------------------------------------
# Tests: GET /mimir/runs
# ---------------------------------------------------------------------------


class TestListRunsEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_list_runs_default_returns_200_with_empty_list(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 0
        assert data["runs"] == []

    async def test_list_runs_response_contains_pagination_metadata(
        self, with_singletons
    ):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs?limit=25&offset=10&user_id=bob")
        data = resp.json()["data"]
        assert data["limit"] == 25
        assert data["offset"] == 10
        assert data["user_id"] == "bob"

    async def test_list_runs_pagination_params_forwarded_to_store(
        self, relay, store
    ):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/runs?user_id=bob&limit=10&offset=20")
        store.list_runs.assert_called_once_with(
            user_id="bob",
            provider=None,
            limit=10,
            offset=20,
            status=None,
        )

    async def test_list_runs_valid_status_filter_passes(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs?status=success")
        assert resp.status_code == 200
        assert resp.json()["data"]["status_filter"] == "success"

    async def test_list_runs_status_filter_forwarded_to_store(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/runs?status=failed")
        store.list_runs.assert_called_once_with(
            user_id="global_user",
            provider=None,
            limit=50,
            offset=0,
            status="failed",
        )

    async def test_list_runs_provider_filter_forwarded_to_store(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs?provider=codex_cli")
        assert resp.status_code == 200
        assert resp.json()["data"]["provider_filter"] == "codex_cli"
        store.list_runs.assert_called_once_with(
            user_id="global_user",
            provider="codex_cli",
            limit=50,
            offset=0,
            status=None,
        )

    async def test_list_runs_invalid_status_returns_422(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs?status=not_a_valid_status")
        assert resp.status_code == 422

    async def test_list_runs_all_valid_statuses_accepted(self, with_singletons):
        valid_statuses = (
            "pending", "running", "success", "failed", "cancelled", "policy_blocked"
        )
        async with _make_client() as client:
            for status in valid_statuses:
                resp = await client.get(f"/mimir/runs?status={status}")
                assert resp.status_code == 200, f"Status '{status}' was unexpectedly rejected"

    async def test_list_runs_store_error_returns_500(self, relay, store):
        store.list_runs = MagicMock(side_effect=Exception("DB connection lost"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        assert resp.status_code == 500

    async def test_list_runs_with_populated_results(self, relay, store):
        store.list_runs = MagicMock(
            return_value=[
                {"run_id": "RUN-A", "status": "success"},
                {"run_id": "RUN-B", "status": "failed"},
            ]
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs")
        data = resp.json()["data"]
        assert data["count"] == 2
        assert len(data["runs"]) == 2


# ---------------------------------------------------------------------------
# Tests: GET /mimir/runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRunEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_get_existing_run_returns_200_with_record(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/TEST-RUN-001")
        assert resp.status_code == 200
        assert resp.json()["data"]["run_id"] == "TEST-RUN-001"

    async def test_get_run_store_called_with_run_id(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/runs/MY-SPECIFIC-ID")
        store.get_run.assert_called_once_with("MY-SPECIFIC-ID")

    async def test_get_nonexistent_run_returns_404(self, relay, store):
        store.get_run = MagicMock(return_value=None)
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/does-not-exist")
        assert resp.status_code == 404

    async def test_get_run_store_error_returns_500(self, relay, store):
        store.get_run = MagicMock(side_effect=Exception("DB read failure"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/TEST-RUN-001")
        assert resp.status_code == 500

    async def test_get_run_wraps_result_in_success_envelope(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/TEST-RUN-001")
        body = resp.json()
        assert body["success"] is True


# ---------------------------------------------------------------------------
# Tests: GET /mimir/runs/{run_id}/audit
# ---------------------------------------------------------------------------


class TestGetRunAuditEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_get_audit_empty_events_returns_200(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/TEST-RUN-001/audit")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["run_id"] == "TEST-RUN-001"
        assert data["count"] == 0
        assert data["events"] == []

    async def test_get_audit_with_events_returns_correct_count(self, relay, store):
        store.get_audit_events = MagicMock(
            return_value=[
                {"event_type": "relay_start", "ts": 1700000000.0},
                {"event_type": "relay_complete", "ts": 1700000001.0},
            ]
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/TEST-RUN-001/audit")
        data = resp.json()["data"]
        assert data["count"] == 2
        assert len(data["events"]) == 2

    async def test_get_audit_limit_param_forwarded_to_store(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/runs/RUN-X/audit?limit=50")
        store.get_audit_events.assert_called_once_with("RUN-X", 50)

    async def test_get_audit_default_limit_is_100(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/runs/RUN-Y/audit")
        store.get_audit_events.assert_called_once_with("RUN-Y", 100)

    async def test_get_audit_store_error_returns_500(self, relay, store):
        store.get_audit_events = MagicMock(side_effect=Exception("DB read failure"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/runs/TEST-RUN-001/audit")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Tests: GET /mimir/settings
# ---------------------------------------------------------------------------


class TestGetSettingsEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_get_settings_returns_200_with_settings_fields(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/settings")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "enabled" in data
        assert "allowed_targets" in data
        assert "hop_max" in data

    async def test_get_settings_default_user_id(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/settings")
        store.get_settings.assert_called_once_with("global_user")

    async def test_get_settings_user_id_forwarded_to_store(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/settings?user_id=alice")
        store.get_settings.assert_called_once_with("alice")

    async def test_get_settings_store_error_returns_500(self, relay, store):
        store.get_settings = MagicMock(side_effect=Exception("Settings not found"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/settings")
        assert resp.status_code == 500

    async def test_get_settings_enabled_field_reflects_store_value(self, relay, store):
        store.get_settings = MagicMock(
            return_value=InteropSettings(enabled=False, user_id="alice")
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/settings?user_id=alice")
        assert resp.json()["data"]["enabled"] is False


# ---------------------------------------------------------------------------
# Tests: POST /mimir/settings
# ---------------------------------------------------------------------------


class TestUpdateSettingsEndpoint:
    pytestmark = pytest.mark.asyncio

    _FULL_BODY = {
        "user_id": "alice",
        "enabled": True,
        "allowed_targets": ["claude_code", "codex_cli"],
        "policy_tools": "allowed",
        "hop_max": 2,
        "memory_context_enabled": True,
        "audit_retention_days": 30,
    }

    async def test_post_settings_returns_200_with_updated_true(self, with_singletons):
        async with _make_client() as client:
            resp = await client.post("/mimir/settings", json=self._FULL_BODY)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["updated"] is True
        assert data["user_id"] == "alice"

    async def test_post_settings_minimal_body_uses_defaults(self, with_singletons):
        """InteropSettings has defaults for all fields — empty body is valid."""
        async with _make_client() as client:
            resp = await client.post("/mimir/settings", json={})
        assert resp.status_code == 200
        assert resp.json()["data"]["updated"] is True

    async def test_post_settings_passes_interop_settings_object_to_store(
        self, relay, store
    ):
        api_module._relay = relay
        api_module._store = store
        body = {
            "user_id": "bob",
            "enabled": False,
            "allowed_targets": ["codex_cli"],
            "policy_tools": "forbidden",
            "hop_max": 1,
            "memory_context_enabled": False,
            "audit_retention_days": 60,
        }
        async with _make_client() as client:
            await client.post("/mimir/settings", json=body)
        store.update_settings.assert_called_once()
        called_settings: InteropSettings = store.update_settings.call_args[0][0]
        assert isinstance(called_settings, InteropSettings)
        assert called_settings.user_id == "bob"
        assert called_settings.enabled is False
        assert called_settings.hop_max == 1
        assert called_settings.policy_tools == "forbidden"

    async def test_post_settings_invalid_policy_tools_returns_422(self, with_singletons):
        """policy_tools must match '^(forbidden|readonly|allowed)$'."""
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/settings",
                json={"policy_tools": "everything"},
            )
        assert resp.status_code == 422

    async def test_post_settings_hop_max_out_of_range_returns_422(self, with_singletons):
        async with _make_client() as client:
            resp = await client.post("/mimir/settings", json={"hop_max": 5})
        assert resp.status_code == 422

    async def test_post_settings_store_error_returns_500(self, relay, store):
        store.update_settings = MagicMock(side_effect=Exception("Write failed"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post("/mimir/settings", json=self._FULL_BODY)
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Tests: GET /mimir/connections
# ---------------------------------------------------------------------------


class TestConnectionsEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_connections_returns_200_with_empty_list(self, with_singletons):
        async with _make_client() as client:
            resp = await client.get("/mimir/connections")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 0
        assert data["connections"] == []

    async def test_connections_user_id_forwarded_to_store(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/connections?user_id=charlie")
        store.list_connections.assert_called_once_with("charlie")

    async def test_connections_with_two_entries_returns_count_2(self, relay, store):
        store.list_connections = MagicMock(
            return_value=[
                {"provider": "claude_code", "status": "active"},
                {"provider": "codex_cli", "status": "expired"},
            ]
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/connections?user_id=charlie")
        data = resp.json()["data"]
        assert data["count"] == 2
        assert data["user_id"] == "charlie"

    async def test_connections_store_error_returns_500(self, relay, store):
        store.list_connections = MagicMock(side_effect=Exception("DB error"))
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.get("/mimir/connections")
        assert resp.status_code == 500

    async def test_connections_default_user_id_is_global_user(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.get("/mimir/connections")
        store.list_connections.assert_called_once_with("global_user")


# ---------------------------------------------------------------------------
# Tests: POST /mimir/audit/purge
# ---------------------------------------------------------------------------


class TestAuditPurgeEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_purge_returns_200_with_deleted_count_and_retention_days(
        self, with_singletons
    ):
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/audit/purge", json={"retention_days": 30}
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["deleted"] == 5  # mock returns 5
        assert data["retention_days"] == 30

    async def test_purge_default_retention_days_is_90(self, with_singletons):
        """Empty body uses the default retention_days=90."""
        async with _make_client() as client:
            resp = await client.post("/mimir/audit/purge", json={})
        assert resp.status_code == 200
        assert resp.json()["data"]["retention_days"] == 90

    async def test_purge_retention_days_forwarded_to_store(self, relay, store):
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            await client.post("/mimir/audit/purge", json={"retention_days": 45})
        store.purge_old_audit_events.assert_called_once_with(45)

    async def test_purge_retention_days_zero_returns_422(self, with_singletons):
        """retention_days must be >= 1 (Pydantic ge=1 constraint)."""
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/audit/purge", json={"retention_days": 0}
            )
        assert resp.status_code == 422

    async def test_purge_retention_days_above_max_returns_422(self, with_singletons):
        """retention_days must be <= 3650 (Pydantic le=3650 constraint)."""
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/audit/purge", json={"retention_days": 3651}
            )
        assert resp.status_code == 422

    async def test_purge_retention_days_at_boundaries_accepted(self, with_singletons):
        """Boundary values (1 and 3650) must be accepted."""
        async with _make_client() as client:
            for days in (1, 3650):
                resp = await client.post(
                    "/mimir/audit/purge", json={"retention_days": days}
                )
                assert resp.status_code == 200, (
                    f"retention_days={days} was unexpectedly rejected"
                )

    async def test_purge_store_error_returns_500(self, relay, store):
        store.purge_old_audit_events = MagicMock(
            side_effect=Exception("Purge operation failed")
        )
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/audit/purge", json={"retention_days": 90}
            )
        assert resp.status_code == 500

    async def test_purge_store_returns_zero_deleted(self, relay, store):
        store.purge_old_audit_events = MagicMock(return_value=0)
        api_module._relay = relay
        api_module._store = store
        async with _make_client() as client:
            resp = await client.post(
                "/mimir/audit/purge", json={"retention_days": 90}
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == 0
