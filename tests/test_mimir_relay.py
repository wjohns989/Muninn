"""
Tests for MimirRelay orchestrator — muninn/mimir/relay.py

Coverage targets:
  - Construction and dependency injection
  - Mode A/B (single-provider) success paths
  - All PolicyError branches (POLICY_BLOCKED paths)
  - RoutingError and provider-level failure paths
  - INTERNAL_ERROR outer boundary
  - Store fire-and-forget behaviour (storage errors never kill relay)
  - _load_settings() fallback chain
  - Mode C (multi-provider reconciliation) — fan-out, token aggregation,
    allowed-target filtering, escalation, no-viable-providers routing error
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from muninn.mimir.models import (
    InteropSettings,
    IRPMode,
    MimirRelayRequest,
    ProviderName,
    ProviderResult,
    ReconciliationResult,
    RelayResult,
    RoutingScore,
    RunStatus,
)
from muninn.mimir.policy import PolicyError
from muninn.mimir.relay import MimirRelay
from muninn.mimir.routing import RoutingError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_request(
    instruction: str = "Hello, world",
    target: str = "auto",
    mode: str = "A",
    user_id: str = "test_user",
    policy: dict | None = None,
    context: dict | None = None,
) -> MimirRelayRequest:
    return MimirRelayRequest(
        instruction=instruction,
        target=target,
        mode=mode,
        user_id=user_id,
        policy=policy,
        context=context or {},
    )


def make_provider_result(
    provider: ProviderName = ProviderName.CLAUDE_CODE,
    raw_output: str = "This is the answer.",
    error: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
    parsed: dict | None = None,
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        raw_output=raw_output,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        parsed=parsed,
    )


def make_routing_score(
    provider: ProviderName = ProviderName.CLAUDE_CODE,
    availability_score: float = 1.0,
    composite: float = 0.80,
) -> RoutingScore:
    return RoutingScore(
        provider=provider,
        availability_score=availability_score,
        composite=composite,
    )


def make_mock_adapter(result: ProviderResult) -> MagicMock:
    """Return a mock adapter whose .call() is an AsyncMock returning result."""
    adapter = MagicMock()
    adapter.call = AsyncMock(return_value=result)
    return adapter


def make_relay(mimir_store=None, metadata_store=None, store=None) -> MimirRelay:
    return MimirRelay(mimir_store=mimir_store or store, metadata_store=metadata_store)


def make_mock_store(
    settings: InteropSettings | None = None,
    enabled: bool = True,
    allowed_targets: list[str] | None = None,
) -> MagicMock:
    """Create a mock store that returns the given (or constructed) settings."""
    store = MagicMock()
    if settings is None:
        settings = InteropSettings(
            enabled=enabled,
            allowed_targets=allowed_targets or ["claude_code", "codex_cli", "gemini_cli"],
        )
    store.get_settings = MagicMock(return_value=settings)
    store.upsert_run = MagicMock()
    store.insert_audit_event = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMimirRelayConstruction:
    def test_construct_without_store(self):
        relay = MimirRelay()
        assert relay._store is None
        assert relay._router is not None
        assert relay._reconciler is not None

    def test_construct_with_store(self):
        store = MagicMock()
        relay = MimirRelay(mimir_store=store)
        assert relay._store is store

    def test_construct_with_metadata_store(self):
        mimir_store = MagicMock()
        metadata_store = MagicMock()
        relay = MimirRelay(mimir_store=mimir_store, metadata_store=metadata_store)
        assert relay._store is mimir_store
        assert relay._reconciler._store is metadata_store

    def test_router_and_reconciler_created_with_correct_types(self):
        from muninn.mimir.reconcile import Reconciler
        from muninn.mimir.routing import MemoryAwareRouter

        relay = MimirRelay()
        assert isinstance(relay._router, MemoryAwareRouter)
        assert isinstance(relay._reconciler, Reconciler)


# ---------------------------------------------------------------------------
# Mode A / B — single-provider success paths
# ---------------------------------------------------------------------------


class TestModeABSuccess:
    pytestmark = pytest.mark.asyncio

    async def test_mode_a_success_no_store(self):
        relay = make_relay()
        result_obj = make_provider_result(raw_output="Advisory answer.")
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request(mode="A"))

        assert isinstance(result, RelayResult)
        assert result.status == RunStatus.SUCCESS
        assert result.output == "Advisory answer."
        assert result.provider == ProviderName.CLAUDE_CODE
        assert result.mode == IRPMode.ADVISORY
        assert result.error_code is None
        assert result.error_message is None

    async def test_mode_b_success(self):
        relay = make_relay()
        result_obj = make_provider_result(
            provider=ProviderName.GEMINI_CLI,
            raw_output='{"plan": "step 1"}',
        )
        relay._router.route = AsyncMock(return_value=ProviderName.GEMINI_CLI)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request(mode="B"))

        assert result.status == RunStatus.SUCCESS
        assert result.provider == ProviderName.GEMINI_CLI
        assert result.mode == IRPMode.STRUCTURED

    async def test_success_with_explicit_target(self):
        relay = make_relay()
        result_obj = make_provider_result(
            provider=ProviderName.CODEX_CLI,
            raw_output="Codex says hi.",
        )
        relay._router.route = AsyncMock(return_value=ProviderName.CODEX_CLI)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request(target="codex_cli"))

        assert result.status == RunStatus.SUCCESS
        assert result.output == "Codex says hi."

    async def test_token_counts_propagated_to_result(self):
        relay = make_relay()
        result_obj = make_provider_result(input_tokens=42, output_tokens=99)
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.input_tokens == 42
        assert result.output_tokens == 99

    async def test_latency_ms_is_non_negative(self):
        relay = make_relay()
        result_obj = make_provider_result()
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.latency_ms >= 0

    async def test_result_contains_non_empty_run_and_irp_ids(self):
        relay = make_relay()
        result_obj = make_provider_result()
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.run_id and len(result.run_id) > 0
        assert result.irp_id and len(result.irp_id) > 0


# ---------------------------------------------------------------------------
# Policy blocked paths
# ---------------------------------------------------------------------------


class TestPolicyBlocked:
    pytestmark = pytest.mark.asyncio

    async def test_interop_disabled_blocks(self):
        store = make_mock_store(enabled=False)
        relay = make_relay(store=store)
        # No router/adapter mock needed — policy guard fires before routing
        result = await relay.run(make_request())

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "INTEROP_DISABLED"

    async def test_target_not_allowed_blocks(self):
        store = make_mock_store(allowed_targets=["claude_code"])
        relay = make_relay(store=store)
        result = await relay.run(make_request(target="codex_cli"))

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "TARGET_NOT_ALLOWED"

    async def test_prompt_too_large_blocks(self):
        # 32001 chars > max_prompt_chars (32000); policy check fires BEFORE routing
        relay = make_relay()
        big_instruction = "x" * 32_001
        result = await relay.run(make_request(instruction=big_instruction))

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "PROMPT_TOO_LARGE"

    async def test_prompt_exactly_at_limit_passes(self):
        # Exactly 32000 chars == max_prompt_chars; must NOT trigger PROMPT_TOO_LARGE
        relay = make_relay()
        instruction = "x" * 32_000
        result_obj = make_provider_result(raw_output="ok")
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request(instruction=instruction))

        assert result.status == RunStatus.SUCCESS

    async def test_output_too_large_blocks(self):
        # Provider returns 16001 chars > max_output_chars (16000)
        relay = make_relay()
        big_output = "y" * 16_001
        result_obj = make_provider_result(raw_output=big_output)
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "OUTPUT_TOO_LARGE"

    async def test_tool_usage_violation_blocks(self):
        # policy.tools=forbidden + parsed output shows totalCalls=2 → TOOL_USAGE_VIOLATION
        relay = make_relay()
        result_obj = make_provider_result(
            parsed={"stats": {"tools": {"totalCalls": 2}}}
        )
        relay._router.route = AsyncMock(return_value=ProviderName.GEMINI_CLI)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request(policy={"tools": "forbidden"}))

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "TOOL_USAGE_VIOLATION"

    async def test_hop_limit_exceeded_blocks_without_router(self):
        # validate_hop_limit fires in _pipeline() BEFORE routing — no router mock required
        relay = make_relay()
        with patch(
            "muninn.mimir.relay.PolicyEngine.validate_hop_limit",
            side_effect=PolicyError("HOP_LIMIT_EXCEEDED", "Hop limit reached."),
        ):
            result = await relay.run(make_request())

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "HOP_LIMIT_EXCEEDED"

    async def test_hop_loop_detected_blocks_after_routing(self):
        # validate_hop_path fires in _run_mode_ab() AFTER routing — router mock required
        relay = make_relay()
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)
        with patch(
            "muninn.mimir.relay.PolicyEngine.validate_hop_path",
            side_effect=PolicyError("HOP_LOOP_DETECTED", "Loop detected."),
        ):
            result = await relay.run(make_request())

        assert result.status == RunStatus.POLICY_BLOCKED
        assert result.error_code == "HOP_LOOP_DETECTED"


# ---------------------------------------------------------------------------
# Routing and provider-level failures
# ---------------------------------------------------------------------------


class TestRoutingAndProviderFailure:
    pytestmark = pytest.mark.asyncio

    async def test_routing_error_returns_routing_failed(self):
        relay = make_relay()
        relay._router.route = AsyncMock(
            side_effect=RoutingError("No providers available.")
        )
        result = await relay.run(make_request())

        assert result.status == RunStatus.FAILED
        assert result.error_code == "ROUTING_FAILED"
        assert "No providers available." in (result.error_message or "")

    async def test_provider_error_in_result_returns_provider_error(self):
        relay = make_relay()
        result_obj = make_provider_result(raw_output="", error="Connection timeout")
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.status == RunStatus.FAILED
        assert result.error_code == "PROVIDER_ERROR"
        assert "Connection timeout" in (result.error_message or "")


# ---------------------------------------------------------------------------
# INTERNAL_ERROR outer boundary (run() outer try/except)
# ---------------------------------------------------------------------------


class TestInternalError:
    pytestmark = pytest.mark.asyncio

    async def test_unexpected_exception_returns_internal_error(self):
        relay = make_relay()
        relay._router.route = AsyncMock(side_effect=RuntimeError("Unexpected failure"))
        result = await relay.run(make_request())

        assert result.status == RunStatus.FAILED
        assert result.error_code == "INTERNAL_ERROR"
        assert "Unexpected failure" in (result.error_message or "")

    async def test_run_never_raises_always_returns_relay_result(self):
        # Even with an uncaught Exception, run() must return a RelayResult
        relay = make_relay()
        relay._router.route = AsyncMock(side_effect=ValueError("bad value from router"))
        result = await relay.run(make_request())

        assert isinstance(result, RelayResult)
        assert result.status == RunStatus.FAILED
        assert result.error_code == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Store fire-and-forget behaviour
# ---------------------------------------------------------------------------


class TestStoreFireAndForget:
    pytestmark = pytest.mark.asyncio

    async def test_upsert_run_exception_does_not_kill_relay(self):
        store = make_mock_store()
        store.upsert_run.side_effect = Exception("DB is down")
        relay = make_relay(store=store)
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)
        result_obj = make_provider_result()

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.status == RunStatus.SUCCESS

    async def test_insert_audit_event_exception_does_not_kill_relay(self):
        store = make_mock_store()
        store.insert_audit_event.side_effect = Exception("Audit DB is down")
        relay = make_relay(store=store)
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)
        result_obj = make_provider_result()

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        assert result.status == RunStatus.SUCCESS

    async def test_get_settings_exception_uses_defaults_and_relay_succeeds(self):
        store = make_mock_store()
        store.get_settings.side_effect = Exception("Settings DB is down")
        relay = make_relay(store=store)
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)
        result_obj = make_provider_result()

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            result = await relay.run(make_request())

        # Default settings allow relay to proceed normally
        assert result.status == RunStatus.SUCCESS

    async def test_upsert_run_called_at_least_twice_per_relay(self):
        # Minimum calls: PENDING (in run()), then final status (in terminal helpers)
        store = make_mock_store()
        relay = make_relay(store=store)
        relay._router.route = AsyncMock(return_value=ProviderName.CLAUDE_CODE)
        result_obj = make_provider_result()

        with patch("muninn.mimir.relay.get_adapter", return_value=make_mock_adapter(result_obj)):
            await relay.run(make_request())

        assert store.upsert_run.call_count >= 2


# ---------------------------------------------------------------------------
# _load_settings() isolation (synchronous unit tests)
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_no_store_returns_default_settings(self):
        relay = make_relay(store=None)
        settings = relay._load_settings("user1")
        assert settings.enabled is True
        assert "claude_code" in settings.allowed_targets

    def test_store_returns_custom_settings(self):
        custom = InteropSettings(user_id="user1", hop_max=3)
        store = MagicMock()
        store.get_settings = MagicMock(return_value=custom)
        relay = make_relay(store=store)
        settings = relay._load_settings("user1")
        assert settings.hop_max == 3

    def test_store_returns_none_falls_back_to_defaults(self):
        store = MagicMock()
        store.get_settings = MagicMock(return_value=None)
        relay = make_relay(store=store)
        settings = relay._load_settings("user1")
        assert settings.enabled is True

    def test_store_raises_falls_back_to_defaults(self):
        store = MagicMock()
        store.get_settings.side_effect = Exception("DB error")
        relay = make_relay(store=store)
        settings = relay._load_settings("user1")
        assert settings.enabled is True


# ---------------------------------------------------------------------------
# Mode C (multi-provider reconciliation)
# ---------------------------------------------------------------------------


class TestModeCReconcile:
    pytestmark = pytest.mark.asyncio

    # Helper: build a full scores dict with configurable availability per provider
    def _make_scores(
        self,
        claude_avail: float = 1.0,
        codex_avail: float = 1.0,
        gemini_avail: float = 1.0,
    ) -> dict[ProviderName, RoutingScore]:
        return {
            ProviderName.CLAUDE_CODE: make_routing_score(
                ProviderName.CLAUDE_CODE, availability_score=claude_avail
            ),
            ProviderName.CODEX_CLI: make_routing_score(
                ProviderName.CODEX_CLI, availability_score=codex_avail
            ),
            ProviderName.GEMINI_CLI: make_routing_score(
                ProviderName.GEMINI_CLI, availability_score=gemini_avail
            ),
        }

    async def test_mode_c_success_consensus_reached(self):
        relay = make_relay()
        relay._router.score_all = AsyncMock(return_value=self._make_scores())
        reconciled = ReconciliationResult(
            synthesis="Consensus reached.", escalated=False
        )
        relay._reconciler.reconcile = AsyncMock(return_value=reconciled)

        adapters = {
            ProviderName.CLAUDE_CODE: make_mock_adapter(
                make_provider_result(
                    provider=ProviderName.CLAUDE_CODE,
                    raw_output="Claude answer.",
                    input_tokens=5,
                    output_tokens=10,
                )
            ),
            ProviderName.CODEX_CLI: make_mock_adapter(
                make_provider_result(
                    provider=ProviderName.CODEX_CLI,
                    raw_output="Codex answer.",
                    input_tokens=6,
                    output_tokens=11,
                )
            ),
            ProviderName.GEMINI_CLI: make_mock_adapter(
                make_provider_result(
                    provider=ProviderName.GEMINI_CLI,
                    raw_output="Gemini answer.",
                    input_tokens=7,
                    output_tokens=12,
                )
            ),
        }
        with patch("muninn.mimir.relay.get_adapter", side_effect=lambda p: adapters[p]):
            result = await relay.run(make_request(mode="C"))

        assert result.status == RunStatus.SUCCESS
        assert result.mode == IRPMode.RECONCILE
        assert result.output == "Consensus reached."
        assert result.reconciliation is not None
        assert result.reconciliation.escalated is False
        assert result.error_code is None

    async def test_mode_c_escalated_result_returns_failed(self):
        relay = make_relay()
        relay._router.score_all = AsyncMock(return_value=self._make_scores())
        reconciled = ReconciliationResult(
            synthesis="No consensus.",
            escalated=True,
            escalation_reason="Providers disagreed on all claims.",
        )
        relay._reconciler.reconcile = AsyncMock(return_value=reconciled)

        adapters = {
            ProviderName.CLAUDE_CODE: make_mock_adapter(
                make_provider_result(provider=ProviderName.CLAUDE_CODE, raw_output="Answer A.")
            ),
            ProviderName.CODEX_CLI: make_mock_adapter(
                make_provider_result(provider=ProviderName.CODEX_CLI, raw_output="Answer B.")
            ),
            ProviderName.GEMINI_CLI: make_mock_adapter(
                make_provider_result(provider=ProviderName.GEMINI_CLI, raw_output="Answer C.")
            ),
        }
        with patch("muninn.mimir.relay.get_adapter", side_effect=lambda p: adapters[p]):
            result = await relay.run(make_request(mode="C"))

        assert result.status == RunStatus.FAILED
        assert result.error_code == "RECONCILIATION_ESCALATED"
        assert "disagreed" in (result.error_message or "")

    async def test_mode_c_no_viable_providers_raises_routing_failed(self):
        # All availability_score == 0.0 → RoutingError raised → ROUTING_FAILED
        relay = make_relay()
        relay._router.score_all = AsyncMock(
            return_value=self._make_scores(
                claude_avail=0.0, codex_avail=0.0, gemini_avail=0.0
            )
        )
        relay._reconciler.reconcile = AsyncMock()  # must NOT be called

        with patch("muninn.mimir.relay.get_adapter"):
            result = await relay.run(make_request(mode="C"))

        assert result.status == RunStatus.FAILED
        assert result.error_code == "ROUTING_FAILED"
        relay._reconciler.reconcile.assert_not_called()

    async def test_mode_c_token_aggregation_skips_error_results(self):
        """
        Providers with result.error set are passed through to the reconciler,
        but their token counts must NOT be accumulated into the relay totals.
        """
        relay = make_relay()
        relay._router.score_all = AsyncMock(return_value=self._make_scores())
        reconciled = ReconciliationResult(synthesis="Partial consensus.", escalated=False)
        relay._reconciler.reconcile = AsyncMock(return_value=reconciled)

        adapters = {
            ProviderName.CLAUDE_CODE: make_mock_adapter(
                make_provider_result(
                    provider=ProviderName.CLAUDE_CODE,
                    raw_output="Claude answer.",
                    input_tokens=10,
                    output_tokens=20,
                )
            ),
            ProviderName.CODEX_CLI: make_mock_adapter(
                # Error result with non-zero tokens — must NOT be counted
                make_provider_result(
                    provider=ProviderName.CODEX_CLI,
                    raw_output="",
                    error="Codex unavailable",
                    input_tokens=500,
                    output_tokens=500,
                )
            ),
            ProviderName.GEMINI_CLI: make_mock_adapter(
                make_provider_result(
                    provider=ProviderName.GEMINI_CLI,
                    raw_output="Gemini answer.",
                    input_tokens=5,
                    output_tokens=15,
                )
            ),
        }
        with patch("muninn.mimir.relay.get_adapter", side_effect=lambda p: adapters[p]):
            result = await relay.run(make_request(mode="C"))

        # Only CLAUDE_CODE (10+20) + GEMINI_CLI (5+15) contribute tokens
        assert result.input_tokens == 15   # 10 + 5 (CODEX_CLI 500 skipped)
        assert result.output_tokens == 35  # 20 + 15 (CODEX_CLI 500 skipped)

    async def test_mode_c_respects_allowed_targets(self):
        """
        Providers absent from allowed_targets are filtered before fan-out.
        The reconciler must only receive results from allowed providers.
        """
        store = make_mock_store(allowed_targets=["claude_code", "gemini_cli"])
        relay = make_relay(store=store)
        relay._router.score_all = AsyncMock(return_value=self._make_scores())
        reconciled = ReconciliationResult(
            synthesis="Two-provider consensus.", escalated=False
        )
        relay._reconciler.reconcile = AsyncMock(return_value=reconciled)

        # get_adapter side_effect raises KeyError for any unlisted provider,
        # which would surface as a fan-out exception if codex_cli were included
        allowed_adapters: dict[ProviderName, MagicMock] = {
            ProviderName.CLAUDE_CODE: make_mock_adapter(
                make_provider_result(provider=ProviderName.CLAUDE_CODE, raw_output="Claude.")
            ),
            ProviderName.GEMINI_CLI: make_mock_adapter(
                make_provider_result(provider=ProviderName.GEMINI_CLI, raw_output="Gemini.")
            ),
        }
        with patch(
            "muninn.mimir.relay.get_adapter",
            side_effect=lambda p: allowed_adapters[p],
        ):
            result = await relay.run(make_request(mode="C"))

        assert result.status == RunStatus.SUCCESS

        # Confirm reconciler did NOT receive CODEX_CLI results
        reconcile_call_args = relay._reconciler.reconcile.call_args
        provider_results_arg = reconcile_call_args.args[1]
        assert ProviderName.CODEX_CLI not in provider_results_arg

    async def test_mode_c_all_errors_still_invokes_reconcile(self):
        """
        When all providers return errors, reconcile() is still called
        (with error results). The reconciler escalates → FAILED.
        Token counts remain zero since error results are skipped.
        """
        relay = make_relay()
        relay._router.score_all = AsyncMock(return_value=self._make_scores())

        # Error results with deliberately non-zero token fields to prove they
        # are not accumulated
        def _make_error_adapter(prov: ProviderName) -> MagicMock:
            return make_mock_adapter(
                make_provider_result(
                    provider=prov,
                    raw_output="",
                    error="Service unavailable",
                    input_tokens=50,
                    output_tokens=50,
                )
            )

        reconciled = ReconciliationResult(
            synthesis="All providers errored.",
            escalated=True,
            escalation_reason="All providers returned errors or empty responses.",
        )
        relay._reconciler.reconcile = AsyncMock(return_value=reconciled)

        adapters = {
            ProviderName.CLAUDE_CODE: _make_error_adapter(ProviderName.CLAUDE_CODE),
            ProviderName.CODEX_CLI: _make_error_adapter(ProviderName.CODEX_CLI),
            ProviderName.GEMINI_CLI: _make_error_adapter(ProviderName.GEMINI_CLI),
        }
        with patch("muninn.mimir.relay.get_adapter", side_effect=lambda p: adapters[p]):
            result = await relay.run(make_request(mode="C"))

        relay._reconciler.reconcile.assert_called_once()
        assert result.status == RunStatus.FAILED
        assert result.error_code == "RECONCILIATION_ESCALATED"
        # No tokens accumulated — all error results skipped the token loop
        assert result.input_tokens == 0
        assert result.output_tokens == 0
