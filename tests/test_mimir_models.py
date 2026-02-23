"""
Tests for Mimir IRP/1 data models (muninn/mimir/models.py).

Coverage
--------
- All enumeration values and member counts
- IRPHop defaults and cap_max_hops validator (hard cap at 4)
- IRPPolicy defaults and pattern validation
- IRPEnvelope alias duality (from_agent / "from") and wire_dict()
- MimirRelayRequest mode / target field validation
- RoutingScore field defaults, compute_composite() arithmetic, clamp, mutation
- InteropSettings defaults and field bounds
- ReconciliationResult defaults
- RelayResult required vs. optional fields
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from muninn.mimir.models import (
    AuditEventType,
    IRPEnvelope,
    IRPHop,
    IRPInput,
    IRPMode,
    IRPNetworkPolicy,
    IRPPolicy,
    IRPRedactionPolicy,
    IRPRequest,
    IRPResponseFormat,
    InteropSettings,
    MimirRelayRequest,
    ProviderName,
    ReconciliationClaim,
    ReconciliationResult,
    RelayResult,
    RoutingScore,
    RunStatus,
)


# ---------------------------------------------------------------------------
# Enumeration coverage
# ---------------------------------------------------------------------------


class TestIRPMode:
    def test_advisory_wire_value(self):
        assert IRPMode.ADVISORY == "A"

    def test_structured_wire_value(self):
        assert IRPMode.STRUCTURED == "B"

    def test_reconcile_wire_value(self):
        assert IRPMode.RECONCILE == "C"

    def test_member_count(self):
        assert len(IRPMode) == 3

    def test_roundtrip_from_wire(self):
        for member in IRPMode:
            assert IRPMode(member.value) is member

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            IRPMode("X")


class TestIRPNetworkPolicy:
    def test_deny_all_wire_value(self):
        assert IRPNetworkPolicy.DENY_ALL == "deny_all"

    def test_local_only_wire_value(self):
        assert IRPNetworkPolicy.LOCAL_ONLY == "local_only"

    def test_allow_all_wire_value(self):
        assert IRPNetworkPolicy.ALLOW_ALL == "allow_all"

    def test_member_count(self):
        assert len(IRPNetworkPolicy) == 3


class TestIRPRedactionPolicy:
    def test_strict(self):
        assert IRPRedactionPolicy.STRICT == "strict"

    def test_balanced(self):
        assert IRPRedactionPolicy.BALANCED == "balanced"

    def test_off(self):
        assert IRPRedactionPolicy.OFF == "off"

    def test_member_count(self):
        assert len(IRPRedactionPolicy) == 3


class TestIRPResponseFormat:
    def test_text(self):
        assert IRPResponseFormat.TEXT == "text"

    def test_json(self):
        assert IRPResponseFormat.JSON == "json"

    def test_markdown(self):
        assert IRPResponseFormat.MARKDOWN == "markdown"

    def test_member_count(self):
        assert len(IRPResponseFormat) == 3


class TestProviderName:
    def test_wire_values(self):
        assert ProviderName.CLAUDE_CODE == "claude_code"
        assert ProviderName.CODEX_CLI == "codex_cli"
        assert ProviderName.GEMINI_CLI == "gemini_cli"
        assert ProviderName.AUTO == "auto"

    def test_member_count(self):
        assert len(ProviderName) == 4

    def test_roundtrip(self):
        for member in ProviderName:
            assert ProviderName(member.value) is member


class TestRunStatus:
    def test_wire_values(self):
        expected = {
            "pending",
            "running",
            "success",
            "failed",
            "cancelled",
            "policy_blocked",
        }
        assert {s.value for s in RunStatus} == expected

    def test_member_count(self):
        assert len(RunStatus) == 6


class TestAuditEventType:
    def test_member_count(self):
        assert len(AuditEventType) == 10

    def test_all_wire_values(self):
        assert AuditEventType.RELAY_START == "relay_start"
        assert AuditEventType.RELAY_COMPLETE == "relay_complete"
        assert AuditEventType.RELAY_FAILED == "relay_failed"
        assert AuditEventType.POLICY_BLOCKED == "policy_blocked"
        assert AuditEventType.HOP_LIMIT_EXCEEDED == "hop_limit_exceeded"
        assert AuditEventType.REDACTION_APPLIED == "redaction_applied"
        assert AuditEventType.PROVIDER_UNAVAILABLE == "provider_unavailable"
        assert AuditEventType.ROUTING_DECISION == "routing_decision"
        assert AuditEventType.RECONCILIATION_START == "reconciliation_start"
        assert AuditEventType.RECONCILIATION_COMPLETE == "reconciliation_complete"


# ---------------------------------------------------------------------------
# IRPHop
# ---------------------------------------------------------------------------


class TestIRPHop:
    def test_defaults(self):
        hop = IRPHop()
        assert hop.count == 0
        assert hop.max == 2
        assert hop.path == []

    def test_cap_max_hops_above_4_clamps_to_4(self):
        hop = IRPHop(max=10)
        assert hop.max == 4

    def test_cap_max_hops_exactly_4_is_unchanged(self):
        hop = IRPHop(max=4)
        assert hop.max == 4

    def test_cap_max_hops_below_4_is_unchanged(self):
        hop = IRPHop(max=3)
        assert hop.max == 3

    def test_explicit_fields(self):
        hop = IRPHop(count=1, max=3, path=["agent-a", "agent-b"])
        assert hop.count == 1
        assert hop.max == 3
        assert hop.path == ["agent-a", "agent-b"]

    def test_count_non_negative(self):
        with pytest.raises(ValidationError):
            IRPHop(count=-1)


# ---------------------------------------------------------------------------
# IRPPolicy
# ---------------------------------------------------------------------------


class TestIRPPolicy:
    def test_defaults(self):
        policy = IRPPolicy()
        assert policy.tools == "allowed"
        assert policy.network == IRPNetworkPolicy.DENY_ALL
        assert policy.redaction == IRPRedactionPolicy.BALANCED
        assert policy.max_prompt_chars == 32_000
        assert policy.max_output_chars == 16_000

    def test_tools_valid_values(self):
        for value in ("forbidden", "readonly", "allowed"):
            p = IRPPolicy(tools=value)
            assert p.tools == value

    def test_tools_invalid_raises(self):
        with pytest.raises(ValidationError):
            IRPPolicy(tools="unrestricted")

    def test_max_prompt_chars_below_min_raises(self):
        with pytest.raises(ValidationError):
            IRPPolicy(max_prompt_chars=100)  # ge=256

    def test_max_output_chars_below_min_raises(self):
        with pytest.raises(ValidationError):
            IRPPolicy(max_output_chars=200)  # ge=256

    def test_max_prompt_chars_at_boundary(self):
        p = IRPPolicy(max_prompt_chars=256)
        assert p.max_prompt_chars == 256


# ---------------------------------------------------------------------------
# IRPEnvelope — alias handling and wire serialisation
# ---------------------------------------------------------------------------

_TEST_ID = "01HWTEST00000000000000001"


def _make_envelope(**kwargs) -> IRPEnvelope:
    """Build a minimal valid IRPEnvelope using the Python-side kwarg ``from_agent``."""
    defaults: dict = {
        "id": _TEST_ID,
        "from_agent": "muninn",
        "to": "auto",
        "request": IRPRequest(instruction="hello world"),
    }
    defaults.update(kwargs)
    return IRPEnvelope(**defaults)


class TestIRPEnvelope:
    def test_from_agent_python_kwarg(self):
        envelope = _make_envelope()
        assert envelope.from_agent == "muninn"

    def test_from_alias_kwarg_accepted(self):
        """IRPEnvelope.model_config populate_by_name=True means 'from' kwarg works."""
        envelope = IRPEnvelope(
            id=_TEST_ID,
            **{"from": "other-agent"},
            to="auto",
            request=IRPRequest(instruction="hi"),
        )
        assert envelope.from_agent == "other-agent"

    def test_from_key_in_model_validate(self):
        """model_validate() with the JSON 'from' key populates from_agent."""
        data = {
            "id": _TEST_ID,
            "from": "json-agent",
            "to": "auto",
            "request": {"instruction": "hi"},
        }
        envelope = IRPEnvelope.model_validate(data)
        assert envelope.from_agent == "json-agent"

    def test_wire_dict_uses_from_key(self):
        envelope = _make_envelope()
        wire = envelope.wire_dict()
        assert "from" in wire
        assert wire["from"] == "muninn"
        assert "from_agent" not in wire

    def test_wire_dict_id_preserved(self):
        envelope = _make_envelope()
        assert envelope.wire_dict()["id"] == _TEST_ID

    def test_wire_dict_mode_is_wire_value(self):
        envelope = _make_envelope()
        wire = envelope.wire_dict()
        assert wire["mode"] == "A"  # IRPMode.ADVISORY.value

    def test_default_mode_advisory(self):
        assert _make_envelope().mode == IRPMode.ADVISORY

    def test_default_hop(self):
        envelope = _make_envelope()
        assert envelope.hop.count == 0
        assert envelope.hop.max == 2

    def test_id_field_is_required(self):
        with pytest.raises(ValidationError):
            IRPEnvelope(
                from_agent="muninn",
                to="auto",
                request=IRPRequest(instruction="hi"),
            )

    def test_populate_by_name_config(self):
        assert IRPEnvelope.model_config.get("populate_by_name") is True

    def test_irp_default_version(self):
        assert _make_envelope().irp == "1"


# ---------------------------------------------------------------------------
# IRPRequest / IRPInput
# ---------------------------------------------------------------------------


class TestIRPRequest:
    def test_instruction_required(self):
        with pytest.raises(ValidationError):
            IRPRequest()  # type: ignore[call-arg]

    def test_defaults(self):
        req = IRPRequest(instruction="do something")
        assert req.inputs == []
        assert req.response_format == IRPResponseFormat.TEXT

    def test_with_inputs(self):
        req = IRPRequest(
            instruction="process this",
            inputs=[IRPInput(name="doc", value="hello", mime_type="text/plain")],
            response_format=IRPResponseFormat.JSON,
        )
        assert len(req.inputs) == 1
        assert req.inputs[0].name == "doc"
        assert req.response_format == IRPResponseFormat.JSON


# ---------------------------------------------------------------------------
# MimirRelayRequest
# ---------------------------------------------------------------------------


class TestMimirRelayRequest:
    def test_defaults(self):
        req = MimirRelayRequest(instruction="do something")
        assert req.target == "auto"
        assert req.mode == "A"
        assert req.from_agent == "muninn"
        assert req.user_id == "global_user"
        assert req.context == {}
        assert req.policy is None

    def test_valid_mode_A(self):
        req = MimirRelayRequest(instruction="test", mode="A")
        assert req.mode == "A"

    def test_valid_mode_B(self):
        req = MimirRelayRequest(instruction="test", mode="B")
        assert req.mode == "B"

    def test_valid_mode_C(self):
        req = MimirRelayRequest(instruction="test", mode="C")
        assert req.mode == "C"

    def test_mode_is_str_not_enum(self):
        req = MimirRelayRequest(instruction="test", mode="C")
        assert isinstance(req.mode, str)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            MimirRelayRequest(instruction="test", mode="D")

    def test_valid_target_auto(self):
        req = MimirRelayRequest(instruction="test", target="auto")
        assert req.target == "auto"

    def test_valid_target_claude_code(self):
        req = MimirRelayRequest(instruction="test", target="claude_code")
        assert req.target == "claude_code"

    def test_valid_target_codex_cli(self):
        req = MimirRelayRequest(instruction="test", target="codex_cli")
        assert req.target == "codex_cli"

    def test_valid_target_gemini_cli(self):
        req = MimirRelayRequest(instruction="test", target="gemini_cli")
        assert req.target == "gemini_cli"

    def test_invalid_target_raises(self):
        with pytest.raises(ValidationError):
            MimirRelayRequest(instruction="test", target="gpt-4")

    def test_policy_dict_accepted(self):
        req = MimirRelayRequest(
            instruction="test",
            policy={"tools": "forbidden", "network": "allowed"},
        )
        assert req.policy == {"tools": "forbidden", "network": "allowed"}


# ---------------------------------------------------------------------------
# RoutingScore
# ---------------------------------------------------------------------------


class TestRoutingScore:
    def test_availability_score_default_is_1(self):
        """availability_score defaults to 1.0, NOT 0.0."""
        score = RoutingScore(provider=ProviderName.CLAUDE_CODE)
        assert score.availability_score == 1.0

    def test_other_defaults(self):
        score = RoutingScore(provider=ProviderName.CLAUDE_CODE)
        assert score.capability_score == 0.5
        assert score.cost_score == 0.5
        assert score.safety_score == 1.0
        assert score.history_score == 0.5
        assert score.composite == 0.5

    def test_compute_composite_all_ones(self):
        score = RoutingScore(
            provider=ProviderName.CLAUDE_CODE,
            capability_score=1.0,
            availability_score=1.0,
            cost_score=1.0,
            safety_score=1.0,
            history_score=1.0,
        )
        score.compute_composite()
        assert score.composite == pytest.approx(1.0, abs=1e-4)

    def test_compute_composite_all_zeros(self):
        score = RoutingScore(
            provider=ProviderName.CODEX_CLI,
            capability_score=0.0,
            availability_score=0.0,
            cost_score=0.0,
            safety_score=0.0,
            history_score=0.0,
        )
        score.compute_composite()
        assert score.composite == 0.0

    def test_compute_composite_default_weights(self):
        """
        Default weights: cap=0.35, avail=0.25, cost=0.15, safety=0.15, hist=0.10
        With availability=0.0 and all others=1.0:
          composite = 0.35 + 0.0 + 0.15 + 0.15 + 0.10 = 0.75
        """
        score = RoutingScore(
            provider=ProviderName.CLAUDE_CODE,
            capability_score=1.0,
            availability_score=0.0,
            cost_score=1.0,
            safety_score=1.0,
            history_score=1.0,
        )
        score.compute_composite()
        assert score.composite == pytest.approx(0.75, abs=1e-4)

    def test_compute_composite_mutates_composite_in_place(self):
        score = RoutingScore(
            provider=ProviderName.CODEX_CLI,
            capability_score=0.8,
            availability_score=1.0,
            cost_score=0.85,
            safety_score=0.88,
            history_score=0.6,
            composite=0.5,  # placeholder
        )
        score.compute_composite()
        expected = round(
            0.8 * 0.35 + 1.0 * 0.25 + 0.85 * 0.15 + 0.88 * 0.15 + 0.6 * 0.10, 4
        )
        assert score.composite == pytest.approx(expected, abs=1e-4)

    def test_compute_composite_clamps_to_1(self):
        """Weights summing > 1 per score don't exceed composite=1.0."""
        score = RoutingScore(
            provider=ProviderName.GEMINI_CLI,
            capability_score=1.0,
            availability_score=1.0,
            cost_score=1.0,
            safety_score=1.0,
            history_score=1.0,
        )
        score.compute_composite(
            w_cap=0.5, w_avail=0.5, w_cost=0.5, w_safety=0.5, w_hist=0.5
        )
        assert score.composite <= 1.0

    def test_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            RoutingScore(provider=ProviderName.CLAUDE_CODE, capability_score=1.1)

    def test_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            RoutingScore(provider=ProviderName.CLAUDE_CODE, history_score=-0.1)


# ---------------------------------------------------------------------------
# InteropSettings
# ---------------------------------------------------------------------------


class TestInteropSettings:
    def test_defaults(self):
        s = InteropSettings()
        assert s.user_id == "global_user"
        assert s.enabled is True
        assert s.allowed_targets == ["claude_code", "codex_cli", "gemini_cli"]
        assert s.policy_tools == "allowed"
        assert s.hop_max == 2
        assert s.memory_context_enabled is True
        assert s.audit_retention_days == 90

    def test_custom_user_id(self):
        s = InteropSettings(user_id="alice")
        assert s.user_id == "alice"

    def test_enabled_false(self):
        s = InteropSettings(enabled=False)
        assert s.enabled is False

    def test_hop_max_below_min_raises(self):
        with pytest.raises(ValidationError):
            InteropSettings(hop_max=0)

    def test_hop_max_above_max_raises(self):
        with pytest.raises(ValidationError):
            InteropSettings(hop_max=5)

    def test_hop_max_at_boundaries(self):
        assert InteropSettings(hop_max=1).hop_max == 1
        assert InteropSettings(hop_max=4).hop_max == 4

    def test_audit_retention_days_below_min_raises(self):
        with pytest.raises(ValidationError):
            InteropSettings(audit_retention_days=0)

    def test_audit_retention_days_above_max_raises(self):
        with pytest.raises(ValidationError):
            InteropSettings(audit_retention_days=3651)

    def test_policy_tools_valid_values(self):
        for value in ("forbidden", "readonly", "allowed"):
            s = InteropSettings(policy_tools=value)
            assert s.policy_tools == value

    def test_policy_tools_invalid_raises(self):
        with pytest.raises(ValidationError):
            InteropSettings(policy_tools="unrestricted")


# ---------------------------------------------------------------------------
# ReconciliationClaim / ReconciliationResult
# ---------------------------------------------------------------------------


class TestReconciliationResult:
    def test_defaults(self):
        result = ReconciliationResult()
        assert result.consensus_claims == []
        assert result.conflicting_claims == []
        assert result.synthesis == ""
        assert result.escalated is False
        assert result.escalation_reason is None

    def test_with_consensus_claim(self):
        claim = ReconciliationClaim(
            provider=ProviderName.CLAUDE_CODE,
            claim_text="The sky is blue",
            confidence=0.8,
        )
        result = ReconciliationResult(consensus_claims=[claim])
        assert len(result.consensus_claims) == 1
        assert result.consensus_claims[0].provider == ProviderName.CLAUDE_CODE
        assert result.consensus_claims[0].confidence == pytest.approx(0.8)

    def test_escalated_with_reason(self):
        result = ReconciliationResult(
            escalated=True,
            escalation_reason="No consensus reached across providers.",
        )
        assert result.escalated is True
        assert result.escalation_reason is not None
        assert "consensus" in result.escalation_reason.lower()

    def test_conflicting_claims(self):
        claim = ReconciliationClaim(
            provider=ProviderName.CODEX_CLI,
            claim_text="Water is H2O",
            confidence=0.5,
        )
        result = ReconciliationResult(conflicting_claims=[claim])
        assert len(result.conflicting_claims) == 1

    def test_claim_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ReconciliationClaim(
                provider=ProviderName.CLAUDE_CODE,
                claim_text="x" * 30,
                confidence=1.5,  # exceeds le=1.0
            )


# ---------------------------------------------------------------------------
# RelayResult
# ---------------------------------------------------------------------------


class TestRelayResult:
    def test_minimum_required_fields(self):
        result = RelayResult(
            run_id="run-001",
            irp_id="irp-001",
            mode=IRPMode.ADVISORY,
            status=RunStatus.SUCCESS,
        )
        assert result.run_id == "run-001"
        assert result.irp_id == "irp-001"
        assert result.mode == IRPMode.ADVISORY
        assert result.status == RunStatus.SUCCESS

    def test_optional_fields_default_to_none_or_empty(self):
        result = RelayResult(
            run_id="run-001",
            irp_id="irp-001",
            mode=IRPMode.ADVISORY,
            status=RunStatus.SUCCESS,
        )
        assert result.provider is None
        assert result.output == ""
        assert result.reconciliation is None
        assert result.latency_ms == 0
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.redaction_count == 0
        assert result.hop_count == 0
        assert result.error_code is None
        assert result.error_message is None

    def test_failed_result(self):
        result = RelayResult(
            run_id="run-002",
            irp_id="irp-002",
            mode=IRPMode.STRUCTURED,
            status=RunStatus.FAILED,
            error_code="PROVIDER_ERROR",
            error_message="Provider returned exit code 1.",
        )
        assert result.status == RunStatus.FAILED
        assert result.error_code == "PROVIDER_ERROR"
        assert "exit code" in result.error_message

    def test_policy_blocked_result(self):
        result = RelayResult(
            run_id="run-003",
            irp_id="irp-003",
            mode=IRPMode.ADVISORY,
            status=RunStatus.POLICY_BLOCKED,
            error_code="RELAY_DISABLED",
        )
        assert result.status == RunStatus.POLICY_BLOCKED

    def test_missing_status_raises(self):
        with pytest.raises(ValidationError):
            RelayResult(  # type: ignore[call-arg]
                run_id="x",
                irp_id="y",
                mode=IRPMode.ADVISORY,
                # status omitted
            )

    def test_with_reconciliation(self):
        recon = ReconciliationResult(synthesis="## Consensus\n- Fact A")
        result = RelayResult(
            run_id="run-004",
            irp_id="irp-004",
            mode=IRPMode.RECONCILE,
            status=RunStatus.SUCCESS,
            reconciliation=recon,
        )
        assert result.reconciliation is not None
        assert "Consensus" in result.reconciliation.synthesis
