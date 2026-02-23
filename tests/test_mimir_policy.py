"""
Tests for muninn.mimir.policy — PolicyEngine and PolicyError.

Covers:
  - PolicyError attributes
  - validate_hop_limit / validate_hop_path
  - redact / redact_prompt / redact_output (all levels)
  - validate_prompt_size / validate_output_size
  - check_no_tools_result
  - validate_interop_enabled / validate_allowed_target
  - hash_prompt
  - build_policy
"""

from __future__ import annotations

import pytest

from muninn.mimir.models import (
    IRPEnvelope,
    IRPHop,
    IRPMode,
    IRPPolicy,
    IRPRedactionPolicy,
    IRPRequest,
    ProviderName,
    ProviderResult,
)
from muninn.mimir.policy import PolicyEngine, PolicyError


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_envelope(
    *,
    instruction: str = "hello world",
    hop: IRPHop | None = None,
    policy: IRPPolicy | None = None,
) -> IRPEnvelope:
    """Create a minimal valid IRPEnvelope for testing."""
    return IRPEnvelope(
        id="01HWTEST00000000000000001",
        from_agent="muninn",
        to="auto",
        mode=IRPMode.ADVISORY,
        hop=hop or IRPHop(),
        policy=policy or IRPPolicy(),
        request=IRPRequest(instruction=instruction),
    )


def _make_provider_result(
    *,
    provider: ProviderName = ProviderName.CLAUDE_CODE,
    raw_output: str = "some output",
    parsed: dict | None = None,
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        raw_output=raw_output,
        parsed=parsed,
    )


# ── PolicyError ───────────────────────────────────────────────────────────────


class TestPolicyError:
    def test_attributes_set_correctly(self):
        err = PolicyError(code="TEST_CODE", message="test message")
        assert err.code == "TEST_CODE"
        assert err.message == "test message"

    def test_str_representation_is_message(self):
        err = PolicyError(code="X", message="boom")
        assert str(err) == "boom"

    def test_is_exception_subclass(self):
        assert issubclass(PolicyError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(PolicyError) as exc_info:
            raise PolicyError(code="ERR", message="something went wrong")
        assert exc_info.value.code == "ERR"
        assert exc_info.value.message == "something went wrong"


# ── validate_hop_limit ────────────────────────────────────────────────────────


class TestValidateHopLimit:
    def test_count_zero_max_two_passes(self):
        env = _make_envelope(hop=IRPHop(count=0, max=2))
        PolicyEngine.validate_hop_limit(env)  # no exception

    def test_count_one_below_max_passes(self):
        env = _make_envelope(hop=IRPHop(count=1, max=2))
        PolicyEngine.validate_hop_limit(env)  # no exception

    def test_count_equal_max_raises(self):
        env = _make_envelope(hop=IRPHop(count=2, max=2))
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_hop_limit(env)
        assert exc_info.value.code == "HOP_LIMIT_EXCEEDED"

    def test_count_exceeds_max_raises(self):
        env = _make_envelope(hop=IRPHop(count=3, max=2))
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_hop_limit(env)
        assert exc_info.value.code == "HOP_LIMIT_EXCEEDED"
        assert "count=3" in exc_info.value.message
        assert "max=2" in exc_info.value.message

    def test_max_4_at_3_passes(self):
        env = _make_envelope(hop=IRPHop(count=3, max=4))
        PolicyEngine.validate_hop_limit(env)

    def test_max_4_at_4_raises(self):
        env = _make_envelope(hop=IRPHop(count=4, max=4))
        with pytest.raises(PolicyError):
            PolicyEngine.validate_hop_limit(env)


# ── validate_hop_path ─────────────────────────────────────────────────────────


class TestValidateHopPath:
    def test_empty_path_passes(self):
        env = _make_envelope(hop=IRPHop(path=[]))
        PolicyEngine.validate_hop_path(env, "claude_code")  # no exception

    def test_new_agent_not_in_path_passes(self):
        env = _make_envelope(hop=IRPHop(path=["muninn", "claude_code"]))
        PolicyEngine.validate_hop_path(env, "gemini_cli")  # no exception

    def test_agent_already_in_path_raises(self):
        env = _make_envelope(hop=IRPHop(path=["muninn", "claude_code"]))
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_hop_path(env, "claude_code")
        assert exc_info.value.code == "HOP_LOOP_DETECTED"
        assert "claude_code" in exc_info.value.message

    def test_origin_agent_loop_detected(self):
        env = _make_envelope(hop=IRPHop(path=["muninn"]))
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_hop_path(env, "muninn")
        assert exc_info.value.code == "HOP_LOOP_DETECTED"


# ── redact ────────────────────────────────────────────────────────────────────


class TestRedact:
    def test_off_level_returns_text_unchanged(self):
        text = "sk-abcdefghijklmnopqrstuvwxyz1234"
        result, count = PolicyEngine.redact(text, IRPRedactionPolicy.OFF)
        assert result == text
        assert count == 0

    def test_off_level_zero_redactions(self):
        text = "AKIAIOSFODNN7EXAMPLE secret key here"
        _, count = PolicyEngine.redact(text, IRPRedactionPolicy.OFF)
        assert count == 0

    def test_balanced_redacts_sk_api_key(self):
        text = "authenticate with sk-abcdefghijklmnopqrstuvwxyz1234"
        result, count = PolicyEngine.redact(text, IRPRedactionPolicy.BALANCED)
        assert count >= 1
        assert "[REDACTED_API_KEY]" in result

    def test_balanced_redacts_pem_private_key(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA1234abcd\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result, count = PolicyEngine.redact(pem, IRPRedactionPolicy.BALANCED)
        assert count >= 1
        assert "[REDACTED_PRIVATE_KEY]" in result

    def test_balanced_redacts_jwt(self):
        jwt = (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36"
        )
        result, count = PolicyEngine.redact(jwt, IRPRedactionPolicy.BALANCED)
        assert count >= 1
        assert "[REDACTED_JWT]" in result

    def test_balanced_redacts_github_token(self):
        token = "ghp_" + "a" * 36
        result, count = PolicyEngine.redact(token, IRPRedactionPolicy.BALANCED)
        assert count >= 1
        assert "[REDACTED_GITHUB_TOKEN]" in result

    def test_strict_redacts_aws_access_key(self):
        text = "AKIAIOSFODNN7EXAMPLE is the access key"
        result, count = PolicyEngine.redact(text, IRPRedactionPolicy.STRICT)
        assert count >= 1
        assert "[REDACTED_AWS_KEY]" in result

    def test_strict_redacts_password_assignment(self):
        text = 'password = "supersecretpassword123"'
        result, count = PolicyEngine.redact(text, IRPRedactionPolicy.STRICT)
        assert count >= 1
        assert "supersecretpassword" not in result

    def test_strict_redacts_connection_string(self):
        text = "postgresql://admin:p@ssw0rd@db.example.com/mydb"
        result, count = PolicyEngine.redact(text, IRPRedactionPolicy.STRICT)
        assert count >= 1
        assert "p@ssw0rd" not in result

    def test_clean_text_returns_zero_count(self):
        text = "Please summarise the quarterly performance metrics."
        result, count = PolicyEngine.redact(text, IRPRedactionPolicy.STRICT)
        assert count == 0
        assert result == text

    def test_multiple_secrets_counted(self):
        text = (
            "sk-aaaaaaaaaaaaaaaaaaaaaaaaa "
            "sk-bbbbbbbbbbbbbbbbbbbbbbbbb"
        )
        _, count = PolicyEngine.redact(text, IRPRedactionPolicy.BALANCED)
        assert count >= 2


# ── redact_prompt ─────────────────────────────────────────────────────────────


class TestRedactPrompt:
    def test_returns_new_envelope_object(self):
        env = _make_envelope(
            instruction="sk-abcdefghijklmnopqrstuvwxyz1234",
            policy=IRPPolicy(redaction=IRPRedactionPolicy.BALANCED),
        )
        new_env, _ = PolicyEngine.redact_prompt(env)
        assert new_env is not env

    def test_original_envelope_not_mutated(self):
        original = "sk-originalkey12345678901234567890"
        env = _make_envelope(
            instruction=original,
            policy=IRPPolicy(redaction=IRPRedactionPolicy.STRICT),
        )
        PolicyEngine.redact_prompt(env)
        assert env.request.instruction == original

    def test_redacts_api_key_in_instruction(self):
        env = _make_envelope(
            instruction="call api with sk-abcdefghijklmnopqrstuvwxyz1234",
            policy=IRPPolicy(redaction=IRPRedactionPolicy.BALANCED),
        )
        new_env, count = PolicyEngine.redact_prompt(env)
        assert count >= 1
        assert "[REDACTED" in new_env.request.instruction

    def test_off_policy_no_redaction(self):
        key = "sk-abcdefghijklmnopqrstuvwxyz1234"
        env = _make_envelope(
            instruction=key,
            policy=IRPPolicy(redaction=IRPRedactionPolicy.OFF),
        )
        new_env, count = PolicyEngine.redact_prompt(env)
        assert count == 0
        assert key in new_env.request.instruction

    def test_clean_instruction_unchanged(self):
        instruction = "summarise the document"
        env = _make_envelope(
            instruction=instruction,
            policy=IRPPolicy(redaction=IRPRedactionPolicy.STRICT),
        )
        new_env, count = PolicyEngine.redact_prompt(env)
        assert count == 0
        assert new_env.request.instruction == instruction


# ── redact_output ─────────────────────────────────────────────────────────────


class TestRedactOutput:
    def test_delegates_to_redact(self):
        text = "sk-secretkeyabcdefghijklmnopqrstu"
        result, count = PolicyEngine.redact_output(text, IRPRedactionPolicy.BALANCED)
        assert count >= 1
        assert "[REDACTED" in result

    def test_off_level_unchanged(self):
        text = "sk-secretkeyabcdefghijklmnopqrstu"
        result, count = PolicyEngine.redact_output(text, IRPRedactionPolicy.OFF)
        assert result == text
        assert count == 0


# ── validate_prompt_size ──────────────────────────────────────────────────────


class TestValidatePromptSize:
    def test_short_instruction_passes(self):
        env = _make_envelope(
            instruction="short",
            policy=IRPPolicy(max_prompt_chars=1000),
        )
        PolicyEngine.validate_prompt_size(env)

    def test_exactly_at_limit_passes(self):
        # min valid max_prompt_chars is 256; use 256 chars exactly
        env = _make_envelope(
            instruction="x" * 256,
            policy=IRPPolicy(max_prompt_chars=256),
        )
        PolicyEngine.validate_prompt_size(env)  # length == max → ok

    def test_one_over_limit_raises(self):
        env = _make_envelope(
            instruction="x" * 257,
            policy=IRPPolicy(max_prompt_chars=256),
        )
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_prompt_size(env)
        assert exc_info.value.code == "PROMPT_TOO_LARGE"
        assert "257" in exc_info.value.message

    def test_error_message_includes_limit(self):
        env = _make_envelope(
            instruction="x" * 300,
            policy=IRPPolicy(max_prompt_chars=256),
        )
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_prompt_size(env)
        assert "256" in exc_info.value.message


# ── validate_output_size ──────────────────────────────────────────────────────


class TestValidateOutputSize:
    def test_short_output_passes(self):
        policy = IRPPolicy(max_output_chars=1000)
        PolicyEngine.validate_output_size("short", policy)

    def test_exactly_at_limit_passes(self):
        # min valid max_output_chars is 256
        policy = IRPPolicy(max_output_chars=256)
        PolicyEngine.validate_output_size("x" * 256, policy)

    def test_one_over_limit_raises(self):
        policy = IRPPolicy(max_output_chars=256)
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_output_size("x" * 257, policy)
        assert exc_info.value.code == "OUTPUT_TOO_LARGE"

    def test_empty_output_passes(self):
        policy = IRPPolicy(max_output_chars=256)
        PolicyEngine.validate_output_size("", policy)


# ── check_no_tools_result ─────────────────────────────────────────────────────


class TestCheckNoToolsResult:
    def test_none_parsed_passes(self):
        result = _make_provider_result(parsed=None)
        PolicyEngine.check_no_tools_result(result)

    def test_empty_dict_parsed_passes(self):
        result = _make_provider_result(parsed={})
        PolicyEngine.check_no_tools_result(result)

    def test_zero_tool_calls_passes(self):
        result = _make_provider_result(
            parsed={"stats": {"tools": {"totalCalls": 0}}}
        )
        PolicyEngine.check_no_tools_result(result)

    def test_one_tool_call_raises(self):
        result = _make_provider_result(
            parsed={"stats": {"tools": {"totalCalls": 1}}}
        )
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.check_no_tools_result(result)
        assert exc_info.value.code == "TOOL_USAGE_VIOLATION"

    def test_multiple_tool_calls_raises(self):
        result = _make_provider_result(
            parsed={"stats": {"tools": {"totalCalls": 5}}}
        )
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.check_no_tools_result(result)
        assert "5" in exc_info.value.message

    def test_missing_tools_stats_passes(self):
        result = _make_provider_result(
            parsed={"stats": {"latency_ms": 100}}
        )
        PolicyEngine.check_no_tools_result(result)

    def test_missing_stats_key_passes(self):
        result = _make_provider_result(parsed={"output": "some text"})
        PolicyEngine.check_no_tools_result(result)


# ── validate_interop_enabled ──────────────────────────────────────────────────


class TestValidateInteropEnabled:
    def test_enabled_true_passes(self):
        PolicyEngine.validate_interop_enabled(True)

    def test_enabled_false_raises(self):
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_interop_enabled(False)
        assert exc_info.value.code == "INTEROP_DISABLED"
        assert "disabled" in exc_info.value.message.lower()


# ── validate_allowed_target ───────────────────────────────────────────────────


class TestValidateAllowedTarget:
    def test_auto_always_passes_regardless_of_list(self):
        PolicyEngine.validate_allowed_target("auto", ["claude_code"])

    def test_target_in_list_passes(self):
        PolicyEngine.validate_allowed_target("claude_code", ["claude_code", "codex_cli"])

    def test_target_not_in_list_raises(self):
        with pytest.raises(PolicyError) as exc_info:
            PolicyEngine.validate_allowed_target(
                "gemini_cli", ["claude_code", "codex_cli"]
            )
        assert exc_info.value.code == "TARGET_NOT_ALLOWED"
        assert "gemini_cli" in exc_info.value.message

    def test_empty_allowed_list_raises_for_specific_target(self):
        with pytest.raises(PolicyError):
            PolicyEngine.validate_allowed_target("claude_code", [])

    def test_all_providers_allowed(self):
        all_targets = ["claude_code", "codex_cli", "gemini_cli"]
        for target in all_targets:
            PolicyEngine.validate_allowed_target(target, all_targets)


# ── hash_prompt ───────────────────────────────────────────────────────────────


class TestHashPrompt:
    def test_returns_16_char_string(self):
        result = PolicyEngine.hash_prompt("hello")
        assert len(result) == 16

    def test_returns_lowercase_hex(self):
        result = PolicyEngine.hash_prompt("hello")
        assert all(c in "0123456789abcdef" for c in result)

    def test_is_deterministic(self):
        text = "deterministic test input"
        assert PolicyEngine.hash_prompt(text) == PolicyEngine.hash_prompt(text)

    def test_different_inputs_produce_different_hashes(self):
        assert PolicyEngine.hash_prompt("aaa") != PolicyEngine.hash_prompt("bbb")

    def test_empty_string_hashed(self):
        result = PolicyEngine.hash_prompt("")
        assert len(result) == 16

    def test_unicode_input(self):
        result = PolicyEngine.hash_prompt("こんにちは世界")
        assert len(result) == 16


# ── build_policy ──────────────────────────────────────────────────────────────


class TestBuildPolicy:
    def test_no_args_returns_default_policy(self):
        policy = PolicyEngine.build_policy()
        assert policy.tools == "allowed"
        assert policy.redaction == IRPRedactionPolicy.BALANCED

    def test_defaults_applied(self):
        policy = PolicyEngine.build_policy(
            defaults={"tools": "forbidden", "max_prompt_chars": 5000}
        )
        assert policy.tools == "forbidden"
        assert policy.max_prompt_chars == 5000

    def test_overrides_win_over_defaults(self):
        policy = PolicyEngine.build_policy(
            defaults={"tools": "allowed"},
            overrides={"tools": "forbidden"},
        )
        assert policy.tools == "forbidden"

    def test_default_and_override_keys_merged(self):
        policy = PolicyEngine.build_policy(
            defaults={"max_prompt_chars": 5000},
            overrides={"max_output_chars": 8000},
        )
        assert policy.max_prompt_chars == 5000
        assert policy.max_output_chars == 8000

    def test_unknown_keys_silently_ignored(self):
        policy = PolicyEngine.build_policy(defaults={"unknown_key": "value"})
        assert isinstance(policy, IRPPolicy)

    def test_redaction_override(self):
        policy = PolicyEngine.build_policy(
            overrides={"redaction": "strict"}
        )
        assert policy.redaction == IRPRedactionPolicy.STRICT

    def test_none_defaults_and_overrides(self):
        policy = PolicyEngine.build_policy(defaults=None, overrides=None)
        assert isinstance(policy, IRPPolicy)
