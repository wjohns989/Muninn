"""
Mimir Interop Relay — Policy Engine
=====================================
Enforces IRP/1 security constraints:
  - Hop limit hard cap (max 4, default 2)
  - Secret redaction (three levels: strict / balanced / off)
  - No-tools policy validation
  - Workspace trust checks

Redaction patterns cover the most common secret shapes seen in agent
workspaces: API keys, PEM blocks, env-var assignments, connection strings,
JWT tokens, OAuth bearer tokens, and GitHub/npm/PyPI credentials.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional, Tuple

from .models import IRPEnvelope, IRPPolicy, IRPRedactionPolicy, ProviderResult

logger = logging.getLogger("Muninn.Mimir.policy")

# ---------------------------------------------------------------------------
# Redaction pattern registry
# ---------------------------------------------------------------------------

# Each entry is (pattern, replacement_label).
# Patterns are compiled once at import time.

# ---------- Strict patterns (catches everything that looks secret-like) ----
_STRICT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # OpenAI / Anthropic / generic API keys
    (re.compile(r"sk-[A-Za-z0-9\-_]{20,}", re.IGNORECASE | re.MULTILINE), "[REDACTED_API_KEY]"),
    # Bearer tokens in Authorization headers
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.~+/]+=*", re.IGNORECASE | re.MULTILINE), "bearer [REDACTED_TOKEN]"),
    # Generic API key assignments  (api_key = "...", API_KEY=...)
    (
        re.compile(
            r"(?i)(api[_\-]?key|apikey|api[_\-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_.~+/]{16,}['\"]?",
            re.IGNORECASE | re.MULTILINE,
        ),
        r"\1=[REDACTED_API_KEY]",
    ),
    # AWS-style access keys
    (
        re.compile(r"(?<![A-Z0-9])(AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}(?![A-Z0-9])", re.IGNORECASE | re.MULTILINE),
        "[REDACTED_AWS_KEY]",
    ),
    # AWS secret access keys
    (
        re.compile(
            r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[:=]\s*[A-Za-z0-9/+]{40}", re.IGNORECASE | re.MULTILINE
        ),
        "[REDACTED_AWS_SECRET]",
    ),
    # PEM private keys
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.MULTILINE
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # PEM certificates (full blocks)
    (
        re.compile(r"-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----", re.IGNORECASE | re.MULTILINE),
        "[REDACTED_CERTIFICATE]",
    ),
    # JWT tokens (header.payload.signature)
    (
        re.compile(r"eyJ[A-Za-z0-9\-_]{2,}\.eyJ[A-Za-z0-9\-_]{2,}\.[A-Za-z0-9\-_]{2,}", re.IGNORECASE | re.MULTILINE),
        "[REDACTED_JWT]",
    ),
    # GitHub tokens (classic + fine-grained)
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36}", re.IGNORECASE | re.MULTILINE), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}", re.IGNORECASE | re.MULTILINE), "[REDACTED_GITHUB_PAT]"),
    # npm auth tokens
    (re.compile(r"npm_[A-Za-z0-9]{36}", re.IGNORECASE | re.MULTILINE), "[REDACTED_NPM_TOKEN]"),
    # PyPI tokens
    (re.compile(r"pypi-[A-Za-z0-9\-_]{70,}", re.IGNORECASE | re.MULTILINE), "[REDACTED_PYPI_TOKEN]"),
    # Generic password assignments
    (
        re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?", re.IGNORECASE | re.MULTILINE),
        r"\1=[REDACTED_PASSWORD]",
    ),
    # Connection strings with embedded credentials
    (
        re.compile(r"(?i)(postgresql|mysql|mongodb|redis)://[^:\s]+:[^@\s]+@[^\s]+", re.IGNORECASE | re.MULTILINE),
        r"\1://[REDACTED_CONN_STRING]",
    ),
    # SSH private keys (raw header detection)
    (
        re.compile(
            r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----",
            re.IGNORECASE | re.MULTILINE,
        ),
        "[REDACTED_SSH_PRIVATE_KEY]",
    ),
    # Generic 40+ hex strings that look like secrets (SHA-style tokens)
    (
        re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40,}(?![0-9a-fA-F])", re.IGNORECASE | re.MULTILINE),
        "[REDACTED_HEX_SECRET]",
    ),
]

# ---------- Balanced patterns (high-signal patterns only) ------------------
_BALANCED_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9\-_]{20,}", re.IGNORECASE | re.MULTILINE), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.~+/]+=*", re.IGNORECASE | re.MULTILINE), "bearer [REDACTED_TOKEN]"),
    (
        re.compile(r"(?<![A-Z0-9])(AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}(?![A-Z0-9])", re.IGNORECASE | re.MULTILINE),
        "[REDACTED_AWS_KEY]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.MULTILINE
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(
            r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----",
            re.IGNORECASE | re.MULTILINE,
        ),
        "[REDACTED_SSH_PRIVATE_KEY]",
    ),
    (
        re.compile(r"eyJ[A-Za-z0-9\-_]{2,}\.eyJ[A-Za-z0-9\-_]{2,}\.[A-Za-z0-9\-_]{2,}", re.IGNORECASE | re.MULTILINE),
        "[REDACTED_JWT]",
    ),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36}", re.IGNORECASE | re.MULTILINE), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}", re.IGNORECASE | re.MULTILINE), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"pypi-[A-Za-z0-9\-_]{70,}", re.IGNORECASE | re.MULTILINE), "[REDACTED_PYPI_TOKEN]"),
]


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyError(Exception):
    """Raised when a relay request violates a configured policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PolicyEngine:
    """
    Stateless policy enforcer for IRP/1 relay calls.

    All public methods are pure functions — they either return a value or
    raise PolicyError.  No side effects outside of logging.
    """

    # ------------------------------------------------------------------
    # Hop limit
    # ------------------------------------------------------------------

    @staticmethod
    def validate_hop_limit(envelope: IRPEnvelope) -> None:
        """Raise PolicyError if hop.count >= hop.max."""
        if envelope.hop.count >= envelope.hop.max:
            raise PolicyError(
                code="HOP_LIMIT_EXCEEDED",
                message=(f"Hop limit reached: count={envelope.hop.count} >= max={envelope.hop.max}. Relay aborted."),
            )

    @staticmethod
    def validate_hop_path(envelope: IRPEnvelope, next_agent: str) -> None:
        """Raise PolicyError if next_agent already appears in hop.path (loop detection)."""
        if next_agent in envelope.hop.path:
            raise PolicyError(
                code="HOP_LOOP_DETECTED",
                message=(f"Loop detected: '{next_agent}' already in relay path {envelope.hop.path}. Relay aborted."),
            )

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    @staticmethod
    def redact(text: str, level: IRPRedactionPolicy) -> Tuple[str, int]:
        """
        Apply redaction to `text` according to `level`.

        Returns (redacted_text, redaction_count).
        """
        if level == IRPRedactionPolicy.OFF:
            return text, 0

        patterns = _STRICT_PATTERNS if level == IRPRedactionPolicy.STRICT else _BALANCED_PATTERNS

        count = 0
        result = text
        for pattern, replacement in patterns:
            new_result, n = pattern.subn(replacement, result)
            if n > 0:
                count += n
                result = new_result

        if count:
            logger.info("Redacted %d secret(s) at level=%s", count, level.value)

        return result, count

    @staticmethod
    def redact_prompt(envelope: IRPEnvelope) -> Tuple[IRPEnvelope, int]:
        """
        Return a copy of the envelope with the instruction text redacted.
        Does NOT mutate the original envelope.
        """
        level = envelope.policy.redaction
        instruction = envelope.request.instruction
        redacted_instruction, count = PolicyEngine.redact(instruction, level)

        # Deep copy via model_dump → reconstruct
        data = envelope.model_dump(by_alias=True)
        data["request"]["instruction"] = redacted_instruction
        new_envelope = IRPEnvelope.model_validate(data)
        return new_envelope, count

    @staticmethod
    def redact_output(text: str, level: IRPRedactionPolicy) -> Tuple[str, int]:
        """Apply output redaction after receiving a provider response."""
        return PolicyEngine.redact(text, level)

    # ------------------------------------------------------------------
    # Size limits
    # ------------------------------------------------------------------

    @staticmethod
    def validate_prompt_size(envelope: IRPEnvelope) -> None:
        """Raise PolicyError if the instruction exceeds max_prompt_chars."""
        length = len(envelope.request.instruction)
        if length > envelope.policy.max_prompt_chars:
            raise PolicyError(
                code="PROMPT_TOO_LARGE",
                message=(f"Instruction length {length} exceeds policy limit {envelope.policy.max_prompt_chars} chars."),
            )

    @staticmethod
    def validate_output_size(text: str, policy: IRPPolicy) -> None:
        """Raise PolicyError if output text exceeds max_output_chars."""
        if len(text) > policy.max_output_chars:
            raise PolicyError(
                code="OUTPUT_TOO_LARGE",
                message=(f"Provider output ({len(text)} chars) exceeds policy limit {policy.max_output_chars} chars."),
            )

    # ------------------------------------------------------------------
    # No-tools enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def check_no_tools_result(result: ProviderResult) -> None:
        """
        Raise PolicyError if a provider result contains tool usage when
        the policy forbids tools.

        The `parsed` field may contain a 'stats' sub-object with
        'tools.totalCalls' (Gemini CLI pattern).
        """
        if not result.parsed:
            return
        stats = result.parsed.get("stats", {})
        tools_stats = stats.get("tools", {})
        total_calls = tools_stats.get("totalCalls", 0)
        if total_calls > 0:
            raise PolicyError(
                code="TOOL_USAGE_VIOLATION",
                message=(f"Provider '{result.provider}' used {total_calls} tool call(s) but policy.tools='forbidden'."),
            )

    # ------------------------------------------------------------------
    # Consent / enabled check
    # ------------------------------------------------------------------

    @staticmethod
    def validate_interop_enabled(enabled: bool) -> None:
        """Raise PolicyError if interop is globally disabled."""
        if not enabled:
            raise PolicyError(
                code="INTEROP_DISABLED",
                message="Mimir interop relay is disabled. Enable via settings.",
            )

    @staticmethod
    def validate_allowed_target(target: str, allowed_targets: list[str]) -> None:
        """Raise PolicyError if the requested target is not in the allow-list."""
        if target != "auto" and target not in allowed_targets:
            raise PolicyError(
                code="TARGET_NOT_ALLOWED",
                message=(f"Target '{target}' is not in the allowed targets list: {allowed_targets}."),
            )

    # ------------------------------------------------------------------
    # Prompt hashing (for audit / dedup)
    # ------------------------------------------------------------------

    @staticmethod
    def hash_prompt(text: str) -> str:
        """Return a short SHA-256 hex digest of the (possibly redacted) prompt."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Build a complete policy from overrides dict
    # ------------------------------------------------------------------

    @staticmethod
    def build_policy(
        defaults: Optional[dict] = None,
        overrides: Optional[dict] = None,
    ) -> IRPPolicy:
        """
        Merge `overrides` on top of `defaults` and return an IRPPolicy.
        Unknown keys in either dict are silently ignored.
        """
        merged: dict = {}
        if defaults:
            for k in ("tools", "network", "redaction", "max_prompt_chars", "max_output_chars"):
                if k in defaults:
                    merged[k] = defaults[k]
        if overrides:
            for k in ("tools", "network", "redaction", "max_prompt_chars", "max_output_chars"):
                if k in overrides:
                    merged[k] = overrides[k]
        return IRPPolicy.model_validate(merged) if merged else IRPPolicy()
