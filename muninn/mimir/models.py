"""
Mimir Interop Relay — Data Models
==================================
Pydantic v2 models for the Interoperability Relay Protocol v1 (IRP/1).

IRP/1 wire format note:
  The JSON envelope uses the field name "from" (reserved Python keyword).
  All Pydantic models use `from_agent` as the Python attribute name and
  serialise to "from" via alias=Field(alias="from").
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IRPMode(str, Enum):
    """Relay execution mode."""
    ADVISORY = "A"       # Advisory text — single-shot, no execution
    STRUCTURED = "B"     # Structured JSON plan — no execution
    RECONCILE = "C"      # Multi-provider reconciliation


class IRPNetworkPolicy(str, Enum):
    ALLOW_ALL = "allow_all"
    LOCAL_ONLY = "local_only"
    DENY_ALL = "deny_all"


class IRPRedactionPolicy(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    OFF = "off"


class IRPResponseFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class ProviderName(str, Enum):
    CLAUDE_CODE = "claude_code"
    CODEX_CLI = "codex_cli"
    GEMINI_CLI = "gemini_cli"
    AUTO = "auto"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    POLICY_BLOCKED = "policy_blocked"


class AuthType(str, Enum):
    API_KEY = "api_key"
    DEVICE_CODE = "device_code"
    ENV_VAR = "env_var"


class ConnectionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class ConsentType(str, Enum):
    RELAY = "relay"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"


class AuditEventType(str, Enum):
    RELAY_START = "relay_start"
    RELAY_COMPLETE = "relay_complete"
    RELAY_FAILED = "relay_failed"
    POLICY_BLOCKED = "policy_blocked"
    HOP_LIMIT_EXCEEDED = "hop_limit_exceeded"
    REDACTION_APPLIED = "redaction_applied"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    ROUTING_DECISION = "routing_decision"
    RECONCILIATION_START = "reconciliation_start"
    RECONCILIATION_COMPLETE = "reconciliation_complete"


# ---------------------------------------------------------------------------
# IRP/1 Sub-objects
# ---------------------------------------------------------------------------

class IRPPolicy(BaseModel):
    """Policy constraints governing the relay call."""
    tools: str = Field(
        default="allowed",
        description="Tool access policy: 'forbidden' | 'readonly' | 'allowed'",
        pattern=r"^(forbidden|readonly|allowed)$",
    )
    network: IRPNetworkPolicy = Field(
        default=IRPNetworkPolicy.DENY_ALL,
        description="Network access policy: 'allow_all' | 'local_only' | 'deny_all'",
    )
    redaction: IRPRedactionPolicy = Field(
        default=IRPRedactionPolicy.BALANCED,
        description="Secret redaction strength applied to prompt and output",
    )
    max_prompt_chars: int = Field(
        default=32_000,
        ge=256,
        le=200_000,
        description="Maximum characters allowed in the forwarded prompt",
    )
    max_output_chars: int = Field(
        default=16_000,
        ge=256,
        le=100_000,
        description="Maximum characters allowed in the relay response",
    )


class IRPHop(BaseModel):
    """Hop tracking for multi-hop relay chains."""
    count: int = Field(default=0, ge=0, description="Current hop number (0-indexed)")
    max: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum hops permitted (hard-capped at 4 for safety)",
    )
    path: List[str] = Field(
        default_factory=list,
        description="Ordered list of agent IDs visited, for loop detection",
    )

    @field_validator("max", mode="before")
    @classmethod
    def cap_max_hops(cls, v: int) -> int:
        """Hard safety cap: never allow more than 4 hops regardless of input."""
        return min(int(v), 4)


class IRPInput(BaseModel):
    """A single named input value for the relay request."""
    name: str
    value: str
    mime_type: str = Field(default="text/plain")


class IRPRequest(BaseModel):
    """The actual relay request payload."""
    instruction: str = Field(
        description="Natural-language instruction forwarded to the target provider",
    )
    inputs: List[IRPInput] = Field(
        default_factory=list,
        description="Named input values for the instruction",
    )
    response_format: IRPResponseFormat = Field(
        default=IRPResponseFormat.TEXT,
        description="Requested output format from the provider",
    )


class IRPTraceEntry(BaseModel):
    """A single entry in the IRP trace log."""
    ts: float = Field(default_factory=time.time)
    hop: int = Field(default=0)
    agent: str
    event: str
    detail: Optional[str] = None


class IRPTrace(BaseModel):
    """Full trace of relay execution for debugging."""
    entries: List[IRPTraceEntry] = Field(default_factory=list)

    def add(self, agent: str, event: str, detail: Optional[str] = None, hop: int = 0) -> None:
        self.entries.append(IRPTraceEntry(agent=agent, event=event, detail=detail, hop=hop))


# ---------------------------------------------------------------------------
# IRP/1 Envelope (wire format)
# ---------------------------------------------------------------------------

class IRPEnvelope(BaseModel):
    """
    IRP/1 message envelope.

    Field `from_agent` maps to JSON key "from" (Python reserved keyword).
    Use model.model_dump(by_alias=True) for wire serialisation.
    """
    model_config = {"populate_by_name": True}

    irp: str = Field(default="1", description="Protocol version; always '1' for IRP/1")
    id: str = Field(description="ULID request identifier")
    ts: float = Field(default_factory=time.time, description="Unix epoch timestamp")
    from_agent: str = Field(alias="from", description="Originating agent identifier")
    to: str = Field(description="Target provider identifier")
    mode: IRPMode = Field(default=IRPMode.ADVISORY)
    hop: IRPHop = Field(default_factory=IRPHop)
    policy: IRPPolicy = Field(default_factory=IRPPolicy)
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context passed through to the target",
    )
    request: IRPRequest
    trace: IRPTrace = Field(default_factory=IRPTrace)

    def wire_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict using wire field names (alias=True)."""
        return self.model_dump(by_alias=True, mode="json")


# ---------------------------------------------------------------------------
# Provider results
# ---------------------------------------------------------------------------

class ProviderResult(BaseModel):
    """Raw result from a single provider execution."""
    provider: ProviderName
    raw_output: str
    parsed: Optional[Dict[str, Any]] = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    redaction_count: int = 0
    error: Optional[str] = None
    available: bool = True


class ReconciliationClaim(BaseModel):
    """A single extracted claim from a provider response."""
    provider: ProviderName
    claim_text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    memory_supported: bool = False
    memory_ids: List[str] = Field(default_factory=list)


class ReconciliationResult(BaseModel):
    """Output of Mode C reconciliation across multiple providers."""
    consensus_claims: List[ReconciliationClaim] = Field(default_factory=list)
    conflicting_claims: List[ReconciliationClaim] = Field(default_factory=list)
    synthesis: str = ""
    escalated: bool = False
    escalation_reason: Optional[str] = None


class RelayResult(BaseModel):
    """Final result returned to the caller of mimir_relay."""
    run_id: str
    irp_id: str
    mode: IRPMode
    provider: Optional[ProviderName] = None
    status: RunStatus
    output: str = ""
    reconciliation: Optional[ReconciliationResult] = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    redaction_count: int = 0
    hop_count: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    trace: IRPTrace = Field(default_factory=IRPTrace)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RoutingScore(BaseModel):
    """Composite routing score for a candidate provider."""
    provider: ProviderName
    capability_score: float = Field(ge=0.0, le=1.0, default=0.5)
    availability_score: float = Field(ge=0.0, le=1.0, default=1.0)
    cost_score: float = Field(ge=0.0, le=1.0, default=0.5)
    safety_score: float = Field(ge=0.0, le=1.0, default=1.0)
    history_score: float = Field(ge=0.0, le=1.0, default=0.5)
    composite: float = Field(ge=0.0, le=1.0, default=0.5)

    def compute_composite(
        self,
        w_cap: float = 0.35,
        w_avail: float = 0.25,
        w_cost: float = 0.15,
        w_safety: float = 0.15,
        w_hist: float = 0.10,
    ) -> None:
        """Recompute composite score with configurable weights."""
        total = (
            self.capability_score * w_cap
            + self.availability_score * w_avail
            + self.cost_score * w_cost
            + self.safety_score * w_safety
            + self.history_score * w_hist
        )
        self.composite = round(min(max(total, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Audit / persistence
# ---------------------------------------------------------------------------

class AuditEvent(BaseModel):
    """An immutable audit event stored in interop_audit_events."""
    id: str
    run_id: str
    ts: float = Field(default_factory=time.time)
    event_type: AuditEventType
    provider: str
    status: RunStatus
    payload: Optional[Dict[str, Any]] = None


class RunRecord(BaseModel):
    """Record persisted in interop_runs table."""
    run_id: str
    irp_id: str
    user_id: str = "global_user"
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    mode: IRPMode
    selected_provider: Optional[ProviderName] = None
    status: RunStatus = RunStatus.PENDING
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    prompt_hash: Optional[str] = None
    redaction_count: int = 0


class ProviderConnection(BaseModel):
    """Row in interop_provider_connections."""
    id: str
    user_id: str = "global_user"
    provider: ProviderName
    auth_type: AuthType
    status: ConnectionStatus = ConnectionStatus.ACTIVE
    scopes: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    last_verified_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InteropSettings(BaseModel):
    """User-configurable interop settings (stored in interop_settings table)."""
    user_id: str = "global_user"
    enabled: bool = True
    allowed_targets: List[str] = Field(
        default_factory=lambda: ["claude_code", "codex_cli", "gemini_cli"]
    )
    policy_tools: str = Field(
        default="allowed",
        pattern=r"^(forbidden|readonly|allowed)$",
    )
    hop_max: int = Field(default=2, ge=1, le=4)
    memory_context_enabled: bool = True
    audit_retention_days: int = Field(default=90, ge=1, le=3650)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MCP tool request / response
# ---------------------------------------------------------------------------

class MimirRelayRequest(BaseModel):
    """
    Input schema for the `mimir_relay` MCP tool.

    All fields match the JSON inputSchema defined in definitions.py.
    """
    instruction: str = Field(
        description="Natural-language instruction to relay to the target provider",
    )
    target: str = Field(
        default="auto",
        description="Target provider: 'auto' | 'claude_code' | 'codex_cli' | 'gemini_cli'",
    )
    mode: str = Field(
        default="A",
        description="Relay mode: 'A' (advisory) | 'B' (structured) | 'C' (reconcile)",
        pattern=r"^(A|B|C)$",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context forwarded to the relay target",
    )
    policy: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional policy overrides (tools, network, redaction, max_chars)",
    )
    from_agent: str = Field(
        default="muninn",
        description="Identifier of the calling agent (used in IRP/1 'from' field)",
    )
    user_id: str = Field(
        default="global_user",
        description="User identifier for audit and consent tracking",
    )

    @model_validator(mode="after")
    def validate_target_enum(self) -> "MimirRelayRequest":
        valid = {"auto", "claude_code", "codex_cli", "gemini_cli"}
        if self.target not in valid:
            raise ValueError(f"target must be one of {valid}, got '{self.target}'")
        return self
