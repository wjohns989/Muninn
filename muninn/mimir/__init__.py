"""
Mimir — Interop Relay MCP Module
==================================
Norse mythology: Mímir is the guardian of the well of wisdom (Mímisbrunnr),
through which Odin gained his knowledge. Here, Mimir acts as the bridge that
connects the Muninn memory system to external AI agents and providers.

Public surface area
-------------------
  MimirRelay          — high-level relay orchestrator (use this directly)
  MemoryAwareRouter   — provider scoring and selection
  Reconciler          — Mode C multi-provider reconciliation
  PolicyEngine        — IRP/1 policy enforcement
  MimirStore          — persistent run/audit/connection storage
  MIMIR_DDL_STATEMENTS — SQLite DDL list for sqlite_metadata._initialize()

All IRP/1 data models are re-exported from .models.
"""

from .models import (
    IRPEnvelope,
    IRPHop,
    IRPInput,
    IRPMode,
    IRPNetworkPolicy,
    IRPPolicy,
    IRPRedactionPolicy,
    IRPRequest,
    IRPResponseFormat,
    IRPTrace,
    IRPTraceEntry,
    MimirRelayRequest,
    ProviderName,
    ProviderResult,
    ReconciliationClaim,
    ReconciliationResult,
    RelayResult,
    RoutingScore,
    RunStatus,
)
from .policy import PolicyEngine, PolicyError
from .reconcile import Reconciler
from .relay import MimirRelay
from .routing import MemoryAwareRouter, RoutingError
from .store import MIMIR_DDL_STATEMENTS, MimirStore

__all__ = [
    # Models
    "IRPEnvelope",
    "IRPHop",
    "IRPInput",
    "IRPMode",
    "IRPNetworkPolicy",
    "IRPPolicy",
    "IRPRedactionPolicy",
    "IRPRequest",
    "IRPResponseFormat",
    "IRPTrace",
    "IRPTraceEntry",
    "MimirRelayRequest",
    "ProviderName",
    "ProviderResult",
    "ReconciliationClaim",
    "ReconciliationResult",
    "RelayResult",
    "RoutingScore",
    "RunStatus",
    # Engine
    "MimirRelay",
    "PolicyEngine",
    "PolicyError",
    "Reconciler",
    "MemoryAwareRouter",
    "RoutingError",
    # Store
    "MimirStore",
    "MIMIR_DDL_STATEMENTS",
]
