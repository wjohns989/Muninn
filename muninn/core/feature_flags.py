"""
Muninn Feature Flags
--------------------
Centralized, env-driven feature toggles for optional capabilities.

All SOTA+ features are gated behind flags to ensure:
1. Backward compatibility — existing v3.0 deployments unaffected
2. Graceful degradation — missing deps don't crash the system
3. Opt-in complexity — users enable features when ready
4. Runtime configurability — flags read from env vars at startup

Design decision: We use a frozen dataclass (not Pydantic) for flags because:
- Flags are immutable after creation (no accidental mutation)
- Zero runtime overhead vs. Pydantic validation
- Simple serialization for health/status endpoints
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any

logger = logging.getLogger("Muninn.Flags")

# Env var prefix for all flags
_PREFIX = "MUNINN_"


def _env_bool(key: str, default: str = "0") -> bool:
    """Read a boolean flag from environment. '1'/'true'/'yes' → True."""
    val = os.environ.get(f"{_PREFIX}{key}", default).strip().lower()
    return val in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class FeatureFlags:
    """
    Immutable feature flag container. All flags are resolved once at
    construction time from environment variables.

    Naming convention: MUNINN_<FLAG_NAME>=1|0

    Phase 1 flags default ON (low cost, high value):
        - explainable_recall: Per-signal attribution in search results
        - instructor_extraction: Pydantic-validated structured extraction
        - platform_abstraction: Cross-platform path/process helpers

    Phase 2 flags default OFF (higher cost, opt-in):
        - conflict_detection: NLI-based contradiction detection on add()
        - semantic_dedup: Cosine-similarity dedup at ingestion
        - adaptive_weights: Entropy-based dynamic signal weighting
        - retrieval_feedback: Persist and apply implicit retrieval feedback

    Phase 3 flags default OFF (require additional deps):
        - memory_chains: Temporal/causal linking between memories
        - multi_source_ingestion: File, conversation, API ingestion
        - python_sdk: Expose MuninnClient for programmatic use
        - otel_genai: OpenTelemetry GenAI trace instrumentation
    """

    # --- Phase 1 (v3.1.0) — default ON ---
    explainable_recall: bool = True
    instructor_extraction: bool = True
    platform_abstraction: bool = True
    goal_compass: bool = True

    # --- Phase 2 (v3.2.0) — default OFF ---
    conflict_detection: bool = False
    semantic_dedup: bool = False
    adaptive_weights: bool = False
    retrieval_feedback: bool = False

    # --- Phase 3 (v3.3.0+) — default OFF ---
    memory_chains: bool = False
    multi_source_ingestion: bool = False
    python_sdk: bool = False
    otel_genai: bool = False
    
    # ColBERT Suite (v3.5.0+)
    colbert: bool = False
    colbert_plaid: bool = False
    colbert_int8: bool = False

    # Phase 13 (v3.10.0) — Advanced Retrieval
    colbert_multivec: bool = False           # Native Qdrant MultiVectorConfig MaxSim
    temporal_query_expansion: bool = False   # NL time-phrase parsing in search()

    # Phase 14 (v3.11.0) — Project-Scoped Memory
    project_scope_strict: bool = False       # Disable fallback retry entirely — zero cross-project leakage

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        """
        Construct flags from environment variables.

        Each flag maps to MUNINN_<UPPER_SNAKE_CASE>=1|0.
        Defaults are defined in the dataclass fields above.
        """
        flags = cls(
            # Phase 1
            explainable_recall=_env_bool("EXPLAIN_RECALL", "1"),
            instructor_extraction=_env_bool("INSTRUCTOR_EXTRACTION", "1"),
            platform_abstraction=_env_bool("PLATFORM_ABSTRACTION", "1"),
            goal_compass=_env_bool("GOAL_COMPASS", "1"),
            # Phase 2
            conflict_detection=_env_bool("CONFLICT_DETECTION", "0"),
            semantic_dedup=_env_bool("SEMANTIC_DEDUP", "0"),
            adaptive_weights=_env_bool("ADAPTIVE_WEIGHTS", "0"),
            retrieval_feedback=_env_bool("RETRIEVAL_FEEDBACK", "0"),
            # Phase 3
            memory_chains=_env_bool("MEMORY_CHAINS", "0"),
            multi_source_ingestion=_env_bool("MULTI_SOURCE_INGESTION", "0"),
            python_sdk=_env_bool("PYTHON_SDK", "0"),
            otel_genai=_env_bool("OTEL_GENAI", "0"),
            # ColBERT
            colbert=_env_bool("COLBERT", "0"),
            colbert_plaid=_env_bool("COLBERT_PLAID", "0"),
            colbert_int8=_env_bool("COLBERT_INT8", "0"),
            # Phase 13 (v3.10.0)
            colbert_multivec=_env_bool("COLBERT_MULTIVEC", "0"),
            temporal_query_expansion=_env_bool("TEMPORAL_QUERY_EXPANSION", "0"),
            # Phase 14 (v3.11.0)
            project_scope_strict=_env_bool("PROJECT_SCOPE_STRICT", "0"),
        )
        _log_active_flags(flags)
        return flags

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for health/status endpoints."""
        return asdict(self)

    @property
    def active_flags(self) -> Dict[str, bool]:
        """Return only the flags that are currently enabled."""
        return {k: v for k, v in self.to_dict().items() if v}

    def is_enabled(self, flag_name: str) -> bool:
        """
        Check if a specific flag is enabled by name.

        Args:
            flag_name: The flag attribute name (e.g., 'explainable_recall').

        Returns:
            True if the flag exists and is enabled, False otherwise.

        Raises:
            AttributeError: If flag_name is not a valid flag.
        """
        if not hasattr(self, flag_name):
            raise AttributeError(
                f"Unknown feature flag: '{flag_name}'. "
                f"Valid flags: {list(self.to_dict().keys())}"
            )
        return getattr(self, flag_name)

    def require(self, flag_name: str) -> None:
        """
        Assert that a flag is enabled. Use at feature entry points.

        Args:
            flag_name: The flag to check.

        Raises:
            RuntimeError: If the flag is disabled.
        """
        if not self.is_enabled(flag_name):
            raise RuntimeError(
                f"Feature '{flag_name}' is disabled. "
                f"Set MUNINN_{flag_name.upper()}=1 to enable."
            )


def _log_active_flags(flags: FeatureFlags) -> None:
    """Log which flags are active at startup."""
    active = flags.active_flags
    if active:
        names = ", ".join(sorted(active.keys()))
        logger.info("Active feature flags: %s", names)
    else:
        logger.info("No feature flags active (baseline v3.0 mode)")


# Module-level singleton for convenience. Initialized lazily.
_global_flags: FeatureFlags | None = None


def get_flags() -> FeatureFlags:
    """
    Get the global FeatureFlags singleton.
    Initializes from environment on first call.

    Thread-safe: worst case two threads create identical immutable instances.
    """
    global _global_flags
    if _global_flags is None:
        from threading import Lock
        _lock = Lock()
        with _lock:
            if _global_flags is None:
                _global_flags = FeatureFlags.from_env()
    return _global_flags


def reset_flags() -> None:
    """Reset the global singleton. Used in testing."""
    global _global_flags
    _global_flags = None
