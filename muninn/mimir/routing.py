"""
Mimir Interop Relay — Memory-Aware Router
==========================================
Routes relay requests to the optimal provider using a composite scoring
model that combines:

  1. Capability score  (0.35) — how well the provider handles this request type
  2. Availability score (0.25) — is the provider reachable right now?
  3. Cost score        (0.15) — estimated cost (lower cost → higher score)
  4. Safety score      (0.15) — policy compliance posture
  5. History score     (0.10) — recent success rate from Muninn memory

Routing algorithm:
  - If target is explicit (not "auto"), validate against allowed list and return
  - Otherwise score all available providers and return the highest-scoring one
  - If no provider scores > 0 (all unavailable), raise RoutingError

Memory integration:
  - Retrieve recent relay run outcomes from Muninn store to feed history_score
  - Store routing decisions back to the audit trail for future scoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .adapters import all_adapters, get_adapter
from .models import (
    AuditEventType,
    IRPEnvelope,
    IRPMode,
    ProviderName,
    RoutingScore,
    RunStatus,
)

logger = logging.getLogger("Muninn.Mimir.routing")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RoutingError(Exception):
    """Raised when no suitable provider can be found."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Provider capability profiles
# ---------------------------------------------------------------------------

# Baseline capability scores by IRP mode.
# These represent our best a-priori estimate; history scores refine them.
_CAPABILITY_DEFAULTS: dict[ProviderName, dict[IRPMode, float]] = {
    ProviderName.CLAUDE_CODE: {
        IRPMode.ADVISORY: 0.90,
        IRPMode.STRUCTURED: 0.85,
        IRPMode.RECONCILE: 0.88,
    },
    ProviderName.CODEX_CLI: {
        IRPMode.ADVISORY: 0.78,
        IRPMode.STRUCTURED: 0.82,
        IRPMode.RECONCILE: 0.75,
    },
    ProviderName.GEMINI_CLI: {
        IRPMode.ADVISORY: 0.85,
        IRPMode.STRUCTURED: 0.80,
        IRPMode.RECONCILE: 0.82,
    },
}

# Baseline cost scores (inverse: lower-cost provider → higher score)
# Relative ordering based on typical API pricing (2025 H1 data).
_COST_DEFAULTS: dict[ProviderName, float] = {
    ProviderName.CODEX_CLI: 0.85,   # cheapest
    ProviderName.GEMINI_CLI: 0.75,
    ProviderName.CLAUDE_CODE: 0.60,
}

# Safety posture: how well each provider supports IRP/1 policy constraints.
# Claude Code and Codex have granular tool controls; Gemini is broader.
_SAFETY_DEFAULTS: dict[ProviderName, float] = {
    ProviderName.CLAUDE_CODE: 0.92,
    ProviderName.CODEX_CLI: 0.88,
    ProviderName.GEMINI_CLI: 0.80,
}


# ---------------------------------------------------------------------------
# MemoryAwareRouter
# ---------------------------------------------------------------------------


class MemoryAwareRouter:
    """
    Scores providers and returns the optimal one for a given IRP envelope.

    Parameters
    ----------
    mimir_store : optional MimirStore
        If provided, recent run history is used to boost/penalise history scores.
    availability_ttl : float
        Seconds to cache provider availability results.
    """

    def __init__(
        self,
        mimir_store=None,
        availability_ttl: float = 60.0,
    ) -> None:
        self._store = mimir_store
        self._availability_ttl = availability_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        envelope: IRPEnvelope,
        allowed_providers: Optional[list[str]] = None,
    ) -> ProviderName:
        """
        Determine the best provider for the given envelope.

        Parameters
        ----------
        envelope : IRPEnvelope
            The IRP request envelope.
        allowed_providers : list[str], optional
            Provider name strings that are allowed; None means all.

        Returns
        -------
        ProviderName
            The selected provider.

        Raises
        ------
        RoutingError
            If no suitable provider is available.
        """
        target = envelope.to

        # Explicit target — validate and short-circuit
        if target != ProviderName.AUTO.value and target != "auto":
            try:
                provider = ProviderName(target)
            except ValueError:
                raise RoutingError(
                    f"Unknown target provider '{target}'. "
                    f"Valid values: {[p.value for p in ProviderName if p != ProviderName.AUTO]}"
                )

            if allowed_providers is not None and target not in allowed_providers:
                raise RoutingError(
                    f"Target '{target}' is not in the allowed providers list: {allowed_providers}"
                )

            # Verify availability before committing
            adapter = get_adapter(provider)
            if not await adapter.check_available():
                raise RoutingError(
                    f"Provider '{target}' is explicitly requested but not available."
                )

            logger.debug("routing: explicit target=%s", target)
            return provider

        # Auto-routing: score all candidates
        candidates = list(ProviderName)
        candidates = [p for p in candidates if p != ProviderName.AUTO]

        if allowed_providers is not None:
            candidates = [p for p in candidates if p.value in allowed_providers]

        if not candidates:
            raise RoutingError("No providers are in the allowed list.")

        scores = await self._score_all(envelope, candidates)

        logger.debug(
            "routing scores: %s",
            {p.value: round(s.composite, 3) for p, s in scores.items()},
        )

        # Filter: only consider providers with availability_score > 0
        viable = {p: s for p, s in scores.items() if s.availability_score > 0.0}

        if not viable:
            raise RoutingError(
                "All candidate providers are currently unavailable. "
                f"Checked: {[p.value for p in candidates]}"
            )

        best = max(viable, key=lambda p: viable[p].composite)
        logger.info(
            "routing: selected provider=%s composite=%.3f",
            best.value,
            viable[best].composite,
        )
        return best

    async def score_all(
        self,
        envelope: IRPEnvelope,
        providers: Optional[list[ProviderName]] = None,
    ) -> dict[ProviderName, RoutingScore]:
        """
        Public scoring endpoint — returns scores for all (or specified) providers.
        Useful for diagnostics and the /providers API endpoint.
        """
        if providers is None:
            providers = [p for p in ProviderName if p != ProviderName.AUTO]
        return await self._score_all(envelope, providers)

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    async def _score_all(
        self, envelope: IRPEnvelope, providers: list[ProviderName]
    ) -> dict[ProviderName, RoutingScore]:
        """Score all providers concurrently and return the RoutingScore map."""
        tasks = {
            provider: asyncio.create_task(self._score_one(envelope, provider))
            for provider in providers
        }
        results: dict[ProviderName, RoutingScore] = {}
        for provider, task in tasks.items():
            try:
                results[provider] = await task
            except Exception as exc:
                logger.warning("scoring failed for %s: %s", provider.value, exc)
                results[provider] = RoutingScore(
                    provider=provider,
                    capability_score=0.0,
                    availability_score=0.0,
                    cost_score=0.0,
                    safety_score=0.0,
                    history_score=0.5,
                    composite=0.0,
                )
        return results

    async def _score_one(
        self, envelope: IRPEnvelope, provider: ProviderName
    ) -> RoutingScore:
        """Compute a full RoutingScore for a single provider."""
        # Availability (async, may hit subprocess)
        adapter = get_adapter(provider)
        available = await adapter.check_available()
        availability_score = 1.0 if available else 0.0

        # Capability
        mode_caps = _CAPABILITY_DEFAULTS.get(provider, {})
        capability_score = mode_caps.get(envelope.mode, 0.5)

        # Cost
        cost_score = _COST_DEFAULTS.get(provider, 0.5)

        # Safety — degrade if policy constraints exceed provider capabilities
        safety_score = _SAFETY_DEFAULTS.get(provider, 0.8)
        if envelope.policy.tools == "forbidden" and provider == ProviderName.GEMINI_CLI:
            # Gemini's no-tools support is less granular
            safety_score = min(safety_score, 0.70)

        # History — query recent run outcomes from store
        history_score = await self._compute_history_score(provider)

        score = RoutingScore(
            provider=provider,
            capability_score=round(capability_score, 4),
            availability_score=availability_score,
            cost_score=round(cost_score, 4),
            safety_score=round(safety_score, 4),
            history_score=round(history_score, 4),
        )
        score.compute_composite()
        return score

    async def _compute_history_score(
        self, provider: ProviderName, window: int = 10
    ) -> float:
        """
        Compute a history score [0.0, 1.0] from recent run outcomes.

        Returns 0.5 (neutral) if no historical data is available.

        Success rate over last `window` runs:
          score = successes / total  (with a Laplace smoothing of 1/1)
        """
        if self._store is None:
            return 0.5

        try:
            runs = self._store.list_runs(
                provider=provider.value,
                limit=window,
            )
            if not runs:
                return 0.5

            # Laplace smoothing: add 1 success and 1 failure
            successes = sum(
                1
                for r in runs
                if r.get("status") == RunStatus.SUCCESS.value
            )
            total = len(runs)
            score = (successes + 1) / (total + 2)
            return round(min(max(score, 0.0), 1.0), 4)

        except Exception as exc:
            logger.debug("history score lookup failed for %s: %s", provider.value, exc)
            return 0.5
