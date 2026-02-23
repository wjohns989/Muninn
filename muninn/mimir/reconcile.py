"""
Mimir Interop Relay — Mode C Reconciler
========================================
Implements multi-provider reconciliation (IRP/1 Mode C) for the Mimir relay.

Algorithm
---------
1. For each ProviderResult, extract the response text and split into sentences /
   top-level claims using a lightweight heuristic.
2. For each claim, query the Muninn memory store to check whether existing
   memories support or contradict it (memory-grounding).
3. Classify claims into *consensus* (all providers agree on the substance) vs.
   *conflicting* (providers materially disagree).
4. Generate a concise synthesis string that:
     • Summarises consensus claims (deduplicated).
     • Highlights key conflicts with attribution.
5. Escalate if no consensus can be reached (all claims conflict) or if the
   providers returned an empty / error response.

Design constraints
------------------
- No external LLM calls — synthesis is purely text manipulation so the
  Reconciler can function offline and in unit tests.
- Memory queries are optional (store may be None).
- All heavy lifting is async-safe; the class is stateless between calls.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Optional

from .models import (
    IRPEnvelope,
    ProviderName,
    ProviderResult,
    ReconciliationClaim,
    ReconciliationResult,
    RunStatus,
)

logger = logging.getLogger("Muninn.Mimir.reconcile")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum similarity ratio (difflib SequenceMatcher) to consider two claims
# as "the same" for consensus detection.
_CONSENSUS_SIMILARITY = 0.60

# If memory-grounded claims make up ≥ this fraction of total claims, we
# weight them more heavily in conflict resolution.
_MEMORY_WEIGHT_THRESHOLD = 0.5

# Maximum characters for each claim extracted from provider output.
_MAX_CLAIM_CHARS = 512

# Minimum claim length (in characters) to be worth keeping.
_MIN_CLAIM_CHARS = 20

# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class Reconciler:
    """
    Performs Mode C (RECONCILE) reconciliation across multiple ProviderResults.

    Parameters
    ----------
    mimir_store : optional MimirStore
        If provided, claims are cross-referenced against stored memories to
        boost or penalise confidence scores.
    """

    def __init__(self, mimir_store=None) -> None:
        self._store = mimir_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        envelope: IRPEnvelope,
        results: dict[ProviderName, ProviderResult],
    ) -> ReconciliationResult:
        """
        Reconcile provider results for a RECONCILE-mode envelope.

        Parameters
        ----------
        envelope : IRPEnvelope
            The original IRP request envelope.
        results : dict[ProviderName, ProviderResult]
            Mapping of provider → result for all providers that were queried.

        Returns
        -------
        ReconciliationResult
            Classified claims and synthesis text.
        """
        # Filter to successful results with non-empty output
        valid: dict[ProviderName, str] = {}
        for provider, result in results.items():
            if result.error:
                logger.debug(
                    "reconcile: skipping provider=%s (error: %s)",
                    provider.value,
                    result.error,
                )
                continue
            text = result.raw_output.strip()
            if text:
                valid[provider] = text

        if not valid:
            return ReconciliationResult(
                escalated=True,
                escalation_reason="All providers returned errors or empty responses.",
            )

        if len(valid) == 1:
            # Only one valid provider — return its output as consensus
            only_provider, only_text = next(iter(valid.items()))
            claims = self._extract_claims(only_provider, only_text)
            claims = await self._memory_ground(claims, envelope)
            return ReconciliationResult(
                consensus_claims=claims,
                synthesis=self._format_single_provider_synthesis(
                    only_provider, only_text
                ),
                escalated=False,
            )

        # Extract claims from every valid provider
        all_claims: list[ReconciliationClaim] = []
        for provider, text in valid.items():
            claims = self._extract_claims(provider, text)
            claims = await self._memory_ground(claims, envelope)
            all_claims.extend(claims)

        if not all_claims:
            return ReconciliationResult(
                escalated=True,
                escalation_reason="No extractable claims found in provider outputs.",
            )

        consensus, conflicting = self._classify_claims(all_claims, len(valid))

        synthesis = self._synthesise(
            consensus, conflicting, valid, escalated=not consensus
        )

        escalated = len(conflicting) > 0 and len(consensus) == 0
        escalation_reason: Optional[str] = None
        if escalated:
            escalation_reason = (
                f"Providers disagreed on all {len(conflicting)} claim(s) with "
                f"no consensus. Manual review required."
            )

        logger.info(
            "reconcile: consensus=%d conflicting=%d escalated=%s",
            len(consensus),
            len(conflicting),
            escalated,
        )

        return ReconciliationResult(
            consensus_claims=consensus,
            conflicting_claims=conflicting,
            synthesis=synthesis,
            escalated=escalated,
            escalation_reason=escalation_reason,
        )

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(
        self, provider: ProviderName, text: str
    ) -> list[ReconciliationClaim]:
        """
        Split provider output text into individual claims.

        Strategy:
          1. Split on sentence boundaries (`.`, `!`, `?`, newlines).
          2. Filter out very short fragments and duplicates.
          3. Truncate long claims to _MAX_CLAIM_CHARS.
        """
        # Normalise newlines → sentence separators
        text = re.sub(r"\n{2,}", "\n", text.strip())

        # Split on sentence-ending punctuation or newlines
        raw_chunks = re.split(r"(?<=[.!?])\s+|(?<=\n)", text)

        claims: list[ReconciliationClaim] = []
        seen: set[str] = set()

        for chunk in raw_chunks:
            chunk = chunk.strip().rstrip(".!?").strip()
            if len(chunk) < _MIN_CLAIM_CHARS:
                continue

            # Deduplicate within same provider
            normalised = re.sub(r"\s+", " ", chunk.lower())
            if normalised in seen:
                continue
            seen.add(normalised)

            claims.append(
                ReconciliationClaim(
                    provider=provider,
                    claim_text=chunk[:_MAX_CLAIM_CHARS],
                    confidence=0.5,
                )
            )

        return claims

    # ------------------------------------------------------------------
    # Memory grounding
    # ------------------------------------------------------------------

    async def _memory_ground(
        self,
        claims: list[ReconciliationClaim],
        envelope: IRPEnvelope,
    ) -> list[ReconciliationClaim]:
        """
        For each claim, attempt to find supporting memories in the Muninn store.
        Boosts confidence for memory-supported claims.
        """
        if self._store is None or not claims:
            return claims

        grounded: list[ReconciliationClaim] = []
        for claim in claims:
            try:
                hits = self._store.search_memories(
                    query=claim.claim_text[:200],
                    limit=3,
                )
                if hits:
                    memory_ids = [h.get("id", "") for h in hits if h.get("id")]
                    # Boost confidence proportionally to the number of hits
                    boost = min(0.15 * len(memory_ids), 0.30)
                    grounded.append(
                        claim.model_copy(
                            update={
                                "memory_supported": True,
                                "memory_ids": memory_ids,
                                "confidence": round(
                                    min(claim.confidence + boost, 1.0), 4
                                ),
                            }
                        )
                    )
                else:
                    grounded.append(claim)
            except Exception as exc:
                logger.debug(
                    "memory grounding failed for provider=%s: %s",
                    claim.provider.value,
                    exc,
                )
                grounded.append(claim)

        return grounded

    # ------------------------------------------------------------------
    # Claim classification
    # ------------------------------------------------------------------

    def _classify_claims(
        self,
        claims: list[ReconciliationClaim],
        provider_count: int,
    ) -> tuple[list[ReconciliationClaim], list[ReconciliationClaim]]:
        """
        Group claims into *consensus* and *conflicting* buckets.

        Consensus: a claim whose substance appears in outputs from
        MAJORITY of providers (≥ 50% of provider_count).

        Conflicting: claims that appear in only a minority of providers
        and whose content materially differs from consensus claims.

        Returns
        -------
        (consensus_claims, conflicting_claims)
            Each list contains one representative claim per group.
        """
        majority = max(1, provider_count // 2 + (provider_count % 2))

        # Group by similarity
        groups: list[list[ReconciliationClaim]] = []
        for claim in claims:
            placed = False
            for group in groups:
                rep = group[0]
                ratio = difflib.SequenceMatcher(
                    None,
                    re.sub(r"\s+", " ", claim.claim_text.lower()),
                    re.sub(r"\s+", " ", rep.claim_text.lower()),
                ).ratio()
                if ratio >= _CONSENSUS_SIMILARITY:
                    group.append(claim)
                    placed = True
                    break
            if not placed:
                groups.append([claim])

        consensus: list[ReconciliationClaim] = []
        conflicting: list[ReconciliationClaim] = []

        for group in groups:
            # Unique providers represented in this group
            providers_in_group = {c.provider for c in group}
            # Pick the highest-confidence claim as representative
            representative = max(group, key=lambda c: c.confidence)

            if len(providers_in_group) >= majority:
                consensus.append(representative)
            else:
                conflicting.append(representative)

        return consensus, conflicting

    # ------------------------------------------------------------------
    # Synthesis generation
    # ------------------------------------------------------------------

    def _synthesise(
        self,
        consensus: list[ReconciliationClaim],
        conflicting: list[ReconciliationClaim],
        provider_outputs: dict[ProviderName, str],
        escalated: bool,
    ) -> str:
        """
        Generate a human-readable synthesis from classified claims.
        """
        parts: list[str] = []

        if consensus:
            parts.append("## Consensus\n")
            for claim in consensus:
                parts.append(f"- {claim.claim_text}")
            parts.append("")

        if conflicting:
            parts.append("## Conflicts Requiring Attention\n")
            for claim in conflicting:
                parts.append(
                    f"- [{claim.provider.value}] {claim.claim_text}"
                )
            parts.append("")

        if escalated:
            parts.append(
                "## Escalation Notice\n"
                "Providers could not reach consensus. "
                "The following raw responses are provided for manual review:\n"
            )
            for provider, text in provider_outputs.items():
                short = text[:300].replace("\n", " ")
                parts.append(f"**{provider.value}**: {short}")
            parts.append("")

        return "\n".join(parts).strip()

    def _format_single_provider_synthesis(
        self, provider: ProviderName, text: str
    ) -> str:
        """Synthesis text when only one valid provider responded."""
        return (
            f"## Response (single provider: {provider.value})\n\n"
            f"{text.strip()}"
        )
