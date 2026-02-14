"""
Muninn NLI-Based Conflict Detection (v3.2.0)
----------------------------------------------
Uses Natural Language Inference (NLI) cross-encoder models to detect
contradictions between new and existing memories during add().

Model: cross-encoder/nli-deberta-v3-small
  - 44MB, Apache-2.0 license
  - 91.65% accuracy on SNLI benchmark
  - Outputs: [contradiction_score, entailment_score, neutral_score]
  - ~10ms per pair on CPU

UNIQUE DIFFERENTIATOR: No competitor (Mem0, Graphiti, Memento, MemoryGraph)
provides NLI-based conflict detection for memory integrity.

Dependencies:
  - transformers (Apache-2.0)
  - torch (BSD-3-Clause)
  Only loaded when conflict_detection feature flag is enabled.
"""

import logging
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

from muninn.core.types import MemoryRecord

logger = logging.getLogger("Muninn.Conflict")

SECONDS_IN_A_DAY = 24 * 60 * 60
SUPERSEDE_AGE_THRESHOLD_DAYS = 7
SUPERSEDE_IMPORTANCE_THRESHOLD = 0.5


class ConflictResolution(str, Enum):
    """Resolution strategy for detected conflicts."""
    SUPERSEDE = "supersede"            # New replaces old (more recent, same topic)
    MERGE = "merge"                    # Combine both into unified memory
    FLAG_FOR_REVIEW = "flag_for_review"  # Return conflict to user/agent
    KEEP_EXISTING = "keep_existing"    # Discard new, keep old


class ConflictResult(BaseModel):
    """Result of a conflict detection check between two memories."""
    new_content: str
    existing_memory_id: str
    existing_content: str
    contradiction_score: float = Field(ge=0.0, le=1.0)
    entailment_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    suggested_resolution: ConflictResolution
    explanation: str = ""


class ConflictDetector:
    """
    NLI-based contradiction detection for memory integrity.

    Uses a cross-encoder model to classify pairs of (new_memory, existing_memory)
    as entailment, contradiction, or neutral. When contradiction score exceeds
    the threshold, a ConflictResult is returned with a suggested resolution.

    The model is loaded lazily on first use and cached for the session lifetime.
    If transformers/torch are not installed, the detector gracefully degrades
    to a no-op (returns empty results).
    """

    # Default model: small, fast, Apache-2.0 licensed
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(
        self,
        model_name: Optional[str] = None,
        contradiction_threshold: float = 0.7,
        similarity_prefilter: float = 0.6,
    ):
        """
        Args:
            model_name: HuggingFace model ID for NLI cross-encoder.
            contradiction_threshold: Minimum contradiction score to flag conflict.
            similarity_prefilter: Only check memories with vector similarity above
                                  this threshold (performance optimization).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.contradiction_threshold = contradiction_threshold
        self.similarity_prefilter = similarity_prefilter

        self._model = None
        self._tokenizer = None
        self._available = False

        # Attempt to load model
        self._initialize()

    def _initialize(self) -> None:
        """Load the NLI model and tokenizer. Gracefully handle missing deps."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import torch  # noqa: F401 — ensure torch is available

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()  # Set to inference mode
            self._available = True
            logger.info(
                "Conflict detector initialized: model=%s, threshold=%.2f",
                self.model_name,
                self.contradiction_threshold,
            )
        except ImportError as e:
            logger.info(
                "Conflict detection unavailable (missing deps: %s). "
                "Install transformers and torch to enable.",
                e,
            )
            self._available = False
        except Exception as e:
            logger.warning("Conflict detector initialization failed: %s", e)
            self._available = False

    @property
    def is_available(self) -> bool:
        """Whether the NLI model is loaded and ready."""
        return self._available

    def detect_conflicts(
        self,
        new_content: str,
        existing_memories: List[MemoryRecord],
    ) -> List[ConflictResult]:
        """
        Check for contradictions between new content and existing memories.

        Args:
            new_content: The new memory text being added.
            existing_memories: Candidate existing memories to check against.
                               Should be pre-filtered by vector similarity
                               (similarity > similarity_prefilter).

        Returns:
            List of ConflictResult for each detected contradiction.
        """
        if not self._available:
            return []

        if not new_content.strip() or not existing_memories:
            return []

        conflicts: List[ConflictResult] = []

        for existing in existing_memories:
            result = self._check_pair(new_content, existing)
            if result is not None:
                conflicts.append(result)

        if conflicts:
            logger.info("Detected %d conflict(s) for new content", len(conflicts))
            logger.debug("Conflict new-content preview: '%s...'", new_content[:60])

        return conflicts

    def _check_pair(
        self,
        new_content: str,
        existing: MemoryRecord,
    ) -> Optional[ConflictResult]:
        """
        Run NLI inference on a single (new, existing) pair.

        Returns ConflictResult if contradiction detected, None otherwise.
        """
        import torch

        try:
            # Tokenize the pair
            features = self._tokenizer(
                new_content,
                existing.content,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Inference
            with torch.no_grad():
                outputs = self._model(**features)
                logits = outputs.logits
                scores = torch.softmax(logits, dim=1)[0]

            # DeBERTa NLI label order: [contradiction, entailment, neutral]
            contradiction_score = scores[0].item()
            entailment_score = scores[1].item()
            neutral_score = scores[2].item()

            if contradiction_score >= self.contradiction_threshold:
                resolution = self._suggest_resolution(
                    existing=existing,
                    contradiction_score=contradiction_score,
                    entailment_score=entailment_score,
                )

                return ConflictResult(
                    new_content=new_content,
                    existing_memory_id=existing.id,
                    existing_content=existing.content,
                    contradiction_score=contradiction_score,
                    entailment_score=entailment_score,
                    neutral_score=neutral_score,
                    suggested_resolution=resolution,
                    explanation=self._generate_explanation(
                        contradiction_score=contradiction_score,
                        resolution=resolution,
                        existing_content=existing.content,
                    ),
                )

        except Exception as e:
            logger.warning(
                "NLI inference failed for pair: %s", e
            )

        return None

    def _suggest_resolution(
        self,
        existing: MemoryRecord,
        contradiction_score: float,
        entailment_score: float,
    ) -> ConflictResolution:
        """
        Determine resolution strategy based on NLI scores and memory metadata.

        Decision tree:
          1. High contradiction + existing has low importance → SUPERSEDE
          2. High contradiction + high entailment (partial) → MERGE
          3. Very high contradiction → FLAG_FOR_REVIEW
          4. Default → FLAG_FOR_REVIEW (safe default)
        """
        import time as _time

        # If existing memory is old (>7 days) and low importance, supersede
        age_days = (_time.time() - existing.created_at) / SECONDS_IN_A_DAY
        if age_days > SUPERSEDE_AGE_THRESHOLD_DAYS and existing.importance < SUPERSEDE_IMPORTANCE_THRESHOLD:
            return ConflictResolution.SUPERSEDE

        # If both contradicting and partially entailing, merge
        if contradiction_score < 0.85 and entailment_score > 0.1:
            return ConflictResolution.MERGE

        # Very high contradiction — flag for human review
        if contradiction_score >= 0.85:
            return ConflictResolution.FLAG_FOR_REVIEW

        # Default: flag for review (safe)
        return ConflictResolution.FLAG_FOR_REVIEW

    @staticmethod
    def _generate_explanation(
        contradiction_score: float,
        resolution: ConflictResolution,
        existing_content: str,
    ) -> str:
        """Generate human-readable conflict explanation."""
        severity = (
            "Strong" if contradiction_score >= 0.85
            else "Moderate" if contradiction_score >= 0.7
            else "Weak"
        )

        resolution_text = {
            ConflictResolution.SUPERSEDE: "Replace old memory with new (more recent)",
            ConflictResolution.MERGE: "Combine both memories into unified version",
            ConflictResolution.FLAG_FOR_REVIEW: "Flag for user/agent review",
            ConflictResolution.KEEP_EXISTING: "Keep existing, discard new",
        }

        return (
            f"{severity} contradiction (score={contradiction_score:.2f}) "
            f"with existing memory: '{existing_content[:80]}...'. "
            f"Suggested action: {resolution_text.get(resolution, 'review')}."
        )
