"""
Muninn Instructor-Based Extractor
-----------------------------------
Structured entity/relation extraction using the Instructor library.

Instructor wraps OpenAI-compatible endpoints with Pydantic model validation
and automatic retry, guaranteeing structured output from ANY backend:
- Ollama (local, free)
- xLAM via llama-cpp-server (local, free)
- vLLM, LM Studio (local)
- OpenAI, Anthropic, etc. (cloud, if desired)

This replaces the fragile raw-JSON-parse approach in the original xLAM
extractor with validated, type-safe structured extraction.

Key advantages over raw prompt → JSON parse:
1. Guaranteed Pydantic-valid output (automatic retry on parse failure)
2. Schema-driven prompting (LLM sees field descriptions as instructions)
3. Unified interface for ALL backends (one extractor, many providers)
4. Confidence scores on each extraction
5. Temporal context awareness
"""

import logging
from typing import Optional

from muninn.core.types import ExtractionResult, Entity, Relation
from muninn.extraction.models import (
    ExtractedMemoryFacts,
    EXTRACTION_SYSTEM_PROMPT,
)

logger = logging.getLogger("Muninn.Extract.Instructor")


class InstructorExtractor:
    """
    Structured extraction using Instructor + Pydantic models.

    Wraps any OpenAI-compatible endpoint (Ollama, xLAM, vLLM, cloud)
    and guarantees ExtractedMemoryFacts output via Pydantic validation.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        max_retries: int = 2,
        timeout: float = 30.0,
    ):
        """
        Initialize the Instructor extractor.

        Args:
            base_url: OpenAI-compatible API base URL.
                      E.g. "http://localhost:11434/v1" for Ollama,
                           "http://localhost:8001/v1" for xLAM.
            model: Model name for the endpoint.
                   E.g. "llama3.2:3b" for Ollama, "xlam" for xLAM.
            api_key: API key (most local endpoints don't need one).
            max_retries: Instructor retry count on validation failure.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self._client = None
        self._available = False

        try:
            import instructor
            from openai import OpenAI

            raw_client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            self._client = instructor.from_openai(
                raw_client,
                mode=instructor.Mode.JSON,
            )
            self._available = True
            logger.info(
                "Instructor extractor ready: %s @ %s",
                model, base_url,
            )
        except ImportError as e:
            logger.info(
                "Instructor not available (missing dependency: %s) "
                "— falling back to rule-based extraction",
                e,
            )
        except Exception as e:
            logger.warning(
                "Instructor init failed: %s — falling back to rule-based",
                e,
            )

    @property
    def is_available(self) -> bool:
        """Whether the Instructor extractor is ready to use."""
        return self._available

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract structured facts from text using Instructor.

        Args:
            text: Input text to extract from (truncated to 3000 chars
                  to respect context limits of smaller models).

        Returns:
            ExtractionResult with entities, relations, and summary.
            Returns empty result on failure (never raises).
        """
        if not self._available:
            return ExtractionResult()

        try:
            # Truncate to avoid context length issues with small models
            truncated = text[:3000]

            facts: ExtractedMemoryFacts = self._client.chat.completions.create(
                model=self.model,
                response_model=ExtractedMemoryFacts,
                max_retries=self.max_retries,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract structured facts from the following text delimited by triple backticks:\n\n```\n{truncated}\n```",
                    },
                ],
            )

            return self._convert_to_extraction_result(facts)

        except Exception as e:
            logger.warning("Instructor extraction failed: %s", e)
            return ExtractionResult()

    def _convert_to_extraction_result(
        self, facts: ExtractedMemoryFacts
    ) -> ExtractionResult:
        """
        Convert Instructor's ExtractedMemoryFacts → Muninn's ExtractionResult.

        This bridge maintains backward compatibility with the existing
        extraction pipeline while gaining structured validation.
        """
        entities = [
            Entity(
                name=e.name,
                entity_type=e.entity_type,
            )
            for e in facts.entities
        ]

        relations = [
            Relation(
                subject=r.subject,
                predicate=r.predicate,
                object=r.object,
                confidence=r.confidence,
            )
            for r in facts.relations
        ]

        return ExtractionResult(
            entities=entities,
            relations=relations,
            summary=facts.summary,
            temporal_context=facts.temporal_context,
        )

    def probe_endpoint(self) -> bool:
        """
        Test if the LLM endpoint is actually reachable and responsive.

        Sends a minimal extraction request to verify connectivity.
        Updates self._available based on result.

        Returns:
            True if endpoint responded successfully.
        """
        if not self._client:
            return False

        try:
            result = self._client.chat.completions.create(
                model=self.model,
                response_model=ExtractedMemoryFacts,
                max_retries=1,
                messages=[
                    {"role": "system", "content": "Extract entities."},
                    {"role": "user", "content": "Python is a programming language."},
                ],
            )
            self._available = True
            logger.info("Instructor endpoint probe successful")
            return True
        except Exception as e:
            logger.warning("Instructor endpoint probe failed: %s", e)
            self._available = False
            return False
