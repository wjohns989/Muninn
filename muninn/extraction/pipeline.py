"""
Muninn Extraction Pipeline
---------------------------
Orchestrates tiered extraction: Rule-based → xLAM → Ollama fallback.
Automatically selects the best available tier at runtime.
"""

import logging
from typing import Optional

from muninn.core.types import ExtractionResult
from muninn.extraction.rules import rule_based_extract

logger = logging.getLogger("Muninn.Extract")


class ExtractionPipeline:
    """
    Tiered extraction pipeline:
      Tier 1: Rule-based (always available, 0ms)
      Tier 2: xLAM chain-of-extraction (if llama-cpp available, ~200ms)
      Tier 3: Ollama LLM (if Ollama running, ~2s)
    """

    def __init__(self, xlam_url: Optional[str] = None, ollama_url: Optional[str] = None):
        self.xlam_url = xlam_url
        self.ollama_url = ollama_url
        self._xlam_available = False
        self._ollama_available = False

        # Probe available tiers
        if xlam_url:
            self._xlam_available = self._check_xlam()
        if ollama_url:
            self._ollama_available = self._check_ollama()

        tier = "Tier 1 (rules only)"
        if self._xlam_available:
            tier = "Tier 2 (xLAM + rules)"
        elif self._ollama_available:
            tier = "Tier 3 (Ollama + rules)"
        logger.info(f"Extraction pipeline initialized: {tier}")

    def _check_xlam(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self.xlam_url}/v1/models", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _check_ollama(self) -> bool:
        try:
            import requests
            resp = requests.get(self.ollama_url, timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    def extract(self, text: str, use_llm: bool = True) -> ExtractionResult:
        """
        Run extraction pipeline. Returns combined results from best available tier.

        Args:
            text: Input text to extract from
            use_llm: Whether to attempt LLM-based extraction (Tier 2/3)
        """
        # Always run Tier 1 (rule-based) — it's free and fast
        result = rule_based_extract(text)

        if not use_llm:
            return result

        # Tier 2: xLAM chain-of-extraction
        if self._xlam_available:
            try:
                xlam_result = self._xlam_extract(text)
                result = self._merge_results(result, xlam_result)
                return result
            except Exception as e:
                logger.warning(f"xLAM extraction failed, falling back: {e}")

        # Tier 3: Ollama LLM extraction (future implementation)
        if self._ollama_available:
            try:
                ollama_result = self._ollama_extract(text)
                result = self._merge_results(result, ollama_result)
                return result
            except Exception as e:
                logger.warning(f"Ollama extraction failed: {e}")

        return result

    def _xlam_extract(self, text: str) -> ExtractionResult:
        """
        xLAM chain-of-extraction using PA-Tool aligned schemas.
        Implements 6-step sequential extraction pipeline.
        """
        import requests
        import json
        from muninn.core.types import Entity, Relation

        entities = []
        relations = []
        summary = None

        # Step 1: Entity extraction
        try:
            resp = requests.post(
                f"{self.xlam_url}/v1/chat/completions",
                json={
                    "model": "xlam",
                    "messages": [
                        {"role": "system", "content": "Extract named entities from the text. Return a JSON array of objects with 'name' and 'entity_type' fields."},
                        {"role": "user", "content": f"Extract entities from: {text[:2000]}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Parse JSON from response
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                entities.append(Entity(
                                    name=item["name"],
                                    entity_type=item.get("entity_type", "concept"),
                                ))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug(f"xLAM entity extraction: {e}")

        # Step 2: Relation extraction (using discovered entities)
        if entities:
            entity_names = [e.name for e in entities[:10]]
            try:
                resp = requests.post(
                    f"{self.xlam_url}/v1/chat/completions",
                    json={
                        "model": "xlam",
                        "messages": [
                            {"role": "system", "content": "Extract relationships between entities. Return a JSON array of objects with 'subject', 'predicate', and 'object' fields."},
                            {"role": "user", "content": f"Entities: {entity_names}\nText: {text[:2000]}\nExtract relationships:"}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and all(k in item for k in ("subject", "predicate", "object")):
                                    relations.append(Relation(
                                        subject=item["subject"],
                                        predicate=item["predicate"],
                                        object=item["object"],
                                    ))
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.debug(f"xLAM relation extraction: {e}")

        # Step 3: Summary generation
        try:
            resp = requests.post(
                f"{self.xlam_url}/v1/chat/completions",
                json={
                    "model": "xlam",
                    "messages": [
                        {"role": "system", "content": "Summarize the key facts in one concise sentence."},
                        {"role": "user", "content": text[:2000]}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                summary = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"xLAM summary: {e}")

        return ExtractionResult(
            entities=entities,
            relations=relations,
            summary=summary,
        )

    def _ollama_extract(self, text: str) -> ExtractionResult:
        """Tier 3: Ollama-based extraction (placeholder for future implementation)."""
        # TODO: Implement Ollama-based extraction as fallback
        return ExtractionResult()

    def _merge_results(self, base: ExtractionResult, additional: ExtractionResult) -> ExtractionResult:
        """Merge extraction results, deduplicating entities."""
        seen_entities = {e.name.lower() for e in base.entities}
        merged_entities = list(base.entities)
        for entity in additional.entities:
            if entity.name.lower() not in seen_entities:
                seen_entities.add(entity.name.lower())
                merged_entities.append(entity)

        seen_relations = {(r.subject.lower(), r.predicate, r.object.lower()) for r in base.relations}
        merged_relations = list(base.relations)
        for rel in additional.relations:
            key = (rel.subject.lower(), rel.predicate, rel.object.lower())
            if key not in seen_relations:
                seen_relations.add(key)
                merged_relations.append(rel)

        return ExtractionResult(
            entities=merged_entities,
            relations=merged_relations,
            summary=additional.summary or base.summary,
            temporal_context=additional.temporal_context or base.temporal_context,
        )
