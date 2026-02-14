"""
Muninn Extraction Pipeline
---------------------------
Orchestrates tiered extraction: Rule-based → Instructor → xLAM → Ollama.

v3.1.0: Added Instructor tier — unified structured extraction via Pydantic
models that works with ANY OpenAI-compatible endpoint (Ollama, xLAM, vLLM,
LM Studio, cloud). Replaces fragile raw-JSON-parse approach.

Tier hierarchy:
  Tier 1: Rule-based (always, 0ms) — guaranteed baseline
  Tier 2: Instructor (if configured + endpoint alive, ~200ms–2s) — structured
  Tier 3: xLAM raw (legacy fallback if Instructor unavailable)
  Merge: Union entities/relations, deduplicate by lowercase name
"""

import logging
from typing import Dict, List, Optional, Tuple

from muninn.core.types import ExtractionResult
from muninn.core.feature_flags import get_flags
from muninn.extraction.rules import rule_based_extract

logger = logging.getLogger("Muninn.Extract")

SUPPORTED_MODEL_PROFILES = ("low_latency", "balanced", "high_reasoning")


class ExtractionPipeline:
    """
    Tiered extraction pipeline with Instructor integration (v3.1.0).

    Tier 1: Rule-based (always available, 0ms)
    Tier 2: Instructor structured extraction (if feature flag ON + endpoint available)
    Tier 3: xLAM chain-of-extraction (legacy, if Instructor unavailable)
    """

    def __init__(
        self,
        xlam_url: Optional[str] = None,
        xlam_model: str = "xLAM",
        ollama_url: Optional[str] = None,
        ollama_model: str = "llama3.2:3b",
        ollama_balanced_model: str = "qwen3:8b",
        ollama_high_reasoning_model: str = "qwen3:32b",
        model_profile: str = "balanced",
        instructor_base_url: Optional[str] = None,
        instructor_model: Optional[str] = None,
        instructor_api_key: str = "not-needed",
    ):
        self.xlam_url = xlam_url
        self.xlam_model = xlam_model
        self.ollama_url = ollama_url
        self._xlam_available = False
        self._ollama_available = False
        self.model_profile = model_profile if model_profile in SUPPORTED_MODEL_PROFILES else "balanced"
        self._instructor_routes_by_profile: Dict[str, List[Tuple[str, "InstructorExtractor"]]] = {}

        # Initialize Instructor extractor (v3.1.0)
        flags = get_flags()
        if flags.is_enabled("instructor_extraction") and instructor_base_url:
            try:
                from muninn.extraction.instructor_extractor import InstructorExtractor
                route_specs_by_profile = {
                    profile: self._build_instructor_route_specs(
                        profile=profile,
                        xlam_url=xlam_url,
                        xlam_model=xlam_model,
                        ollama_url=ollama_url,
                        ollama_model=ollama_model,
                        ollama_balanced_model=ollama_balanced_model,
                        ollama_high_reasoning_model=ollama_high_reasoning_model,
                    )
                    for profile in SUPPORTED_MODEL_PROFILES
                }
                if instructor_model:
                    explicit_base_url = self._normalize_openai_base_url(instructor_base_url)
                    route_specs_by_profile[self.model_profile].insert(
                        0,
                        (
                            f"instructor-explicit:{instructor_model}",
                            explicit_base_url,
                            instructor_model,
                        ),
                    )
                extractor_cache: Dict[Tuple[str, str], InstructorExtractor] = {}
                for profile, specs in route_specs_by_profile.items():
                    self._instructor_routes_by_profile[profile] = []
                    for label, base_url, model in specs:
                        cache_key = (base_url, model)
                        extractor = extractor_cache.get(cache_key)
                        if extractor is None:
                            extractor = InstructorExtractor(
                                base_url=base_url,
                                model=model,
                                api_key=instructor_api_key,
                            )
                            extractor_cache[cache_key] = extractor
                        self._instructor_routes_by_profile[profile].append((label, extractor))
            except Exception as e:
                logger.warning("Instructor extractor init failed: %s", e)

        # Probe legacy tiers
        if xlam_url:
            self._xlam_available = self._check_xlam()
        if ollama_url:
            self._ollama_available = self._check_ollama()

        tier = "Tier 1 (rules only)"
        if self._has_available_instructor_route(self.model_profile):
            tier = f"Tier 2 (Instructor + rules, profile={self.model_profile})"
        elif self._xlam_available:
            tier = "Tier 2 (xLAM legacy + rules)"
        elif self._ollama_available:
            tier = "Tier 3 (Ollama + rules)"
        logger.info("Extraction pipeline initialized: %s", tier)

    @staticmethod
    def _normalize_openai_base_url(url: str) -> str:
        clean = url.rstrip("/")
        if clean.endswith("/v1"):
            return clean
        return f"{clean}/v1"

    @classmethod
    def _build_instructor_route_specs(
        cls,
        profile: str,
        xlam_url: Optional[str],
        xlam_model: str,
        ollama_url: Optional[str],
        ollama_model: str,
        ollama_balanced_model: str,
        ollama_high_reasoning_model: str,
    ) -> List[Tuple[str, str, str]]:
        """
        Build deterministic instructor routes for a profile.

        Returns ordered tuples of (route_label, base_url, model_name).
        """
        normalized_profile = profile if profile in SUPPORTED_MODEL_PROFILES else "balanced"
        route_specs: List[Tuple[str, str, str]] = []
        if normalized_profile == "low_latency":
            candidate_specs = [
                ("ollama-low", ollama_url, ollama_model),
                ("xlam", xlam_url, xlam_model),
            ]
        elif normalized_profile == "high_reasoning":
            candidate_specs = [
                ("xlam", xlam_url, xlam_model),
                ("ollama-high", ollama_url, ollama_high_reasoning_model),
                ("ollama-balanced", ollama_url, ollama_balanced_model),
                ("ollama-low", ollama_url, ollama_model),
            ]
        else:
            candidate_specs = [
                ("xlam", xlam_url, xlam_model),
                ("ollama-balanced", ollama_url, ollama_balanced_model),
                ("ollama-low", ollama_url, ollama_model),
            ]

        seen: set[Tuple[str, str]] = set()
        for label, base_url, model in candidate_specs:
            if not base_url or not model:
                continue
            normalized_url = cls._normalize_openai_base_url(base_url)
            dedup_key = (normalized_url, model)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            route_specs.append((f"{label}:{model}", normalized_url, model))
        return route_specs

    def _has_available_instructor_route(self, profile: str) -> bool:
        routes = self._instructor_routes_by_profile.get(profile, [])
        return any(extractor.is_available for _, extractor in routes)

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

    def extract(
        self,
        text: str,
        use_llm: bool = True,
        model_profile: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Run extraction pipeline. Returns combined results from best available tier.

        v3.1.0 routing:
          Tier 1 (rules) → always runs first (0ms, guaranteed baseline)
          Tier 2 (Instructor) → if feature flag ON + endpoint available
          Tier 3 (xLAM legacy) → fallback if Instructor unavailable
          Merge: Union entities/relations, deduplicate by lowercase name

        Args:
            text: Input text to extract from.
            use_llm: Whether to attempt LLM-based extraction (Tier 2/3).
        """
        # Always run Tier 1 (rule-based) — it's free and fast
        result = rule_based_extract(text)

        if not use_llm:
            return result

        selected_profile = (
            model_profile
            if model_profile in SUPPORTED_MODEL_PROFILES
            else self.model_profile
        )
        if selected_profile not in SUPPORTED_MODEL_PROFILES:
            selected_profile = "balanced"

        # Tier 2: Instructor structured extraction (v3.1.0)
        for route_label, extractor in self._instructor_routes_by_profile.get(selected_profile, []):
            if not extractor.is_available:
                continue
            try:
                instructor_result = extractor.extract(text)
                if instructor_result.entities or instructor_result.relations:
                    logger.debug("Instructor route succeeded: %s", route_label)
                    result = self._merge_results(result, instructor_result)
                    return result
                logger.debug("Instructor route returned empty: %s", route_label)
            except Exception as e:
                logger.warning(
                    "Instructor route failed (%s), trying next route: %s",
                    route_label,
                    e,
                )

        # Tier 3: xLAM chain-of-extraction (legacy)
        if self._xlam_available:
            try:
                xlam_result = self._xlam_extract(text)
                result = self._merge_results(result, xlam_result)
                return result
            except Exception as e:
                logger.warning("xLAM extraction failed: %s", e)

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
                    "model": self.xlam_model,
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
                        "model": self.xlam_model,
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
                    "model": self.xlam_model,
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
        """
        Tier 3: Ollama-based extraction.

        Deprecated in v3.1.0: Instructor now handles Ollama extraction
        via the unified OpenAI-compatible interface. This method remains
        as a no-op fallback for backward compatibility.
        """
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
