"""
Muninn Rule-Based Extraction
-----------------------------
Tier 1 extraction: zero-latency, always-available entity and relation extraction
using regex patterns, heuristics, and lightweight NLP. No LLM required.
"""

import re
import logging
from typing import List, Tuple

from muninn.core.types import Entity, Relation, ExtractionResult

logger = logging.getLogger("Muninn.Extract.Rules")

# --- Entity Patterns ---

# Technology/framework patterns
TECH_PATTERNS = [
    r'\b(React|Vue|Angular|Next\.?js|Nuxt|Svelte|Django|Flask|FastAPI|Express|Spring|Rails)\b',
    r'\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|C#|Ruby|PHP|Swift|Kotlin)\b',
    r'\b(PostgreSQL|MySQL|MongoDB|Redis|SQLite|Qdrant|Kuzu|Elasticsearch|DynamoDB)\b',
    r'\b(Docker|Kubernetes|AWS|Azure|GCP|Vercel|Netlify|Cloudflare|Supabase|Firebase)\b',
    r'\b(Git|GitHub|GitLab|Bitbucket|npm|pip|cargo|Maven|Gradle)\b',
    r'\b(TensorFlow|PyTorch|Transformers|LangChain|LlamaIndex|OpenAI|Anthropic|Ollama)\b',
    r'\b(Linux|Windows|macOS|Ubuntu|Debian|RHEL|Alpine)\b',
]

# File path patterns
FILE_PATTERN = re.compile(
    r'(?:^|\s)([a-zA-Z_][\w/\\.-]*\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|html|css|md|yaml|yml|json|toml|sql|sh|bat))\b'
)

# URL pattern
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')

# Email pattern
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Version patterns
VERSION_PATTERN = re.compile(r'\b[vV]?(\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?)\b')

# Date-like patterns (basic)
DATE_PATTERN = re.compile(
    r'\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
    re.IGNORECASE
)

# --- Relation Patterns ---

PREFERENCE_PATTERNS = [
    (r'(?:I |we |user )(?:prefer|like|love|enjoy|use|recommend)s?\s+(.+?)(?:\.|$)', "prefers"),
    (r'(?:always |usually |typically )(?:use|prefer|choose)\s+(.+?)(?:\.|$)', "prefers"),
    (r'(.+?)\s+is (?:my|our|the) (?:preferred|favorite|go-to|default)', "prefers"),
]

DEPENDENCY_PATTERNS = [
    (r'(.+?)\s+(?:depends on|requires|needs|uses)\s+(.+?)(?:\.|$)', "depends_on"),
    (r'(.+?)\s+is (?:built with|powered by|based on)\s+(.+?)(?:\.|$)', "uses"),
]

CREATION_PATTERNS = [
    (r'(?:I |we )(?:created|built|made|wrote|developed)\s+(.+?)(?:\.|$)', "created"),
    (r'(.+?)\s+was (?:created|built|made|written|developed)\s+(?:by|in|with)\s+(.+?)(?:\.|$)', "created"),
]


def extract_entities_rule_based(text: str) -> List[Entity]:
    """Extract entities using regex patterns and heuristics."""
    entities = []
    seen = set()

    # Technology entities
    for pattern in TECH_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(1)
            key = name.lower()
            if key not in seen:
                seen.add(key)
                entities.append(Entity(name=name, entity_type="tech"))

    # File paths
    for match in FILE_PATTERN.finditer(text):
        path = match.group(1)
        if path not in seen:
            seen.add(path)
            entities.append(Entity(name=path, entity_type="file"))

    # URLs
    for match in URL_PATTERN.finditer(text):
        url = match.group()
        if url not in seen:
            seen.add(url)
            entities.append(Entity(name=url, entity_type="location"))

    # Emails
    for match in EMAIL_PATTERN.finditer(text):
        email = match.group()
        if email not in seen:
            seen.add(email)
            entities.append(Entity(name=email, entity_type="person"))

    return entities


def extract_relations_rule_based(text: str, entities: List[Entity]) -> List[Relation]:
    """Extract relations using pattern matching against known entities."""
    relations = []
    entity_names = {e.name.lower(): e.name for e in entities}

    # Preference relations
    for pattern, predicate in PREFERENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            obj_text = match.group(1).strip()
            # Check if the object matches a known entity
            for ename_lower, ename in entity_names.items():
                if ename_lower in obj_text.lower():
                    relations.append(Relation(
                        subject="user",
                        predicate="prefers",
                        object=ename,
                    ))
                    break

    # Dependency relations
    for pattern, predicate in DEPENDENCY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                subj_text = groups[0].strip()
                obj_text = groups[1].strip()
                # Match against known entities
                subj_entity = None
                obj_entity = None
                for ename_lower, ename in entity_names.items():
                    if ename_lower in subj_text.lower():
                        subj_entity = ename
                    if ename_lower in obj_text.lower():
                        obj_entity = ename
                if subj_entity and obj_entity:
                    relations.append(Relation(
                        subject=subj_entity,
                        predicate=predicate,
                        object=obj_entity,
                    ))

    return relations


def extract_temporal_context(text: str) -> str:
    """Extract temporal references from text."""
    dates = DATE_PATTERN.findall(text)
    if dates:
        return "; ".join(dates[:5])
    return ""


def rule_based_extract(text: str) -> ExtractionResult:
    """
    Complete rule-based extraction pipeline.
    Zero-latency, always available, no dependencies beyond stdlib + regex.
    """
    entities = extract_entities_rule_based(text)
    relations = extract_relations_rule_based(text, entities)
    temporal = extract_temporal_context(text)

    # Generate a simple summary (first sentence or first 200 chars)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    summary = sentences[0][:200] if sentences else text[:200]

    return ExtractionResult(
        entities=entities,
        relations=relations,
        summary=summary,
        temporal_context=temporal if temporal else None,
    )
