"""
Muninn Extraction Models
-------------------------
Pydantic schemas for structured entity/relation extraction.

These models serve dual purpose:
1. Response schemas for Instructor-based LLM extraction (structured output)
2. Validation layer for rule-based extraction results

Using Pydantic v2 with Field descriptions enables Instructor to
generate JSON schema automatically, providing the LLM with clear
extraction instructions without fragile prompt engineering.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Valid type enums (as constrained strings for LLM compatibility)
# ---------------------------------------------------------------------------

VALID_ENTITY_TYPES = {
    "person", "org", "tech", "concept", "project",
    "file", "preference", "location", "event",
}

VALID_PREDICATES = {
    "uses", "prefers", "created", "depends_on", "knows",
    "part_of", "works_with", "located_in", "manages",
    "authored", "contributes_to", "migrated_from",
}


# ---------------------------------------------------------------------------
# Extraction response models
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """A named entity extracted from text."""
    name: str = Field(
        description="Entity name, properly capitalized. "
        "E.g. 'Python', 'John Smith', 'GitHub Actions'."
    )
    entity_type: str = Field(
        description="One of: person, org, tech, concept, project, "
        "file, preference, location, event."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Extraction confidence from 0.0 to 1.0."
    )


class ExtractedRelation(BaseModel):
    """A relationship between two entities extracted from text."""
    subject: str = Field(
        description="Source entity name (must match an extracted entity)."
    )
    predicate: str = Field(
        description="Relationship type: uses, prefers, created, depends_on, "
        "knows, part_of, works_with, located_in, manages, authored, "
        "contributes_to, migrated_from."
    )
    object: str = Field(
        description="Target entity name (must match an extracted entity)."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Extraction confidence from 0.0 to 1.0."
    )
    temporal_context: Optional[str] = Field(
        default=None,
        description="Time context if mentioned: 'recently', 'since 2024', etc."
    )


class ExtractedMemoryFacts(BaseModel):
    """
    Complete structured extraction result from a text passage.

    Instructor uses this model as the response_model, generating JSON schema
    automatically for the LLM to fill. The LLM sees the field descriptions
    as instructions.
    """
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="Named entities found in the text. Include people, "
        "technologies, organizations, concepts, projects, files, "
        "preferences, and locations."
    )
    relations: List[ExtractedRelation] = Field(
        default_factory=list,
        description="Relationships between extracted entities. "
        "Each relation connects a subject to an object via a predicate."
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description="Atomic factual statements extracted from the text. "
        "Each fact should be a single, self-contained statement."
    )
    summary: Optional[str] = Field(
        default=None,
        description="One concise sentence summarizing the key information."
    )
    temporal_context: Optional[str] = Field(
        default=None,
        description="Overall temporal context: 'present', 'past', "
        "specific dates, or time references found in the text."
    )


# ---------------------------------------------------------------------------
# System prompt for Instructor extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise memory extraction engine. Given text from a "
    "conversation or document, extract structured facts including:\n"
    "- Named entities (people, technologies, organizations, concepts, "
    "projects, files, preferences, locations)\n"
    "- Relationships between entities\n"
    "- Key factual statements (atomic, self-contained)\n"
    "- A one-sentence summary\n\n"
    "Rules:\n"
    "1. Only extract what is EXPLICITLY stated — never infer or assume.\n"
    "2. Entity names must be properly capitalized.\n"
    "3. Each key_fact must be a single atomic statement.\n"
    "4. If no entities or relations are found, return empty lists.\n"
    "5. Confidence should reflect certainty: 1.0 = explicit statement, "
    "0.5 = implied, 0.3 = uncertain.\n"
)
