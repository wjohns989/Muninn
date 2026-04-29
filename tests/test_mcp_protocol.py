import pytest

from muninn.mcp.protocol import (
    SUPPORTED_PROTOCOL_VERSIONS,
    JSON_SCHEMA_2020_12,
    SUPPORTED_MODEL_PROFILES,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
)

def test_protocol_constants():
    """Test that protocol constants are correctly defined."""
    assert isinstance(SUPPORTED_PROTOCOL_VERSIONS, tuple)
    assert "2025-11-25" in SUPPORTED_PROTOCOL_VERSIONS
    assert "2024-11-05" in SUPPORTED_PROTOCOL_VERSIONS
    assert len(SUPPORTED_PROTOCOL_VERSIONS) > 0
    assert JSON_SCHEMA_2020_12 == "https://json-schema.org/draft/2020-12/schema"
    assert isinstance(SUPPORTED_MODEL_PROFILES, tuple)
    assert "low_latency" in SUPPORTED_MODEL_PROFILES
    assert "balanced" in SUPPORTED_MODEL_PROFILES
    assert "high_reasoning" in SUPPORTED_MODEL_PROFILES

def test_error_codes():
    """Test that standard error codes have the correct values."""
    # Standard MCP Error Codes
    assert INVALID_REQUEST == -32600
    assert METHOD_NOT_FOUND == -32601
