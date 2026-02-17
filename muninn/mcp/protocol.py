"""
Muninn MCP Protocol Constants & Schemas
"""

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"
SUPPORTED_MODEL_PROFILES = ("low_latency", "balanced", "high_reasoning")

# Standard MCP Error Codes
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
PARSE_ERROR = -32700

# Server configuration
SERVER_URL = "http://localhost:8000"

def negotiate_protocol_version(version: str | None) -> str | None:
    if not version:
        return SUPPORTED_PROTOCOL_VERSIONS[0]
    if version in SUPPORTED_PROTOCOL_VERSIONS:
        return version
    return None

# Muninn Specific Error Codes
BACKEND_UNAVAILABLE = -32000
TASK_NOT_FOUND = -32001
TIMEOUT = -32002
