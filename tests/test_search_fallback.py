import json
import pytest
import mcp_wrapper
from typing import Any, Dict, List

class MockResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

def test_search_memory_fallback_triggered_when_empty_and_auto_project(monkeypatch):
    """
    Test that search_memory retries without project filter when:
    1. Project filter was auto-applied
    2. First search returns empty results
    3. Fallback env var is enabled (default)
    """
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn", "branch": "main"})
    
    # Trace calls to verify logic
    calls = []

    def _fake_request(method, url, **kwargs):
        payload = kwargs.get("json", {})
        filters = payload.get("filters", {})
        calls.append(filters)
        
        # First call has project filter -> return empty
        if "project" in filters:
            return MockResponse({"success": True, "data": []})
        
        # Second call has no project filter -> return hit
        return MockResponse({"success": True, "data": [{"id": "1", "content": "found it"}]})

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-search-fallback",
        {
            "name": "search_memory",
            "arguments": {
                "query": "hello",
                "limit": 5,
            },
        },
    )

    # Verify two calls were made
    assert len(calls) == 2
    assert calls[0]["project"] == "muninn"  # First call had project
    assert "project" not in calls[1]        # Second call dropped project

    # Verify result came from the second call
    assert sent
    result_text = sent[0]["result"]["content"][0]["text"]
    assert "found it" in result_text

def test_search_memory_fallback_skipped_if_results_found(monkeypatch):
    """Test that fallback is NOT triggered if the first search finds results."""
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn", "branch": "main"})
    
    calls = []

    def _fake_request(method, url, **kwargs):
        payload = kwargs.get("json", {})
        filters = payload.get("filters", {})
        calls.append(filters)
        
        return MockResponse({"success": True, "data": [{"id": "1", "content": "found immediately"}]})

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-search-ok",
        {
            "name": "search_memory",
            "arguments": {
                "query": "hello",
            },
        },
    )

    # Verify only one call was made
    assert len(calls) == 1
    assert calls[0]["project"] == "muninn"
    
    result_text = sent[0]["result"]["content"][0]["text"]
    assert "found immediately" in result_text

def test_search_memory_fallback_skipped_if_project_explicitly_set(monkeypatch):
    """Test that fallback is skipped if the user explicitly provided the project filter."""
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn", "branch": "main"})
    
    calls = []

    def _fake_request(method, url, **kwargs):
        payload = kwargs.get("json", {})
        filters = payload.get("filters", {})
        calls.append(filters)
        return MockResponse({"success": True, "data": []})

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-search-explicit",
        {
            "name": "search_memory",
            "arguments": {
                "query": "hello",
                "filters": {"project": "other_project"}
            },
        },
    )

    # Verify only one call was made because project was explicit
    assert len(calls) == 1
    assert calls[0]["project"] == "other_project"

def test_search_memory_fallback_disabled_via_env(monkeypatch):
    """Test that fallback can be disabled via environment variable."""
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn", "branch": "main"})
    monkeypatch.setenv("MUNINN_MCP_SEARCH_PROJECT_FALLBACK", "false")
    
    calls = []

    def _fake_request(method, url, **kwargs):
        payload = kwargs.get("json", {})
        filters = payload.get("filters", {})
        calls.append(filters)
        return MockResponse({"success": True, "data": []})

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-search-disabled",
        {
            "name": "search_memory",
            "arguments": {
                "query": "hello",
            },
        },
    )

    # Verify only one call was made
    assert len(calls) == 1
    assert calls[0]["project"] == "muninn"