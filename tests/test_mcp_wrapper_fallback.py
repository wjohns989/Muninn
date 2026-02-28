import json
import pytest
import mcp_wrapper
import requests

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

def test_search_memory_project_fallback(monkeypatch):
    """
    Verify that search_memory retries without the project filter 
    if the initial scoped search returns no results.
    """
    requests_history = []

    def mock_request(method, url, **kwargs):
        import copy
        json_body = kwargs.get("json", {})
        requests_history.append(copy.deepcopy(json_body))
        filters = json_body.get("filters", {})

        # Scenario:
        # 1. First call with project="muninn" -> return empty
        # 2. Second call without project -> return results
        if filters.get("project") == "muninn":
            return MockResponse({"success": True, "data": []})
        else:
            return MockResponse({
                "success": True, 
                "data": [{"content": "found global memory", "score": 0.9, "memory_type": "text"}]
            })

    # Setup mocks
    # Patch handlers directly as mcp_wrapper facade delegates to it
    monkeypatch.setattr("muninn.mcp.handlers.make_request_with_retry", mock_request)
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn", "branch": "main"})
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.ensure_server_running", lambda: True)
    monkeypatch.setattr(mcp_wrapper, "_backend_circuit_open", lambda: False)
    monkeypatch.setattr(mcp_wrapper, "_startup_recovery_allowed", lambda epoch: True)
    # Patch handlers directly as mcp_wrapper facade delegates to it
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn", "branch": "main"})
    
    # Mock send_json_rpc to capture output
    sent_messages = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent_messages.append(msg))

    # Execute tool call
    mcp_wrapper.handle_call_tool("msg-1", {
        "name": "search_memory",
        "arguments": {"query": "test query"}
    })

    # Assertions
    # 1. Should have made 2 requests to the backend
    assert len(requests_history) == 2
    assert requests_history[0]["filters"]["project"] == "muninn"
    assert "project" not in requests_history[1]["filters"]

    # 2. Should have returned the global memory result
    assert len(sent_messages) == 1
    result_text = sent_messages[0]["result"]["content"][0]["text"]
    assert "found global memory" in result_text
    assert "No relevant memories found" not in result_text

def test_search_memory_no_fallback_on_manual_filter(monkeypatch):
    """
    Verify that search_memory does NOT fallback if the user 
    explicitly provided a project filter (even if it returns no results).
    """
    requests_history = []

    def mock_request(method, url, **kwargs):
        requests_history.append(kwargs.get("json", {}))
        return MockResponse({"success": True, "data": []})

    monkeypatch.setattr("muninn.mcp.handlers.make_request_with_retry", mock_request)
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "muninn_mcp", "branch": "main"})
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr("muninn.mcp.handlers.ensure_server_running", lambda: True)
    monkeypatch.setattr(mcp_wrapper, "_backend_circuit_open", lambda: False)
    
    sent_messages = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent_messages.append(msg))

    # Execute tool call with EXPLICIT filter
    mcp_wrapper.handle_call_tool("msg-2", {
        "name": "search_memory",
        "arguments": {
            "query": "test query",
            "filters": {"project": "other_project"}
        }
    })

    # Should only make 1 request because the filter was NOT auto-injected
    assert len(requests_history) == 1
    assert requests_history[0]["filters"]["project"] == "other_project"
    assert "[]" in sent_messages[0]["result"]["content"][0]["text"]