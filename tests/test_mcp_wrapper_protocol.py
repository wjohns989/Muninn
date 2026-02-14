import copy

import pytest

import mcp_wrapper


@pytest.fixture(autouse=True)
def reset_session_state():
    previous = copy.deepcopy(mcp_wrapper._SESSION_STATE)
    mcp_wrapper._SESSION_STATE.clear()
    mcp_wrapper._SESSION_STATE.update({
        "negotiated": False,
        "initialized": False,
        "protocol_version": mcp_wrapper.SUPPORTED_PROTOCOL_VERSIONS[0],
    })
    yield
    mcp_wrapper._SESSION_STATE.clear()
    mcp_wrapper._SESSION_STATE.update(previous)


def test_negotiate_protocol_supported():
    assert mcp_wrapper._negotiate_protocol_version("2025-11-25") == "2025-11-25"
    assert mcp_wrapper._negotiate_protocol_version("2025-06-18") == "2025-06-18"
    assert mcp_wrapper._negotiate_protocol_version("2024-11-05") == "2024-11-05"


def test_negotiate_protocol_unsupported():
    assert mcp_wrapper._negotiate_protocol_version("2023-01-01") is None


def test_negotiate_protocol_default():
    assert mcp_wrapper._negotiate_protocol_version(None) == mcp_wrapper.SUPPORTED_PROTOCOL_VERSIONS[0]


def test_initialize_rejects_unsupported_protocol(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper.handle_initialize("req-1", {"protocolVersion": "1999-01-01"})

    assert len(sent) == 1
    assert sent[0]["id"] == "req-1"
    assert "error" in sent[0]
    assert sent[0]["error"]["code"] == -32602
    assert "Unsupported protocol version" in sent[0]["error"]["message"]


def test_initialize_rejects_non_object_params(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-init-bad",
        "method": "initialize",
        "params": "bad",
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "initialize params must be an object" in sent[0]["error"]["message"]


def test_unknown_request_method_returns_method_not_found(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-unknown",
        "method": "unknown/method",
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32601
    assert "Method not found" in sent[0]["error"]["message"]


def test_unknown_notification_method_is_ignored(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "method": "unknown/notification",
    })

    assert sent == []


def test_initialized_notification_requires_prior_initialize(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })

    assert mcp_wrapper._SESSION_STATE["initialized"] is False
    assert sent == []


def test_tools_list_before_initialized_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tools-list",
        "method": "tools/list",
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32600
    assert "Server not initialized" in sent[0]["error"]["message"]


def test_tools_call_invalid_params_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tools-call",
        "method": "tools/call",
        "params": {"name": "", "arguments": []},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "tools/call requires non-empty string name" in sent[0]["error"]["message"]


def test_list_tools_adds_json_schema_and_annotations(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper.handle_list_tools("req-2")

    assert len(sent) == 1
    tools = sent[0]["result"]["tools"]
    assert tools

    by_name = {tool["name"]: tool for tool in tools}
    assert "record_retrieval_feedback" in by_name
    assert "search_memory" in by_name
    assert "ingest_sources" in by_name

    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["$schema"] == mcp_wrapper.JSON_SCHEMA_2020_12
        assert "annotations" in tool
        assert "readOnlyHint" in tool["annotations"]

    assert by_name["search_memory"]["annotations"]["readOnlyHint"] is True
    assert by_name["record_retrieval_feedback"]["annotations"]["readOnlyHint"] is False
    assert by_name["ingest_sources"]["annotations"]["readOnlyHint"] is False
    feedback_props = by_name["record_retrieval_feedback"]["inputSchema"]["properties"]
    assert "rank" in feedback_props
    assert "sampling_prob" in feedback_props
    ingest_props = by_name["ingest_sources"]["inputSchema"]["properties"]
    assert "sources" in ingest_props


def test_tool_schemas_have_consistent_contract(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper.handle_list_tools("req-schema")
    tools = sent[0]["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert len(names) == len(set(names))

    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties"), dict)
        for required_field in schema.get("required", []):
            assert required_field in schema["properties"]


def test_ingest_sources_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)
    monkeypatch.setattr(mcp_wrapper, "get_git_info", lambda: {"project": "muninn", "branch": "main"})

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "INGEST_COMPLETED"}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-ingest",
        {
            "name": "ingest_sources",
            "arguments": {
                "sources": ["/tmp/a.txt"],
                "recursive": True,
                "chunk_size_chars": 500,
            },
        },
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/ingest")
    assert captured["json"]["sources"] == ["/tmp/a.txt"]
    assert captured["json"]["project"] == "muninn"
    assert captured["json"]["recursive"] is True
    assert captured["json"]["chunk_size_chars"] == 500
    assert sent
    assert sent[0]["id"] == "req-ingest"
