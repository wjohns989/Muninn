import copy
import io

import pytest

import mcp_wrapper


def _sample_task(
    *,
    task_id: str,
    status: str,
    result: dict | None = None,
) -> dict:
    payload = {
        "taskId": task_id,
        "status": status,
        "createdAt": "2026-02-15T04:00:00Z",
        "lastUpdatedAt": "2026-02-15T04:00:00Z",
        "ttl": 60000,
    }
    if result is not None:
        payload["result"] = result
    return payload


@pytest.fixture(autouse=True)
def reset_session_state():
    previous = copy.deepcopy(mcp_wrapper._SESSION_STATE)
    mcp_wrapper._SESSION_STATE.clear()
    mcp_wrapper._SESSION_STATE.update({
        "negotiated": False,
        "initialized": False,
        "protocol_version": mcp_wrapper.SUPPORTED_PROTOCOL_VERSIONS[0],
        "client_capabilities": {},
        "client_elicitation_modes": tuple(),
        "tasks": {},
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


def test_read_rpc_message_supports_json_line():
    stream = io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    msg = mcp_wrapper._read_rpc_message(stream)
    assert msg is not None
    assert msg["method"] == "ping"


def test_read_rpc_message_supports_content_length_framing():
    payload = b'{"jsonrpc":"2.0","id":2,"method":"ping"}'
    framed = (
        b"Content-Length: "
        + str(len(payload)).encode("ascii")
        + b"\r\nContent-Type: application/json\r\n\r\n"
        + payload
    )
    stream = io.BytesIO(framed)
    msg = mcp_wrapper._read_rpc_message(stream)
    assert msg is not None
    assert msg["method"] == "ping"
    assert msg["id"] == 2


def test_read_rpc_message_invalid_content_length_returns_none():
    stream = io.BytesIO(b"Content-Length: nope\r\n\r\n{}")
    assert mcp_wrapper._read_rpc_message(stream) is None


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


def test_initialize_includes_startup_warnings(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(
        mcp_wrapper,
        "_collect_startup_warnings",
        lambda: [
            "Muninn server is not reachable at http://localhost:42069. Start it with: python server.py",
            "Ollama is not reachable at http://localhost:11434. Start it with: ollama serve",
        ],
    )

    mcp_wrapper.handle_initialize("req-startup", {"protocolVersion": "2025-11-25"})

    assert len(sent) == 1
    instructions = sent[0]["result"]["instructions"]
    assert "Startup checks:" in instructions
    assert "python server.py" in instructions
    assert "ollama serve" in instructions


def test_initialize_without_warnings_uses_base_instructions(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "_collect_startup_warnings", lambda: [])

    mcp_wrapper.handle_initialize("req-clean", {"protocolVersion": "2025-11-25"})

    assert len(sent) == 1
    instructions = sent[0]["result"]["instructions"]
    assert "Startup checks:" not in instructions
    assert "cross-assistant continuity" in instructions


def test_initialize_includes_session_model_profile(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "_collect_startup_warnings", lambda: [])
    monkeypatch.setenv("MUNINN_OPERATOR_MODEL_PROFILE", "high_reasoning")

    mcp_wrapper.handle_initialize("req-profile", {"protocolVersion": "2025-11-25"})

    assert len(sent) == 1
    instructions = sent[0]["result"]["instructions"]
    assert "Session model profile: high_reasoning" in instructions


def test_initialize_advertises_tasks_capability(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "_collect_startup_warnings", lambda: [])

    mcp_wrapper.handle_initialize("req-capabilities", {"protocolVersion": "2025-11-25"})

    assert len(sent) == 1
    capabilities = sent[0]["result"]["capabilities"]
    assert capabilities["tools"]["listChanged"] is False
    assert capabilities["tasks"]["list"] == {}
    assert capabilities["tasks"]["cancel"] == {}


def test_initialize_elicitation_empty_object_defaults_to_form_mode(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "_collect_startup_warnings", lambda: [])

    mcp_wrapper.handle_initialize(
        "req-elicitation-default",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "elicitation": {}
            },
        },
    )

    assert len(sent) == 1
    assert mcp_wrapper._SESSION_STATE["client_elicitation_modes"] == ("form",)


def test_initialize_elicitation_modes_include_form_and_url(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "_collect_startup_warnings", lambda: [])

    mcp_wrapper.handle_initialize(
        "req-elicitation-modes",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "elicitation": {
                    "form": {},
                    "url": {},
                }
            },
        },
    )

    assert len(sent) == 1
    assert mcp_wrapper._SESSION_STATE["client_elicitation_modes"] == ("form", "url")


def test_collect_startup_warnings_when_dependencies_unavailable(monkeypatch):
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: False)
    monkeypatch.setattr(mcp_wrapper, "check_and_start_ollama", lambda: False)

    warnings = mcp_wrapper._collect_startup_warnings(
        autostart_server=True,
        autostart_ollama=True,
    )

    assert any("Muninn server is not reachable" in warning for warning in warnings)
    assert any("ollama serve" in warning for warning in warnings)


def test_bootstrap_dependencies_on_launch_honors_flags(monkeypatch):
    calls = {"server": 0, "ollama": 0}
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_ON_LAUNCH", "1")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "1")
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_OLLAMA", "0")

    def _server():
        calls["server"] += 1
        return True

    def _ollama():
        calls["ollama"] += 1
        return True

    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", _server)
    monkeypatch.setattr(mcp_wrapper, "check_and_start_ollama", _ollama)

    mcp_wrapper._bootstrap_dependencies_on_launch()
    assert calls["server"] == 1
    assert calls["ollama"] == 0


def test_bootstrap_dependencies_on_launch_disabled(monkeypatch):
    calls = {"server": 0, "ollama": 0}
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_ON_LAUNCH", "0")

    monkeypatch.setattr(
        mcp_wrapper,
        "ensure_server_running",
        lambda: calls.__setitem__("server", calls["server"] + 1) or True,
    )
    monkeypatch.setattr(
        mcp_wrapper,
        "check_and_start_ollama",
        lambda: calls.__setitem__("ollama", calls["ollama"] + 1) or True,
    )

    mcp_wrapper._bootstrap_dependencies_on_launch()
    assert calls["server"] == 0
    assert calls["ollama"] == 0


def test_add_memory_injects_operator_profile_into_metadata(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)
    monkeypatch.setattr(mcp_wrapper, "get_git_info", lambda: {"project": "muninn", "branch": "main"})
    monkeypatch.setenv("MUNINN_OPERATOR_MODEL_PROFILE", "balanced")

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-add-memory",
        {
            "name": "add_memory",
            "arguments": {
                "content": "hello world",
                "metadata": {"source": "test"},
            },
        },
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/add")
    metadata = captured["json"]["metadata"]
    assert metadata["source"] == "test"
    assert metadata["operator_model_profile"] == "balanced"


def test_operation_specific_profile_overrides_generic_profile(monkeypatch):
    monkeypatch.setenv("MUNINN_OPERATOR_MODEL_PROFILE", "balanced")
    monkeypatch.setenv("MUNINN_OPERATOR_INGESTION_MODEL_PROFILE", "high_reasoning")

    metadata = mcp_wrapper._inject_operator_profile_metadata({}, operation="ingest")
    assert metadata["operator_model_profile"] == "high_reasoning"

    runtime_metadata = mcp_wrapper._inject_operator_profile_metadata({}, operation="add")
    assert runtime_metadata["operator_model_profile"] == "balanced"


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


def test_tasks_list_before_initialized_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-list",
        "method": "tasks/list",
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32600
    assert "Server not initialized" in sent[0]["error"]["message"]


def test_tasks_list_invalid_params_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-list-invalid",
        "method": "tasks/list",
        "params": [],
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "tasks/list params must be an object" in sent[0]["error"]["message"]


def test_tasks_list_returns_empty_collection(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-list-ok",
        "method": "tasks/list",
        "params": {"cursor": "cursor-1"},
    })

    assert len(sent) == 1
    assert sent[0]["id"] == "req-tasks-list-ok"
    assert sent[0]["result"]["tasks"] == []


def test_tasks_get_invalid_params_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-get-invalid",
        "method": "tasks/get",
        "params": [],
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "tasks/get params must be an object" in sent[0]["error"]["message"]


def test_tasks_get_unknown_task_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-get-unknown",
        "method": "tasks/get",
        "params": {"taskId": "task-unknown"},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "unknown taskId" in sent[0]["error"]["message"]


def test_tasks_get_unknown_task_does_not_reflect_raw_task_id(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-get-unknown-safe",
        "method": "tasks/get",
        "params": {"taskId": "<script>alert(1)</script>"},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "<script>" not in sent[0]["error"]["message"]
    assert "unknown taskId" in sent[0]["error"]["message"]


def test_tasks_get_returns_task(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(task_id="task-1", status="working"),
    }

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-get-ok",
        "method": "tasks/get",
        "params": {"taskId": "task-1"},
    })

    assert len(sent) == 1
    assert sent[0]["id"] == "req-tasks-get-ok"
    assert sent[0]["result"]["taskId"] == "task-1"
    assert sent[0]["result"]["status"] == "working"


def test_tasks_result_requires_terminal_status(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(task_id="task-1", status="working"),
    }

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-result-not-done",
        "method": "tasks/result",
        "params": {"taskId": "task-1"},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32001
    assert "not complete" in sent[0]["error"]["message"]


def test_tasks_result_returns_payload_for_completed_task(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(
            task_id="task-1",
            status="completed",
            result={"content": [{"type": "text", "text": "ok"}]},
        ),
    }

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-result-ok",
        "method": "tasks/result",
        "params": {"taskId": "task-1"},
    })

    assert len(sent) == 1
    assert sent[0]["id"] == "req-tasks-result-ok"
    assert sent[0]["result"]["content"][0]["text"] == "ok"


def test_tasks_result_rejects_terminal_task_without_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(task_id="task-1", status="cancelled"),
    }

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-result-cancelled-missing",
        "method": "tasks/result",
        "params": {"taskId": "task-1"},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32001
    assert "result is unavailable for status 'cancelled'" in sent[0]["error"]["message"]


def test_tasks_cancel_rejects_terminal_task(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(task_id="task-1", status="completed"),
    }

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-cancel-terminal",
        "method": "tasks/cancel",
        "params": {"taskId": "task-1"},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32001
    assert "already terminal" in sent[0]["error"]["message"]


def test_tasks_cancel_updates_task_state(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(task_id="task-1", status="working"),
    }

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-cancel-ok",
        "method": "tasks/cancel",
        "params": {"taskId": "task-1"},
    })

    assert len(sent) == 1
    assert sent[0]["id"] == "req-tasks-cancel-ok"
    assert sent[0]["result"]["status"] == "cancelled"
    assert "Cancelled by client request." in sent[0]["result"]["statusMessage"]


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
    assert "get_model_profiles" in by_name
    assert "set_model_profiles" in by_name
    assert "get_model_profile_events" in by_name
    assert "ingest_sources" in by_name
    assert "discover_legacy_sources" in by_name
    assert "ingest_legacy_sources" in by_name

    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["$schema"] == mcp_wrapper.JSON_SCHEMA_2020_12
        assert "annotations" in tool
        assert "readOnlyHint" in tool["annotations"]
        assert "destructiveHint" in tool["annotations"]
        assert "idempotentHint" in tool["annotations"]
        assert "openWorldHint" in tool["annotations"]
        assert tool["execution"]["taskSupport"] == "forbidden"

    assert by_name["search_memory"]["annotations"]["readOnlyHint"] is True
    assert by_name["get_model_profiles"]["annotations"]["readOnlyHint"] is True
    assert by_name["get_model_profile_events"]["annotations"]["readOnlyHint"] is True
    assert by_name["set_model_profiles"]["annotations"]["readOnlyHint"] is False
    assert by_name["record_retrieval_feedback"]["annotations"]["readOnlyHint"] is False
    assert by_name["ingest_sources"]["annotations"]["readOnlyHint"] is False
    assert by_name["discover_legacy_sources"]["annotations"]["readOnlyHint"] is True
    assert by_name["ingest_legacy_sources"]["annotations"]["readOnlyHint"] is False
    assert by_name["delete_memory"]["annotations"]["destructiveHint"] is True
    assert by_name["delete_all_memories"]["annotations"]["destructiveHint"] is True
    assert by_name["search_memory"]["annotations"]["idempotentHint"] is True
    assert by_name["set_model_profiles"]["annotations"]["idempotentHint"] is True
    assert by_name["update_memory"]["annotations"]["idempotentHint"] is True
    assert by_name["import_handoff"]["annotations"]["idempotentHint"] is True
    feedback_props = by_name["record_retrieval_feedback"]["inputSchema"]["properties"]
    assert "rank" in feedback_props
    assert "sampling_prob" in feedback_props
    ingest_props = by_name["ingest_sources"]["inputSchema"]["properties"]
    assert "sources" in ingest_props
    assert "chronological_order" in ingest_props
    legacy_discover_props = by_name["discover_legacy_sources"]["inputSchema"]["properties"]
    assert "roots" in legacy_discover_props
    legacy_ingest_props = by_name["ingest_legacy_sources"]["inputSchema"]["properties"]
    assert "selected_source_ids" in legacy_ingest_props
    set_profile_props = by_name["set_model_profiles"]["inputSchema"]["properties"]
    assert "runtime_model_profile" in set_profile_props
    assert "legacy_ingestion_model_profile" in set_profile_props
    assert "source" in set_profile_props
    profile_event_props = by_name["get_model_profile_events"]["inputSchema"]["properties"]
    assert "limit" in profile_event_props


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
    assert captured["json"]["chronological_order"] == "none"
    assert sent
    assert sent[0]["id"] == "req-ingest"


def test_get_model_profiles_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"active": {"runtime_model_profile": "low_latency"}}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-get-profiles",
        {
            "name": "get_model_profiles",
            "arguments": {},
        },
    )

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/profiles/model")
    assert captured["json"] is None
    assert sent
    assert sent[0]["id"] == "req-get-profiles"


def test_set_model_profiles_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "MODEL_PROFILE_POLICY_UPDATED"}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-set-profiles",
        {
            "name": "set_model_profiles",
            "arguments": {
                "runtime_model_profile": "low_latency",
                "ingestion_model_profile": "balanced",
            },
        },
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/profiles/model")
    assert captured["json"]["runtime_model_profile"] == "low_latency"
    assert captured["json"]["ingestion_model_profile"] == "balanced"
    assert captured["json"]["source"] == "mcp_tool"
    assert sent
    assert sent[0]["id"] == "req-set-profiles"


def test_set_model_profiles_requires_field(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    mcp_wrapper.handle_call_tool(
        "req-set-profiles-empty",
        {
            "name": "set_model_profiles",
            "arguments": {},
        },
    )

    assert sent
    assert sent[0]["id"] == "req-set-profiles-empty"
    assert sent[0]["error"]["code"] == -32603
    assert "requires at least one profile field" in sent[0]["error"]["message"]


def test_get_model_profile_events_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "MODEL_PROFILE_EVENTS", "count": 1}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-profile-events",
        {
            "name": "get_model_profile_events",
            "arguments": {"limit": 12},
        },
    )

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/profiles/model/events")
    assert captured["params"]["limit"] == 12
    assert sent
    assert sent[0]["id"] == "req-profile-events"


def test_discover_legacy_sources_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "LEGACY_DISCOVERY_COMPLETED"}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-legacy-discover",
        {
            "name": "discover_legacy_sources",
            "arguments": {
                "providers": ["codex_cli", "serena_memory"],
                "max_results_per_provider": 25,
            },
        },
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/ingest/legacy/discover")
    assert captured["json"]["providers"] == ["codex_cli", "serena_memory"]
    assert captured["json"]["max_results_per_provider"] == 25
    assert sent
    assert sent[0]["id"] == "req-legacy-discover"


def test_ingest_legacy_sources_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)
    monkeypatch.setattr(mcp_wrapper, "get_git_info", lambda: {"project": "muninn", "branch": "main"})

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "LEGACY_INGEST_COMPLETED"}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-legacy-import",
        {
            "name": "ingest_legacy_sources",
            "arguments": {
                "selected_source_ids": ["src_123"],
                "chunk_size_chars": 700,
                "chronological_order": "oldest_first",
            },
        },
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/ingest/legacy/import")
    assert captured["json"]["selected_source_ids"] == ["src_123"]
    assert captured["json"]["project"] == "muninn"
    assert captured["json"]["chunk_size_chars"] == 700
    assert captured["json"]["chronological_order"] == "oldest_first"
    assert sent
    assert sent[0]["id"] == "req-legacy-import"
