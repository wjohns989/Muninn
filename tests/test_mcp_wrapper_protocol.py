import copy
import io
import threading
import time

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


@pytest.fixture(autouse=True)
def reset_transport_and_backend_state():
    mcp_wrapper._TRANSPORT_CLOSED.clear()
    with mcp_wrapper._BACKEND_CIRCUIT_LOCK:
        previous = dict(mcp_wrapper._BACKEND_CIRCUIT_STATE)
        mcp_wrapper._BACKEND_CIRCUIT_STATE["consecutive_failures"] = 0
        mcp_wrapper._BACKEND_CIRCUIT_STATE["open_until_epoch"] = 0.0
    yield
    mcp_wrapper._TRANSPORT_CLOSED.clear()
    with mcp_wrapper._BACKEND_CIRCUIT_LOCK:
        mcp_wrapper._BACKEND_CIRCUIT_STATE.update(previous)


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


def test_read_rpc_message_invalid_content_length_skips_to_next_message():
    stream = io.BytesIO(
        b"Content-Length: nope\r\n\r\n"
        b'{"jsonrpc":"2.0","id":3,"method":"ping"}\n'
    )
    msg = mcp_wrapper._read_rpc_message(stream)
    assert msg is not None
    assert msg["id"] == 3
    assert msg["method"] == "ping"


def test_read_rpc_message_invalid_framed_payload_skips_to_next_message():
    invalid_payload = b"{bad-json}"
    valid_payload = b'{"jsonrpc":"2.0","id":4,"method":"ping"}'
    framed = (
        b"Content-Length: "
        + str(len(invalid_payload)).encode("ascii")
        + b"\r\n\r\n"
        + invalid_payload
        + b"Content-Length: "
        + str(len(valid_payload)).encode("ascii")
        + b"\r\n\r\n"
        + valid_payload
    )
    stream = io.BytesIO(framed)
    msg = mcp_wrapper._read_rpc_message(stream)
    assert msg is not None
    assert msg["id"] == 4
    assert msg["method"] == "ping"


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
    assert capabilities["tasks"]["requests"]["tools/call"] == {}
    assert capabilities["tasks"]["notifications"]["status"] == {}


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


def test_handle_call_tool_skips_preflight_when_autostart_disabled(monkeypatch):
    sent = []
    calls = {"ensure": 0}
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "0")
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    def _ensure():
        calls["ensure"] += 1
        return True

    class _Resp:
        def json(self):
            return {"success": True}

    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", _ensure)
    monkeypatch.setattr(mcp_wrapper, "_backend_circuit_open", lambda now_epoch=None: False)
    monkeypatch.setattr(mcp_wrapper, "get_git_info", lambda: {"project": "muninn", "branch": "main"})
    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", lambda *a, **k: _Resp())

    mcp_wrapper.handle_call_tool(
        "req-search-no-autostart",
        {"name": "search_memory", "arguments": {"query": "hello", "limit": 1}},
    )

    assert calls["ensure"] == 0
    assert sent
    assert sent[0]["id"] == "req-search-no-autostart"


def test_handle_call_tool_skips_preflight_when_backend_circuit_open(monkeypatch):
    sent = []
    calls = {"ensure": 0}
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "1")
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    def _ensure():
        calls["ensure"] += 1
        return True

    class _Resp:
        def json(self):
            return {"success": True}

    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", _ensure)
    monkeypatch.setattr(mcp_wrapper, "_backend_circuit_open", lambda now_epoch=None: True)
    monkeypatch.setattr(mcp_wrapper, "get_git_info", lambda: {"project": "muninn", "branch": "main"})
    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", lambda *a, **k: _Resp())

    mcp_wrapper.handle_call_tool(
        "req-search-circuit-open",
        {"name": "search_memory", "arguments": {"query": "hello", "limit": 1}},
    )

    assert calls["ensure"] == 0
    assert sent
    assert sent[0]["id"] == "req-search-circuit-open"


def test_handle_call_tool_skips_preflight_when_deadline_budget_low(monkeypatch):
    sent = []
    calls = {"ensure": 0}
    monkeypatch.setenv("MUNINN_MCP_AUTOSTART_SERVER", "1")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", "1")
    monkeypatch.setenv("MUNINN_MCP_STARTUP_RECOVERY_MIN_BUDGET_SEC", "2")
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    def _ensure():
        calls["ensure"] += 1
        return True

    class _Resp:
        def json(self):
            return {"success": True}

    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", _ensure)
    monkeypatch.setattr(mcp_wrapper, "_backend_circuit_open", lambda now_epoch=None: False)
    monkeypatch.setattr(mcp_wrapper, "get_git_info", lambda: {"project": "muninn", "branch": "main"})
    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", lambda *a, **k: _Resp())

    mcp_wrapper.handle_call_tool(
        "req-search-low-budget",
        {"name": "search_memory", "arguments": {"query": "hello", "limit": 1}},
    )

    assert calls["ensure"] == 0
    assert sent
    assert sent[0]["id"] == "req-search-low-budget"


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


def test_background_dispatch_selection():
    assert mcp_wrapper._should_dispatch_in_background({"method": "tasks/result"}) is True
    assert mcp_wrapper._should_dispatch_in_background({"method": "tools/call"}) is False
    assert mcp_wrapper._should_dispatch_in_background({"method": "ping"}) is False


def test_background_dispatch_selection_can_enable_tools_call(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_BACKGROUND_TOOLS_CALL", "1")
    assert mcp_wrapper._should_dispatch_in_background({"method": "tools/call"}) is True


def test_dispatch_guard_logs_generic_message(monkeypatch):
    logged = []
    monkeypatch.setattr(
        mcp_wrapper,
        "_dispatch_rpc_message",
        lambda _msg: (_ for _ in ()).throw(ValueError("bad\nuser")),
    )
    monkeypatch.setattr(mcp_wrapper.logger, "error", lambda msg: logged.append(msg))

    mcp_wrapper._dispatch_rpc_message_guarded({"method": "tasks/result"})
    assert logged == ["An unexpected error occurred during RPC dispatch."]


def test_dispatch_guard_returns_internal_error_for_request_id(monkeypatch):
    sent = []
    monkeypatch.setattr(
        mcp_wrapper,
        "_dispatch_rpc_message",
        lambda _msg: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    mcp_wrapper._dispatch_rpc_message_guarded(
        {"jsonrpc": "2.0", "id": "req-dispatch-crash", "method": "tasks/result"}
    )

    assert len(sent) == 1
    assert sent[0]["id"] == "req-dispatch-crash"
    assert sent[0]["error"]["code"] == -32603
    assert "Internal error during request dispatch." in sent[0]["error"]["message"]


def test_send_json_rpc_marks_transport_closed_on_broken_pipe(monkeypatch):
    sent = []

    class _StdoutStub:
        def write(self, _value):
            sent.append("write")
            return 0

        def flush(self):
            raise BrokenPipeError("pipe closed")

    monkeypatch.setattr(mcp_wrapper.sys, "stdout", _StdoutStub())

    mcp_wrapper.send_json_rpc({"jsonrpc": "2.0", "id": "req", "result": {}})
    assert mcp_wrapper._TRANSPORT_CLOSED.is_set() is True
    writes_before = len(sent)

    # No-op once transport is closed.
    mcp_wrapper.send_json_rpc({"jsonrpc": "2.0", "id": "req-2", "result": {}})
    assert len(sent) == writes_before


def test_submit_background_dispatch_uses_executor(monkeypatch):
    calls = []

    class _StubFuture:
        def __init__(self):
            self.callbacks = []

        def add_done_callback(self, cb):
            self.callbacks.append(cb)

    class _StubExecutor:
        def submit(self, fn, *args):
            calls.append((fn, args))
            return _StubFuture()

    monkeypatch.setattr(mcp_wrapper, "_get_dispatch_executor", lambda: _StubExecutor())
    payload = {"method": "tasks/result", "id": "req"}
    mcp_wrapper._submit_background_dispatch(payload)
    assert len(calls) == 1
    assert calls[0][0] is mcp_wrapper._dispatch_rpc_message_guarded
    assert calls[0][1][0] == payload


def test_submit_background_dispatch_rejects_when_queue_saturated(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))

    class _SaturatedSemaphore:
        def acquire(self, blocking=True):
            return False

    monkeypatch.setattr(mcp_wrapper, "_DISPATCH_QUEUE_SEMAPHORE", _SaturatedSemaphore())

    mcp_wrapper._submit_background_dispatch(
        {"jsonrpc": "2.0", "id": "req-busy", "method": "tools/call"}
    )
    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32001
    assert "dispatch queue is saturated" in sent[0]["error"]["message"]


def test_make_request_with_retry_fast_fails_when_circuit_is_open(monkeypatch):
    calls = {"request": 0}
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)

    def _never_called(_method, _url, **_kwargs):
        calls["request"] += 1
        raise AssertionError("requests.request should not be called when circuit is open")

    monkeypatch.setattr(mcp_wrapper.requests, "request", _never_called)
    with mcp_wrapper._BACKEND_CIRCUIT_LOCK:
        mcp_wrapper._BACKEND_CIRCUIT_STATE["consecutive_failures"] = (
            mcp_wrapper._BACKEND_CIRCUIT_FAILURE_THRESHOLD
        )
        mcp_wrapper._BACKEND_CIRCUIT_STATE["open_until_epoch"] = time.time() + 10

    with pytest.raises(mcp_wrapper._BackendCircuitOpenError):
        mcp_wrapper.make_request_with_retry("GET", "http://localhost:42069/health", timeout=0.1)
    assert calls["request"] == 0


def test_make_request_with_retry_clamps_timeout_to_remaining_deadline(monkeypatch):
    observed = {}

    class _Resp:
        status_code = 200

    def _fake_request(_method, _url, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr(mcp_wrapper.requests, "request", _fake_request)

    deadline_epoch = time.monotonic() + 0.05
    mcp_wrapper.make_request_with_retry(
        "GET",
        "http://localhost:42069/health",
        timeout=10.0,
        deadline_epoch=deadline_epoch,
    )

    timeout_value = observed["timeout"]
    assert isinstance(timeout_value, float)
    assert 0 < timeout_value <= 0.06


def test_make_request_with_retry_fails_before_request_when_deadline_exhausted(monkeypatch):
    calls = {"request": 0}

    def _never_called(_method, _url, **_kwargs):
        calls["request"] += 1
        raise AssertionError("requests.request should not be called after deadline exhaustion")

    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: True)
    monkeypatch.setattr(mcp_wrapper.requests, "request", _never_called)

    with pytest.raises(mcp_wrapper._RequestDeadlineExceededError):
        mcp_wrapper.make_request_with_retry(
            "GET",
            "http://localhost:42069/health",
            timeout=1.0,
            deadline_epoch=time.monotonic() - 1.0,
        )
    assert calls["request"] == 0


def test_get_tool_call_deadline_seconds_uses_explicit_override(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", "42")
    monkeypatch.setenv("MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC", "120")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "10")

    assert mcp_wrapper._get_tool_call_deadline_seconds() == 42.0


def test_get_tool_call_deadline_seconds_can_disable_budget(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", "0")
    assert mcp_wrapper._get_tool_call_deadline_seconds() is None


def test_get_tool_call_deadline_seconds_derives_from_host_timeout_and_margin(monkeypatch):
    monkeypatch.delenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", raising=False)
    monkeypatch.setenv("MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC", "90")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "15")

    assert mcp_wrapper._get_tool_call_deadline_seconds() == 75.0


def test_get_tool_call_deadline_seconds_clamps_to_minimum_when_margin_exceeds_host(monkeypatch):
    monkeypatch.delenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", raising=False)
    monkeypatch.setenv("MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC", "5")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "9")

    assert mcp_wrapper._get_tool_call_deadline_seconds() == 1.0


def test_get_tool_call_deadline_seconds_clamps_explicit_value_above_host_safe_budget(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", "119")
    monkeypatch.setenv("MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC", "120")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "10")
    monkeypatch.delenv("MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN", raising=False)

    assert mcp_wrapper._get_tool_call_deadline_seconds() == 110.0


def test_get_tool_call_deadline_seconds_allows_explicit_overrun_when_enabled(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", "119")
    monkeypatch.setenv("MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC", "120")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "10")
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN", "1")

    assert mcp_wrapper._get_tool_call_deadline_seconds() == 119.0


def test_format_tool_result_text_truncates_large_payload(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS", "512")
    text = mcp_wrapper._format_tool_result_text(
        {"success": True, "data": {"blob": "x" * 1000}},
        "discover_legacy_sources",
    )

    assert len(text) <= 512
    assert "truncated" in text


def test_public_tool_error_message_redacts_connection_details():
    msg = mcp_wrapper._public_tool_error_message(
        mcp_wrapper.requests.ConnectionError("socket timeout to 10.0.0.7")
    )
    assert "10.0.0.7" not in msg
    assert "Unable to reach backend service" in msg


def test_handle_call_tool_truncates_large_json_response(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)
    monkeypatch.setenv("MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS", "512")

    class _Resp:
        def json(self):
            return {"success": True, "data": {"blob": "z" * 2000}}

    monkeypatch.setattr(
        mcp_wrapper,
        "make_request_with_retry",
        lambda *_a, **_k: _Resp(),
    )

    mcp_wrapper.handle_call_tool(
        "req-truncate",
        {
            "name": "discover_legacy_sources",
            "arguments": {},
        },
    )

    assert sent
    text = sent[0]["result"]["content"][0]["text"]
    assert len(text) <= 512
    assert "truncated" in text


def test_make_request_with_retry_skips_startup_recovery_when_deadline_budget_low(monkeypatch):
    calls = {"ensure": 0}

    def _ensure():
        calls["ensure"] += 1
        return True

    def _always_fail(_method, _url, **_kwargs):
        raise mcp_wrapper.requests.ConnectionError("boom")

    monkeypatch.setenv("MUNINN_MCP_STARTUP_RECOVERY_MIN_BUDGET_SEC", "1")
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", _ensure)
    monkeypatch.setattr(mcp_wrapper.requests, "request", _always_fail)

    with pytest.raises((mcp_wrapper.requests.ConnectionError, mcp_wrapper._RequestDeadlineExceededError)):
        mcp_wrapper.make_request_with_retry(
            "GET",
            "http://localhost:42069/health",
            timeout=0.2,
            deadline_epoch=time.monotonic() + 0.05,
        )
    assert calls["ensure"] == 0


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
        "params": {},
    })

    assert len(sent) == 1
    assert sent[0]["id"] == "req-tasks-list-ok"
    assert sent[0]["result"]["tasks"] == []


def test_tasks_list_invalid_cursor_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-list-bad-cursor",
        "method": "tasks/list",
        "params": {"cursor": "abc"},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "opaque cursor token" in sent[0]["error"]["message"]


def test_tasks_list_supports_cursor_pagination(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True
    mcp_wrapper._SESSION_STATE["tasks"] = {
        "task-1": _sample_task(task_id="task-1", status="completed"),
        "task-2": _sample_task(task_id="task-2", status="completed"),
        "task-3": _sample_task(task_id="task-3", status="completed"),
    }
    mcp_wrapper._SESSION_STATE["tasks"]["task-1"]["lastUpdatedAt"] = "2026-02-15T04:00:01Z"
    mcp_wrapper._SESSION_STATE["tasks"]["task-2"]["lastUpdatedAt"] = "2026-02-15T04:00:02Z"
    mcp_wrapper._SESSION_STATE["tasks"]["task-3"]["lastUpdatedAt"] = "2026-02-15T04:00:03Z"

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-list-page-1",
        "method": "tasks/list",
        "params": {"limit": 2},
    })
    assert len(sent) == 1
    page1 = sent[0]["result"]
    assert len(page1["tasks"]) == 2
    assert page1["tasks"][0]["taskId"] == "task-3"
    assert page1["tasks"][1]["taskId"] == "task-2"
    assert page1["nextCursor"] != "2"
    assert mcp_wrapper._decode_task_cursor(page1["nextCursor"]) == 2

    sent.clear()
    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tasks-list-page-2",
        "method": "tasks/list",
        "params": {"cursor": page1["nextCursor"], "limit": 2},
    })
    assert len(sent) == 1
    page2 = sent[0]["result"]
    assert [task["taskId"] for task in page2["tasks"]] == ["task-1"]
    assert "nextCursor" not in page2


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
    done = threading.Event()

    def _invoke_result():
        mcp_wrapper._dispatch_rpc_message({
            "jsonrpc": "2.0",
            "id": "req-tasks-result-blocking",
            "method": "tasks/result",
            "params": {"taskId": "task-1"},
        })
        done.set()

    thread = threading.Thread(target=_invoke_result, daemon=True)
    thread.start()
    time.sleep(0.05)
    assert done.is_set() is False
    assert sent == []

    with mcp_wrapper._TASKS_CONDITION:
        task = mcp_wrapper._SESSION_STATE["tasks"]["task-1"]
        mcp_wrapper._set_task_state_locked(
            task,
            status="completed",
            status_message="done",
            result={"content": [{"type": "text", "text": "ok"}]},
        )
        mcp_wrapper._TASKS_CONDITION.notify_all()

    thread.join(timeout=1.0)
    assert done.is_set() is True
    assert len(sent) == 1
    assert sent[0]["id"] == "req-tasks-result-blocking"
    assert sent[0]["result"]["content"][0]["text"] == "ok"
    assert sent[0]["result"]["_meta"]["io.modelcontextprotocol/related-task"]["taskId"] == "task-1"


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
    assert sent[0]["result"]["_meta"]["io.modelcontextprotocol/related-task"]["taskId"] == "task-1"


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
    assert sent[0]["error"]["code"] == -32603
    assert "without a result payload" in sent[0]["error"]["message"]


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
    assert sent[0]["error"]["code"] == -32602
    assert "terminal state" in sent[0]["error"]["message"]


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

    response = next(msg for msg in sent if msg.get("id") == "req-tasks-cancel-ok")
    assert response["result"]["status"] == "cancelled"
    assert "Task cancelled by client request." in response["result"]["statusMessage"]
    notifications = [msg for msg in sent if msg.get("method") == "notifications/tasks/status"]
    assert notifications
    assert notifications[-1]["params"]["task"]["status"] == "cancelled"


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


def test_tools_call_task_must_be_object(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tools-call-task-invalid",
        "method": "tools/call",
        "params": {"name": "search_memory", "arguments": {"query": "x"}, "task": []},
    })

    assert len(sent) == 1
    assert sent[0]["error"]["code"] == -32602
    assert "task must be an object" in sent[0]["error"]["message"]


def test_tools_call_with_task_returns_create_task_and_completes(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    mcp_wrapper._SESSION_STATE["negotiated"] = True
    mcp_wrapper._SESSION_STATE["initialized"] = True

    class _InlineThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    def _fake_worker(task_id: str, _name: str, _arguments: dict):
        with mcp_wrapper._TASKS_CONDITION:
            task = mcp_wrapper._SESSION_STATE["tasks"][task_id]
            mcp_wrapper._set_task_state_locked(
                task,
                status="completed",
                status_message="done",
                result={"content": [{"type": "text", "text": "ok"}]},
            )
            task_snapshot = mcp_wrapper._public_task(task)
            mcp_wrapper._TASKS_CONDITION.notify_all()
        mcp_wrapper._emit_task_status_notification(task_snapshot)

    monkeypatch.setattr(mcp_wrapper.threading, "Thread", _InlineThread)
    monkeypatch.setattr(mcp_wrapper, "_run_tool_call_task_worker", _fake_worker)

    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tools-call-task",
        "method": "tools/call",
        "params": {
            "name": "search_memory",
            "arguments": {"query": "hello"},
            "task": {"ttl": 1234},
        },
    })

    create_response = next(msg for msg in sent if msg.get("id") == "req-tools-call-task")
    assert create_response["result"]["task"]["status"] == "working"
    assert create_response["result"]["task"]["ttl"] == 1234
    assert create_response["result"]["task"]["pollInterval"] == mcp_wrapper._TASK_POLL_INTERVAL_MS
    assert "io.modelcontextprotocol/model-immediate-response" in create_response["result"]["_meta"]

    notifications = [msg for msg in sent if msg.get("method") == "notifications/tasks/status"]
    assert notifications
    assert any(msg["params"]["task"]["status"] == "completed" for msg in notifications)

    task_id = create_response["result"]["task"]["taskId"]
    sent.clear()
    mcp_wrapper._dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-tools-call-task-result",
        "method": "tasks/result",
        "params": {"taskId": task_id},
    })
    assert len(sent) == 1
    assert sent[0]["result"]["content"][0]["text"] == "ok"
    assert sent[0]["result"]["_meta"]["io.modelcontextprotocol/related-task"]["taskId"] == task_id


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
    assert "set_user_profile" in by_name
    assert "get_user_profile" in by_name
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
        assert tool["execution"]["taskSupport"] == "optional"

    assert by_name["search_memory"]["annotations"]["readOnlyHint"] is True
    assert by_name["get_model_profiles"]["annotations"]["readOnlyHint"] is True
    assert by_name["get_model_profile_events"]["annotations"]["readOnlyHint"] is True
    assert by_name["get_user_profile"]["annotations"]["readOnlyHint"] is True
    assert by_name["set_user_profile"]["annotations"]["readOnlyHint"] is False
    assert by_name["set_model_profiles"]["annotations"]["readOnlyHint"] is False
    assert by_name["record_retrieval_feedback"]["annotations"]["readOnlyHint"] is False
    assert by_name["ingest_sources"]["annotations"]["readOnlyHint"] is False
    assert by_name["discover_legacy_sources"]["annotations"]["readOnlyHint"] is True
    assert by_name["ingest_legacy_sources"]["annotations"]["readOnlyHint"] is False
    assert by_name["delete_memory"]["annotations"]["destructiveHint"] is True
    assert by_name["delete_all_memories"]["annotations"]["destructiveHint"] is True
    assert by_name["search_memory"]["annotations"]["idempotentHint"] is True
    assert by_name["set_user_profile"]["annotations"]["idempotentHint"] is True
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
    set_user_profile_props = by_name["set_user_profile"]["inputSchema"]["properties"]
    assert "profile" in set_user_profile_props
    assert "merge" in set_user_profile_props
    assert "source" in set_user_profile_props
    get_user_profile_props = by_name["get_user_profile"]["inputSchema"]["properties"]
    assert get_user_profile_props == {}
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


def test_set_user_profile_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "USER_PROFILE_UPDATED"}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-set-user-profile",
        {
            "name": "set_user_profile",
            "arguments": {
                "profile": {"skills": ["python"]},
                "merge": False,
            },
        },
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/profile/user/set")
    assert captured["json"]["profile"] == {"skills": ["python"]}
    assert captured["json"]["merge"] is False
    assert captured["json"]["source"] == "mcp_tool"
    assert sent
    assert sent[0]["id"] == "req-set-user-profile"


def test_get_user_profile_tool_call_payload(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "USER_PROFILE_LOADED", "profile": {}}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-get-user-profile",
        {
            "name": "get_user_profile",
            "arguments": {},
        },
    )

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/profile/user/get")
    assert captured["params"]["user_id"] == "global_user"
    assert sent
    assert sent[0]["id"] == "req-get-user-profile"


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


def test_delete_memory_tool_call_url_encodes_memory_id_and_sets_deadline(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_wrapper, "send_json_rpc", lambda msg: sent.append(msg))
    monkeypatch.setattr(mcp_wrapper, "ensure_server_running", lambda: None)
    monkeypatch.setenv("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC", "30")

    captured = {}

    class _Resp:
        def json(self):
            return {"success": True, "data": {"event": "DELETE_COMPLETED"}}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["deadline_epoch"] = kwargs.get("deadline_epoch")
        return _Resp()

    monkeypatch.setattr(mcp_wrapper, "make_request_with_retry", _fake_request)

    mcp_wrapper.handle_call_tool(
        "req-delete",
        {
            "name": "delete_memory",
            "arguments": {"memory_id": "folder/item?x=1"},
        },
    )

    assert captured["method"] == "DELETE"
    assert captured["url"].endswith("/delete/folder%2Fitem%3Fx%3D1")
    assert isinstance(captured["deadline_epoch"], float)
    assert sent
    assert sent[0]["id"] == "req-delete"
