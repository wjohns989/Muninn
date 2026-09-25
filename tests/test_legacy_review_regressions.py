import io
from unittest.mock import MagicMock

import mcp_wrapper
import server
import silent_mcp


def test_existing_server_health_rejects_generic_healthy_service(monkeypatch) -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {"status": "ok"}
    monkeypatch.setattr(server.requests, "get", MagicMock(return_value=response))

    assert server._existing_server_healthy("127.0.0.1", 42069) is False


def test_existing_server_health_accepts_muninn_signature(monkeypatch) -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {"status": "ok", "backend": "muninn-native"}
    monkeypatch.setattr(server.requests, "get", MagicMock(return_value=response))

    assert server._existing_server_healthy("127.0.0.1", 42069) is True


def test_silent_proxy_reads_content_length_body_without_newline() -> None:
    body = b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
    framed = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body

    raw_message, payload, is_framed = silent_mcp._read_client_message(io.BytesIO(framed))

    assert raw_message == framed
    assert payload == body
    assert is_framed is True


def test_stdio_task_is_created_in_same_session_used_by_worker(monkeypatch) -> None:
    worker_args = {}

    class NoopThread:
        def __init__(self, target, args=(), **kwargs):
            worker_args["args"] = args

        def start(self):
            pass

    monkeypatch.setitem(mcp_wrapper._SESSION_STATE, "tasks", {})
    monkeypatch.setattr("muninn.mcp.handlers.threading.Thread", NoopThread)

    mcp_wrapper.handle_call_tool_with_task(
        "request-1",
        "search_memory",
        {"query": "synthetic"},
        {"ttl": 1000},
    )

    assert mcp_wrapper._SESSION_STATE["tasks"]
    assert worker_args["args"][0] == "default"
