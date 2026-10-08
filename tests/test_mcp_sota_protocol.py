import json
import pytest
from muninn.mcp.main import _negotiate_protocol_version, _dispatch_rpc_message, _REAL_SESSION_STATE, _TRANSPORT_CLOSED

@pytest.fixture(autouse=True)
def reset_mcp_state():
    _REAL_SESSION_STATE.clear()
    _REAL_SESSION_STATE.update({
        "negotiated": False,
        "initialized": False,
        "protocol_version": "2024-11-05",
        "client_capabilities": {},
        "client_info": {},
        "client_elicitation_modes": tuple(),
        "tasks": {},
    })
    _TRANSPORT_CLOSED.clear()
    yield

def test_negotiate_protocol():
    from muninn.mcp.handlers import SUPPORTED_PROTOCOL_VERSIONS
    assert _negotiate_protocol_version("2025-11-25") == "2025-11-25"
    assert _negotiate_protocol_version("unsupported") is None
    assert _negotiate_protocol_version(None) == SUPPORTED_PROTOCOL_VERSIONS[0]

def test_initialize_flow(monkeypatch):
    sent = []
    def mock_send(msg): sent.append(msg)
    monkeypatch.setattr("muninn.mcp.main._server.send_rpc", mock_send)
    monkeypatch.setattr("muninn.mcp.main.ensure_server_running", lambda: True)

    # 1. Initialize
    _dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    })

    assert len(sent) == 1
    assert sent[0]["id"] == "req-1"
    assert "result" in sent[0]
    assert _REAL_SESSION_STATE["negotiated"] is True

    # 2. Initialized notification
    _dispatch_rpc_message({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    })
    assert _REAL_SESSION_STATE["initialized"] is True

def test_uninitialized_call(monkeypatch):
    sent = []
    def mock_send(msg): sent.append(msg)
    monkeypatch.setattr("muninn.mcp.main._server.send_rpc", mock_send)

    _dispatch_rpc_message({
        "jsonrpc": "2.0",
        "id": "req-2",
        "method": "tools/list"
    })

    assert len(sent) == 1
    assert "error" in sent[0]
    assert "not initialized" in sent[0]["error"]["message"].lower()
