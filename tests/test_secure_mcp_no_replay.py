"""A lost response is not permission to repeat a model call or one-use page."""
from types import SimpleNamespace

import pytest
import requests

from muninn.mcp import handlers
from muninn.mcp import requests as transport


@pytest.mark.parametrize("operation,args", [
    (handlers._do_analyze_secure_history, {"capability": "fixture-only-capability"}),
    (handlers._do_read_secure_history_transcript_page, {"cursor": "fixture-only-cursor"}),
])
@pytest.mark.parametrize("failure", ["timeout", "connection", "503"])
def test_unknown_post_outcome_is_not_replayed(monkeypatch, operation, args, failure):
    attempted = []
    sleeps = []
    monkeypatch.setattr(transport, "get_token", lambda: "fixture-only-auth")
    monkeypatch.setattr(transport, "is_circuit_open", lambda: False)
    monkeypatch.setattr(transport, "mark_failure", lambda error: None)
    monkeypatch.setattr(transport, "mark_success", lambda: None)
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)
    # Exercise the real retry loop, not a mock of the handler's retry argument.
    monkeypatch.setattr(handlers, "make_request_with_retry", transport._make_request_with_retry_internal)
    def request(method, url, **kwargs):
        attempted.append((method, url))  # Provider/one-use side effect occurred.
        if failure == "timeout":
            raise requests.ReadTimeout("fixture response lost after processing")
        if failure == "connection":
            raise requests.ConnectionError("fixture response connection lost")
        response = requests.Response()
        response.status_code = 503
        response.url = url
        response._content = b'{"detail":"fixture unavailable"}'
        return response
    monkeypatch.setattr(transport.requests, "request", request)
    with pytest.raises((requests.ReadTimeout, requests.ConnectionError, requests.HTTPError)):
        operation(args, None)
    assert len(attempted) == 1
    assert attempted[0][0] == "POST"
    assert sleeps == []


def test_analysis_payload_and_success_are_preserved(monkeypatch):
    observed = {}
    def request(method, url, **kwargs):
        observed.update(method=method, url=url, **kwargs)
        return SimpleNamespace(json=lambda: {"success": True, "data": {"status": "ok"}})
    monkeypatch.setattr(handlers, "make_request_with_retry", request)
    result = handlers._do_analyze_secure_history({"capability": "fixture", "allow_remote": True,
                                                  "prefer_remote": True}, 123.0)
    assert observed["json"] == {"capability": "fixture", "allow_remote": True, "prefer_remote": True}
    assert observed["deadline_epoch"] == 123.0 and observed["timeout"] == 180.0
    assert observed["max_retries"] == 0
    assert result == {"success": True, "data": {"status": "ok"}}


def test_read_only_poll_keeps_its_existing_retry_contract(monkeypatch):
    attempts = []
    monkeypatch.setattr(transport, "get_token", lambda: "fixture-only-auth")
    monkeypatch.setattr(transport, "is_circuit_open", lambda: False)
    monkeypatch.setattr(transport, "mark_failure", lambda error: None)
    monkeypatch.setattr(transport, "mark_success", lambda: None)
    monkeypatch.setattr(transport.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(handlers, "make_request_with_retry", transport._make_request_with_retry_internal)
    def request(method, url, **kwargs):
        attempts.append(method)
        if len(attempts) == 1:
            raise requests.ReadTimeout("fixture transient poll")
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"success":true,"data":{"state":"pending"}}'
        return response
    monkeypatch.setattr(transport.requests, "request", request)
    result = handlers._do_poll_secure_history_analysis({"job_id": "fixture-only-job"}, None)
    assert result["data"]["state"] == "pending"
    assert attempts == ["GET", "GET"]
