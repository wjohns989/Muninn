"""Keep one compact private read tool; ambiguous modes never issue HTTP."""

import pytest

from muninn.mcp import handlers
from muninn.mcp.definitions import CORE_TOOLS, PRIVATE_MAIN_TOKEN_TOOLS, TOOLS_SCHEMAS


def test_explicit_review_mode_routes_to_private_queue_and_preserves_core20(monkeypatch):
    calls = []

    class Reply:
        def json(self):
            return {"success": True, "data": {"matches": [], "next_cursor": "opaque"}}

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return Reply()

    monkeypatch.setattr(handlers, "make_request_with_retry", request)
    result = handlers._do_search_cited_memories({"review_only": True, "limit": 1, "cursor": "opaque"}, None)
    assert calls[0][0][1].endswith("/history/secure/memories/review-queue")
    assert calls[0][1]["json"] == {"limit": 1, "cursor": "opaque"}
    assert result["data"]["next_cursor"] == "opaque"
    assert len(CORE_TOOLS) <= 20 and "search_cited_memories" in PRIVATE_MAIN_TOKEN_TOOLS
    assert next(s for s in TOOLS_SCHEMAS if s["name"] == "search_cited_memories")["inputSchema"]["anyOf"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"review_only": True, "query": None},
        {"review_only": True, "query": ""},
        {"review_only": True, "query": "citations"},
        {"review_only": "true"},
        {"query": "citations", "cursor": None},
        {"query": "citations", "cursor": "opaque"},
        {"review_only": 1},
    ],
)
def test_invalid_or_ambiguous_modes_fail_before_http(monkeypatch, arguments):
    monkeypatch.setattr(handlers, "make_request_with_retry", lambda *a, **kw: pytest.fail("must not issue HTTP"))
    with pytest.raises(ValueError):
        handlers._do_search_cited_memories(arguments, None)
