"""Wire-proof validation is offline here; the opt-in probe exercises real data."""
import json

import pytest

from scripts.smoke_user_bridge_live import _review_page, _review_progress


def page():
    return {"matches": [{"id": "candidate-1", "source_ref": "source-1",
                         "state": "provisional", "text": "public", "quote": "public"}],
            "limit": 1, "has_more": True,
            "next_cursor": "private-capability", "snapshot_events": 12,
            "credential_or_withheld_excluded": True}


def test_accepts_exact_text_and_structured_pages():
    value = page()
    assert _review_page({"structuredContent": value}) == value
    assert _review_page({"content": [{"type": "text", "text": json.dumps(value)}]}) == value


@pytest.mark.parametrize("changes", [
    {"limit": 2}, {"matches": [{}, {}]}, {"matches": [{}]}, {"has_more": 1},
    {"next_cursor": None}, {"credential_or_withheld_excluded": False},
    {"snapshot_events": "12"},
    {"matches": [{"id": "candidate-1", "source_ref": "source-1", "state": "provisional"}]},
])
def test_rejects_invalid_bound_privacy_and_continuation(changes):
    with pytest.raises(RuntimeError, match="review_response_invalid"):
        _review_page({"structuredContent": {**page(), **changes}})


def test_tool_failure_never_counts_as_proof():
    with pytest.raises(RuntimeError, match="review_call_failed"):
        _review_page({"isError": True, "structuredContent": page()})


def test_rejects_truncated_text_without_exposing_it():
    with pytest.raises(RuntimeError, match="^review_response_invalid$"):
        _review_page({"content": [{"type": "text", "text": "private malformed payload"}]})


def test_repeated_page_never_proves_continuation():
    with pytest.raises(RuntimeError, match="review_continuation_invalid"):
        _review_progress(page(), page())


@pytest.mark.parametrize("changes", [
    {"snapshot_events": 13}, {"next_cursor": "private-capability"},
    {"matches": page()["matches"]},
])
def test_anchor_cursor_and_candidate_progress_are_required(changes):
    following = {**page(), "next_cursor": "new-capability",
                 "matches": [{**page()["matches"][0], "id": "candidate-2"}], **changes}
    with pytest.raises(RuntimeError, match="review_continuation_invalid"):
        _review_progress(page(), following)


def test_accepts_distinct_last_page_on_same_snapshot():
    following = {**page(), "has_more": False, "next_cursor": None,
                 "matches": [{**page()["matches"][0], "id": "candidate-2"}]}
    _review_progress(page(), following)
