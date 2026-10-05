"""Offline diagnostic boundaries; these tests never dispatch a model."""
import pytest

from scripts.compare_gpt_oss import FrozenSource, MODEL, response_matches


@pytest.mark.parametrize("status,model,expected", [
    (200, MODEL, True), (503, MODEL, False), (200, "other", False),
])
def test_provider_or_identity_failure_cannot_continue(status, model, expected):
    assert response_matches(status, {"model": model, "choices": [],
                                     "usage": {"cost": .001}}) is expected


def test_exact_quote_in_one_authenticated_range():
    window = {"text": "Keep encrypted backups.",
              "citation_ranges": [{"start": 0, "length": 23}]}
    proposal = {"type": "decision", "text": "Keep encrypted backups.",
                "quote": window["text"], "start": 0}
    assert FrozenSource(window).validated_proposals({}, [proposal]) == [proposal]


def test_quote_cannot_cross_citation_ranges():
    window = {"text": "Keep encrypted backups.",
              "citation_ranges": [{"start": 0, "length": 5}, {"start": 5, "length": 18}]}
    with pytest.raises(ValueError, match="Unsupported citation"):
        FrozenSource(window).validated_proposals({}, [{"type": "decision", "text": "Back up.",
            "quote": window["text"], "start": 0}])
