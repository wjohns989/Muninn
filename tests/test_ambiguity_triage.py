"""Synthetic-only tests; no Ollama call or real credential is used."""

import json

import httpx
import pytest

from muninn.history.ambiguity_triage import (
    CandidateForReview, classify_local, deterministic_decision,
)
from scripts import triage_credential_ambiguity as runner


def test_deterministic_references_are_not_vault_values():
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "${SECRET_REF}")
    result = deterministic_decision(item)
    assert (result.decision, result.basis) == ("rejected", "local-rule")
    assert "SECRET_REF" not in str(result)


def test_nontrivial_candidate_requires_local_model():
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "abc-123$def")
    assert deterministic_decision(item) is None


def test_local_model_is_loopback_only_and_returns_decisions_without_values(monkeypatch):
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "synthetic-987654")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "items": [{"index": 0, "class": "possible_credential", "confidence": 0.99}],
        })}})

    transport = httpx.MockTransport(handler)
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    result = classify_local([item], model="local-test")
    assert requests[0].url.host == "127.0.0.1"
    assert json.loads(requests[0].content)["keep_alive"] == 0
    assert result[0].decision == "deferred"
    assert "synthetic-987654" not in str(result)
    with pytest.raises(ValueError, match="loopback"):
        classify_local([item], model="local-test", base_url="https://example.com")


def test_malformed_model_reply_defers(monkeypatch):
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "synthetic-987654")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "message": {"content": "synthetic-987654"},
    }))
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    assert classify_local([item], model="local-test")[0].decision == "deferred"


def test_bounded_rule_pass_updates_group_without_returning_candidate(monkeypatch, tmp_path):
    class FakeStore:
        def __init__(self, _root):
            self.decisions = []

        def list_ambiguity_groups(self, *, status, limit):
            assert (status, limit) == ("pending", 1)
            return [{"representative_id": "one", "name": "SERVICE_API_KEY",
                     "reason": "unparsed_value", "count": 7}]

        def reveal_ambiguity(self, _id, *, passphrase):
            assert passphrase == "test-passphrase"
            return "${SECRET_REFERENCE}"

        def decide_ambiguity_group(self, _id, **kwargs):
            self.decisions.append(kwargs["decision"])

        def ambiguity_status(self):
            return {"pending": 0, "rejected": 7}

    instance = FakeStore(tmp_path)
    monkeypatch.setattr(runner, "CredentialStore", lambda _root: instance)
    report = runner.run(root=tmp_path, passphrase="test-passphrase", limit=1,
                        model_limit=0, model="qwen2.5:7b", apply=True,
                        base_url="http://127.0.0.1:11434")
    assert report["rule_rejected"] == 1
    assert instance.decisions == ["rejected"]
    assert "SECRET_REFERENCE" not in str(report)
    with pytest.raises(ValueError, match="loopback"):
        runner.run(root=tmp_path, passphrase="test-passphrase", limit=1,
                   model_limit=0, model="qwen2.5:7b", apply=False,
                   base_url="https://example.com")
