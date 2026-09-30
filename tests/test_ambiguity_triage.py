"""Synthetic-only tests; no Ollama call or real credential is used."""

import io
import json
import sys

import httpx
import pytest

from muninn.history.ambiguity_triage import (
    CandidateForReview, classify_local, deterministic_decision,
)
from scripts import triage_credential_ambiguity as runner
from muninn.history.credential_store import AmbiguousCandidate, CredentialStore, source_fingerprint


class _TTY(io.StringIO):
    def isatty(self):
        return True


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
    classify_local([item], model="local-test", keep_alive="30s")
    assert json.loads(requests[1].content)["keep_alive"] == "30s"
    assert result[0].decision == "deferred"
    assert "synthetic-987654" not in str(result)
    with pytest.raises(ValueError, match="loopback"):
        classify_local([item], model="local-test", base_url="https://example.com")
    with pytest.raises(ValueError, match="loopback"):
        classify_local([item], model="local-test", base_url="http://localhost:11434")


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
            return {"pending": 3, "rejected": 7}

    instance = FakeStore(tmp_path)
    monkeypatch.setattr(runner, "CredentialStore", lambda _root: instance)
    report = runner.run(root=tmp_path, passphrase="test-passphrase", limit=1,
                        model_limit=0, model="qwen2.5:7b", apply=True,
                        base_url="http://127.0.0.1:11434")
    assert report["rule_rejected"] == 1
    assert instance.decisions == ["rejected"]
    assert report["left_pending"] == 3
    assert "SECRET_REFERENCE" not in str(report)
    with pytest.raises(ValueError, match="loopback"):
        runner.run(root=tmp_path, passphrase="test-passphrase", limit=1,
                   model_limit=0, model="qwen2.5:7b", apply=False,
                   base_url="https://example.com")
    with pytest.raises(ValueError, match="loopback"):
        runner.run(root=tmp_path, passphrase="test-passphrase", limit=1,
                   model_limit=0, model="qwen2.5:7b", apply=False,
                   base_url="http://localhost:11434")


def test_interactive_triage_validates_backups_before_and_after(tmp_path, monkeypatch):
    passphrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    store = CredentialStore.create(root, passphrase)
    store.scan_source(
        passphrase=passphrase, source_hash=source_fingerprint("synthetic-source"),
        project="test", origin="transcript",
        findings=[AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value",
                                     "${SYNTHETIC_REF}", "")],
    )
    before, after = tmp_path / "before", tmp_path / "after"
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["triage_credential_ambiguity", "--root", str(root),
                                  "--model-limit", "0", "--max-pages", "2",
                                  "--backup-before", str(before), "--backup-after", str(after),
                                  "--apply"])
    monkeypatch.setattr(runner.getpass, "getpass", lambda _prompt: passphrase)
    assert runner.main() == 0
    assert CredentialStore(before).ambiguity_status() == {"pending": 1}
    assert CredentialStore(after).ambiguity_status() == {"rejected": 1}
    assert "SYNTHETIC_REF" not in output.getvalue()
    assert "validated_pre_triage_backup" in output.getvalue()
    assert "validated_post_triage_backup" in output.getvalue()


def test_triage_reports_user_review_still_needed(tmp_path, monkeypatch):
    passphrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    store = CredentialStore.create(root, passphrase)
    store.scan_source(
        passphrase=passphrase, source_hash=source_fingerprint("empty-source"),
        project="test", origin="transcript",
        findings=[AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", "", "")],
    )
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["triage_credential_ambiguity", "--root", str(root),
                                  "--model-limit", "0", "--max-pages", "2", "--apply"])
    monkeypatch.setattr(runner.getpass, "getpass", lambda _prompt: passphrase)
    assert runner.main() == 2
    assert store.ambiguity_status() == {"deferred": 1}
    assert "review_resolved" in output.getvalue()
    assert "false" in output.getvalue()


def test_triage_rejects_alias_and_in_vault_backup_destinations(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    CredentialStore.create(root, "synthetic passphrase long enough")
    before = tmp_path / "before"
    alias = tmp_path / "other" / ".." / "before"
    (tmp_path / "other").mkdir()
    monkeypatch.setattr(sys, "argv", ["triage_credential_ambiguity", "--root", str(root),
                                  "--backup-before", str(before),
                                  "--backup-after", str(alias), "--apply"])
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
    monkeypatch.setattr(sys, "argv", ["triage_credential_ambiguity", "--root", str(root),
                                  "--backup-before", str(root / "inside"), "--apply"])
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2


def test_triage_failure_reports_backup_state_without_exception_text(tmp_path, monkeypatch):
    passphrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    CredentialStore.create(root, passphrase)
    before, after = tmp_path / "before", tmp_path / "after"
    output = _TTY()
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["triage_credential_ambiguity", "--root", str(root),
                                  "--backup-before", str(before),
                                  "--backup-after", str(after), "--apply"])
    monkeypatch.setattr(runner.getpass, "getpass", lambda _prompt: passphrase)

    def fail(**_kwargs):
        raise RuntimeError("synthetic-sensitive-context-never-log")

    monkeypatch.setattr(runner, "run", fail)
    assert runner.main() == 1
    assert before.exists() and not after.exists()
    report = json.loads(output.getvalue().splitlines()[-1])
    assert report["backup_state"] == "validated_pre_triage_backup"
    assert report["post_backup_unavailable"] is True
    assert "synthetic-sensitive-context-never-log" not in output.getvalue()
