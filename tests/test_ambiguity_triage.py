"""Synthetic-only tests; no Ollama call or real credential is used."""

import io
import json
import sys
import time

import httpx
import pytest

from muninn.history.ambiguity_triage import (
    CandidateForReview,
    classify_local,
    deterministic_decision,
)
from muninn.history.auto_routing import GpuState
from muninn.history.credential_store import AmbiguousCandidate, CredentialStore, source_fingerprint
from scripts import triage_credential_ambiguity as runner


class _TTY(io.StringIO):
    def isatty(self):
        return True


class _ReviewSource:
    def prepare(self, row, *, candidate=None):
        return row["id"]

    def inputs(self, prepared, row, candidate):
        yield 0, CandidateForReview(row["id"], row["name"], row["reason"], candidate,
                                   {"provider": "codex", "context": "synthetic source context"})

    def cached(self, *args):
        return None

    def record(self, *args):
        pass


def test_deterministic_references_are_not_vault_values():
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "${SECRET_REF}")
    result = deterministic_decision(item)
    assert (result.decision, result.basis) == ("rejected", "local-rule")
    assert "SECRET_REF" not in str(result)


def test_nontrivial_candidate_requires_local_model():
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "abc-123$def")
    assert deterministic_decision(item) is None


def test_local_model_is_loopback_only_and_returns_decisions_without_values(monkeypatch):
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "synthetic-987654",
                              {"context": "synthetic source context", "provider": "codex"})
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
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "synthetic-987654",
                              {"context": "synthetic source context", "provider": "codex"})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "message": {"content": "synthetic-987654"},
    }))
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    assert classify_local([item], model="local-test")[0].decision == "deferred"


def test_bounded_rule_pass_updates_occurrence_without_returning_candidate(monkeypatch, tmp_path):
    class FakeStore:
        def __init__(self, _root):
            self.decisions = []

        def list_ambiguities(self, *, status, limit):
            assert (status, limit) == ("pending", 1)
            return [{"id": "one", "name": "SERVICE_API_KEY",
                     "reason": "unparsed_value", "count": 7}]

        def reveal_ambiguity(self, _id, *, passphrase):
            assert passphrase == "test-passphrase"
            return "${SECRET_REFERENCE}"

        def decide_ambiguity(self, _id, **kwargs):
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


def test_two_triage_pages_reuse_requested_resident_model(monkeypatch, tmp_path):
    class FakeStore:
        pending = ["one", "two"]

        def list_ambiguities(self, *, status, limit):
            assert (status, limit) == ("pending", 1)
            return [{"id": self.pending[0], "name": "SERVICE_API_KEY",
                     "reason": "unparsed_value", "count": 1}] if self.pending else []

        def reveal_ambiguity(self, _id, *, passphrase):
            assert passphrase == "test-passphrase"
            return "synthetic-987654"

        def decide_ambiguity(self, record_id, **kwargs):
            assert kwargs["decision"] == "rejected"
            self.pending.remove(record_id)

        def ambiguity_status(self):
            return {"pending": len(self.pending), "rejected": 2 - len(self.pending)}

    store = FakeStore()
    monkeypatch.setattr(runner, "CredentialStore", lambda _root: store)
    monkeypatch.setattr(runner, "probe_gpu", lambda: GpuState(7_500, 16_376, 1, time.time()))
    loaded = iter([(), ("qwen2.5:7b",), ("qwen2.5:7b",), ("qwen2.5:7b",)])
    monkeypatch.setattr(runner, "probe_ollama", lambda _url: (
        [{"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024, "digest": "synthetic-weights"}], next(loaded)))
    requests = []

    def handler(request):
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "items": [{"index": 0, "class": "not_credential", "confidence": 1.0}],
        })}})

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    reports = [runner.run(root=tmp_path, passphrase="test-passphrase", limit=1,
                          model_limit=1, model="qwen2.5:7b", apply=True,
                          base_url="http://127.0.0.1:11434", keep_alive="30s", review_source=_ReviewSource())
               for _ in range(2)]
    assert [report["model_rejected"] for report in reports] == [1, 1]
    assert [body["model"] for body in requests] == ["qwen2.5:7b", "qwen2.5:7b"]
    assert all(body["keep_alive"] == "30s" for body in requests)
    assert store.ambiguity_status()["pending"] == 0


def test_model_without_original_source_context_never_dispatches(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Missing provenance dispatched a model")
    monkeypatch.setattr(httpx, "Client", forbidden)
    item = CandidateForReview("one", "SERVICE_API_KEY", "unparsed_value", "synthetic-987654")
    assert classify_local([item], model="local-test")[0].decision == "deferred"


def test_triage_reports_quota_separately_from_transient_gpu_contention(tmp_path, monkeypatch):
    class Store:
        def list_ambiguities(self, **kwargs):
            return [{"id": "one", "name": "SERVICE_API_KEY", "reason": "unparsed_value"}]
        def reveal_ambiguity(self, *args, **kwargs):
            return "synthetic-987654"
        def ambiguity_status(self):
            return {"pending": 1}

    class Source(_ReviewSource):
        saved = []
        def inputs(self, prepared, row, candidate):
            for page in range(4):
                yield page, CandidateForReview(row["id"], row["name"], row["reason"], candidate,
                                              {"context": f"synthetic source {page}"})
        def record(self, prepared, page, identity, result):
            self.saved.append(page)

    source = Source()
    monkeypatch.setattr(runner, "CredentialStore", lambda _root: Store())
    utilization = iter([0, 85, 0, 0])
    monkeypatch.setattr(runner, "probe_gpu", lambda: GpuState(
        7_500, 16_376, next(utilization), time.time()))
    monkeypatch.setattr(runner, "probe_ollama", lambda _url: (
        [{"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024, "digest": "synthetic-weights"}], ()))
    from muninn.history.ambiguity_triage import ReviewDecision
    monkeypatch.setattr(runner, "classify_local", lambda items, **kwargs: [
        ReviewDecision(items[0].id, "rejected", "local-model")])
    report = runner.run(root=tmp_path, passphrase="synthetic phrase", limit=2, model_limit=2,
                        model="qwen2.5:7b", apply=True, base_url="http://127.0.0.1:11434",
                        review_source=source)
    assert report["model_route"] == "model_limit_reached"
    assert report["contexts_route_deferred"] == 1
    assert report["contexts_quota_deferred"] == 1
    assert report["model_calls"] == 2 and source.saved == [1, 2]
    assert report["model_rejected"] == 0 and report["next_cursor"] is None


def test_model_decision_cannot_spread_to_another_source_in_same_value_group(tmp_path, monkeypatch):
    passphrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    store = CredentialStore.create(root, passphrase)
    for source in ["first", "second"]:
        store.scan_source(passphrase=passphrase, source_hash=source_fingerprint(source),
                          project="codex", origin="transcript", findings=[
                              AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", "synthetic-987654", "")])
    assert store.list_ambiguity_groups()[0]["count"] == 2
    monkeypatch.setattr(runner, "probe_gpu", lambda: GpuState(7_500, 16_376, 1, time.time()))
    monkeypatch.setattr(runner, "probe_ollama", lambda _url: (
        [{"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024, "digest": "synthetic-weights"}], ()))
    from muninn.history.ambiguity_triage import ReviewDecision
    monkeypatch.setattr(runner, "classify_local", lambda items, **kwargs: [
        ReviewDecision(items[0].id, "rejected", "local-model")])
    report = runner.run(root=root, passphrase=passphrase, limit=1, model_limit=1, model="qwen2.5:7b",
                        apply=True, base_url="http://127.0.0.1:11434", review_source=_ReviewSource())
    assert report["model_rejected"] == 1
    assert store.ambiguity_status() == {"pending": 1, "rejected": 1}


def test_all_contexts_must_finish_and_previous_model_decisions_are_reused(tmp_path, monkeypatch):
    passphrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    store = CredentialStore.create(root, passphrase)
    store.scan_source(passphrase=passphrase, source_hash=source_fingerprint("first"),
                      project="codex", origin="transcript", findings=[
                          AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", "synthetic-987654", "")])

    class Multiple(_ReviewSource):
        results = {}
        def inputs(self, prepared, row, candidate):
            for page in range(2):
                yield page, CandidateForReview(row["id"], row["name"], row["reason"], candidate,
                                              {"context": f"synthetic source {page}"})
        def cached(self, prepared, page, identity):
            return self.results.get((prepared, page, identity))
        def record(self, prepared, page, identity, result):
            self.results[(prepared, page, identity)] = result

    source = Multiple()
    monkeypatch.setattr(runner, "probe_gpu", lambda: GpuState(7_500, 16_376, 1, time.time()))
    monkeypatch.setattr(runner, "probe_ollama", lambda _url: (
        [{"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024, "digest": "synthetic-weights"}], ()))
    from muninn.history.ambiguity_triage import ReviewDecision
    calls = []
    def classify(items, **kwargs):
        calls.append(items[0].source_context)
        return [ReviewDecision(items[0].id, "rejected", "local-model")]
    monkeypatch.setattr(runner, "classify_local", classify)
    reports = [runner.run(root=root, passphrase=passphrase, limit=1, model_limit=1, model="qwen2.5:7b",
                          apply=True, base_url="http://127.0.0.1:11434", review_source=source) for _ in range(2)]
    assert reports[0]["model_rejected"] == 0 and reports[0]["left_pending"] == 1
    assert reports[1]["contexts_reused"] == 1 and reports[1]["model_rejected"] == 1
    assert len(calls) == 2 and store.ambiguity_status() == {"rejected": 1}


@pytest.mark.parametrize("tamper_after_model", [False, True])
def test_real_context_store_review_cache_commit_and_late_integrity_gate(tmp_path, monkeypatch,
                                                                       tamper_after_model):
    from muninn.history.ambiguity_triage import ReviewDecision
    from muninn.history.credential_discovery import ExtractionStats, iter_transcript_findings
    from muninn.history.credential_review_source import CredentialReviewSource
    from muninn.history.secure_archive import SecureHistoryArchive
    from muninn.history.secure_projection_store import ProjectionIntegrityError
    phrase = "synthetic passphrase long enough"
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"type": "session_meta", "payload": {"cwd": "C:/synthetic-project"}},
        *[{"type": "event_msg", "timestamp": "2026-09-30T12:00:00Z", "payload": {
            "type": "user_message", "message": "SERVICE_API_KEY=abc-123$def"}}] * 2,
    ]) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", phrase)
    archive.archive_file(path, "codex")
    entry = archive._load_manifest()["files"][str(path.resolve())][0]
    root = tmp_path / "vault"
    store = CredentialStore.create(root, phrase)
    store.scan_source(passphrase=phrase, project="codex", origin="transcript",
                      source_hash=source_fingerprint(
                          f"{archive.vault_id}:{entry['blob']}:{entry['sha256']}"),
                      findings=iter_transcript_findings([path.read_bytes()], ExtractionStats(),
                                                        include_ambiguous=True))
    assert store.ambiguity_status() == {"pending": 1}
    source = CredentialReviewSource(archive)
    prepared = source.prepare(store.list_ambiguities(status="pending", limit=1)[0])
    monkeypatch.setattr(runner, "probe_gpu", lambda: GpuState(7_500, 16_376, 1, time.time()))
    monkeypatch.setattr(runner, "probe_ollama", lambda _url: (
        [{"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024, "digest": "synthetic-weights"}], ()))
    calls = []

    def classify(items, **kwargs):
        assert items[0].source_context["time_basis"] != "unknown"
        calls.append(items[0].id)
        if tamper_after_model and len(calls) == 1:
            with source.contexts._connect() as db:
                db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                           "WHERE attempt=? AND ordinal=1", (prepared[3],))
        return [ReviewDecision(items[0].id, "rejected", "local-model")]

    monkeypatch.setattr(runner, "classify_local", classify)
    kwargs = dict(root=root, passphrase=phrase, limit=1, model_limit=1,
                  model="qwen2.5:7b", apply=True, base_url="http://127.0.0.1:11434",
                  review_source=source)
    if tamper_after_model:
        with pytest.raises(ProjectionIntegrityError):
            runner.run(**kwargs)
        assert store.ambiguity_status() == {"pending": 1}
    else:
        first = runner.run(**kwargs)
        assert first["model_calls"] == 1 and first["left_pending"] == 1
        second = runner.run(**kwargs)
        assert second["contexts_reused"] == 1 and second["model_rejected"] == 1
        assert len(calls) == 2 and store.ambiguity_status() == {"rejected": 1}


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
    assert store.ambiguity_status() == {"pending": 1}
    assert "review_resolved" in output.getvalue()
    assert "false" in output.getvalue()


def test_pending_unknown_does_not_starve_later_rows(tmp_path):
    phrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    store = CredentialStore.create(root, phrase)
    for source, value in [("first", ""), ("second", "${SYNTHETIC_REF}")]:
        store.scan_source(passphrase=phrase, source_hash=source_fingerprint(source),
                          project="test", origin="transcript", findings=[
                              AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", value, "")])
    first = runner.run(root=root, passphrase=phrase, limit=1, model_limit=0,
                       model="qwen2.5:7b", apply=True, base_url="http://127.0.0.1:11434")
    assert first["deferred_for_user"] == 1 and store.ambiguity_status() == {"pending": 2}
    second = runner.run(root=root, passphrase=phrase, limit=1, model_limit=0,
                        model="qwen2.5:7b", apply=True, base_url="http://127.0.0.1:11434",
                        after=first["next_cursor"])
    assert second["rule_rejected"] == 1 and store.ambiguity_status() == {"pending": 1, "rejected": 1}


def test_changed_model_digest_never_classifies_under_old_identity(tmp_path, monkeypatch):
    phrase = "synthetic passphrase long enough"
    root = tmp_path / "vault"
    store = CredentialStore.create(root, phrase)
    store.scan_source(passphrase=phrase, source_hash=source_fingerprint("first"),
                      project="test", origin="transcript", findings=[
                          AmbiguousCandidate("SERVICE_API_KEY", "unparsed_value", "synthetic-987654", "")])
    monkeypatch.setattr(runner, "probe_gpu", lambda: GpuState(7_500, 16_376, 1, time.time()))
    digests = iter(["first-weights", "changed-weights"])
    monkeypatch.setattr(runner, "probe_ollama", lambda _url: (
        [{"name": "qwen2.5:7b", "size": 4_700 * 1024 * 1024, "digest": next(digests)}], ()))
    monkeypatch.setattr(runner, "classify_local", lambda *_args, **_kwargs: pytest.fail("changed weights dispatched"))
    result = runner.run(root=root, passphrase=phrase, limit=1, model_limit=1,
                        model="qwen2.5:7b", apply=True, base_url="http://127.0.0.1:11434",
                        review_source=_ReviewSource())
    assert result["model_route"] == "model_identity_changed" and result["model_calls"] == 0
    assert result["next_cursor"] is None and store.ambiguity_status() == {"pending": 1}


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


@pytest.mark.parametrize("status_code", [None, 400, 404, 429, 500, 503])
def test_triage_failure_reports_backup_state_without_exception_text(tmp_path, monkeypatch, status_code):
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
        if status_code is not None:
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat?private=context",
                                    headers={"Authorization": "synthetic-sensitive-header"})
            response = httpx.Response(status_code, request=request,
                                      text="synthetic-sensitive-body-never-log")
            raise httpx.HTTPStatusError("synthetic-sensitive-context-never-log",
                                        request=request, response=response)
        raise RuntimeError("synthetic-sensitive-context-never-log")

    monkeypatch.setattr(runner, "run", fail)
    assert runner.main() == 1
    assert before.exists() and not after.exists()
    report = json.loads(output.getvalue().splitlines()[-1])
    assert report["backup_state"] == "validated_pre_triage_backup"
    assert report["post_backup_unavailable"] is True
    assert report.get("http_status_code") == status_code
    assert "synthetic-sensitive-context-never-log" not in output.getvalue()
    assert "synthetic-sensitive-body-never-log" not in output.getvalue()
    assert "synthetic-sensitive-header" not in output.getvalue()
    assert "private=context" not in output.getvalue()
