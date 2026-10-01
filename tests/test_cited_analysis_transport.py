"""Private cited transport and real encrypted worker replay; no live inference."""
import json
import asyncio
from contextlib import asynccontextmanager

import pytest

pytestmark = pytest.mark.usefixtures("fake_strict_remote_admission")

from muninn.history import secure_analysis as analysis
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.memory_ledger import MemoryLedgerIntegrityError
from muninn.history.service import HistoryService
from tests.test_analysis_publication_journal import queued, bind_stage, expire, publish


def transport(monkeypatch, source, stage, *, local=True, output=None, digest_changes=False):
    seen = []
    window = source.reopen(stage["window"])
    content = output or json.dumps({**stage["result"]["analysis"], "proposals": stage["proposals"]})
    @asynccontextmanager
    async def slot():
        yield
    class Response:
        def raise_for_status(self):
            pass
        def json(self, **kwargs):
            return {"message": {"content": content}, "model": "fixture-model",
                    "choices": [{"message": {"content": content}}]}
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, *, json, **kwargs):
            seen.append((url, json))
            return Response()
    monkeypatch.setattr(analysis.httpx, "AsyncClient", Client)
    monkeypatch.setattr("muninn.extraction.ollama_slot.async_ollama_slot", slot)
    monkeypatch.setattr(analysis, "_select_local", lambda base: ("fixture-model" if local else None, "gpu_busy"))
    digests = iter(["a" * 64, "b" * 64] if digest_changes else ["a" * 64] * 2)
    monkeypatch.setattr(analysis, "_weights_digest", lambda *args: next(digests))
    monkeypatch.setattr(analysis, "_remote_eligible", lambda *args, **kwargs: kwargs["allow_remote"])
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda **kwargs: True)
    monkeypatch.setattr(analysis.Provider, "from_env", lambda *args: analysis.Provider(
        "openrouter", "https://openrouter.ai/api/v1", ["fixture-model"], "fixture-key"))
    return seen, window


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [True, False])
async def test_private_cited_response_binds_schema_input_and_identity(tmp_path, monkeypatch, local):
    journal, archive, job, stage, source = queued(tmp_path)
    seen, window = transport(monkeypatch, source, stage, local=local)
    outcome = await analysis.analyze_cited_window(object(), source, stage["window"], allow_remote=not local)
    assert outcome["status"] == "ok"
    extraction = outcome["extraction"]
    assert extraction["window"] == stage["window"] and extraction["proposals"] == stage["proposals"]
    assert len(extraction["model_identity"]) == 64
    body = seen[0][1]
    assert json.loads(body["messages"][1]["content"]) == {k: v for k, v in window.items() if k != "project_ref"}
    if local:
        assert body["keep_alive"] == 0 and "proposals" in body["format"]["required"]
    else:
        assert "proposals" in body["response_format"]["json_schema"]["schema"]["required"]
        assert body["provider"]["zdr"] is True
    assert source.ledger.verify_all()["candidates"] == 0  # Transport is not publication.


@pytest.mark.asyncio
async def test_changed_local_weights_cannot_acquire_false_identity(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    seen, window = transport(monkeypatch, source, stage, digest_changes=True)
    with pytest.raises(RuntimeError, match="identity changed"):
        await analysis.analyze_cited_window(object(), source, stage["window"])
    assert len(seen) == 1 and source.ledger.verify_all()["candidates"] == 0


@pytest.mark.asyncio
async def test_reuse_miss_rechecks_weights_before_any_model_post(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    seen, window = transport(monkeypatch, source, stage, digest_changes=True)
    observed = []
    async def miss(model, digest, options, base):
        observed.append((model, digest, options, base))
        return False
    with pytest.raises(RuntimeError, match="before inference"):
        await analysis.analyze_cited_window(object(), source, stage["window"], reuse_completed=miss)
    assert observed[0][2] == {"temperature": 0.1}
    assert seen == [] and source.ledger.verify_all()["candidates"] == 0


@pytest.mark.asyncio
async def test_bad_citation_does_not_return_private_stage(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    output = json.dumps({**stage["result"]["analysis"], "proposals": [{
        **stage["proposals"][0], "quote": "not in this source"}]})
    transport(monkeypatch, source, stage, output=output)
    result = await analysis.analyze_cited_window(object(), source, stage["window"])
    assert result["reason"] == "local_output_invalid" and "extraction" not in result


@pytest.mark.asyncio
async def test_model_cannot_override_ordinary_enrichment_type_boundary(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    output = json.dumps({**stage["result"]["analysis"], "proposals": [{
        **stage["proposals"][0], "type": "possible_credential"}]})
    transport(monkeypatch, source, stage, output=output)
    result = await analysis.analyze_cited_window(object(), source, stage["window"])
    assert result["reason"] == "local_output_invalid" and "extraction" not in result
    assert source.ledger.verify_all()["candidates"] == 0


def test_unique_exact_quote_can_resolve_incorrect_model_coordinate(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    output = json.dumps({**stage["result"]["analysis"], "proposals": [{
        **stage["proposals"][0], "start": 7}]})
    result = analysis._cited_outcome(output, source, stage["window"], "ollama", "fixture-model", "a" * 64)
    assert result["extraction"]["proposals"][0]["start"] == 0
    assert result["extraction"]["proposals"][0]["quote"] == stage["proposals"][0]["quote"]
    assert source.ledger.verify_all()["candidates"] == 0


def test_invalid_quote_does_not_discard_independent_valid_citation(tmp_path):
    journal, archive, job, stage, source = queued(tmp_path)
    valid = stage["proposals"][0]
    output = json.dumps({**stage["result"]["analysis"], "proposals": [
        valid, {**valid, "quote": "not in this source"},
    ]})
    result = analysis._cited_outcome(output, source, stage["window"],
                                     "ollama", "fixture-model", "a" * 64)
    assert result["extraction"]["proposals"] == [valid]
    assert source.ledger.verify_all()["candidates"] == 0
    refs = source.record_proposals(result["extraction"]["window"],
                                   result["extraction"]["proposals"],
                                   model_identity=result["extraction"]["model_identity"])
    assert len(refs) == 1 and source.ledger.verify_all()["candidates"] == 1


def test_repeated_exact_quote_with_bad_coordinate_stays_ambiguous():
    class Source:
        def reopen(self, descriptor):
            return {"text": "repeat repeat", "citation_ranges": [{"start": 0, "length": 13}]}
        def validated_proposals(self, descriptor, proposals):
            raise ValueError("ambiguous")
    output = json.dumps({"summary": "Repeated context.", "decisions": [], "open_items": [],
        "uncertainty": "Not settled.", "proposals": [{"type": "decision", "text": "Repeated.",
            "quote": "repeat", "start": 1}]})
    with pytest.raises(analysis.ModelOutputInvalid):
        analysis._cited_outcome(output, Source(), {}, "ollama", "fixture-model", "a" * 64)


def test_coordinate_repair_cannot_bypass_source_range_validation():
    seen = []
    class Source:
        def reopen(self, descriptor):
            return {"text": "left right", "citation_ranges": [
                {"start": 0, "length": 5}, {"start": 5, "length": 5}]}
        def validated_proposals(self, descriptor, proposals):
            seen.append(proposals[0]["start"])
            raise ValueError("quote crosses a source range")
    output = json.dumps({"summary": "Context.", "decisions": [], "open_items": [],
        "uncertainty": "Not settled.", "proposals": [{"type": "decision", "text": "Context.",
            "quote": "left right", "start": 1}]})
    with pytest.raises(analysis.ModelOutputInvalid):
        analysis._cited_outcome(output, Source(), {}, "ollama", "fixture-model", "a" * 64)
    assert seen == [0]


def test_valid_declared_repeat_coordinate_is_preserved():
    seen = []
    class Source:
        def reopen(self, descriptor):
            return {"text": "repeat repeat", "citation_ranges": [{"start": 0, "length": 13}]}
        def validated_proposals(self, descriptor, proposals):
            seen.append(proposals[0]["start"])
    output = json.dumps({"summary": "Context.", "decisions": [], "open_items": [],
        "uncertainty": "Not settled.", "proposals": [{"type": "decision", "text": "Repeated.",
            "quote": "repeat", "start": 7}]})
    result = analysis._cited_outcome(output, Source(), {}, "ollama", "fixture-model", "a" * 64)
    assert seen == [7] and result["extraction"]["proposals"][0]["start"] == 7


@pytest.mark.asyncio
async def test_whole_unit_denial_precedes_budget_or_remote_marker(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    seen, window = transport(monkeypatch, source, stage, local=False)
    monkeypatch.setattr(source, "remote_input", lambda descriptor: None)
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda **kwargs: pytest.fail("budget queried"))
    async def marker():
        pytest.fail("remote marker set")
    result = await analysis.analyze_cited_window(object(), source, stage["window"],
                                               allow_remote=True, before_remote=marker)
    assert result["reason"] == "source_not_remote_safe" and not seen


@pytest.mark.asyncio
async def test_actual_envelope_extra_field_is_screened_before_budget(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    seen, window = transport(monkeypatch, source, stage, local=False)
    original = analysis.Provider.request_body
    def injected(provider, messages):
        return {**original(provider, messages), "extra": {"nested": ["TOKEN=CANARY-SECRET-91919"]}}
    monkeypatch.setattr(analysis.Provider, "request_body", injected)
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda **kwargs: pytest.fail("budget queried"))
    result = await analysis.analyze_cited_window(object(), source, stage["window"], allow_remote=True)
    assert result["reason"] == "source_not_remote_safe" and not seen


@pytest.mark.asyncio
async def test_envelope_mutation_during_marker_is_blocked_as_proven_unsent(tmp_path, monkeypatch):
    journal, archive, job, stage, source = queued(tmp_path)
    seen, window = transport(monkeypatch, source, stage, local=False)
    original = analysis.Provider.request_body
    bodies, cleared = [], []
    def capture(provider, messages):
        body = original(provider, messages)
        bodies.append(body)
        return body
    monkeypatch.setattr(analysis.Provider, "request_body", capture)
    async def marker():
        bodies[0]["extra"] = "TOKEN=CANARY-SECRET-91919"
        return True
    async def unsent():
        cleared.append(True)
        return True
    result = await analysis.analyze_cited_window(object(), source, stage["window"],
        allow_remote=True, before_remote=marker, remote_not_sent=unsent)
    assert result["reason"] == "source_not_remote_safe" and not seen and cleared == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("already_published", [True, False])
async def test_worker_recovers_stage_without_model_dispatch(tmp_path, monkeypatch, already_published):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.mark_remote_dispatched(job.job_id, job.lease_token)
    if already_published:
        assert journal.begin_publication(job.job_id, job.lease_token)
        publish(source, stage)
    expire(journal, job)
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    monkeypatch.setattr(service, "_require_secure_archive", lambda: archive)
    async def forbidden(*args, **kwargs):
        pytest.fail("staged recovery redispatched inference")
    monkeypatch.setattr(analysis, "analyze_cited_window", forbidden)
    assert await service._process_secure_analysis_once()
    status = journal.get_analysis_job(job.job_id)
    assert status["state"] == "succeeded" and len(status["memory_refs"]) == 1
    assert source.ledger.verify_all()["candidates"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("integrity", [True, False])
async def test_worker_publication_failures_never_retry_inference(tmp_path, monkeypatch, integrity):
    journal, archive, job, stage, source = queued(tmp_path)
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    expire(journal, job)
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    monkeypatch.setattr(service, "_require_secure_archive", lambda: archive)
    def failure(*args, **kwargs):
        raise MemoryLedgerIntegrityError("fixture failure") if integrity else OSError("fixture failure")
    monkeypatch.setattr(CitedAnalysisSource, "record_proposals", failure)
    async def forbidden(*args, **kwargs):
        pytest.fail("publication failure redispatched inference")
    monkeypatch.setattr(analysis, "analyze_cited_window", forbidden)
    assert await service._process_secure_analysis_once()
    status = journal.get_analysis_job(job.job_id)
    assert status["state"] == ("failed" if integrity else "publication_pending")
    assert source.ledger.verify_all()["candidates"] == 0


def test_complete_request_screening_covers_json_escaped_strings():
    assert not analysis._request_safe({"messages": [{"content": "TOKEN=CANARY-SECRET-91919"}]})
    assert not analysis._request_safe({"extra": {"content": "C:\\Users\\user\\private.env"}})
    assert analysis._request_safe({"messages": [{"content": "Keep SQLite citations."}]})


def test_bundled_public_model_ids_are_configuration_not_credentials():
    body = {"model": analysis.llm_settings.DEFAULT_MODEL,
            "models": [analysis.llm_settings.DEFAULT_MODEL, *analysis.llm_settings.FALLBACK_MODELS],
            "messages": [{"content": "Keep SQLite citations."}]}
    assert analysis._request_safe(body)
    assert not analysis._request_safe({**body, "model": "TOKEN=CANARY-SECRET-91919"})
    assert not analysis._request_safe({**body, "extra": "TOKEN=CANARY-SECRET-91919"})
    assert not analysis._request_safe({**body, "messages": [{
        "content": analysis.llm_settings.FALLBACK_MODELS[-1]}]})


@pytest.mark.asyncio
async def test_cited_consent_generation_is_bound_before_source_await(tmp_path, monkeypatch):
    from muninn.history.remote_policy import write_policy
    journal, archive, job, stage, source = queued(tmp_path)
    fallback = lambda: (False, 1.0, 20.0, False)
    write_policy(tmp_path, enabled=True, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=fallback)
    class History:
        data_dir = tmp_path
    eligible = analysis._remote_eligible
    seen, window = transport(monkeypatch, source, stage, local=False)
    monkeypatch.setattr(analysis, "_remote_eligible", eligible)
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda **kwargs: pytest.fail("new consent acquired"))
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.to_thread
    async def delayed(func, *args, **kwargs):
        if getattr(func, "__name__", None) == "reopen":
            entered.set()
            await release.wait()
        return await original(func, *args, **kwargs)
    monkeypatch.setattr(analysis.asyncio, "to_thread", delayed)
    pending = asyncio.create_task(analysis.analyze_cited_window(
        History(), source, stage["window"], allow_remote=True, prefer_remote=True))
    await asyncio.wait_for(entered.wait(), timeout=2)
    write_policy(tmp_path, enabled=False, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=fallback)
    write_policy(tmp_path, enabled=True, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=fallback)
    release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    assert result["status"] == "deferred" and not seen


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [True, False])
async def test_production_archive_secret_outside_window_blocks_remote(tmp_path, monkeypatch, legacy):
    journal, archive, job, stage, source = queued(tmp_path)
    path = tmp_path / "risky.jsonl"
    text = "Keep needle citations. " + "Safe context. " * 500 + " TOKEN=CANARY-SECRET-91919"
    path.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": text}}) + "\n", encoding="utf-8")
    archive.archive_file(path, "codex")
    entry = archive._load_manifest()["files"][str(path.resolve())][0]
    cap = SecureHistoryBlindIndex(archive)._capability(entry, 0, "needle")
    source = CitedAnalysisSource(archive)
    descriptor = source.prepare(cap)
    assert "CANARY-SECRET" not in source.reopen(descriptor)["text"]
    assert source.remote_input(descriptor) is None
    transport(monkeypatch, source, {**stage, "window": descriptor}, local=False)
    monkeypatch.setattr(analysis, "guarded_openrouter_available", lambda **kwargs: pytest.fail("budget queried"))
    monkeypatch.setattr(analysis.Provider, "from_env", lambda *args: pytest.fail("remote credential accessed"))
    class History:
        def _require_secure_archive(self):
            return archive
        def _secure_model_window(self, capability):
            return "Keep needle citations."
    if legacy:
        result = await analysis.analyze_secure_hit(History(), cap, allow_remote=True, prefer_remote=True)
    else:
        result = await analysis.analyze_cited_window(History(), source, descriptor, allow_remote=True)
    assert result["reason"] == "source_not_remote_safe"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["gpu_telemetry_unavailable", "no_eligible_model_fits",
    "ollama_model_already_resident", "no_chat_model_fits", "source_not_remote_safe"])
async def test_resource_deferral_keeps_immutable_job_eligible_for_later_local_work(tmp_path, monkeypatch, reason):
    journal, archive, job, stage, source = queued(tmp_path)
    expire(journal, job)
    service = HistoryService(None, tmp_path / "service", home=tmp_path)
    service._capture_journal = journal
    monkeypatch.setattr(service, "_require_secure_archive", lambda: archive)
    async def unavailable(*args, **kwargs):
        return {"status": "deferred", "provider": None, "model": None, "reason": reason}
    monkeypatch.setattr(analysis, "analyze_cited_window", unavailable)
    assert await service._process_secure_analysis_once()
    status = journal.get_analysis_job(job.job_id)
    assert status["state"] == "retry" and status["error_code"] == reason
    assert journal._analysis_row(journal._publication_row(job.job_id)).window is not None


def test_zdr_preview_requires_explicit_enabled_managed_policy_root(tmp_path):
    from scripts.smoke_memory_ledger_archive import preview_policy_root, run
    from muninn.history.remote_policy import write_policy
    with pytest.raises(ValueError, match="explicit policy root"):
        run(tmp_path / "not-an-archive", zdr_analysis_preview=True)
    with pytest.raises(ValueError, match="enabled managed policy"):
        preview_policy_root(tmp_path)
    policy_dir = tmp_path / "different-policy-directory"
    policy_dir.mkdir()
    write_policy(policy_dir, enabled=True, daily_usd=1, monthly_usd=20,
                 override_ceiling=False, fallback=lambda: (False, 1, 20, False))
    assert preview_policy_root(policy_dir) == policy_dir.resolve()
