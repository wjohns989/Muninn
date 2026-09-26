"""Thread understanding: key setup, model routing with zero data retention, validated storage."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from muninn.history import insights, llm_settings
from muninn.history.importer import collect, import_history, read_thread
from muninn.history.insights import Provider, analyze_threads, parse_reply, render_turns, validate_reply
from muninn.history.vault import HistoryVault
from muninn.store.sqlite_metadata import SQLiteMetadataStore

sys.path.insert(0, str(Path(__file__).parent))
from test_history_import import T0, FakeMemory, home, relay  # noqa: E402,F401  (fixtures)

GOOD = {"summary": "Built the login page; logout pending.", "status": "in_progress", "topics": ["Auth", "frontend"],
        "insights": [{"kind": "decision", "text": "Sessions use JWT cookies.", "turn": 0, "scope": "project"},
                     {"kind": "open_item", "text": "Add logout confirmation.", "turn": 2, "scope": "project"}]}


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    for var in ("OPENROUTER_API_KEY", "MUNINN_OPENROUTER_API_KEY", "MUNINN_INSIGHTS_PROVIDER",
                "MUNINN_INSIGHTS_MODEL", "MUNINN_INSIGHTS_WINDOW_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MUNINN_CONFIG_DIR", str(tmp_path / "config"))


# --- key and model settings -----------------------------------------------------------------

def test_key_is_saved_privately_and_env_wins(monkeypatch):
    assert llm_settings.api_key() is None and llm_settings.should_prompt()
    path = llm_settings.save_key("sk-or-saved", model="deepseek/deepseek-v4.1-flash")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert llm_settings.api_key() == "sk-or-saved" and llm_settings.key_source() == str(path)
    assert llm_settings.models()[0] == "deepseek/deepseek-v4.1-flash"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert llm_settings.api_key() == "sk-or-env" and "OPENROUTER_API_KEY" in llm_settings.key_source()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    llm_settings.decline()
    assert llm_settings.api_key() is None and not llm_settings.should_prompt()


def test_default_models_are_luna_pro_then_zdr_fallbacks():
    assert llm_settings.models() == ["openai/gpt-6-luna-pro", "deepseek/deepseek-v4-flash",
                                     "google/gemini-3.5-flash-lite"]


def test_batch_variants_are_used_as_their_direct_model(monkeypatch):
    # ':batch' ids only work through the asynchronous Batch API (which stores data up to 30 days).
    llm_settings.save_key("sk-or-k", model="openai/gpt-6-luna-pro:batch")
    assert llm_settings.models()[0] == "openai/gpt-6-luna-pro"
    monkeypatch.setenv("MUNINN_INSIGHTS_MODEL", "deepseek/deepseek-v4-flash:batch")
    assert llm_settings.models()[0] == "deepseek/deepseek-v4-flash"
    chosen = Provider.from_env(model="openai/gpt-6-luna:batch")
    assert chosen.model == "openai/gpt-6-luna" and len(chosen.models) == 3   # OpenRouter's limit
    assert len(chosen.request_body([])["models"]) <= 3
    assert oct(llm_settings.settings_path().parent.stat().st_mode & 0o777) == "0o700"


def test_first_run_prompt_saves_a_verified_key(monkeypatch, capsys):
    from muninn import cli

    answers = iter(["bad-key", "sk-or-good"])
    monkeypatch.setattr("getpass.getpass", lambda prompt: next(answers))
    monkeypatch.setattr(llm_settings, "verify_key",
                        lambda key: (key == "sk-or-good", "accepted" if key == "sk-or-good" else "rejected"))
    assert cli._prompt_openrouter_key(first_run=True) is True
    assert llm_settings.api_key() == "sk-or-good"
    assert "zero data retention" in capsys.readouterr().out


def test_first_run_skip_chooses_ollama_and_stops_asking(monkeypatch):
    from muninn import cli

    monkeypatch.setattr("getpass.getpass", lambda prompt: "")
    assert cli._prompt_openrouter_key(first_run=True) is False
    assert not llm_settings.should_prompt()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    called = []
    monkeypatch.setattr(cli, "_prompt_openrouter_key", lambda **kw: called.append(kw))
    cli._maybe_first_run_openrouter()
    assert called == []


def test_verify_key_uses_openrouter_key_endpoint(monkeypatch):
    seen = {}

    def fake_get(url, headers, timeout):
        seen.update(url=url, auth=headers["Authorization"])
        return httpx.Response(200, json={"data": {"label": "muninn", "limit": None}})

    monkeypatch.setattr(httpx, "get", fake_get)
    ok, message = llm_settings.verify_key("sk-or-x")
    assert ok and "muninn" in message and seen == {"url": "https://openrouter.ai/api/v1/key", "auth": "Bearer sk-or-x"}


# --- request shape --------------------------------------------------------------------------------

def test_openrouter_request_enforces_zdr_schema_and_only_supported_parameters():
    llm_settings.save_key("sk-or-k")
    provider = Provider.from_env()
    body = provider.request_body([{"role": "user", "content": "x"}])
    assert provider.name == "openrouter" and body["models"][0] == "openai/gpt-6-luna-pro"
    assert body["provider"] == {"zdr": True, "data_collection": "deny", "require_parameters": True}
    schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema" and schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) == {"summary", "status", "topics", "insights", "supersedes"}
    insight = schema["schema"]["properties"]["insights"]["items"]
    assert set(insight["required"]) == set(insight["properties"]) == {"kind", "text", "turn", "scope", "current"}
    # GPT-6 Luna's ZDR endpoints do not accept these; with require_parameters they would exclude it.
    assert "temperature" not in body and "max_tokens" not in body
    assert Provider.from_env(model="deepseek/deepseek-v4.1-flash").models[0] == "deepseek/deepseek-v4.1-flash"


def test_missing_key_explains_how_to_set_it():
    assert Provider.from_env().name == "ollama"
    with pytest.raises(ValueError, match="openrouter set"):
        Provider.from_env("openrouter")


# --- validation ---------------------------------------------------------------------------------

def test_validation_normalizes_before_storage():
    reply = json.dumps({"summary": "Done. token=supersecretvalue1", "status": "completed", "topics": ["Auth", " "],
                        "insights": [
                            {"kind": "decision", "text": "Use JWT", "turn": 99, "scope": "global"},
                            {"kind": "decision", "text": "use jwt", "turn": 1, "scope": "project"},
                            {"kind": "preference", "text": "Prefers pnpm", "turn": 0, "scope": "global"}]})
    result = validate_reply(f"```json\n{reply}\n```", turn_count=3)
    assert "supersecretvalue1" not in result["summary"] and result["topics"] == ["auth"]
    assert result["insights"] == [
        {"kind": "decision", "text": "Use JWT", "turn": None, "scope": "project", "current": True},  # bad turn
        {"kind": "preference", "text": "Prefers pnpm", "turn": 0, "scope": "global", "current": True},
    ]
    assert validate_reply(json.dumps({"summary": "", "status": "completed", "topics": [], "insights": [],
                                      "supersedes": ["N1", "N9"]}), notes={"N1": "mem-1"})["supersedes"] == ["mem-1"]
    with pytest.raises(Exception):
        validate_reply(json.dumps({"summary": "x", "status": "weird", "topics": [], "insights": []}))
    lenient = parse_reply(json.dumps({"status": "weird", "insights": [{"kind": "odd", "text": "Kept anyway"}]}))
    assert lenient["status"] == "completed" and lenient["insights"][0]["kind"] == "fact"


# --- end to end with a simulated OpenRouter ----------------------------------------------------------

@pytest.fixture
def imported(home, tmp_path):  # noqa: F811
    vault = HistoryVault(tmp_path / "vault", home=home)
    memory = FakeMemory(SQLiteMetadataStore(tmp_path / "metadata.db"))
    vault.sync()
    asyncio.run(import_history(memory, vault, apply=True, providers=["claude_code"]))
    yield SimpleNamespace(vault=vault, memory=memory, store=memory._metadata)
    vault.close()


def _openrouter(answers, sent):
    replies = iter(answers)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints/zdr"):
            endpoint = {"model_id": "openai/gpt-6-luna-pro", "supported_parameters": ["structured_outputs"],
                        "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}}
            return httpx.Response(200, json={"data": [endpoint]})
        sent.append((request.headers.get("authorization"), json.loads(request.content)))
        content = next(replies)
        return httpx.Response(200, json={"model": "openai/gpt-6-luna", "provider": "Azure",
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0002},
                                         "choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(respond)


def test_render_uses_whole_conversation_and_redacts(imported):
    thread = next(t for t in collect(imported.vault).threads if t.key == "claude_code:c-111")
    windows = render_turns(thread)
    assert len(windows) == 1 and "### Turn 2" in windows[0] and "Compaction summary" in windows[0]
    assert "x" * 6000 in windows[0]            # long replies are no longer cut to 2.5k characters
    assert "sk-abcdef" not in windows[0]
    assert len(render_turns(thread, window_chars=3000)) > 1


def test_analysis_validates_retries_and_stores(imported, monkeypatch):
    llm_settings.save_key("sk-or-k")
    monkeypatch.setattr(insights, "HANDOFF_MAX_AGE_DAYS", 100_000)
    sent = []
    # First thread: an invalid reply, then a valid one after the schema error is sent back.
    answers = ["not json at all", json.dumps(GOOD), json.dumps(GOOD)]
    dry = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=False,
                                      transport=_openrouter(answers, [])))
    assert dry["threads"] == 2 and dry["approx_cost_usd"] is not None and dry["zero_data_retention"]

    report = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, concurrency=1,
                                         transport=_openrouter(answers, sent)))
    assert not report["errors"] and report["schema_retries"] == 1 and report["lenient_parses"] == 0
    assert report["calls"] == 3 and report["cost_usd"] == pytest.approx(0.0006)
    assert report["models_used"] == {"openai/gpt-6-luna": 3}
    assert "does not match the required JSON schema" in sent[1][1]["messages"][-1]["content"]
    assert all(body["provider"]["zdr"] and "sk-abcdef" not in json.dumps(body) for _, body in sent)

    stored = [r for r in imported.store.get_all(limit=500) if (r.metadata or {}).get("kind") == "thread_insight"]
    decision = next(r for r in stored if r.metadata["thread_id"] == "claude_code:c-111"
                    and r.metadata["insight_kind"] == "decision")
    assert decision.created_at == pytest.approx(T0) and decision.project == "webapp"
    assert decision.metadata["insight_model"] == "openai/gpt-6-luna"
    view = asyncio.run(read_thread(imported.memory, "claude_code:c-111"))
    assert {i["kind"] for i in view["insights"]} == {"decision", "open_item"}
    assert all(e["kind"] != "thread_insight" for e in view["entries"])
    thread = imported.store.list_history_threads("webapp", topic="auth")[0]
    assert thread["status"] == "in_progress" and thread["topics"] == ["auth", "frontend"]
    handoffs = imported.store.list_handoffs("global_user", "webapp", ["open"])
    assert handoffs and handoffs[0]["details"]["next_steps"] == ["Add logout confirmation."]


def test_reanalysis_replaces_insights_and_giant_threads_are_merged(imported, monkeypatch):
    llm_settings.save_key("sk-or-k")
    monkeypatch.setenv("MUNINN_INSIGHTS_WINDOW_TOKENS", "2000")   # force several windows
    part = json.dumps({**GOOD, "insights": GOOD["insights"][:1]})
    merged = json.dumps({**GOOD, "summary": "Whole conversation."})
    sent = []
    many = [part] * 20 + [merged]

    def answers():
        for content in many:
            yield content

    first = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, concurrency=1,
                                        project="webapp", limit=1, transport=_openrouter(answers(), sent)))
    assert first["calls"] > 2
    assert "parts" in sent[-1][1]["messages"][-1]["content"]         # final merge call
    key = imported.store.list_history_threads("webapp", limit=1)[0]["thread_key"]
    count = lambda: sum(1 for r in imported.store.get_thread_memories(key, 0, 1000)  # noqa: E731
                        if (r.metadata or {}).get("kind") == "thread_insight")
    before = count()
    imported.store.set_history_analysis(key, status="completed", topics=[], analyzed_turns=0)  # thread "grew"
    second = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, concurrency=1,
                                         project="webapp", limit=1, transport=_openrouter(answers(), [])))
    assert second["replaced_insights"] == before and count() == before


def test_catalog_filters(imported):
    store = imported.store
    assert {t["agent"] for t in store.list_history_threads(limit=50)} == {"claude-code", "claude-desktop"}
    assert [t["thread_key"] for t in store.list_history_threads(agent="claude-desktop")] == ["claude_code:c-222"]
    assert store.list_history_threads(text="Desktop login")[0]["thread_key"] == "claude_code:c-222"
    assert store.list_history_threads(since=T0 + 10**9) == []



def test_analysis_reads_projects_in_order_across_apps(relay, monkeypatch):  # noqa: F811
    """Codex decides, Claude Code later changes it in between Codex's turns; the old decision is retired."""
    from test_history_import import _codex

    llm_settings.save_key("sk-or-k")
    with relay.codex_file.open("a") as handle:   # Codex continues after Claude's work
        for row in _codex("cx-1", str(relay.repo), [(400, "codex step 5", "done 5")])[1:]:
            handle.write(json.dumps(row) + "\n")
    relay.vault.sync()
    asyncio.run(import_history(relay.memory, relay.vault, apply=True))
    requests = []

    def answer(body):
        text = body["messages"][-1]["content"]
        requests.append(text)
        if "Agent: Codex" in text:
            reply = {**GOOD, "status": "completed", "insights": [
                {"kind": "decision", "text": "The API uses REST.", "turn": 0, "scope": "project", "current": True}],
                "supersedes": []}
        elif "claude step 7" in text:
            reply = {**GOOD, "status": "completed", "supersedes": [], "insights": [
                {"kind": "fact", "text": "A temporary mock server was used.", "turn": 0, "scope": "project",
                 "current": False}]}
        else:
            reply = {**GOOD, "status": "completed", "insights": [
                {"kind": "decision", "text": "The API moves to GraphQL.", "turn": 0, "scope": "project",
                 "current": True}], "supersedes": ["N1"]}
        return json.dumps(reply)

    def respond(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"model": "openai/gpt-6-luna", "usage": {},
                                         "choices": [{"message": {"content": answer(body)}}]})

    report = asyncio.run(analyze_threads(relay.memory, relay.vault, apply=True, concurrency=4,
                                         transport=httpx.MockTransport(respond)))
    assert not report["errors"] and report["threads"] == 3
    codex_request = next(r for r in requests if "Agent: Codex" in r)
    assert "Meanwhile" in codex_request and "claude_code:cc-1" in codex_request
    claude_request = next(r for r in requests if "Agent: Claude Code" in r and "claude step 7" not in r)
    assert "[N1]" in claude_request and "The API uses REST." in claude_request

    insights_by_text = {r.content.split("\n")[0]: r for r in relay.store.get_all(limit=200)
                        if (r.metadata or {}).get("kind") == "thread_insight"}
    rest = insights_by_text["Decision: The API uses REST."]
    assert rest.archived and rest.metadata["superseded"] is True
    graphql = insights_by_text["Decision: The API moves to GraphQL."]
    assert not graphql.archived
    assert report["superseded_insights"] == 1
    replaced_here = insights_by_text["Fact: A temporary mock server was used."]
    assert replaced_here.archived and replaced_here.metadata["superseded"] is True


# --- refusals ------------------------------------------------------------------------------------

def _reply(content="", **choice):
    return 200, {"model": "openai/gpt-6-luna-pro", "usage": {"prompt_tokens": 10, "cost": 0.0001},
                 "choices": [{"message": {"content": content, **choice.pop("message", {})}, **choice}]}


REFUSALS = {
    "refusal field": _reply(message={"refusal": "I can't help with that."}),
    "content filter stop": _reply(json.dumps(GOOD)[:40], finish_reason="content_filter"),
    "native safety stop": _reply("", finish_reason="stop", native_finish_reason="SAFETY"),
    "plain-text refusal": _reply("I'm sorry, but I can't help with summarizing this conversation."),
    "refusal as the summary": _reply(json.dumps({**GOOD, "summary": "I cannot assist with this request.",
                                                 "insights": []})),
    "empty reply": _reply(""),
    "moderation error": (403, {"error": {
        "code": 403, "message": "Input was flagged by moderation",
        "metadata": {"reasons": ["sexual"], "model_slug": "openai/gpt-6-luna-pro"}}}),
    "azure content filter": (400, {"error": {"code": "content_filter", "message": "The response was filtered "
                                             "due to the prompt triggering Azure OpenAI's content management policy."}}),
}


def _scripted(replies, sent):
    queue = iter(replies)

    def respond(request):
        body = json.loads(request.content)
        sent.append(body)
        status, payload = next(queue)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("signal", sorted(REFUSALS))
def test_a_refusal_is_never_stored_and_the_next_model_answers(imported, signal):
    llm_settings.save_key("sk-or-k")
    sent = []
    good = (200, {"model": "deepseek/deepseek-v4-flash", "usage": {}, "choices": [{"message": {
        "content": json.dumps(GOOD)}, "finish_reason": "stop"}]})
    report = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, project="webapp", limit=1,
                                         transport=_scripted([REFUSALS[signal], good], sent)))
    assert not report["errors"] and not report["refused_threads"] and report["refusals"] == 1
    # OpenRouter's own fallback only covers errors: the refused request is re-sent to the next model.
    assert sent[1]["model"] == "deepseek/deepseek-v4-flash" and "openai/gpt-6-luna-pro" not in sent[1]["models"]
    stored = [r for r in imported.store.get_all(limit=500) if (r.metadata or {}).get("kind") == "thread_insight"]
    assert stored and all(r.metadata["insight_model"] == "deepseek/deepseek-v4-flash" for r in stored)
    assert not any("can't" in r.content or "cannot" in r.content for r in stored)


def test_when_every_model_refuses_nothing_is_stored_and_the_thread_says_why(imported):
    llm_settings.save_key("sk-or-k")
    key = imported.store.list_history_threads("webapp", limit=1)[0]["thread_key"]
    summary_before = imported.store.get(imported.store.get_history_thread(key)["summary_memory_id"]).content
    refusal = REFUSALS["plain-text refusal"]
    report = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, project="webapp", limit=1,
                                         transport=_scripted([refusal] * 3, [])))
    assert report["refusals"] == 3 and report["insights"] == 0 and not report["errors"]
    refused = report["refused_threads"][0]
    assert refused["thread"] == key and "every model refused" in refused["reason"]
    state = imported.store.get_history_thread(key)
    assert state["status"] is None and state["analysis_error"].startswith("refused:")
    assert imported.store.get(state["summary_memory_id"]).content == summary_before   # untouched
    assert not [r for r in imported.store.get_all(limit=500) if (r.metadata or {}).get("kind") == "thread_insight"]
    assert len(imported.store.get_thread_memories(key, 0, 1000)) > 0                   # imported turns remain

    # Not retried on every run (that would pay for the same refusal again)...
    pending = lambda **kw: [t["thread_key"] for t in imported.store.list_history_threads(  # noqa: E731
        "webapp", 50, needs_analysis=True, **kw)]
    assert key not in pending() and key in pending(retry_refused=True)
    # ...unless asked, e.g. with another model; a success clears the error.
    good = _reply(json.dumps(GOOD), finish_reason="stop")
    retried = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, project="webapp", limit=1,
                                          retry_refused=True, transport=_scripted([good], [])))
    assert retried["threads"] == 1 and retried["insights"] == 2
    assert imported.store.get_history_thread(key)["analysis_error"] is None


def test_an_unreadable_reply_is_an_error_not_an_empty_result(imported):
    llm_settings.save_key("sk-or-k")
    garbage = _reply("{not json")
    report = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True, project="webapp", limit=1,
                                         transport=_scripted([garbage, garbage], [])))
    assert report["errors"] and report["insights"] == 0
    key = imported.store.list_history_threads("webapp", limit=1)[0]["thread_key"]
    assert imported.store.get_history_thread(key)["status"] is None       # not marked "completed"


def test_refusal_wording_only_matches_refusals():
    from muninn.history.insights import looks_like_refusal

    for text in ("I'm sorry, but I can't help with that.", "I cannot assist with this request.",
                 "Sorry, I won't summarize this.", "As an AI, I must decline.", "I’m unable to process this content.",
                 "Unfortunately, I can't provide a summary of this conversation."):
        assert looks_like_refusal(text), text
    for text in ("The user asked the assistant to fix the login page.", "Implemented the parser; I/O is pending.",
                 "In this conversation the assistant said it cannot reproduce the bug.",
                 "Sorry-state handling was added to the checkout flow."):
        assert not looks_like_refusal(text, anywhere=True), text
    assert looks_like_refusal("The chat is a story draft with explicit scenes, so I can't summarize it.", anywhere=True)
    assert not looks_like_refusal("The chat is a story draft, so I can't summarize it.")   # start only
    kept = validate_reply(json.dumps({**GOOD, "insights": [
        {"kind": "fact", "text": "I cannot provide details about this content.", "turn": 0, "scope": "project"},
        {"kind": "fact", "text": "Drafts live in drafts/.", "turn": 0, "scope": "project"}]}))
    assert [i["text"] for i in kept["insights"]] == ["Drafts live in drafts/."]
