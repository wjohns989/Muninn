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


def test_default_models_are_luna_then_zdr_fallbacks():
    assert llm_settings.models() == ["openai/gpt-6-luna", "deepseek/deepseek-v4-flash",
                                     "google/gemini-3.5-flash-lite"]


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
    assert provider.name == "openrouter" and body["models"][0] == "openai/gpt-6-luna"
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
            endpoint = {"model_id": "openai/gpt-6-luna", "supported_parameters": ["structured_outputs"],
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
