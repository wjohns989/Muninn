"""Thread understanding: provider selection, zero data retention, parsing, and stored insights."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from muninn.history import insights
from muninn.history.importer import import_history
from muninn.history.insights import Provider, analyze_threads, parse_reply, render_turns
from muninn.history.vault import HistoryVault
from muninn.store.sqlite_metadata import SQLiteMetadataStore

sys.path.insert(0, str(Path(__file__).parent))
from test_history_import import T0, FakeMemory, home  # noqa: E402,F401  (fixture)


def test_provider_choice_and_zdr(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "MUNINN_OPENROUTER_API_KEY", "MUNINN_INSIGHTS_PROVIDER", "MUNINN_INSIGHTS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert Provider.from_env().name == "ollama"
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Provider.from_env("openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    chosen = Provider.from_env()
    assert chosen.name == "openrouter" and chosen.base_url == "https://openrouter.ai/api/v1"
    body = chosen.request_body([{"role": "user", "content": "x"}])
    assert body["provider"] == {"zdr": True, "data_collection": "deny"}
    assert "provider" not in Provider.from_env("ollama").request_body([])


def test_parse_reply_normalizes_and_redacts():
    reply = """```json
    {"summary": "Built login. token=supersecretvalue1", "status": "weird", "topics": ["Auth", " "],
     "insights": [
       {"kind": "decision", "text": "Use JWT cookies", "turn": 2, "scope": "global"},
       {"kind": "preference", "text": "User prefers pnpm", "turn": 0, "scope": "global"},
       {"kind": "nonsense", "text": "key sk-abcdefghijklmnopqrstuvwxyz0123 lives in env"},
       {"kind": "fact", "text": "  "}]}
    ```"""
    parsed = parse_reply(reply)
    assert parsed["status"] == "completed" and parsed["topics"] == ["auth"]
    assert "supersecretvalue1" not in parsed["summary"]
    kinds = [(i["kind"], i["scope"]) for i in parsed["insights"]]
    assert kinds == [("decision", "project"), ("preference", "global"), ("fact", "project")]
    assert "abcdefghijk" not in parsed["insights"][2]["text"]


@pytest.fixture
def imported(home, tmp_path, monkeypatch):  # noqa: F811
    for var in ("OPENROUTER_API_KEY", "MUNINN_OPENROUTER_API_KEY", "MUNINN_INSIGHTS_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    vault = HistoryVault(tmp_path / "vault", home=home)
    memory = FakeMemory(SQLiteMetadataStore(tmp_path / "metadata.db"))
    vault.sync()
    asyncio.run(import_history(memory, vault, apply=True, providers=["claude_code"]))
    yield SimpleNamespace(vault=vault, memory=memory, store=memory._metadata)
    vault.close()


def test_render_turns_is_redacted_and_indexed(imported):
    from muninn.history.importer import collect

    thread = next(t for t in collect(imported.vault).threads if t.key == "claude_code:c-111")
    text = "".join(render_turns(thread))
    assert "### Turn 0" in text and "### Turn 2" in text and "Compaction summary" in text
    assert "sk-abcdef" not in text


def test_analysis_stores_dated_insights_and_updates_the_catalog(imported, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setattr(insights, "HANDOFF_MAX_AGE_DAYS", 100_000)
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((request.headers.get("authorization"), body))
        answer = {"summary": "Built the login page; logout pending.", "status": "in_progress",
                  "topics": ["auth", "frontend"],
                  "insights": [{"kind": "decision", "text": "Sessions use JWT cookies.", "turn": 0},
                               {"kind": "open_item", "text": "Add logout confirmation.", "turn": 2}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    dry = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=False))
    assert dry["threads"] == 2 and dry["approx_input_tokens"] > 0 and not sent

    report = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=True,
                                         transport=httpx.MockTransport(respond)))
    assert report["zero_data_retention"] and report["threads"] == 2 and not report["errors"]
    assert all(auth == "Bearer or-key" and body["provider"]["zdr"] is True for auth, body in sent)
    assert all("sk-abcdef" not in json.dumps(body) for _, body in sent)

    stored = [r for r in imported.store.get_all(limit=500) if (r.metadata or {}).get("kind") == "thread_insight"]
    decision = next(r for r in stored if r.metadata["thread_id"] == "claude_code:c-111"
                    and r.metadata["insight_kind"] == "decision")
    assert decision.created_at == pytest.approx(T0) and decision.project == "webapp"
    assert decision.content.startswith("Decision: Sessions use JWT cookies.")
    thread = imported.store.list_history_threads("webapp", topic="auth")[0]
    assert thread["status"] == "in_progress" and thread["topics"] == ["auth", "frontend"]
    summary = imported.store.get(imported.store.get_history_thread("claude_code:c-111")["summary_memory_id"])
    assert "status: in_progress" in summary.content and "logout pending" in summary.content
    handoffs = imported.store.list_handoffs("global_user", "webapp", ["open"])
    assert handoffs and handoffs[0]["details"]["next_steps"] == ["Add logout confirmation."]
    assert report["handoffs"] == len({h["details"]["thread_id"] for h in handoffs})

    again = asyncio.run(analyze_threads(imported.memory, imported.vault, apply=False))
    assert again["threads"] == 0  # nothing new since the last analysis


def test_catalog_filters(imported):
    store = imported.store
    assert {t["agent"] for t in store.list_history_threads(limit=50)} == {"claude-code", "claude-desktop"}
    assert [t["thread_key"] for t in store.list_history_threads(agent="claude-desktop")] == ["claude_code:c-222"]
    assert store.list_history_threads(text="Desktop login")[0]["thread_key"] == "claude_code:c-222"
    assert store.list_history_threads(since=T0 + 10**9) == []
