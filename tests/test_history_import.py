"""Local AI history: locations, the never-delete vault, parsers, and ordered project-tagged import."""

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from muninn.core.types import MemoryRecord
from muninn.history import parsers
from muninn.history.importer import PART_CHARS, collect, import_history, read_thread, redact
from muninn.history.locations import history_sources
from muninn.history.vault import HistoryVault, read_text, version_path
from muninn.store.sqlite_metadata import SQLiteMetadataStore

T0 = 1_780_000_000.0  # 2026-05-28


def iso(offset: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(T0 + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def claude_rows(cwd: str, session="c-111", entry="cli"):
    base = {"sessionId": session, "cwd": cwd, "gitBranch": "feat/login", "entrypoint": entry, "isSidechain": False}
    return [
        {**base, "type": "user", "timestamp": iso(0), "message": {"role": "user", "content": "Add a login page"}},
        {**base, "type": "assistant", "timestamp": iso(5), "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/login.tsx"}},
            {"type": "text", "text": "Created the login page. Key sk-abcdefghijklmnopqrstuvwxyz0123 removed."}]}},
        {**base, "type": "user", "timestamp": iso(6), "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "ok"}]}},
        {**base, "type": "user", "timestamp": iso(7), "message": {"role": "user", "content": [
            {"type": "text", "text": "<task-notification>done</task-notification>"}]}},
        {**base, "type": "user", "isSidechain": True, "timestamp": iso(8),
         "message": {"role": "user", "content": "subagent prompt"}},
        {**base, "type": "user", "timestamp": iso(60), "isCompactSummary": True, "message": {
            "role": "user", "content": "This session is being continued from a previous conversation. Summary: "
                                       "we chose JWT cookies."}},
        {**base, "type": "user", "timestamp": iso(70), "message": {"role": "user", "content": (
            "<command-message>review</command-message>\n<command-name>/review</command-name>\n"
            "<command-args>src/login.tsx</command-args>")}},
        {**base, "type": "assistant", "timestamp": iso(75), "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "npm test -- login"}},
            {"type": "text", "text": "Review done. " + "x" * (PART_CHARS + 500)}]}},
        {**base, "type": "attachment", "timestamp": iso(80), "attachment": {
            "type": "queued_command", "prompt": "also add a logout button", "timestamp": iso(80)}},
        {**base, "type": "assistant", "timestamp": iso(85), "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Logout button added."}]}},
    ]


def codex_rows(cwd: str):
    return [
        {"timestamp": iso(1000), "type": "session_meta", "payload": {
            "id": "0199-codex-aaaa", "timestamp": iso(1000), "cwd": cwd, "originator": "codex_desktop",
            "git": {"branch": "main", "repository_url": "git@github.com:me/Shop.git"}}},
        {"timestamp": iso(1001), "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>x"}]}},
        {"timestamp": iso(1002), "type": "event_msg",
         "payload": {"type": "user_message", "message": "Fix checkout tax"}},
        {"timestamp": iso(1002), "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix checkout tax"}]}},
        {"timestamp": iso(1003), "type": "response_item", "payload": {
            "type": "custom_tool_call", "name": "apply_patch", "call_id": "1",
            "input": "*** Begin Patch\n*** Update File: shop/tax.py\n*** End Patch"}},
        {"timestamp": iso(1004), "type": "event_msg", "payload": {"type": "agent_message", "message": "Tax fixed."}},
        {"timestamp": iso(1004), "type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Tax fixed."}]}},
        {"timestamp": iso(1100), "type": "compacted", "payload": {"message": "Earlier: tax uses banker's rounding."}},
    ]


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    repo = home / "code" / "webapp"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    # Claude Code (CLI) and a Claude Desktop Code-tab session
    jsonl(home / ".claude" / "projects" / "-home-code-webapp" / "c-111.jsonl", claude_rows(str(repo)))
    jsonl(home / ".claude" / "projects" / "-home-code-webapp" / "c-222.jsonl",
          claude_rows(str(repo), session="c-222", entry="claude-desktop")[:2])
    meta = home / ".config" / "Claude" / "claude-code-sessions" / "acct" / "ws" / "local_1.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"cliSessionId": "c-222", "title": "Desktop login work"}))
    jsonl(home / ".claude" / "history.jsonl", [
        {"display": "Add a login page", "timestamp": (T0 + 0) * 1000, "project": str(repo), "sessionId": "c-111"},
        {"display": "old prompt from a deleted session", "timestamp": (T0 - 86400 * 90) * 1000, "project": str(repo)},
    ])
    (home / ".claude" / "settings.json").write_text(json.dumps({"cleanupPeriodDays": 30}))
    # Codex: new layout, early layout (archived), compressed, thread titles, credentials
    jsonl(home / ".codex" / "sessions" / "2026" / "05" / "28" / "rollout-2026-05-28T01-00-00-0199-codex-aaaa.jsonl",
          codex_rows("/somewhere/Shop"))
    jsonl(home / ".codex" / "archived_sessions" / "rollout-2025-01-02-old.jsonl", [
        {"id": "old-1", "timestamp": iso(-500000), "instructions": "sys"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "early codex question"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "early answer"}]},
    ])
    zstandard = pytest.importorskip("zstandard")
    raw = "\n".join(json.dumps(r) for r in codex_rows("/somewhere/Shop"))
    raw = raw.replace("0199-codex-aaaa", "0199-codex-zzzz").replace("Fix checkout tax", "Zipped request")
    zst = home / ".codex" / "archived_sessions" / "rollout-2026-05-01-0199-codex-zzzz.jsonl.zst"
    zst.write_bytes(zstandard.ZstdCompressor().compress(raw.encode()))
    db = sqlite3.connect(home / ".codex" / "state_5.sqlite")
    db.execute("CREATE TABLE threads (id TEXT, title TEXT, archived INTEGER)")
    db.execute("INSERT INTO threads VALUES ('0199-codex-aaaa', 'Checkout tax fix', 0)")
    db.commit()
    db.close()
    (home / ".codex" / "auth.json").write_text('{"token": "secret"}')
    # Gemini CLI: project folder named by sha256 of the project root
    gem = home / ".gemini" / "tmp" / hashlib.sha256(str(repo).encode()).hexdigest() / "chats" / "session-1.json"
    gem.parent.mkdir(parents=True)
    gem.write_text(json.dumps({"sessionId": "g-1", "startTime": iso(2000), "messages": [
        {"type": "user", "timestamp": iso(2000), "content": "Explain the auth flow"},
        {"type": "gemini", "timestamp": iso(2001), "content": [{"text": "It uses cookies."}]}]}))
    (home / ".gemini" / "oauth_creds.json").write_text('{"refresh_token": "secret"}')
    # Data exports in Downloads
    downloads = home / "Downloads"
    downloads.mkdir()
    (downloads / "conversations.json").write_text(json.dumps([{
        "conversation_id": "gpt-1", "title": "Trip plan", "create_time": T0 + 3000, "current_node": "n3",
        "mapping": {
            "n1": {"message": None, "parent": None},
            "n2": {"parent": "n1", "message": {"author": {"role": "user"}, "create_time": T0 + 3000,
                                                "content": {"content_type": "text", "parts": ["Plan Rome"]}}},
            "n2b": {"parent": "n1", "message": {"author": {"role": "user"}, "create_time": T0 + 2999,
                                                 "content": {"content_type": "text", "parts": ["abandoned edit"]}}},
            "n3": {"parent": "n2", "message": {"author": {"role": "assistant"}, "create_time": T0 + 3001,
                                                "content": {"content_type": "text", "parts": ["Day 1: Colosseum"]}}},
        }}]))
    with zipfile.ZipFile(downloads / "claude-export.zip", "w") as archive:
        archive.writestr("conversations.json", json.dumps([{
            "uuid": "cl-1", "name": "Essay", "created_at": iso(4000), "chat_messages": [
                {"sender": "human", "text": "Outline my essay", "created_at": iso(4000)},
                {"sender": "assistant", "content": [{"type": "text", "text": "I. Intro"}], "created_at": iso(4001)}]}]))
    return home


class FakeMemory:
    """MuninnMemory's add/update over a real metadata store (no embeddings needed)."""

    def __init__(self, store):
        self._metadata = store

    async def add(self, content, user_id, agent_id=None, metadata=None, memory_type=None, provenance=None,
                  scope="project", **_):
        metadata = dict(metadata or {}, user_id=user_id)
        record = MemoryRecord(content=content, project=metadata.get("project", "global"), scope=scope,
                              source_agent=agent_id or "unknown", branch=metadata.get("branch"), metadata=metadata)
        self._metadata.add(record)
        return {"id": record.id, "event": "ADD"}

    async def update(self, memory_id, data=None, metadata_patch=None, archived=None, **_):
        record = self._metadata.get(memory_id)
        fields = {"content": data or record.content,
                  "metadata": dict(record.metadata or {}, **(metadata_patch or {}))}
        if archived is not None:
            fields["archived"] = int(archived)
        self._metadata.update(memory_id, **fields)
        return {"id": memory_id}

    async def delete(self, memory_id):
        self._metadata.delete(memory_id)
        return {"id": memory_id, "event": "DELETE"}


@pytest.fixture
def env(home, tmp_path):
    vault = HistoryVault(tmp_path / "vault", home=home)
    store = SQLiteMetadataStore(tmp_path / "metadata.db")
    yield SimpleNamespace(home=home, vault=vault, store=store, memory=FakeMemory(store))
    vault.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# --- locations ---------------------------------------------------------------

def test_locations_follow_the_apps_relocation_settings(home, tmp_path, monkeypatch):
    sources = {s.provider: s for s in history_sources(home)}
    assert sources["claude_code"].home == home / ".claude" and sources["claude_code"].retention["days"] == 30
    assert sources["codex"].extras["state_db"] == [home / ".codex" / "state_5.sqlite"]
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "moved-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "moved-codex"))
    moved = {s.provider: s for s in history_sources(home)}
    assert moved["claude_code"].home == tmp_path / "moved-claude" and moved["claude_code"].relocated_by
    assert moved["codex"].home == tmp_path / "moved-codex" and moved["codex"].relocated_by == "CODEX_HOME"


# --- vault ---------------------------------------------------------------------

def test_vault_copies_everything_but_credentials(env):
    report = env.vault.sync()
    assert report["new"] >= 10 and not report["errors"]
    names = {Path(f.source_path).name for f in env.vault.files()}
    assert {"auth.json", "oauth_creds.json", "settings.json"}.isdisjoint(names)
    assert {"c-111.jsonl", "history.jsonl", "state_5.sqlite", "conversations.json", "claude-export.zip"} <= names
    copy = next(f for f in env.vault.files() if f.source_path.endswith("c-111.jsonl"))
    assert copy.path.name.endswith(".gz") and oct(copy.path.stat().st_mode & 0o777) == "0o600"
    assert "Add a login page" in read_text(copy.path)
    assert env.vault.sync()["unchanged"] == report["new"]


def test_vault_keeps_what_the_app_deletes_or_rewrites(env):
    env.vault.sync()
    transcript = env.home / ".claude" / "projects" / "-home-code-webapp" / "c-111.jsonl"
    original = transcript.read_text()
    transcript.write_text(original.splitlines()[0] + "\n")  # rewritten shorter
    os.utime(transcript, (T0 + 10, T0 + 10))
    report = env.vault.sync()
    assert report["versions_kept"] == 1
    kept = next(f for f in env.vault.files() if f.source_path == str(transcript.resolve()))
    assert read_text(version_path(kept.path, 1)) == original
    transcript.unlink()  # the app's cleanup deletes it
    assert env.vault.sync()["newly_missing"] == 1
    kept = next(f for f in env.vault.files() if f.source_path == str(transcript.resolve()))
    assert kept.missing_since and kept.path.exists()
    assert env.vault.status()["totals"]["preserved_after_deletion"] == 1


# --- parsers ----------------------------------------------------------------------

def test_claude_parser_keeps_the_conversation_and_drops_noise(home):
    text = (home / ".claude" / "projects" / "-home-code-webapp" / "c-111.jsonl").read_text()
    session = parsers.parse_claude_code(text, "c-111.jsonl")
    assert [t.user for t in session.turns] == ["Add a login page", "/review src/login.tsx", "also add a logout button"]
    assert session.turns[0].files == ["src/login.tsx"] and "hidden" not in session.turns[0].assistant
    assert session.turns[1].actions == ["ran: npm test -- login"]
    assert session.turns[2].assistant == "Logout button added."
    assert [c.text[-21:] for c in session.compactions] == ["we chose JWT cookies."]
    assert session.branch == "feat/login" and session.agent == "claude-code"


def test_codex_parser_prefers_events_and_reads_old_layouts(home):
    new = parsers.parse_codex(json.dumps(codex_rows("/x")[0]) + "\n" + "\n".join(
        json.dumps(r) for r in codex_rows("/x")[1:]), "rollout.jsonl", {"0199-codex-aaaa": "Checkout tax fix"})
    assert [(t.user, t.assistant) for t in new.turns] == [("Fix checkout tax", "Tax fixed.")]
    assert new.turns[0].files == ["shop/tax.py"] and new.title == "Checkout tax fix"
    assert new.repo_url.endswith("Shop.git") and new.surface == "codex_desktop" and len(new.compactions) == 1
    old = parsers.parse_codex((home / ".codex" / "archived_sessions" / "rollout-2025-01-02-old.jsonl").read_text(), "x")
    assert old.session_id == "old-1" and old.turns[0].assistant == "early answer"


def test_export_parsers_follow_the_visible_branch():
    chatgpt = parsers.parse_export([{
        "id": "g", "title": "t", "current_node": "b", "mapping": {
            "a": {"parent": None, "message": {"author": {"role": "user"}, "content": {"parts": ["q"]}}},
            "x": {"parent": None, "message": {"author": {"role": "user"}, "content": {"parts": ["dropped"]}}},
            "b": {"parent": "a", "message": {"author": {"role": "assistant"}, "content": {"parts": ["ans"]}}}}}])
    assert [(t.user, t.assistant) for t in chatgpt[0].turns] == [("q", "ans")]


def test_redaction():
    text = redact("key sk-abcdefghijklmnopqrstuvwxyz0123 and postgres://u:hunter22@db/x password=supersecret1")
    assert "abcdefghijk" not in text and "hunter22" not in text and "supersecret1" not in text


# --- import -----------------------------------------------------------------------------

def test_collect_files_threads_under_their_projects(env):
    env.vault.sync()
    collected = collect(env.vault)
    by_key = {t.key: t for t in collected.threads}
    assert by_key["claude_code:c-111"].project == "webapp"
    assert by_key["claude_code:c-222"].session.agent == "claude-desktop"
    assert by_key["claude_code:c-222"].session.title == "Desktop login work"
    assert by_key["codex:0199-codex-aaaa"].project == "Shop"          # from the git remote
    assert by_key["codex:0199-codex-aaaa"].session.title == "Checkout tax fix"
    assert by_key["codex:0199-codex-zzzz"].session.turns[0].user == "Zipped request"   # .zst archive
    assert by_key["gemini_cli:g-1"].project == "webapp"               # sha256 project folder matched
    assert by_key["chatgpt:gpt-1"].project is None and by_key["claude_ai:cl-1"].session.turns
    assert [t.session.started_at for t in collected.threads] == sorted(t.session.started_at for t in collected.threads)


def test_import_is_ordered_complete_and_incremental(env):
    env.vault.sync()
    dry = run(import_history(env.memory, env.vault, apply=False))
    assert dry["threads_new"] == dry["threads"] and env.store.count() == 0
    assert dry["recovered_prompts"] == 1 and dry["by_project"]["webapp"]["threads"] == 3

    report = run(import_history(env.memory, env.vault, apply=True))
    # The compressed Codex rollout repeats the other rollout's compaction (as a fork does): stored once.
    assert report["turn_memories"] >= 8 and report["compaction_memories"] == 2
    thread = run(read_thread(env.memory, "claude_code:c-111"))
    entries = thread["entries"]
    kinds = [(e["kind"], e["turn_index"], e["part"]) for e in entries]
    assert kinds == [("conversation_turn", 0, 1), ("compaction_summary", None, 1),
                     ("conversation_turn", 1, 1), ("conversation_turn", 1, 2), ("conversation_turn", 2, 1)]
    assert [e["created_at"] for e in entries] == sorted(e["created_at"] for e in entries)
    assert entries[0]["created_at"] == pytest.approx(T0)
    assert "sk-abcdef" not in entries[0]["content"]
    assert "Claude Code · webapp (branch feat/login)" in entries[0]["content"]
    first = env.store.get(entries[0]["id"])
    assert first.project == "webapp" and first.scope == "project" and first.source_agent == "claude-code"
    assert first.metadata["directory"].endswith("webapp") and first.metadata["files"] == ["src/login.tsx"]
    summary_id = env.store.get_history_thread("claude_code:c-111")["summary_memory_id"]

    count = env.store.count()
    again = run(import_history(env.memory, env.vault, apply=True))
    assert again["turn_memories"] == 0 and again["recovered_prompts"] == 0 and env.store.count() == count

    # The live session continues: only the new turn is added and the summary is refreshed in place.
    transcript = env.home / ".claude" / "projects" / "-home-code-webapp" / "c-111.jsonl"
    with transcript.open("a") as handle:
        handle.write(json.dumps({"sessionId": "c-111", "type": "user", "timestamp": iso(200), "cwd": "x",
                                 "message": {"role": "user", "content": "ship it"}}) + "\n")
    env.vault.sync()
    grown = run(import_history(env.memory, env.vault, apply=True))
    assert grown["threads_grown"] == 1 and grown["turn_memories"] == 1
    assert env.store.count() == count + 1
    assert env.store.get_history_thread("claude_code:c-111")["summary_memory_id"] == summary_id
    assert "ship it" in env.store.get(summary_id).content


def test_recovered_prompt_keeps_its_original_time_and_project(env):
    env.vault.sync()
    run(import_history(env.memory, env.vault, apply=True))
    recovered = [r for r in env.store.get_all(limit=500) if (r.metadata or {}).get("kind") == "recovered_prompt"]
    assert len(recovered) == 1
    assert recovered[0].created_at == pytest.approx(T0 - 86400 * 90) and recovered[0].project == "webapp"


def test_since_limits_the_import(env):
    env.vault.sync()
    report = run(import_history(env.memory, env.vault, apply=False, since=T0 + 2500))
    assert set(report["by_agent"]) == {"chatgpt", "claude-ai"}


def test_service_status_and_auto_import_after_first_apply(env, monkeypatch):
    from muninn.history.service import HistoryService

    monkeypatch.delenv("MUNINN_HISTORY_AUTO_IMPORT", raising=False)
    service = HistoryService(env.memory, env.vault.root, home=env.home, interval_minutes=5)

    async def scenario():
        await service.sync()
        status = service.status()
        assert status["vault"]["totals"]["files"] >= 10 and status["sync_interval_minutes"] == 5
        assert any("cleanupPeriodDays" in w for w in status["warnings"])
        assert not service.auto_import_enabled()
        await service.run_import(apply=False)
        assert not service.auto_import_enabled()          # a dry run changes nothing
        await service.run_import(apply=True, providers=["codex"])
        assert not service.auto_import_enabled()          # a partial import does not opt in
        await service.run_import(apply=True)
        assert service.auto_import_enabled()
        monkeypatch.setenv("MUNINN_HISTORY_AUTO_IMPORT", "0")
        assert not service.auto_import_enabled()
        listing = service.status()["sources"]
        assert {s["provider"] for s in listing} == {"claude_code", "codex", "gemini_cli"}
        await service.stop()

    run(scenario())


def test_extra_homes_are_vaulted_without_collisions(env, tmp_path, monkeypatch):
    other = tmp_path / "windows-home"
    jsonl(other / ".claude" / "projects" / "-home-code-webapp" / "c-111.jsonl",
          claude_rows("/mnt/c/code/webapp", session="c-win"))
    monkeypatch.setenv("MUNINN_HISTORY_HOMES", str(other))
    env.vault.sync()
    transcripts = env.vault.files(provider="claude_code", kind="transcript")
    same_name = [f for f in transcripts if f.path.name == "c-111.jsonl.gz"]
    assert len(same_name) == 2 and len({f.path for f in same_name}) == 2
    assert "claude_code:c-win" in {t.key for t in collect(env.vault).threads}



# --- one project, several apps, one timeline ----------------------------------------------

def _claude(session, cwd, turns, entry="cli"):
    rows = []
    for at, user, reply in turns:
        base = {"sessionId": session, "cwd": cwd, "gitBranch": "main", "entrypoint": entry}
        rows.append({**base, "type": "user", "timestamp": iso(at), "message": {"role": "user", "content": user}})
        rows.append({**base, "type": "assistant", "timestamp": iso(at + 5),
                     "message": {"role": "assistant", "content": [{"type": "text", "text": reply}]}})
    return rows


def _codex(session, cwd, turns):
    rows = [{"timestamp": iso(turns[0][0]), "type": "session_meta",
             "payload": {"id": session, "cwd": cwd, "originator": "codex_desktop"}}]
    for at, user, reply in turns:
        rows.append({"timestamp": iso(at), "type": "event_msg", "payload": {"type": "user_message", "message": user}})
        rows.append({"timestamp": iso(at + 5), "type": "event_msg",
                     "payload": {"type": "agent_message", "message": reply}})
    return rows


@pytest.fixture
def relay(tmp_path, monkeypatch):
    """Codex starts, hands to Claude Code, then Codex continues its own thread; a resume copies turns."""
    for var in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "MUNINN_HISTORY_HOMES"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    repo = home / "code" / "Relay"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    codex_file = home / ".codex" / "sessions" / "2026" / "05" / "28" / "rollout-2026-05-28T00-00-00-cx-1.jsonl"
    jsonl(codex_file, _codex("cx-1", str(repo), [(0, "codex step 1", "done 1"), (100, "codex step 2", "done 2")]))
    claude_first = [(200, "claude step 3", "done 3"), (300, "claude step 4", "done 4")]
    jsonl(home / ".claude" / "projects" / "-relay" / "cc-1.jsonl", _claude("cc-1", str(repo), claude_first))
    # `claude --resume`: a new session file that starts with a copy of cc-1, then new work.
    jsonl(home / ".claude" / "projects" / "-relay" / "cc-2.jsonl",
          _claude("cc-2", str(repo), claude_first + [(600, "claude step 7", "done 7")]))
    vault = HistoryVault(tmp_path / "vault", home=home)
    memory = FakeMemory(SQLiteMetadataStore(tmp_path / "metadata.db"))
    yield SimpleNamespace(home=home, repo=repo, codex_file=codex_file, vault=vault, memory=memory,
                          store=memory._metadata)
    vault.close()


def test_resumed_sessions_do_not_duplicate_turns(relay):
    relay.vault.sync()
    dry = run(import_history(relay.memory, relay.vault, apply=False))
    assert dry["duplicate_turns"] == 2
    report = run(import_history(relay.memory, relay.vault, apply=True))
    assert report["duplicate_turns"] == 2 and report["turn_memories"] == 5   # 2 codex + 2 claude + 1 new
    resumed = relay.store.get_history_thread("claude_code:cc-2")
    assert resumed["continues_thread"] == "claude_code:cc-1" and resumed["duplicate_turns"] == 2
    texts = [e["content"] for e in run(read_thread(relay.memory, "claude_code:cc-2"))["entries"]]
    assert len(texts) == 1 and "claude step 7" in texts[0]
    again = run(import_history(relay.memory, relay.vault, apply=True))
    assert again["turn_memories"] == 0 and again["duplicate_turns"] == 0


def test_project_timeline_interleaves_apps_in_time_order(relay):
    from muninn.core import handoffs
    from muninn.history.importer import read_project_timeline

    relay.vault.sync()
    run(import_history(relay.memory, relay.vault, apply=True))
    run(handoffs.create_handoff(relay.memory, project="Relay", from_agent="codex", summary="Continue in Claude",
                                to_agent="claude-code"))
    handoff = relay.store.list_handoffs("global_user", "Relay", None)[0]
    relay.store._get_conn().execute("UPDATE agent_handoffs SET created_at = ? WHERE id = ?", (T0 + 150, handoff["id"]))
    relay.store._get_conn().commit()
    # Codex later continues its own thread after Claude's work (a handoff back).
    with relay.codex_file.open("a") as handle:
        for row in _codex("cx-1", str(relay.repo), [(400, "codex step 5", "done 5")])[1:]:
            handle.write(json.dumps(row) + "\n")
    os.utime(relay.codex_file, (T0 + 900, T0 + 900))
    relay.vault.sync()
    grown = run(import_history(relay.memory, relay.vault, apply=True))
    assert grown["turn_memories"] == 1 and grown["duplicate_turns"] == 0

    timeline = run(read_project_timeline(relay.memory, "Relay"))
    order = [(e["kind"], e["agent"], (e.get("content") or "").split("User: ")[-1].split("\n")[0])
             for e in timeline["entries"]]
    assert order == [
        ("conversation_turn", "codex", "codex step 1"),
        ("conversation_turn", "codex", "codex step 2"),
        ("handoff", "codex", order[2][2]),
        ("conversation_turn", "claude-code", "claude step 3"),
        ("conversation_turn", "claude-code", "claude step 4"),
        ("conversation_turn", "codex", "codex step 5"),
        ("conversation_turn", "claude-code", "claude step 7"),
    ]
    switches = [(e.get("agent_switch_from"), e["agent"]) for e in timeline["entries"] if e.get("agent_switch_from")]
    assert switches == [("codex", "claude-code"), ("claude-code", "codex"), ("codex", "claude-code")]
    page = run(read_project_timeline(relay.memory, "Relay", limit=2))
    assert page["next_offset"] == 2 and [e["kind"] for e in page["entries"]].count("conversation_turn") == 2


def test_windows_transcripts_name_their_project_on_any_os():
    from muninn.core.projects import project_for_directory

    repo = r"C:\Users\user\VSCodeProjects\ChatConverter"
    assert project_for_directory(repo) == "ChatConverter"
    assert project_for_directory(repo + r"\.claude\worktrees\fix-ui") == "ChatConverter"
    assert project_for_directory(r"C:\Users\user") is None and project_for_directory("C:\\") is None
    assert project_for_directory("/home/someone-else") is None
    assert project_for_directory("/Users/user/code/Proj") == "Proj"


def test_a_session_belongs_to_the_folder_it_was_started_in():
    """The agent may cd into subfolders; the launch folder (the one Claude Code files it under) names it."""
    from muninn.history import parsers

    root = r"C:\Users\user\VSCodeProjects\StoryApp"
    rows = (_claude("s-1", root, [(0, "start", "ok")])
            + _claude("s-1", root + r"\story-engine\recovery", [(60, "recover drafts", "done")]))
    session = parsers.parse_claude_code("\n".join(json.dumps(r) for r in rows))
    assert session.cwd == root and len(session.turns) == 2
