"""Synthetic-only proof: ordinary memory is not a credential transport/store."""
import asyncio
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from muninn.core.credential_boundary import (
    CredentialMemoryError, project_credentials, require_credential_free,
)
from muninn.core.ingestion_manager import IngestionManager
from muninn.core.memory import MuninnMemory
from muninn.core.types import AddMemoryRequest, ExtractionResult, MemoryRecord, MemoryType, Provenance
from muninn.retrieval.hybrid import HybridRetriever
from muninn.store.sqlite_metadata import SQLiteMetadataStore


FAKE = "OnlySynthetic!$c;D42"


@pytest.mark.parametrize("text", [
    f"WiFi Password: {FAKE}", f"PASSWORD='{FAKE}'", f'API_KEY="{FAKE}"',
    f"api key = `{FAKE}`", f'PASSWORD="first line\n{FAKE}"',
    f"PASSWORD=\n{FAKE}", f"WiFi Password: prefix with spaces {FAKE}",
    f'PASSWORD="escaped\\\"quote {FAKE}"', f"postgres://synthetic-user:{FAKE}@localhost/database",
    f'PASSWORD="""first line\n{FAKE}"""', f'PASSWORD="{FAKE}\nnot closed',
    "Authorization: Bearer only-synthetic-bearer", "Cookie: a=synthetic; b=also-synthetic",
    "Use sk-onlysynthetictestmaterial here.",
    "-----BEGIN PRIVATE KEY-----\nsynthetic\n-----END PRIVATE KEY-----",
])
def test_complete_credential_values_are_not_projected(text):
    projected = project_credentials(text)
    assert "REDACTED" in projected
    assert FAKE not in projected
    assert "only-synthetic-bearer" not in projected
    assert "also-synthetic" not in projected
    assert "sk-onlysynthetictestmaterial" not in projected
    with pytest.raises(CredentialMemoryError) as error:
        require_credential_free(text)
    assert FAKE not in str(error.value)
    assert "vault" in str(error.value).lower()
    assert project_credentials(projected) == projected


def test_searchable_location_and_presence_metadata_remain_usable():
    metadata = {"service": "OpenRouter", "credential_name": "OPENROUTER_API_KEY",
                "source_path": "C:\\project\\.env", "status": "configured",
                "token": "issued_at", "sha256": "a" * 64,
                "api_key": {"status": "present", "source_path": ".env"}}
    assert project_credentials(metadata) == metadata
    require_credential_free(metadata)


def test_nested_credential_values_and_dictionary_keys_are_guarded():
    payload = {"safe": ["unchanged", {"PASSWORD": FAKE}],
               "sk-onlysyntheticdictionarykey": {"source": ".env"}}
    projected = project_credentials(payload)
    assert FAKE not in json.dumps(projected)
    assert "sk-onlysyntheticdictionarykey" not in json.dumps(projected)
    assert payload["safe"][1]["PASSWORD"] == FAKE  # no mutation
    with pytest.raises(CredentialMemoryError):
        require_credential_free(payload)


def test_ingestion_rejects_before_telemetry_or_models():
    memory = SimpleNamespace(_otel=MagicMock(), config=SimpleNamespace(),
                             _embed=AsyncMock(), _extract_with_profile=AsyncMock())
    manager = IngestionManager(memory)
    with pytest.raises(CredentialMemoryError):
        asyncio.run(manager.process_add(f"PASSWORD={FAKE}", user_id="global_user", agent_id=None,
                                       metadata=None, namespace="global", memory_type=MemoryType.EPISODIC,
                                       provenance=Provenance.USER_EXPLICIT))
    memory._otel.add_event.assert_not_called()
    memory._embed.assert_not_called()
    memory._extract_with_profile.assert_not_called()


@pytest.mark.parametrize("change", [
    {"data": f"PASSWORD={FAKE}"}, {"metadata_patch": {"password": FAKE}},
    {"metadata": {"nested": [{"api_key": FAKE}]}},
])
def test_update_rejects_before_read_or_model_calls(change):
    memory = object.__new__(MuninnMemory)
    memory._initialized = True
    memory._metadata = MagicMock()
    with pytest.raises(CredentialMemoryError):
        asyncio.run(memory.update("synthetic-id", **change))
    memory._metadata.get.assert_not_called()


def test_sqlite_projects_legacy_records_without_rewriting_them(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    original = f"Device settings\nWiFi Password: {FAKE}\nPort: 8000"
    record = MemoryRecord(content="safe initial", metadata={"user_id": "global_user"})
    store.add(record)
    conn = store._get_conn()
    conn.execute("UPDATE memories SET content=? WHERE id=?", (original, record.id))
    conn.commit()  # emulate a pre-boundary record, synthetic only
    views = [store.get(record.id), *store.get_all(), *store.get_by_ids([record.id])]
    assert views
    assert all(FAKE not in r.content and "Port: 8000" in r.content for r in views)
    with pytest.raises(CredentialMemoryError):
        store.update(record.id, content=views[0].content, metadata=views[0].metadata)
    assert store.update(record.id, importance=0.8)
    assert conn.execute("SELECT content FROM memories WHERE id=?", (record.id,)).fetchone()[0] == original
    store.close()


def test_sqlite_direct_add_and_update_cannot_bypass_boundary(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    with pytest.raises(CredentialMemoryError):
        store.add(MemoryRecord(content=f"PASSWORD={FAKE}"))
    record = MemoryRecord(content="safe")
    store.add(record)
    with pytest.raises(CredentialMemoryError):
        store.update(record.id, metadata={"password": FAKE})
    assert store.get(record.id).content == "safe"
    assert store.count() == 1
    store.close()


def test_legacy_projection_reaches_real_rerank_get_and_briefing(tmp_path):
    from muninn.core.handoffs import project_context
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    record = MemoryRecord(content="safe", project="synthetic-project", scope="global",
                          metadata={"user_id": "global_user"})
    store.add(record)
    conn = store._get_conn()
    original = f"WiFi Password: {FAKE}\nCamera settings"
    conn.execute("UPDATE memories SET content=? WHERE id=?", (original, record.id))
    conn.commit()
    retriever = object.__new__(HybridRetriever)
    reranker = MagicMock()
    def rerank(**kwargs):
        assert all(FAKE not in text for text in kwargs["documents"])
        return [(record.id, 0.8)]
    reranker.rerank.side_effect = rerank
    retriever.reranker = reranker
    rows = store.get_by_ids([record.id])
    results = retriever._rerank_candidates("camera", [(record.id, 0.7)], {r.id: r for r in rows}, 1)
    assert FAKE not in results[0].memory.content
    memory = object.__new__(MuninnMemory)
    memory._initialized = True
    memory._metadata = store
    assert FAKE not in json.dumps(asyncio.run(memory.get(record.id)))
    context = asyncio.run(project_context(memory, project="synthetic-project"))
    assert FAKE not in json.dumps(context)
    assert "Camera settings" in json.dumps(context)
    conn.execute("UPDATE memories SET archived=1 WHERE id=?", (record.id,))
    conn.commit()
    with pytest.raises(CredentialMemoryError):
        asyncio.run(memory.restore(record.id))
    assert conn.execute("SELECT content FROM memories WHERE id=?", (record.id,)).fetchone()[0] == original
    store.close()


def test_legacy_elo_preserves_original_metadata(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    record = MemoryRecord(content="safe")
    store.add(record)
    conn = store._get_conn()
    conn.execute("UPDATE memories SET metadata=? WHERE id=?", (json.dumps({"password": FAKE}), record.id))
    conn.commit()
    assert store.update_elo_rating(record.id, 1234)
    raw = json.loads(conn.execute("SELECT metadata FROM memories WHERE id=?", (record.id,)).fetchone()[0])
    assert raw == {"password": FAKE, "elo_rating": 1234.0}
    assert FAKE not in json.dumps(store.get(record.id).metadata)
    store.close()


def test_profiles_goals_handoffs_guard_writes_and_preserve_originals(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    with pytest.raises(CredentialMemoryError):
        store.set_user_profile(user_id="synthetic-user", profile={"password": FAKE})
    goal_args = dict(user_id="synthetic-user", namespace="global", project="p", constraints=[])
    with pytest.raises(CredentialMemoryError):
        store.set_project_goal(**goal_args, goal_statement=f"PASSWORD={FAKE}")
    handoff = dict(id="synthetic-handoff", user_id="synthetic-user", project="p", title="safe",
                   summary="safe", details={}, from_agent="synthetic-agent", created_at=1)
    store.add_handoff(handoff)
    with pytest.raises(CredentialMemoryError):
        store.transition_handoff(handoff["id"], "completed", agent="synthetic-agent", note=f"PASSWORD={FAKE}")
    store.set_user_profile(user_id="synthetic-user", profile={"safe": "yes"})
    store.set_project_goal(**goal_args, goal_statement="safe")
    conn = store._get_conn()
    conn.execute("UPDATE user_profiles SET profile_json=?", (json.dumps({"password": FAKE}),))
    conn.execute("UPDATE project_goals SET goal_statement=?", (f"PASSWORD={FAKE}",))
    conn.execute("UPDATE agent_handoffs SET details_json=?", (json.dumps({"password": FAKE}),))
    conn.commit()
    views = [store.get_user_profile(user_id="synthetic-user"), store.get_handoff(handoff["id"]),
             store.get_project_goal(user_id="synthetic-user", namespace="global", project="p")]
    assert FAKE not in json.dumps(views)
    with pytest.raises(CredentialMemoryError):
        store.set_user_profile(user_id="synthetic-user", profile=views[0]["profile"])
    with pytest.raises(CredentialMemoryError):
        store.set_project_goal(**goal_args, goal_statement=views[2]["goal_statement"])
    assert FAKE in conn.execute("SELECT profile_json FROM user_profiles").fetchone()[0]
    assert FAKE in conn.execute("SELECT goal_statement FROM project_goals").fetchone()[0]
    assert FAKE in conn.execute("SELECT details_json FROM agent_handoffs").fetchone()[0]
    store.close()


def test_actual_http_add_handler_screens_before_chunking_without_server_import():
    # Actual handler, without importing server/env/log/service setup.
    from fastapi import HTTPException
    source = Path(__file__).resolve().parents[1] / "server.py"
    handler = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                   if isinstance(n, ast.AsyncFunctionDef) and n.name == "add_memory_endpoint")
    handler.decorator_list = []
    memory = SimpleNamespace(add=AsyncMock())
    namespace = dict(memory=memory, logger=MagicMock(), asyncio=asyncio, HTTPException=HTTPException,
                     CredentialMemoryError=CredentialMemoryError, require_credential_free=require_credential_free,
                     AddMemoryRequest=AddMemoryRequest, Provenance=Provenance)
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(HTTPException) as error:
        asyncio.run(namespace["add_memory_endpoint"](AddMemoryRequest(content="normal. " * 150 + f"PASSWORD={FAKE}")))
    assert error.value.status_code == 400
    assert FAKE not in error.value.detail
    memory.add.assert_not_called()


def test_model_generated_credentials_are_rejected_before_embedding():
    memory = SimpleNamespace(_otel=MagicMock(), config=SimpleNamespace(extraction=SimpleNamespace(
        runtime_model_profile="balanced", model_profile="balanced")), _embed=AsyncMock(),
        _extract_with_profile=AsyncMock(return_value=ExtractionResult(summary=f"PASSWORD={FAKE}")),
        _extract_entity_names=lambda extraction: [])
    with pytest.raises(CredentialMemoryError):
        asyncio.run(IngestionManager(memory).process_add("safe text", user_id="global_user", agent_id=None,
                                                       metadata=None, namespace="global", memory_type=MemoryType.EPISODIC,
                                                       provenance=Provenance.USER_EXPLICIT))
    memory._embed.assert_not_called()


def test_consolidation_skips_legacy_projections_but_advances_cursor():
    from muninn.consolidation.daemon import ConsolidationDaemon
    protected = MemoryRecord(id="a", content="safe projection")
    protected._credential_projection = True
    ordinary = MemoryRecord(id="b", content="normal")
    daemon = object.__new__(ConsolidationDaemon)
    daemon._dry_run = False
    daemon._batch_size = 1
    daemon.metadata = MagicMock()
    daemon.metadata.get_meta.return_value = ""
    daemon.metadata.get_for_consolidation.side_effect = [[protected], [ordinary]]
    assert daemon._next_batch("decay") == []
    daemon.metadata.set_meta.assert_called_with("consolidation_cursor:decay", "a")
    assert daemon._next_batch("decay") == [ordinary]


def test_guarded_rewrite_is_atomic_even_when_legacy_connection_commits(tmp_path, monkeypatch):
    import threading
    import sqlite3
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    record = MemoryRecord(content="safe")
    store.add(record)
    conn = store._get_conn()
    original = f"PASSWORD={FAKE}"
    conn.execute("UPDATE memories SET content=? WHERE id=?", (original, record.id))
    conn.commit()
    reading, release = threading.Event(), threading.Event()
    hydrate = store._row_to_record
    def pause(row):
        reading.set()
        assert release.wait(3)
        return hydrate(row)
    monkeypatch.setattr(store, "_row_to_record", pause)
    errors = []
    def rewrite():
        try:
            store.update(record.id, content="a projection")
        except CredentialMemoryError:
            errors.append("rejected")
    thread = threading.Thread(target=rewrite)
    thread.start()
    try:
        assert reading.wait(3)
        conn.commit()  # must not release the separate screening transaction
        with sqlite3.connect(str(store.db_path), timeout=0.01) as contender:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("UPDATE memories SET content='concurrent rewrite' WHERE id=?", (record.id,))
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and errors == ["rejected"]
    assert conn.execute("SELECT content FROM memories WHERE id=?", (record.id,)).fetchone()[0] == original
    store.close()


def test_fixture_detects_the_old_raw_hydration_defect(tmp_path, monkeypatch):
    from muninn.store import sqlite_metadata
    store = SQLiteMetadataStore(tmp_path / "ordinary.db")
    record = MemoryRecord(content="safe")
    store.add(record)
    conn = store._get_conn()
    conn.execute("UPDATE memories SET content=? WHERE id=?", (f"PASSWORD={FAKE}", record.id))
    conn.commit()
    # Targeted old-defect mutation, on synthetic temporary data only.
    monkeypatch.setattr(sqlite_metadata, "project_credentials", lambda value: value)
    assert FAKE in store.get(record.id).content
    store.close()
