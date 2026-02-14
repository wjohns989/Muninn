"""Tests for ingestion parser adapters and chat contextualization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from muninn.ingestion.parser import infer_source_type, parse_source


def test_jsonl_chat_contextualization(tmp_path: Path):
    payload = (
        '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hello world"}]}}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}\n'
    )
    source = tmp_path / "session.jsonl"
    source.write_text(payload, encoding="utf-8")

    parsed = parse_source(source, "jsonl")

    assert "[user] hello world" in parsed
    assert "[assistant] hi there" in parsed


def test_sqlite_chat_contextualization(tmp_path: Path):
    source = tmp_path / "state.vscdb"
    conn = sqlite3.connect(str(source))
    try:
        conn.execute("CREATE TABLE chatMessages (role TEXT, content TEXT)")
        conn.execute("INSERT INTO chatMessages(role, content) VALUES ('user', 'Need build status')")
        conn.execute("INSERT INTO chatMessages(role, content) VALUES ('assistant', 'Build is green')")
        conn.commit()
    finally:
        conn.close()

    parsed = parse_source(source, "sqlite")

    assert "# table: chatMessages" in parsed
    assert "[user] Need build status" in parsed
    assert "[assistant] Build is green" in parsed


def test_vscdb_extension_maps_to_sqlite(tmp_path: Path):
    source = tmp_path / "state.vscdb"
    source.write_bytes(b"")

    assert infer_source_type(source) == "sqlite"
