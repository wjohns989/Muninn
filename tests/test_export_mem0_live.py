"""Mem0 history export must use final live state, never intermediate/deleted text."""

import sqlite3

import pytest

from scripts.export_mem0_live import live_records


def _history(path, events):
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE history (
            id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT, new_memory TEXT,
            event TEXT, created_at TEXT, updated_at TEXT, is_deleted INTEGER,
            actor_id TEXT, role TEXT)""")
        conn.executemany("INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?)", events)


def test_exports_only_latest_live_text_and_original_date(tmp_path):
    path = tmp_path / "history.db"
    _history(path, [
        ("e1", "kept", None, "old", "ADD", "2020-01-01", None, 0, "u", "user"),
        ("e2", "deleted", None, "private", "ADD", "2020-02-01", None, 0, "u", "user"),
        ("e3", "kept", "old", "new", "UPDATE", "2020-01-01", "2021-01-01", 0, "u", "user"),
        ("e4", "deleted", "private", None, "DELETE", None, None, 1, "u", "user"),
    ])
    records, report = live_records(path)
    assert report == {"events": 4, "identities": 2, "deleted": 1,
                      "live": 1, "distinct_content": 1}
    assert records[0]["id"] == "kept"
    assert records[0]["content"] == "new"
    assert records[0]["created_at"] == "2020-01-01"


def test_rejects_event_after_delete(tmp_path):
    path = tmp_path / "history.db"
    _history(path, [
        ("e1", "id", None, "old", "ADD", None, None, 0, None, None),
        ("e2", "id", "old", None, "DELETE", None, None, 1, None, None),
        ("e3", "id", "old", "revived", "UPDATE", None, None, 0, None, None),
    ])
    with pytest.raises(ValueError, match="event order"):
        live_records(path)
