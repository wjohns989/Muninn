"""Tests for legacy discovery source scanning behavior."""

from __future__ import annotations

from pathlib import Path

from muninn.ingestion.discovery import _iter_paths, discover_legacy_sources


def test_custom_root_discovery_includes_sqlite_artifacts(tmp_path: Path):
    root = tmp_path / "archive"
    nested = root / "sessions"
    nested.mkdir(parents=True)
    jsonl_path = nested / "chat.jsonl"
    sqlite_path = nested / "state.vscdb"
    jsonl_path.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")
    sqlite_path.write_bytes(b"SQLite format 3\x00")

    discovered = discover_legacy_sources(
        home=tmp_path,
        roots=[str(root)],
        include_unsupported=False,
        max_results_per_provider=200,
    )
    custom_paths = {
        str(item["path"])
        for item in discovered
        if item["provider"] == "custom_root"
    }

    assert str(jsonl_path.resolve()) in custom_paths
    assert str(sqlite_path.resolve()) in custom_paths


def test_iter_paths_handles_repeated_base_segments(tmp_path: Path):
    base = tmp_path / "same"
    deep = base / "nested" / "same"
    deep.mkdir(parents=True)
    target = deep / "session.json"
    target.write_text("{}", encoding="utf-8")

    pattern = base / "**" / "same" / "*.json"
    found = {str(path.resolve()) for path in _iter_paths(pattern)}

    assert str(target.resolve()) in found
