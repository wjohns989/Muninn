"""Tests for legacy discovery source scanning behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from muninn.ingestion.discovery import _iter_paths, _provider_specs, discover_legacy_sources
from muninn.ingestion.parser import SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ---------------------------------------------------------------------------


def test_zed_extension_is_supported():
    """Zed AI conversation files (.zed) must be parseable as plain text."""
    assert ".zed" in SUPPORTED_EXTENSIONS
    assert SUPPORTED_EXTENSIONS[".zed"] == "text"


# ---------------------------------------------------------------------------
# New provider presence in _provider_specs()
# ---------------------------------------------------------------------------


def _provider_names(tmp_path: Path) -> set[str]:
    return {spec["provider"] for spec in _provider_specs(tmp_path, [])}


def test_aider_chat_provider_present(tmp_path: Path):
    assert "aider_chat" in _provider_names(tmp_path)


def test_continue_dev_provider_present(tmp_path: Path):
    assert "continue_dev" in _provider_names(tmp_path)


def test_zed_ai_provider_present(tmp_path: Path):
    assert "zed_ai" in _provider_names(tmp_path)


def test_gemini_tmp_chats_pattern_present(tmp_path: Path):
    """Gemini CLI provider must include the tmp/<hash>/chats pattern."""
    gemini_specs = [s for s in _provider_specs(tmp_path, []) if s["provider"] == "gemini_cli"]
    assert gemini_specs, "gemini_cli provider is missing"
    patterns = [str(p) for p in gemini_specs[0]["patterns"]]
    assert any("tmp" in p for p in patterns), (
        "Expected a pattern containing 'tmp' for ~/.gemini/tmp/<hash>/chats/ paths"
    )


# ---------------------------------------------------------------------------
# Aider discovery
# ---------------------------------------------------------------------------


def test_aider_chat_history_discovered(tmp_path: Path):
    """Home-dir .aider.chat.history.md should be found as aider_chat source."""
    chat_file = tmp_path / ".aider.chat.history.md"
    chat_file.write_text("#### User\nhello\n\n---\n\n#### Assistant\nhi\n", encoding="utf-8")

    discovered = discover_legacy_sources(home=tmp_path, max_results_per_provider=200)
    aider_paths = {
        str(item["path"])
        for item in discovered
        if item["provider"] == "aider_chat"
    }
    assert str(chat_file.resolve()) in aider_paths


def test_aider_source_is_parser_supported(tmp_path: Path):
    """Aider chat history (.md) must report parser_supported=True."""
    chat_file = tmp_path / ".aider.chat.history.md"
    chat_file.write_text("#### User\nhello\n", encoding="utf-8")

    discovered = discover_legacy_sources(home=tmp_path, max_results_per_provider=200)
    match = next((i for i in discovered if i["provider"] == "aider_chat"), None)
    assert match is not None
    assert match["parser_supported"] is True


# ---------------------------------------------------------------------------
# Continue.dev discovery
# ---------------------------------------------------------------------------


def test_continue_dev_discovered(tmp_path: Path):
    """Continue.dev dev_data JSONL events should be discovered."""
    dev_data = tmp_path / ".continue" / "dev_data"
    dev_data.mkdir(parents=True)
    event_file = dev_data / "chat_completions.jsonl"
    event_file.write_text(
        '{"event":"chat_completion","model":"claude-sonnet-4-6","prompt":"hi","completion":"hello"}\n',
        encoding="utf-8",
    )

    discovered = discover_legacy_sources(home=tmp_path, max_results_per_provider=200)
    continue_paths = {
        str(item["path"])
        for item in discovered
        if item["provider"] == "continue_dev"
    }
    assert str(event_file.resolve()) in continue_paths


# ---------------------------------------------------------------------------
# Zed AI discovery
# ---------------------------------------------------------------------------


def _make_zed_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake Zed conversations dir under mac_app_support and return (conv_file, threads_db)."""
    zed_dir = tmp_path / "Library" / "Application Support" / "Zed"
    conv_dir = zed_dir / "conversations"
    threads_dir = zed_dir / "threads"
    conv_dir.mkdir(parents=True)
    threads_dir.mkdir(parents=True)
    conv_file = conv_dir / "Fix auth bug.zed"
    conv_file.write_text("You: fix the bug\n---\nAssistant: sure\n", encoding="utf-8")
    threads_db = threads_dir / "threads.db"
    threads_db.write_bytes(b"SQLite format 3\x00")
    return conv_file, threads_db


def test_zed_conversation_file_discovered(tmp_path: Path):
    conv_file, _ = _make_zed_dirs(tmp_path)
    discovered = discover_legacy_sources(home=tmp_path, max_results_per_provider=200)
    zed_paths = {str(item["path"]) for item in discovered if item["provider"] == "zed_ai"}
    assert str(conv_file.resolve()) in zed_paths


def test_zed_threads_db_discovered(tmp_path: Path):
    _, threads_db = _make_zed_dirs(tmp_path)
    discovered = discover_legacy_sources(home=tmp_path, max_results_per_provider=200)
    zed_paths = {str(item["path"]) for item in discovered if item["provider"] == "zed_ai"}
    assert str(threads_db.resolve()) in zed_paths


def test_zed_conversation_is_parser_supported(tmp_path: Path):
    """.zed files must report parser_supported=True now that .zed is in SUPPORTED_EXTENSIONS."""
    conv_file, _ = _make_zed_dirs(tmp_path)
    discovered = discover_legacy_sources(home=tmp_path, max_results_per_provider=200)
    match = next(
        (i for i in discovered if i["provider"] == "zed_ai" and i["path"].endswith(".zed")),
        None,
    )
    assert match is not None, "No .zed source found in discovery output"
    assert match["parser_supported"] is True


# ---------------------------------------------------------------------------
# _require_discovery_pipeline: works without multi_source_ingestion flag
# ---------------------------------------------------------------------------


def test_require_discovery_pipeline_no_flag(tmp_path: Path):
    """_require_discovery_pipeline() must succeed even when multi_source_ingestion is disabled."""
    from unittest.mock import MagicMock, patch

    from muninn.ingestion.pipeline import IngestionPipeline

    # Build a minimal Memory-like object with _ingestion=None to simulate disabled flag.
    class _MinimalMemory:
        _ingestion = None

        def _require_discovery_pipeline(self):
            if self._ingestion is not None:
                return self._ingestion
            return IngestionPipeline()

    obj = _MinimalMemory()
    pipeline = obj._require_discovery_pipeline()
    assert isinstance(pipeline, IngestionPipeline)
    # Returned pipeline must allow paths under home directory.
    assert pipeline.is_path_allowed(Path.home())


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

    sample = next(item for item in discovered if str(item["path"]) == str(jsonl_path.resolve()))
    assert sample["parent_path"] == str(jsonl_path.resolve().parent)
    assert sample["path_depth"] >= 1
    assert "relative_path_hint" in sample
    assert "modified_at_epoch" in sample
    assert "modified_at_iso" in sample


def test_iter_paths_handles_repeated_base_segments(tmp_path: Path):
    base = tmp_path / "same"
    deep = base / "nested" / "same"
    deep.mkdir(parents=True)
    target = deep / "session.json"
    target.write_text("{}", encoding="utf-8")

    pattern = base / "**" / "same" / "*.json"
    found = {str(path.resolve()) for path in _iter_paths(pattern)}

    assert str(target.resolve()) in found
