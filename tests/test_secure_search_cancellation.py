"""Long encrypted searches can stop without publishing partial evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from muninn.history.blind_index import SearchCancelled, SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive

PASSPHRASE = "correct horse battery archive recovery"


def _index(tmp_path: Path) -> tuple[SecureHistoryBlindIndex, SecureHistoryArchive]:
    source = tmp_path / "chat.jsonl"
    source.write_text("archived constellation " * 80000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    return index, archive


def test_search_can_cancel_before_start(tmp_path: Path) -> None:
    index, _archive = _index(tmp_path)
    with pytest.raises(SearchCancelled):
        index.search("constellation", should_cancel=lambda: True)


def test_search_cancels_during_authenticated_candidate_and_discards_result(
    tmp_path: Path, monkeypatch,
) -> None:
    index, archive = _index(tmp_path)
    original = archive._verify_entry
    cancelled = False

    def interrupt_after_first_chunk(entry, *, collect, on_chunk=None):
        def accept(chunk: bytes) -> None:
            nonlocal cancelled
            if on_chunk:
                on_chunk(chunk)
            cancelled = True

        return original(entry, collect=collect, on_chunk=accept)

    monkeypatch.setattr(archive, "_verify_entry", interrupt_after_first_chunk)
    with pytest.raises(SearchCancelled):
        index.search("constellation", should_cancel=lambda: cancelled)


def test_fetch_capabilities_minted_only_after_full_candidate_verification(
    tmp_path: Path, monkeypatch,
) -> None:
    index, archive = _index(tmp_path)
    original = archive._verify_entry
    verified = False
    minted = False

    def verify(entry, *, collect, on_chunk=None):
        nonlocal verified
        result = original(entry, collect=collect, on_chunk=on_chunk)
        verified = True
        return result

    original_capability = index._capability

    def capability(entry, version, term):
        nonlocal minted
        assert verified
        minted = True
        return original_capability(entry, version, term)

    monkeypatch.setattr(archive, "_verify_entry", verify)
    monkeypatch.setattr(index, "_capability", capability)
    result = index.search("constellation")
    assert minted and len(result["matches"]) == 1
