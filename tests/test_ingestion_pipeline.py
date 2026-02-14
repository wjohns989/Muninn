"""Tests for multi-source ingestion parsing and fail-open behavior."""

from pathlib import Path

import pytest

from muninn.ingestion.pipeline import IngestionPipeline


def test_ingestion_pipeline_processes_supported_files(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("alpha\n" + ("beta " * 300), encoding="utf-8")

    json_file = tmp_path / "meta.json"
    json_file.write_text('{"k":"v","n":1}', encoding="utf-8")

    pipeline = IngestionPipeline(chunk_size_chars=200, chunk_overlap_chars=20, min_chunk_chars=20)
    report = pipeline.ingest([str(txt_file), str(json_file)])

    assert report.total_sources == 2
    assert report.processed_sources == 2
    assert report.total_chunks >= 2
    for src in report.source_results:
        assert src.status == "processed"
        assert src.chunks
        for chunk in src.chunks:
            assert chunk.metadata["source_path"] == src.source_path
            assert chunk.metadata["chunk_count"] == chunk.chunk_count
            assert len(chunk.source_sha256) == 64


def test_ingestion_pipeline_fail_open_for_missing_and_oversized(tmp_path):
    missing = tmp_path / "missing.txt"
    big = tmp_path / "big.txt"
    big.write_text("x" * 1024, encoding="utf-8")

    pipeline = IngestionPipeline(
        max_file_size_bytes=128,
        chunk_size_chars=100,
        chunk_overlap_chars=10,
        min_chunk_chars=10,
    )
    report = pipeline.ingest([str(missing), str(big)])

    assert report.total_sources == 2
    assert report.processed_sources == 0
    assert report.skipped_sources == 2
    reasons = {entry.skipped_reason for entry in report.source_results}
    assert "source_not_found" in reasons
    assert "file_too_large" in reasons


def test_ingestion_pipeline_directory_expansion_respects_recursive(tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)

    (root / "a.txt").write_text("A", encoding="utf-8")
    (nested / "b.txt").write_text("B", encoding="utf-8")

    pipeline = IngestionPipeline(chunk_size_chars=10, chunk_overlap_chars=0, min_chunk_chars=1)

    report_non_recursive = pipeline.ingest([str(root)], recursive=False)
    assert report_non_recursive.total_sources == 1

    report_recursive = pipeline.ingest([str(root)], recursive=True)
    assert report_recursive.total_sources == 2


def test_ingestion_pipeline_chronological_order_and_metadata(tmp_path):
    older = tmp_path / "older.txt"
    newer = tmp_path / "newer.txt"
    older.write_text("old content", encoding="utf-8")
    newer.write_text("new content", encoding="utf-8")

    # Ensure deterministic mtime ordering across filesystems.
    older_epoch = 1_700_000_000
    newer_epoch = older_epoch + 100
    older.touch()
    newer.touch()
    import os
    os.utime(older, (older_epoch, older_epoch))
    os.utime(newer, (newer_epoch, newer_epoch))

    pipeline = IngestionPipeline(chunk_size_chars=64, chunk_overlap_chars=0, min_chunk_chars=1)
    report_oldest = pipeline.ingest(
        [str(newer), str(older)],
        chronological_order="oldest_first",
    )
    report_newest = pipeline.ingest(
        [str(older), str(newer)],
        chronological_order="newest_first",
    )

    assert report_oldest.source_results[0].source_path.endswith("older.txt")
    assert report_newest.source_results[0].source_path.endswith("newer.txt")

    first_chunk = report_oldest.source_results[0].chunks[0]
    assert first_chunk.metadata["source_mtime_epoch"] == older_epoch
    assert first_chunk.metadata["source_ingest_order"] == 0
    assert first_chunk.metadata["chronological_order"] == "oldest_first"


def test_ingestion_pipeline_blocks_sources_outside_allowed_roots(tmp_path, tmp_path_factory):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    blocked_root = tmp_path_factory.mktemp("blocked")
    blocked_file = blocked_root / "secrets.txt"
    blocked_file.write_text("do not ingest", encoding="utf-8")

    pipeline = IngestionPipeline(
        allowed_roots=[str(allowed_root)],
        chunk_size_chars=64,
        chunk_overlap_chars=0,
        min_chunk_chars=1,
    )
    report = pipeline.ingest([str(blocked_file)])

    assert report.total_sources == 1
    assert report.processed_sources == 0
    assert report.skipped_sources == 1
    assert report.source_results[0].skipped_reason == "outside_allowed_roots"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size_chars": 0}, "chunk_size_chars"),
        ({"chunk_size_chars": 100, "chunk_overlap_chars": 100}, "chunk_overlap_chars"),
        ({"chunk_size_chars": 100, "chunk_overlap_chars": 10, "min_chunk_chars": 101}, "min_chunk_chars"),
    ],
)
def test_ingestion_pipeline_rejects_invalid_runtime_limits(tmp_path, kwargs, message):
    source = tmp_path / "a.txt"
    source.write_text("hello world", encoding="utf-8")
    pipeline = IngestionPipeline()

    with pytest.raises(ValueError, match=message):
        pipeline.ingest([str(source)], **kwargs)
