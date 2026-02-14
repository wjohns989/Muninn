"""Tests for multi-source ingestion parsing and fail-open behavior."""

from pathlib import Path

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

    pipeline = IngestionPipeline(max_file_size_bytes=128, chunk_size_chars=100, chunk_overlap_chars=10)
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
