# Multi-Source Ingestion Guide

Date: 2026-02-14

## Overview

Muninn Phase 3B adds feature-gated multi-source ingestion with fail-open behavior.

- Feature flag: `MUNINN_MULTI_SOURCE_INGESTION=1`
- Browser UI: `GET /` (Control Center for discovery/import and project ingestion)
- REST endpoint: `POST /ingest`
- REST endpoints: `POST /ingest/legacy/discover`, `POST /ingest/legacy/import`
- MCP tool: `ingest_sources`
- MCP tools: `discover_legacy_sources`, `ingest_legacy_sources`
- MCP tools: `get_periodic_ingestion_status`, `run_periodic_ingestion`, `start_periodic_ingestion`, `stop_periodic_ingestion`
- SDK methods:
  - `MuninnClient.ingest_sources(...)`
  - `MuninnClient.discover_legacy_sources(...)`
  - `MuninnClient.ingest_legacy_sources(...)`
  - `MuninnClient.periodic_ingestion_status()`
  - `MuninnClient.run_periodic_ingestion()`
  - `MuninnClient.start_periodic_ingestion()`
  - `MuninnClient.stop_periodic_ingestion()`
  - `AsyncMuninnClient` parity for all three methods

## Supported Source Types

- Native: `.txt`, `.md`, `.markdown`, `.json`, `.jsonl`, `.ndjson`, `.csv`, `.tsv`, `.html`, `.htm`
- SQLite-backed chat/session stores: `.vscdb`, `.db`, `.sqlite`, `.sqlite3`
- Optional dependency-backed: `.pdf` (`pypdf`), `.docx` (`python-docx`)

## Safety + Reliability Model

Ingestion is fail-open:

- Missing sources are recorded and skipped.
- Oversized sources are skipped by policy.
- Sources outside configured allow-list roots are skipped (`outside_allowed_roots`).
- Parser failures are isolated per source and do not abort the batch.
- Add-time failures are isolated per chunk and reported in output.
- Optional chronological ordering supports timeline-preserving imports:
  - `none` (default deterministic path order)
  - `oldest_first` (by file modification time)
  - `newest_first` (by file modification time)

Every ingested chunk includes provenance metadata:

- `source_path`, `source_name`, `source_type`
- `source_sha256`, `source_size_bytes`
- `chunk_index`, `chunk_count`, `char_start`, `char_end`
- `source_mtime_epoch`, `source_mtime_iso`, `source_ingest_order`, `chronological_order`
- Legacy import adds contextual metadata when available:
  - `legacy_import`, `legacy_source_id`, `legacy_source_provider`
  - `legacy_source_category`, `legacy_source_confidence`, `legacy_source_notes`

## Configuration

Environment variables:

- `MUNINN_MULTI_SOURCE_INGESTION=1`
- `MUNINN_INGESTION_MAX_FILE_BYTES` (default `5242880`)
- `MUNINN_INGESTION_CHUNK_SIZE_CHARS` (default `1200`)
- `MUNINN_INGESTION_CHUNK_OVERLAP_CHARS` (default `150`)
- `MUNINN_INGESTION_MIN_CHUNK_CHARS` (default `120`)
- `MUNINN_INGESTION_ALLOWED_ROOTS` (optional, path-separated list of allowed source roots)

Periodic ingestion scheduler environment variables:

- `MUNINN_PERIODIC_INGESTION_ENABLED` (`1`/`0`, default `0`)
- `MUNINN_PERIODIC_INGESTION_RUN_ON_START` (`1`/`0`, default `0`)
- `MUNINN_PERIODIC_INGESTION_INTERVAL_SECONDS` (default `900`, clamped to minimum `5`)
- `MUNINN_PERIODIC_INGESTION_FAILURE_BACKOFF_MULTIPLIER` (default `2.0`, minimum `1.0`)
- `MUNINN_PERIODIC_INGESTION_MAX_BACKOFF_SECONDS` (default `3600`, minimum `5`)
- `MUNINN_PERIODIC_INGESTION_JITTER_RATIO` (default `0.1`, clamped to `0..1`)
- `MUNINN_PERIODIC_INGESTION_SOURCES` (path-separated source list, required for scheduler runs)
- `MUNINN_PERIODIC_INGESTION_RECURSIVE` (`1`/`0`, default `0`)
- `MUNINN_PERIODIC_INGESTION_CHRONOLOGICAL_ORDER` (`none|oldest_first|newest_first`)
- `MUNINN_PERIODIC_INGESTION_MODEL_PROFILE` (`low_latency|balanced|high_reasoning`, optional)
- `MUNINN_PERIODIC_INGESTION_SKIP_EXTRACTION` (`1`/`0`, default `0`; bypasses extraction pipeline for bulk-speed imports)
- `MUNINN_PERIODIC_INGESTION_EXTRACT_TIMEOUT_SECONDS` (optional per-memory extraction timeout, seconds)
- `MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SECONDS` (optional full periodic run timeout, seconds)
- `MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SKIP_WARMUP_RUNS` (optional integer, default `0`; skip run-timeout enforcement for first N periodic runs to absorb cold-start latency)
- `MUNINN_PERIODIC_INGESTION_METADATA_JSON` (optional JSON object)
- `MUNINN_PERIODIC_INGESTION_USER_ID`, `MUNINN_PERIODIC_INGESTION_NAMESPACE`, `MUNINN_PERIODIC_INGESTION_PROJECT`
- Optional periodic overrides:
  - `MUNINN_PERIODIC_INGESTION_MAX_FILE_SIZE_BYTES`
  - `MUNINN_PERIODIC_INGESTION_CHUNK_SIZE_CHARS`
  - `MUNINN_PERIODIC_INGESTION_CHUNK_OVERLAP_CHARS`
  - `MUNINN_PERIODIC_INGESTION_MIN_CHUNK_CHARS`

Runtime hard bounds enforced by pipeline:

- `max_file_size_bytes`: `1..104857600` (100 MB)
- `chunk_size_chars`: `1..20000`
- `chunk_overlap_chars`: `0..5000`, and must be `< chunk_size_chars`
- `min_chunk_chars`: `1..chunk_size_chars`

If `MUNINN_INGESTION_ALLOWED_ROOTS` is unset, Muninn defaults to a safe allow-list:

- user home directory,
- current working directory,
- system temp directory.

## REST Example

```bash
curl -X POST "http://localhost:42069/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["./docs", "./notes/project.md"],
    "recursive": true,
    "chronological_order": "oldest_first",
    "project": "muninn",
    "namespace": "global"
  }'
```

## SDK Example

```python
from muninn import Memory

client = Memory(base_url="http://localhost:42069")
result = client.ingest_sources(
    sources=["./docs", "./notes/project.md"],
    recursive=True,
    chronological_order="oldest_first",
    project="muninn",
)
print(result["added_memories"], result["skipped_chunks"], result["failed_chunks"])
```

## Legacy Discovery + Import (REST)

```bash
curl -X POST "http://localhost:42069/ingest/legacy/discover" \
  -H "Content-Type: application/json" \
  -d '{
    "providers": ["codex_cli", "claude_code", "serena_memory"],
    "max_results_per_provider": 100
  }'
```

```bash
curl -X POST "http://localhost:42069/ingest/legacy/import" \
  -H "Content-Type: application/json" \
  -d '{
    "selected_source_ids": ["src_123", "src_456"],
    "chronological_order": "oldest_first",
    "project": "muninn",
    "namespace": "global"
  }'
```

## Legacy Discovery + Import (SDK)

```python
from muninn import Memory

client = Memory(base_url="http://localhost:42069")

catalog = client.discover_legacy_sources(
    providers=["codex_cli", "claude_code", "serena_memory"]
)
selected = [
    source["source_id"]
    for source in catalog["sources"]
    if source["parser_supported"]
]
result = client.ingest_legacy_sources(
    selected_source_ids=selected,
    chronological_order="oldest_first",
    project="muninn",
)
print(result["added_memories"], result["selected_supported_sources"])
```

## Operational Notes

- For PDF and DOCX parsing support, install ingestion extras:
  - `pip install "muninn-mcp[ingestion]"`
- Use conservative size limits for untrusted corpora.
- Prefer directory-level ingestion with recursion and stable source roots to preserve deterministic paths/checksums.
- For SQLite ingestion, only read-only bounded scans are performed (table allowlisting + row limits) to reduce blast radius.
- Periodic control-plane endpoints are token-protected:
  - `GET /ingest/periodic/status`
  - `POST /ingest/periodic/run`
  - `POST /ingest/periodic/start`
  - `POST /ingest/periodic/stop`
- Scheduler status includes reliability diagnostics:
  - `consecutive_failures`
  - `last_scheduled_sleep_seconds`
  - `last_run_elapsed_seconds`
  - `last_run_timeout_enforced`
  - configured backoff/jitter parameters
- If `MUNINN_PERIODIC_INGESTION_MODEL_PROFILE` is set, periodic runs inject
  `metadata.operator_model_profile` automatically so extraction follows the
  selected runtime profile.
- If `MUNINN_PERIODIC_INGESTION_SKIP_EXTRACTION=1`, periodic runs inject
  `metadata.muninn_skip_extraction=true`, bypassing extraction for higher
  ingestion throughput in bulk/log replay scenarios.
- If `MUNINN_PERIODIC_INGESTION_EXTRACT_TIMEOUT_SECONDS` is set, periodic runs
  inject `metadata.muninn_extraction_timeout_seconds` so slow extraction falls
  back to empty extraction instead of stalling ingestion.
- If `MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SECONDS` is set, each periodic run
  is bounded with `asyncio.wait_for`; timeout events are counted as scheduler
  failures and trigger backoff.
- If `MUNINN_PERIODIC_INGESTION_RUN_TIMEOUT_SKIP_WARMUP_RUNS` is set, timeout
  enforcement is deferred for the first `N` periodic runs to avoid false
  timeout failures during model/provider cold-start.
