# Multi-Source Ingestion Guide

Date: 2026-02-14

## Overview

Muninn Phase 3B adds feature-gated multi-source ingestion with fail-open behavior.

- Feature flag: `MUNINN_MULTI_SOURCE_INGESTION=1`
- REST endpoint: `POST /ingest`
- MCP tool: `ingest_sources`
- SDK methods: `MuninnClient.ingest_sources(...)` and `AsyncMuninnClient.ingest_sources(...)`

## Supported Source Types

- Native: `.txt`, `.md`, `.markdown`, `.json`, `.csv`, `.tsv`, `.html`, `.htm`
- Optional dependency-backed: `.pdf` (`pypdf`), `.docx` (`python-docx`)

## Safety + Reliability Model

Ingestion is fail-open:

- Missing sources are recorded and skipped.
- Oversized sources are skipped by policy.
- Parser failures are isolated per source and do not abort the batch.
- Add-time failures are isolated per chunk and reported in output.

Every ingested chunk includes provenance metadata:

- `source_path`, `source_name`, `source_type`
- `source_sha256`, `source_size_bytes`
- `chunk_index`, `chunk_count`, `char_start`, `char_end`

## Configuration

Environment variables:

- `MUNINN_MULTI_SOURCE_INGESTION=1`
- `MUNINN_INGESTION_MAX_FILE_BYTES` (default `5242880`)
- `MUNINN_INGESTION_CHUNK_SIZE_CHARS` (default `1200`)
- `MUNINN_INGESTION_CHUNK_OVERLAP_CHARS` (default `150`)
- `MUNINN_INGESTION_MIN_CHUNK_CHARS` (default `120`)

## REST Example

```bash
curl -X POST "http://localhost:42069/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["./docs", "./notes/project.md"],
    "recursive": true,
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
    project="muninn",
)
print(result["added_memories"], result["skipped_chunks"], result["failed_chunks"])
```

## Operational Notes

- For PDF and DOCX parsing support, install ingestion extras:
  - `pip install "muninn-mcp[ingestion]"`
- Use conservative size limits for untrusted corpora.
- Prefer directory-level ingestion with recursion and stable source roots to preserve deterministic paths/checksums.
