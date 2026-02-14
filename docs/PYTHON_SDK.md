# Muninn Python SDK Guide

Date: 2026-02-14

## Overview

Muninn now ships a first-party Python SDK for direct REST integration:

- Sync client: `MuninnClient`
- Async client: `AsyncMuninnClient`
- Mem0-style aliases: `Memory`, `AsyncMemory`

All clients target `MUNINN_SERVER_URL` when set, otherwise `http://localhost:42069`.

## Installation

No extra package is required when using `muninn-mcp`; SDK dependencies are already included.

## Quickstart (sync)

```python
from muninn import Memory

client = Memory()

add_result = client.add(
    content="The release gate must fail on significant regressions.",
    user_id="global_user",
    metadata={"project": "muninn"},
)

search_result = client.search(
    "What is the release gate rule?",
    user_id="global_user",
    explain=True,
)

print(add_result["id"])
print(search_result[0]["memory"]["content"])
```

## Quickstart (async)

```python
import asyncio
from muninn import AsyncMemory

async def main():
    async with AsyncMemory() as client:
        await client.set_project_goal(
            project="muninn",
            goal_statement="Deliver Phase 3 with production-grade verification",
            constraints=["local-first", "backward-compatible"],
        )
        goal = await client.get_project_goal(project="muninn")
        print(goal)

asyncio.run(main())
```

## Supported Methods

The sync and async clients expose equivalent methods:

- `health()`
- `add(...)`
- `search(...)`
- `set_project_goal(...)`
- `get_project_goal(...)`
- `export_handoff(...)`
- `import_handoff(...)`
- `record_retrieval_feedback(...)`
- `ingest_sources(...)`
- `discover_legacy_sources(...)`
- `ingest_legacy_sources(...)`
- `get_all(...)`
- `update(...)`
- `delete(...)`
- `delete_all(...)`
- `delete_batch(...)`
- `get_graph(...)`
- `handover(...)`
- `federated_search(...)`
- `run_consolidation()`
- `consolidation_status()`

For ingestion methods, use `chronological_order` (`none`, `oldest_first`, `newest_first`) when you need timeline-preserving imports.

## Error Handling

The SDK raises typed exceptions:

- `MuninnConnectionError`: request could not reach server (network/down host).
- `MuninnAPIError`: server returned an HTTP error or `{"success": false}` payload.

```python
from muninn.sdk import Memory, MuninnConnectionError, MuninnAPIError

client = Memory()
try:
    client.search("release policy")
except MuninnConnectionError:
    # server unavailable
    ...
except MuninnAPIError as err:
    # server responded with error status/payload
    print(err.status_code, err.path)
```

## Context Manager Ergonomics

Use context managers to guarantee transport cleanup:

```python
from muninn.sdk import MuninnClient

with MuninnClient() as client:
    status = client.health()
    print(status["status"])
```

```python
from muninn.sdk import AsyncMuninnClient

async with AsyncMuninnClient() as client:
    result = await client.health()
```
