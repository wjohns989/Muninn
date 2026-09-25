"""
Muninn Maintenance Operations
-----------------------------
Store-level operations that run inside the one process that owns the stores:

- ``reindex``: rebuild derived indexes (vectors, BM25) from SQLite, the source
  of truth. Used after an embedding-model change, a lost or corrupt vector
  store, or when an older ``metadata.db`` is placed into a fresh install.
- ``import_memories``: add memories exported from another system (including
  the pre-3.0 Mem0-based Muninn) while keeping their original timestamps.

Both default to dry runs that only report what would change.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from muninn.core.types import MemoryRecord, Provenance

if TYPE_CHECKING:  # pragma: no cover
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.Maintenance")

_CONTENT_KEYS = ("content", "memory", "text", "data")
_CREATED_KEYS = ("created_at", "createdAt", "timestamp")
# Keys consumed by the importer; everything else in `metadata` is preserved.
_RESERVED_METADATA = {"user_id", "provenance"}


def vector_payload(record: MemoryRecord) -> Dict[str, Any]:
    """The payload MuninnMemory.add writes with every vector."""
    return {
        "content": record.content[:500],
        "memory_type": record.memory_type.value,
        "namespace": record.namespace,
        "importance": record.importance,
        "user_id": (record.metadata or {}).get("user_id", "global_user"),
        "project": record.project,
        "branch": record.branch,
        "scope": record.scope,
        "media_type": getattr(record.media_type, "value", record.media_type),
    }


def _live_pages(memory: "MuninnMemory", page_size: int) -> Iterable[List[MemoryRecord]]:
    cursor = ""
    while True:
        page = memory._metadata.get_for_consolidation(limit=page_size, archived=False, after_id=cursor)
        if page:
            yield page
        if len(page) < page_size:
            return
        cursor = page[-1].id


async def reindex(
    memory: "MuninnMemory",
    *,
    vectors: bool = True,
    bm25: bool = True,
    recreate_vectors: bool = False,
    dry_run: bool = True,
    page_size: int = 200,
) -> Dict[str, Any]:
    """Rebuild vectors and/or BM25 for every live memory from the metadata store."""
    memory._check_initialized()
    live = sum(len(page) for page in _live_pages(memory, page_size))
    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "live_memories": live,
        "vector_points_before": memory._vectors.count(),
        "vectors": vectors,
        "bm25": bm25,
        "recreate_vectors": recreate_vectors,
    }
    if dry_run:
        return report

    if bm25:
        await memory._rebuild_bm25()
        report["bm25_documents"] = memory._bm25.size

    if vectors:
        if recreate_vectors:
            # Drops every point and recreates the collection at the configured
            # dimensions (needed when the embedding model's size changes).
            await asyncio.to_thread(memory._vectors.delete_all)
        embedded = failed = 0
        for page in _live_pages(memory, page_size):
            for record in page:
                try:
                    embedding = await memory._embed(record.content)
                    await asyncio.to_thread(
                        memory._vectors.upsert, record.id, embedding, vector_payload(record)
                    )
                    embedded += 1
                except Exception as exc:  # keep going; report the count
                    failed += 1
                    logger.warning("Reindex failed for %s: %s", record.id, type(exc).__name__)
        report.update({"vectors_embedded": embedded, "vectors_failed": failed})
        report["vector_points_after"] = memory._vectors.count()

    logger.info("Reindex complete: %s", report)
    return report


def _parse_timestamp(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Accept milliseconds as well as seconds.
        return float(value) / 1000.0 if value > 1e11 else float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def normalize_legacy_record(
    raw: Dict[str, Any],
    *,
    user_id: str = "global_user",
    namespace: str = "global",
    source: str = "legacy",
) -> Optional[Dict[str, Any]]:
    """Map one exported memory (Muninn, Mem0 or similar JSON) to add() arguments.

    Returns None when the record has no text. The original user id and id are
    kept in metadata; memories import under ``user_id`` so default searches find them.
    """
    content = next((raw[k] for k in _CONTENT_KEYS if isinstance(raw.get(k), str) and raw[k].strip()), None)
    if content is None:
        return None
    raw_metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    metadata = {k: v for k, v in raw_metadata.items() if k not in _RESERVED_METADATA}
    legacy_user = raw.get("user_id") or raw_metadata.get("user_id")
    if legacy_user and legacy_user != user_id:
        metadata["legacy_user_id"] = legacy_user
    if raw.get("id") is not None:
        metadata["legacy_id"] = str(raw["id"])
    metadata["import_source"] = source
    project = raw.get("project") or raw_metadata.get("project")
    if project:
        metadata["project"] = project
    created_at = next(
        (ts for ts in (_parse_timestamp(raw.get(k)) for k in _CREATED_KEYS) if ts is not None), None
    )
    return {
        "content": content.strip(),
        "created_at": created_at,
        "user_id": user_id,
        "namespace": raw.get("namespace") or namespace,
        # Pre-3.0 memories were not project-scoped; keep them visible everywhere.
        "scope": "project" if project else "global",
        "metadata": metadata,
    }


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


async def import_memories(
    memory: "MuninnMemory",
    records: Iterable[Dict[str, Any]],
    *,
    user_id: str = "global_user",
    namespace: str = "global",
    source: str = "legacy",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Import exported memories, skipping exact duplicates and keeping original timestamps."""
    memory._check_initialized()
    existing = set()
    for page in _live_pages(memory, 500):
        existing.update(content_hash(r.content) for r in page)

    report = {"dry_run": dry_run, "read": 0, "invalid": 0, "duplicates": 0,
              "imported": 0, "merged": 0, "skipped": 0}
    for raw in records:
        report["read"] += 1
        item = normalize_legacy_record(raw, user_id=user_id, namespace=namespace, source=source)
        if item is None:
            report["invalid"] += 1
            continue
        digest = content_hash(item["content"])
        if digest in existing:
            report["duplicates"] += 1
            continue
        existing.add(digest)
        if dry_run:
            report["imported"] += 1
            continue
        result = await memory.add(
            content=item["content"],
            user_id=item["user_id"],
            metadata=item["metadata"],
            namespace=item["namespace"],
            provenance=Provenance.INGESTED,
            scope=item["scope"],
        )
        event = result.get("event")
        if event == "ADD" and result.get("id"):
            report["imported"] += 1
            if item["created_at"] is not None:
                await asyncio.to_thread(memory._metadata.update, result["id"], created_at=item["created_at"])
        elif event == "DEDUP_MERGED":
            report["merged"] += 1
        else:
            report["skipped"] += 1
    logger.info("Import complete: %s", {k: v for k, v in report.items()})
    return report
