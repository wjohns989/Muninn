"""Memory-bounded, content-addressed storage for image memories."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Sequence

logger = logging.getLogger("Muninn.Media.Image")

ALLOWED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".avif"}
)
COPY_CHUNK_BYTES = 1024 * 1024
_IMAGE_STORAGE_LOCK = asyncio.Lock()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_managed_copy(source: Path, images_dir: Path) -> tuple[Path, str, bool]:
    image_hash = _stream_sha256(source)
    destination = images_dir / f"{image_hash}{source.suffix.lower()}"
    if destination.exists():
        return destination, image_hash, False

    temporary = images_dir / f".{uuid.uuid4().hex}.tmp"
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=COPY_CHUNK_BYTES)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, image_hash, True


def _remove_managed_files(
    images_dir: Path,
    stored_names: Sequence[str],
    referenced_names: set[str],
) -> list[str]:
    """Remove only unreferenced files contained directly in managed storage."""
    images_dir = images_dir.resolve()
    removed: list[str] = []
    for stored_name in dict.fromkeys(stored_names):
        if not stored_name or Path(stored_name).name != stored_name:
            logger.warning("Skipped unsafe managed image filename during cleanup")
            continue
        if stored_name in referenced_names:
            continue
        candidate = (images_dir / stored_name).resolve()
        try:
            candidate.relative_to(images_dir)
        except ValueError:
            logger.warning("Skipped managed image outside storage root")
            continue
        if candidate.is_file():
            try:
                candidate.unlink()
            except OSError as exc:
                logger.warning(
                    "Managed image cleanup failed (%s); file retained for later retry",
                    type(exc).__name__,
                )
            else:
                removed.append(stored_name)
    return removed


async def cleanup_managed_images(
    *,
    metadata_store: Any,
    images_dir: Path,
    stored_names: Sequence[str],
) -> list[str]:
    """Delete managed files after their final memory reference is removed."""
    candidates = list(dict.fromkeys(name for name in stored_names if name))
    if not candidates:
        return []
    async with _IMAGE_STORAGE_LOCK:
        referenced_names = await asyncio.to_thread(
            metadata_store.get_referenced_image_names,
            candidates,
        )
        removed = await asyncio.to_thread(
            _remove_managed_files,
            images_dir,
            candidates,
            referenced_names,
        )
    if removed:
        logger.info("Removed %d unreferenced managed image file(s)", len(removed))
    return removed


async def store_image_memory(
    *,
    memory: Any,
    image_path: str,
    description: str,
    images_dir: Path,
    metadata: Dict[str, Any] | None,
    linked_memory_ids: Sequence[str] | None,
    user_id: str,
    namespace: str,
    scope: str,
    max_bytes: int,
) -> Dict[str, Any]:
    """Copy an image into managed storage and add its searchable description."""
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Image file does not exist")
    if source.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {source.suffix}")
    size_bytes = source.stat().st_size
    if size_bytes > max_bytes:
        raise ValueError(f"Image exceeds maximum size of {max_bytes} bytes")
    if size_bytes <= 0:
        raise ValueError("Image file is empty")

    images_dir = images_dir.resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    image_id = str(uuid.uuid4())
    linked_ids = list(dict.fromkeys(linked_memory_ids or ()))

    async with _IMAGE_STORAGE_LOCK:
        destination, image_hash, created = await asyncio.to_thread(
            _prepare_managed_copy, source, images_dir
        )
        scoped_metadata = dict(metadata or {})
        scoped_metadata.update(
            {
                "image_id": image_id,
                "image_hash": image_hash,
                "image_size_bytes": size_bytes,
                "image_stored_name": destination.name,
                "image_original_name": source.name,
            }
        )
        if linked_ids:
            scoped_metadata["linked_memory_ids"] = linked_ids
        scoped_metadata.setdefault("muninn_skip_extraction", True)

        try:
            result = await memory.add(
                content=description,
                user_id=user_id,
                agent_id=None,
                metadata=scoped_metadata,
                namespace=namespace,
                scope=scope,
                media_type="image",
            )
        except Exception:
            if created:
                await asyncio.to_thread(destination.unlink, missing_ok=True)
            raise

    memory_id = result.get("id") if isinstance(result, dict) else None
    link_failures: list[str] = []
    if linked_ids and memory_id:
        for linked_id in linked_ids:
            try:
                record = await asyncio.to_thread(memory._metadata.get, linked_id)
                if record is None:
                    link_failures.append(linked_id)
                    continue
                linked_images = list((record.metadata or {}).get("linked_image_ids", []))
                if memory_id not in linked_images:
                    linked_images.append(memory_id)
                    await memory.update(
                        linked_id,
                        metadata_patch={"linked_image_ids": linked_images},
                    )
            except Exception:
                logger.warning("Failed to cross-link image memory to %s", linked_id)
                link_failures.append(linked_id)

    return {
        "image_id": image_id,
        "memory_id": memory_id,
        "stored_name": destination.name,
        "size_bytes": size_bytes,
        "linked_memory_ids": linked_ids,
        "link_failures": link_failures,
        "memory_result": result,
    }
