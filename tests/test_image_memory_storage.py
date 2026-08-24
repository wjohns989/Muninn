from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_image_storage_streams_deduplicates_and_omits_original_path(tmp_path, monkeypatch):
    from muninn.media.image_memory import store_image_memory

    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic-image-bytes" * 1000)
    images_dir = tmp_path / "managed" / "images"
    memory = SimpleNamespace(add=AsyncMock(side_effect=[{"id": "m1"}, {"id": "m2"}]))

    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("eager read")))
    first = await store_image_memory(
        memory=memory,
        image_path=str(source),
        description="Synthetic image description",
        images_dir=images_dir,
        metadata={},
        linked_memory_ids=[],
        user_id="benchmark-user",
        namespace="benchmark",
        scope="project",
        max_bytes=1024 * 1024,
    )
    second = await store_image_memory(
        memory=memory,
        image_path=str(source),
        description="Second reference to the same synthetic image",
        images_dir=images_dir,
        metadata={},
        linked_memory_ids=[],
        user_id="benchmark-user",
        namespace="benchmark",
        scope="project",
        max_bytes=1024 * 1024,
    )

    assert first["stored_name"] == second["stored_name"]
    assert "stored_path" not in first
    assert len(list(images_dir.glob("*.png"))) == 1
    add_metadata = memory.add.await_args_list[0].kwargs["metadata"]
    assert "image_original_path" not in add_metadata
    assert add_metadata["image_original_name"] == "source.png"
    assert add_metadata["muninn_skip_extraction"] is True


@pytest.mark.asyncio
async def test_image_storage_rejects_oversized_files(tmp_path):
    from muninn.media.image_memory import store_image_memory

    source = tmp_path / "large.png"
    source.write_bytes(b"x" * 32)

    with pytest.raises(ValueError, match="maximum"):
        await store_image_memory(
            memory=SimpleNamespace(add=AsyncMock()),
            image_path=str(source),
            description="Synthetic",
            images_dir=tmp_path / "images",
            metadata={},
            linked_memory_ids=[],
            user_id="u",
            namespace="n",
            scope="project",
            max_bytes=16,
        )


@pytest.mark.asyncio
async def test_new_managed_copy_is_removed_if_memory_insert_fails(tmp_path):
    from muninn.media.image_memory import store_image_memory

    source = tmp_path / "source.webp"
    source.write_bytes(b"synthetic")
    images_dir = tmp_path / "images"
    memory = SimpleNamespace(add=AsyncMock(side_effect=RuntimeError("synthetic insert failure")))

    with pytest.raises(RuntimeError, match="synthetic insert failure"):
        await store_image_memory(
            memory=memory,
            image_path=str(source),
            description="Synthetic",
            images_dir=images_dir,
            metadata={},
            linked_memory_ids=[],
            user_id="u",
            namespace="n",
            scope="project",
            max_bytes=1024,
        )

    assert not list(images_dir.glob("*.webp"))
    assert not list(images_dir.glob("*.tmp"))
