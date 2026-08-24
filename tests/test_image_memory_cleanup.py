import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from muninn.core.memory import MuninnMemory
from muninn.core.types import MediaType, MemoryRecord, MemoryType, Provenance
from muninn.store.sqlite_metadata import SQLiteMetadataStore


def _image_record(memory_id: str, stored_name: str) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content="Managed image",
        memory_type=MemoryType.EPISODIC,
        media_type=MediaType.IMAGE,
        provenance=Provenance.USER_EXPLICIT,
        metadata={
            "user_id": "user-1",
            "image_stored_name": stored_name,
        },
    )


def _mock_memory(tmp_path) -> MuninnMemory:
    memory = MuninnMemory()
    memory.config.data_dir = tmp_path
    memory._initialized = True
    memory._metadata = MagicMock()
    memory._vectors = MagicMock()
    memory._graph = MagicMock()
    memory._bm25 = MagicMock()
    return memory


@pytest.mark.parametrize(
    "remaining_references,should_exist",
    [(set(), False), ({"image.png"}, True)],
)
def test_delete_cleans_managed_image_only_after_last_reference(
    tmp_path, remaining_references, should_exist
) -> None:
    memory = _mock_memory(tmp_path)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "image.png"
    image_path.write_bytes(b"synthetic-image")
    memory._metadata.get.return_value = _image_record("image-memory", image_path.name)
    memory._metadata.get_referenced_image_names.return_value = remaining_references

    asyncio.run(memory.delete("image-memory"))

    assert image_path.exists() is should_exist


def test_delete_all_cleans_managed_images_for_deleted_user(tmp_path) -> None:
    memory = _mock_memory(tmp_path)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_paths = [images_dir / "first.png", images_dir / "second.png"]
    for image_path in image_paths:
        image_path.write_bytes(b"synthetic-image")
    records = [
        _image_record("image-1", image_paths[0].name),
        _image_record("image-2", image_paths[1].name),
    ]
    memory._metadata.get_all.return_value = records
    memory._metadata.delete_all.return_value = len(records)
    memory._metadata.get_referenced_image_names.return_value = set()

    result = asyncio.run(memory.delete_all(user_id="user-1"))

    assert result["deleted_count"] == 2
    assert not any(image_path.exists() for image_path in image_paths)


def test_delete_succeeds_when_managed_image_unlink_fails(
    tmp_path, monkeypatch, caplog
) -> None:
    memory = _mock_memory(tmp_path)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "locked.png"
    image_path.write_bytes(b"synthetic-image")
    memory._metadata.get.return_value = _image_record("image-memory", image_path.name)
    memory._metadata.get_referenced_image_names.return_value = set()
    original_unlink = Path.unlink

    def fail_managed_unlink(path: Path, *args, **kwargs):
        if path == image_path:
            raise PermissionError("synthetic locked file")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_managed_unlink)

    result = asyncio.run(memory.delete("image-memory"))

    assert result == {"id": "image-memory", "event": "DELETE"}
    assert image_path.exists()
    assert "managed image cleanup failed" in caplog.text.lower()


def test_sqlite_reports_only_managed_image_names_still_referenced(tmp_path) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.db")
    try:
        store.add(_image_record("image-1", "shared.png"))
        store.add(_image_record("image-2", "other.png"))

        assert store.get_referenced_image_names(
            ["shared.png", "missing.png"]
        ) == {"shared.png"}
    finally:
        store.close()
