from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_image_request_model_preserves_scope_and_links():
    from muninn.core.types import AddImageMemoryRequest

    request = AddImageMemoryRequest(
        image_path="C:/synthetic/example.png",
        description="Synthetic image",
        scope="global",
        linked_memory_ids=["memory-1"],
    )
    assert request.scope == "global"
    assert request.linked_memory_ids == ["memory-1"]


@pytest.mark.asyncio
async def test_add_image_endpoint_delegates_to_managed_storage(tmp_path, monkeypatch):
    import server
    from muninn.core.types import AddImageMemoryRequest

    fake_memory = SimpleNamespace(config=SimpleNamespace(data_dir=str(tmp_path)))
    store = AsyncMock(return_value={"memory_id": "memory-1", "stored_name": "hash.png"})
    monkeypatch.setattr(server, "memory", fake_memory)
    monkeypatch.setattr("muninn.media.image_memory.store_image_memory", store)

    response = await server.add_image_memory_endpoint(
        AddImageMemoryRequest(
            image_path=str(tmp_path / "source.png"),
            description="Synthetic image",
            linked_memory_ids=["memory-0"],
        )
    )

    assert response["success"] is True
    assert response["data"]["memory_id"] == "memory-1"
    assert store.await_args.kwargs["images_dir"] == tmp_path / "images"


def test_mcp_image_tool_is_public_and_forwards_to_backend(monkeypatch):
    from muninn.mcp.definitions import TOOLS_SCHEMAS
    from muninn.mcp.handlers import _do_call_tool_logic

    response = MagicMock()
    response.json.return_value = {"success": True, "data": {"memory_id": "memory-1"}}
    request = MagicMock(return_value=response)
    monkeypatch.setattr("muninn.mcp.handlers.make_request_with_retry", request)
    monkeypatch.setattr(
        "muninn.mcp.handlers.get_git_info",
        lambda: {"project": "synthetic-project", "branch": "synthetic-branch"},
    )

    result = _do_call_tool_logic(
        "add_image_memory",
        {"image_path": "C:/synthetic/example.png", "description": "Synthetic image"},
        None,
    )

    assert any(schema["name"] == "add_image_memory" for schema in TOOLS_SCHEMAS)
    assert result["success"] is True
    assert request.call_args.args[1].endswith("/add-image")


@pytest.mark.asyncio
async def test_linked_images_are_batch_enriched_without_original_paths(monkeypatch):
    import server

    image_record = SimpleNamespace(
        id="image-memory-1",
        content="Synthetic image description",
        metadata={
            "image_stored_name": "content-hash.png",
            "image_original_path": "C:/private/source.png",
        },
    )
    metadata = MagicMock()
    metadata.get_by_ids.return_value = [image_record]
    monkeypatch.setattr(server, "memory", SimpleNamespace(_metadata=metadata))
    results = [{"id": "text-memory-1", "metadata": {"linked_image_ids": ["image-memory-1"]}}]

    enriched = await server._enrich_with_linked_images(results)

    linked = enriched[0]["linked_images"][0]
    assert linked["image_url"] == "/images/content-hash.png"
    assert "image_original_path" not in linked
    metadata.get_by_ids.assert_called_once_with(["image-memory-1"])


@pytest.mark.asyncio
async def test_linked_images_enrich_typed_search_results(monkeypatch):
    import server
    from muninn.core.types import MemoryRecord, SearchResult

    image_record = SimpleNamespace(
        id="image-memory-1",
        content="Synthetic image description",
        metadata={"image_stored_name": "content-hash.png"},
    )
    metadata = MagicMock()
    metadata.get_by_ids.return_value = [image_record]
    monkeypatch.setattr(server, "memory", SimpleNamespace(_metadata=metadata))
    result = SearchResult(
        memory=MemoryRecord(
            content="Text memory",
            metadata={"linked_image_ids": ["image-memory-1"]},
        )
    )

    enriched = await server._enrich_with_linked_images([result])

    assert enriched[0].linked_images[0]["image_url"] == "/images/content-hash.png"
