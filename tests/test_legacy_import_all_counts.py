from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_import_all_sums_added_memories_across_batches(monkeypatch):
    import server

    sources = [{"source_id": f"src-{i}", "parser_supported": True} for i in range(60)]
    sources.append({"source_id": "unsupported", "parser_supported": False})
    fake_memory = SimpleNamespace(
        discover_legacy_sources=AsyncMock(return_value={"sources": sources}),
        ingest_legacy_sources=AsyncMock(
            side_effect=[
                {"added_memories": 7, "event": "LEGACY_INGEST_COMPLETED"},
                {"added_memories": 3, "event": "LEGACY_INGEST_COMPLETED"},
            ]
        ),
    )
    monkeypatch.setattr(server, "memory", fake_memory)

    response = await server.ingest_all_legacy_sources_endpoint()

    assert response["success"] is True
    assert response["data"]["imported"] == 10
    assert response["data"]["total_supported"] == 60
    assert response["data"]["batches"] == 2
    first_batch = fake_memory.ingest_legacy_sources.await_args_list[0].kwargs["selected_source_ids"]
    assert len(first_batch) == 50
    assert "unsupported" not in first_batch


@pytest.mark.asyncio
async def test_ingest_reports_disabled_feature_as_409(monkeypatch):
    from fastapi import HTTPException

    import server
    from muninn.core.feature_flags import FeatureDisabledError

    fake_memory = SimpleNamespace(
        ingest_sources=AsyncMock(side_effect=FeatureDisabledError(
            "Feature 'multi_source_ingestion' is disabled. Set MUNINN_MULTI_SOURCE_INGESTION=1 to enable."))
    )
    monkeypatch.setattr(server, "memory", fake_memory)

    with pytest.raises(HTTPException) as caught:
        await server.ingest_sources_endpoint(server.IngestSourcesRequest(sources=["x"]))

    assert caught.value.status_code == 409
    assert "MUNINN_MULTI_SOURCE_INGESTION=1" in caught.value.detail
