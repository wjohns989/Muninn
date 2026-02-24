"""
Tests for Vision Adapter (Phase 20).
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from muninn.extraction.vision_adapter import VisionAdapter

@pytest.fixture
def vision_adapter():
    return VisionAdapter(
        enabled=True,
        provider="ollama",
        base_url="http://localhost:11434",
        model="llava"
    )

def test_vision_adapter_disabled():
    adapter = VisionAdapter(enabled=False)
    assert adapter.describe_image_sync("dummy.png") is None

def test_vision_adapter_file_not_found(vision_adapter):
    assert vision_adapter.describe_image_sync("non_existent.png") is None

@patch("requests.post")
def test_describe_image_sync_success(mock_post, vision_adapter, tmp_path):
    # Setup dummy image
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake image data")
    
    # Mock response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "A beautiful sunset over the mountains."}
    mock_post.return_value = mock_resp
    
    description = vision_adapter.describe_image_sync(str(img_path))
    
    assert description == "A beautiful sunset over the mountains."
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "llava"
    assert "images" in kwargs["json"]
    assert len(kwargs["json"]["images"]) == 1

@patch("requests.post")
def test_describe_image_sync_error(mock_post, vision_adapter, tmp_path):
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake image data")
    
    # Mock error response
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_post.return_value = mock_resp
    
    description = vision_adapter.describe_image_sync(str(img_path))
    
    assert description is None

@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post")
async def test_describe_image_async_success(mock_post, vision_adapter, tmp_path):
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake image data")
    
    # Mock context manager and response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"response": "Async description."})
    
    mock_context = MagicMock()
    mock_context.__aenter__.return_value = mock_resp
    mock_post.return_value = mock_context
    
    description = await vision_adapter.describe_image(str(img_path))
    
    assert description == "Async description."

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)
