"""
Tests for Audio Adapter (Phase 20).
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from muninn.extraction.audio_adapter import AudioAdapter

@pytest.fixture
def audio_adapter():
    return AudioAdapter(
        enabled=True,
        provider="openai_compatible",
        base_url="http://localhost:8000/v1",
        model="whisper-1"
    )

def test_audio_adapter_disabled():
    adapter = AudioAdapter(enabled=False)
    assert adapter.transcribe_audio_sync("dummy.mp3") is None

def test_audio_adapter_file_not_found(audio_adapter):
    assert audio_adapter.transcribe_audio_sync("non_existent.mp3") is None

@patch("requests.post")
def test_transcribe_audio_sync_success(mock_post, audio_adapter, tmp_path):
    # Setup dummy audio
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"fake audio data")
    
    # Mock response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "Hello, this is a test transcription."}
    mock_post.return_value = mock_resp
    
    transcript = audio_adapter.transcribe_audio_sync(str(audio_path))
    
    assert transcript == "Hello, this is a test transcription."
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "audio/transcriptions" in args[0]
    assert kwargs["data"]["model"] == "whisper-1"

@patch("requests.post")
def test_transcribe_audio_sync_error(mock_post, audio_adapter, tmp_path):
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"fake audio data")
    
    # Mock error response
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_post.return_value = mock_resp
    
    transcript = audio_adapter.transcribe_audio_sync(str(audio_path))
    
    assert transcript is None
