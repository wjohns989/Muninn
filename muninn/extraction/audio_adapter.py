"""
Audio Adapter for Multimodal Ingestion (Phase 20).

Handles speech-to-text transcription for audio files to enable semantic
search via text embeddings.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger("Muninn.Audio")

class AudioAdapter:
    """
    Adapter for Speech-to-Text (STT) models.
    
    Transcribes audio files into text for ingestion into the memory core.
    """

    def __init__(
        self,
        enabled: bool = False,
        provider: str = "openai_compatible",
        base_url: str = "http://localhost:8000/v1",
        model: str = "whisper-1",
        api_key: str = "not-needed",
        timeout_seconds: float = 60.0,
    ):
        self.enabled = enabled
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout_seconds

    def transcribe_audio_sync(self, audio_path: str) -> Optional[str]:
        """
        Synchronous version of transcribe_audio for use in worker processes.
        """
        if not self.enabled:
            return None

        path = Path(audio_path)
        if not path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            return None

        try:
            if self.provider == "openai_compatible":
                return self._transcribe_openai_sync(path)
            else:
                logger.warning("Unsupported audio provider: %s", self.provider)
                return None
        except Exception as e:
            logger.error("Audio transcription failed for %s: %s", audio_path, e)
            return None

    def _transcribe_openai_sync(self, path: Path) -> Optional[str]:
        """Call OpenAI-compatible /audio/transcriptions endpoint."""
        url = f"{self.base_url}/audio/transcriptions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Files payload for multipart/form-data
        files = {
            "file": (path.name, path.open("rb"), "audio/mpeg")
        }
        data = {
            "model": self.model,
            "response_format": "json"
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if resp.status_code != 200:
                logger.error("Audio transcription error %d: %s", resp.status_code, resp.text)
                return None
                
            result = resp.json()
            return result.get("text", "").strip()
        finally:
            # Ensure file handle is closed
            if "file" in files:
                files["file"][1].close()
