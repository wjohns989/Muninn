"""
Vision Adapter for Multimodal Ingestion (Phase 20).

Wraps multimodal LLMs (Ollama LLaVA, Moondream, Llama 3.2 Vision) to generate
textual descriptions of images for embedding and retrieval.
"""

import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import aiohttp

logger = logging.getLogger("Muninn.Vision")

class VisionAdapter:
    """
    Adapter for Vision-Language Models (VLMs).
    
    Generates dense captions for images to enable semantic search via text embeddings.
    """

    def __init__(
        self,
        enabled: bool = False,
        provider: str = "ollama",
        base_url: str = "http://localhost:11434",
        model: str = "llava",
        timeout_seconds: float = 30.0,
    ):
        self.enabled = enabled
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds

    async def describe_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> Optional[str]:
        """
        Generate a text description of the image at the given path.
        """
        if not self.enabled:
            return None

        path = Path(image_path)
        if not path.exists():
            logger.warning("Image file not found: %s", image_path)
            return None

        try:
            if self.provider == "ollama":
                return await self._describe_ollama(path, prompt)
            else:
                logger.warning("Unsupported vision provider: %s", self.provider)
                return None
        except Exception as e:
            logger.error("Vision generation failed for %s: %s", image_path, e)
            return None

    async def _describe_ollama(self, path: Path, prompt: str) -> Optional[str]:
        """Call Ollama /api/generate with image data."""
        with path.open("rb") as f:
            image_bytes = f.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("Ollama vision error %d: %s", resp.status, text)
                    return None
                
                result = await resp.json()
                return result.get("response", "").strip()
