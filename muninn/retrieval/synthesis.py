"""
Scout Synthesis — optional LLM narration of multi-hop discovery paths.

Uses the Anthropic SDK (claude-haiku) to generate a concise 2-3 sentence
explanation of what Scout found and why those memories are relevant.

Gracefully degrades to empty string when:
  - the anthropic SDK is not installed
  - ANTHROPIC_API_KEY is not set in the environment
  - the API call fails for any reason
"""

import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger("Muninn.Synthesis")

_SYNTHESIS_MODEL = "claude-haiku-4-5-20251001"
_SYNTHESIS_MAX_TOKENS = 200
_SYNTHESIS_SNIPPET_CHARS = 120
_SYNTHESIS_MAX_SNIPPETS = 6


async def synthesize_hunt_results(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Generate a concise LLM narration explaining what Scout discovered and why
    those memories are relevant to the query.

    Args:
        query:   The original hunt query string.
        results: List of memory dicts from MuninnMemory.hunt() — each dict has
                 at minimum ``memory`` (content str), ``memory_type``, ``score``.

    Returns:
        A 2-3 sentence narrative string, or "" on any failure.
    """
    if not results:
        return ""

    try:
        from anthropic import AsyncAnthropic  # optional dependency
    except ImportError:
        logger.debug("Synthesis skipped: anthropic SDK not installed")
        return ""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.debug("Synthesis skipped: ANTHROPIC_API_KEY not configured")
        return ""

    try:
        client = AsyncAnthropic(api_key=api_key)
        snippets = "\n".join(
            f"- [{r.get('memory_type', 'memory')}] {str(r.get('memory', ''))[:_SYNTHESIS_SNIPPET_CHARS]}"
            for r in results[:_SYNTHESIS_MAX_SNIPPETS]
        )
        prompt = (
            "You are Muninn, an AI memory assistant. In 2-3 concise sentences, "
            "explain what was discovered during this multi-hop memory hunt and "
            "why these memories are relevant to the query. "
            "Focus on the conceptual connections, not just listing what was found.\n\n"
            f"Query: {query}\n"
            f"Discovered ({len(results)} memories):\n{snippets}\n\n"
            "Discovery summary:"
        )
        message = await client.messages.create(
            model=_SYNTHESIS_MODEL,
            max_tokens=_SYNTHESIS_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        logger.warning("Synthesis failed: %s", exc)
        return ""
