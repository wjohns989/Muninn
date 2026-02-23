"""
Mimir provider adapters package.

Exports:
  BaseAdapter      — abstract base class
  ClaudeCodeAdapter — `claude` CLI adapter
  CodexAdapter     — `codex` CLI adapter
  GeminiAdapter    — `gemini` CLI adapter
  get_adapter      — factory: ProviderName → BaseAdapter instance
"""

from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .codex_cli import CodexAdapter
from .gemini_cli import GeminiAdapter

from ..models import ProviderName

# Singleton adapter registry (one instance per provider)
_ADAPTER_REGISTRY: dict[ProviderName, BaseAdapter] = {
    ProviderName.CLAUDE_CODE: ClaudeCodeAdapter(),
    ProviderName.CODEX_CLI: CodexAdapter(),
    ProviderName.GEMINI_CLI: GeminiAdapter(),
}


def get_adapter(provider: ProviderName) -> BaseAdapter:
    """Return the singleton adapter for the given provider."""
    adapter = _ADAPTER_REGISTRY.get(provider)
    if adapter is None:
        raise ValueError(f"No adapter registered for provider '{provider.value}'")
    return adapter


def all_adapters() -> list[BaseAdapter]:
    """Return all registered adapters (excluding AUTO)."""
    return list(_ADAPTER_REGISTRY.values())


__all__ = [
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "GeminiAdapter",
    "get_adapter",
    "all_adapters",
]
