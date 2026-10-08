"""A missing FastEmbed runtime must not silently consume GPU or index zero vectors."""

from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest

from muninn.core.memory import MuninnMemory
from muninn.extraction import ollama_slot


def _memory():
    memory = MuninnMemory.__new__(MuninnMemory)
    memory.config = SimpleNamespace(embedding=SimpleNamespace(
        ollama_url="http://localhost:11434", model="nomic-embed-text",
        ollama_keep_alive="0", dimensions=768,
    ))
    return memory


def test_fallback_embedding_explicitly_uses_cpu_and_unloads(monkeypatch):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, body=json)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"embedding": [0.1, 0.2]})

    monkeypatch.setattr(ollama_slot, "ollama_slot", lambda: nullcontext())
    monkeypatch.setattr(httpx, "post", post)
    assert _memory()._ollama_embed("test") == [0.1, 0.2]
    assert sent["body"]["options"]["num_gpu"] == 0
    assert sent["body"]["keep_alive"] == "0"


def test_fallback_failure_does_not_return_a_fake_searchable_vector(monkeypatch):
    monkeypatch.setattr(ollama_slot, "ollama_slot", lambda: nullcontext())

    def fail(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(RuntimeError, match="CPU embedding"):
        _memory()._ollama_embed("test")
