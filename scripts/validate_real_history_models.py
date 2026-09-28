"""Read one imported thread and test actual Ollama/OpenRouter understanding without writes.

Only loopback Muninn/Ollama endpoints are accepted. OpenRouter requires the
MUNINN_OPENROUTER_API_KEY environment variable and a provider-enforced $1/day
key cap. The complete selected thread must fit the requested input limit; this
tool never silently samples or truncates, or stores generated understanding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx

from muninn.history.auto_routing import (
    choose_route,
    guarded_openrouter_available,
    probe_gpu,
    probe_ollama,
)
from muninn.history.importer import collect, redact
from muninn.history.insights import CallStats, Provider, render_turns, understand
from muninn.history.vault import VaultFile


def _loopback(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Only local HTTP endpoints are allowed")
    return url.rstrip("/")


class ReadOnlyVault:
    """The same manifest listing as HistoryVault, without a second writer."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._db = sqlite3.connect((self.root / "manifest.db").as_uri() + "?mode=ro", uri=True)
        self._db.row_factory = sqlite3.Row

    def files(self, provider: str | None = None, kind: str | None = None) -> list[VaultFile]:
        query, params = "SELECT * FROM vault_files WHERE 1=1", []
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        rows = self._db.execute(query + " ORDER BY mtime", params).fetchall()
        return [VaultFile(row["source_path"], row["provider"], row["kind"],
                          self.root / row["vault_rel"], row["size"], row["mtime"],
                          row["missing_since"]) for row in rows]

    def close(self) -> None:
        self._db.close()


async def _run(server_url: str, ollama_url: str, vault_root: Path,
               thread_key: str, provider_name: str, model: str | None,
               max_chars: int) -> int:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{server_url}/history/threads/{quote(thread_key, safe='')}?limit=1",
        )
        response.raise_for_status()
        data = response.json()["data"]
    source_path = data["thread"]["source_path"]
    vault = ReadOnlyVault(vault_root)
    try:
        collected = collect(vault, sources=[source_path])
    finally:
        vault.close()
    if collected.errors:
        print(json.dumps({"ok": False, "reason": "vault_parse_error",
                          "error_count": len(collected.errors)}))
        return 2
    threads = [thread for thread in collected.threads if thread.key == thread_key]
    if len(threads) != 1:
        print(json.dumps({"ok": False, "reason": "thread_not_found_in_vault"}))
        return 2
    thread = threads[0]
    if provider_name == "openrouter":
        if not os.environ.get("MUNINN_OPENROUTER_API_KEY", "").strip():
            print(json.dumps({"ok": False, "reason": "environment_key_required",
                              "inference_sent": False}))
            return 2
        if not guarded_openrouter_available(1.0):
            print(json.dumps({"ok": False, "reason": "daily_key_cap_unverified",
                              "inference_sent": False}))
            return 2
        provider = Provider.from_env("openrouter", model)
        body = provider.request_body([{"role": "user", "content": ""}])
        if body.get("provider") != {"zdr": True, "data_collection": "deny",
                                    "require_parameters": True}:
            print(json.dumps({"ok": False, "reason": "zdr_request_guard_failed",
                              "inference_sent": False}))
            return 2
    else:
        if not model:
            raise ValueError("Ollama requires --model")
        provider = Provider("ollama", f"{ollama_url}/v1", [model])
    windows = render_turns(thread, provider.window_chars)
    input_chars = sum(len(window) for window in windows)
    if input_chars > max_chars or len(windows) != 1:
        print(json.dumps({"ok": False, "reason": "thread_exceeds_input_limit",
                          "chars": input_chars, "windows": len(windows)}))
        return 2
    if provider_name == "ollama":
        gpu = probe_gpu()
        installed, loaded = probe_ollama(ollama_url)
        if gpu is not None:
            gpu = replace(gpu, loaded_models=loaded)
        route = choose_route(gpu, installed, model_hints=(model,))
        if route.provider != "ollama" or route.model != model:
            print(json.dumps({"ok": False, "reason": route.reason, "inference_sent": False}))
            return 2
    started = time.monotonic()
    stats = CallStats()
    async with httpx.AsyncClient(timeout=300.0) as client:
        result = await understand(provider, client, thread, stats=stats)
    resident = []
    if provider_name == "ollama":
        _, resident = probe_ollama(ollama_url)
    # Redact again because a model may echo source text in its answer.
    safe_result = json.loads(redact(json.dumps(result, ensure_ascii=False)))
    print(json.dumps({
        "ok": True, "provider": provider_name,
        "requested_models": provider.models,
        "source_turns": len(thread.session.turns),
        "input_chars": input_chars,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "model_resident_after": model in resident if provider_name == "ollama" else None,
        "response_models": stats.models,
        "reported_cost_usd": round(stats.cost, 6) if provider_name == "openrouter" else None,
        "calls": stats.calls,
        "result": safe_result,
    }, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--thread-key", required=True)
    parser.add_argument("--provider", choices=("ollama", "openrouter"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--max-chars", type=int, default=50_000)
    args = parser.parse_args()
    try:
        server_url = _loopback(args.server_url)
        ollama_url = _loopback(os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434"))
        return asyncio.run(_run(server_url, ollama_url, Path(args.vault_root), args.thread_key,
                                args.provider, args.model, args.max_chars))
    except (httpx.HTTPError, KeyError, ValueError, TypeError, sqlite3.Error,
            RuntimeError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
