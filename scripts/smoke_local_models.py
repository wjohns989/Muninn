"""Check local Ollama insight extraction on synthetic chats without storing data.

Usage: python scripts/smoke_local_models.py MODEL [MODEL ...]
Each request is gated on current GPU headroom and uses keep_alive=0.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import httpx

from muninn.history.auto_routing import choose_route, probe_gpu, probe_ollama
from muninn.history.insights import SYSTEM_PROMPT, Provider, validate_reply

CASES = (
    ("decision", "User: Keep SQLite as the local cache, and do not send transcripts to cloud services.\n"
     "Assistant: I used SQLite with WAL mode.\n"
     "User: I considered DuckDB, but keep SQLite.\n"
     "Assistant: Confirmed; SQLite remains the choice."),
    ("open_item", "User: The optional importer fails with an ImportError.\n"
     "Assistant: I installed the dependency and its unit tests pass.\n"
     "User: Keep the importer disabled until its live behavior is verified.\n"
     "Assistant: It remains disabled; live validation is still pending."),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    base = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    with httpx.Client(timeout=240.0) as client:
        for model in args.models:
            for label, conversation in CASES:
                gpu = probe_gpu()
                installed, loaded = probe_ollama(base)
                if gpu is not None:
                    gpu = replace(gpu, loaded_models=loaded)
                route = choose_route(gpu, installed, model_hints=(model,))
                if route.provider != "ollama" or route.model != model:
                    print(json.dumps({"model": model, "case": label,
                                      "skipped": route.reason, "free_mib": route.free_mib}))
                    continue
                provider = Provider("ollama", f"{base}/v1", [model])
                body = provider.request_body([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": conversation},
                ])
                body["keep_alive"] = 0
                started = time.monotonic()
                try:
                    response = client.post(f"{base}/api/chat", json=body)
                    response.raise_for_status()
                    raw = response.json().get("message", {}).get("content", "")
                    result = validate_reply(raw, turn_count=2)
                    _, resident = probe_ollama(base)
                    print(json.dumps({"model": model, "case": label,
                                      "elapsed_seconds": round(time.monotonic() - started, 1),
                                      "status": result["status"],
                                      "insights": result["insights"],
                                      "resident_after": list(resident)}, ensure_ascii=False))
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    print(json.dumps({"model": model, "case": label,
                                      "error_type": type(exc).__name__}))
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
