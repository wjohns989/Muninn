"""Exercise installed models on real Muninn code and redacted local history.

No result is persisted. OpenRouter receives checked-in source by default;
private redacted history requires an explicit per-run flag. Ollama requests
use keep_alive=0 and GPU gating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import httpx

from muninn.history import llm_settings
from muninn.history.auto_routing import (
    choose_route,
    guarded_openrouter_available,
    probe_gpu,
    probe_ollama,
)
from muninn.history.insights import Provider
from muninn.history.safe_span import sanitize_agent_span

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "muninn" / "history" / "service.py"
QUESTION = (
    "Read this actual Muninn implementation. Return only a JSON object with "
    "strict_auto_analysis (boolean), archive_index_is_cpu_only (boolean), "
    "ollama_unloads_after_request (boolean or null if not shown), and evidence "
    "(one short sentence). Do not guess from intended design; use the code.\n\n"
)
PUBLIC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["strict_auto_analysis", "archive_index_is_cpu_only",
                 "ollama_unloads_after_request", "evidence"],
    "properties": {
        "strict_auto_analysis": {"type": "boolean"},
        "archive_index_is_cpu_only": {"type": "boolean"},
        "ollama_unloads_after_request": {"type": ["boolean", "null"]},
        "evidence": {"type": "string"},
    },
}
PRIVATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "decisions", "open_items", "uncertainty"],
    "properties": {
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "open_items": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"},
    },
}


def _public_case() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    selected = lines[431:556]
    if not any("if strict_history_mode():" in line for line in selected):
        raise RuntimeError("Code case changed; reselect evidence before testing")
    return QUESTION + "\n".join(f"{n}: {line}" for n, line in enumerate(selected, 432))


def _user_token() -> str:
    token = os.environ.get("MUNINN_AUTH_TOKEN", "")
    if token:
        return token
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")[0])
    raise RuntimeError("MUNINN_AUTH_TOKEN is unavailable")


def _private_case(query: str) -> str:
    headers = {"Authorization": f"Bearer {_user_token()}"}
    base = "http://127.0.0.1:42069"
    with httpx.Client(timeout=90.0, trust_env=False) as client:
        search = client.post(f"{base}/history/secure/search", headers=headers,
                             json={"query": query, "limit": 1})
        search.raise_for_status()
        matches = search.json()["data"]["matches"]
        if not matches:
            raise RuntimeError("No local encrypted-history match")
        response = client.post(f"{base}/history/secure/fetch", headers=headers,
                               json={"capability": matches[0]["fetch_capability"],
                                     "max_chars": 3000})
        response.raise_for_status()
        span = response.json()["data"]["redacted_text"]
    # The second pass is defense in depth. This remains local-only; it must not
    # be used as proof that any private input is safe for remote model egress.
    safe = sanitize_agent_span(span, max_chars=3000)
    return (
        "Extract only durable decisions, constraints, and unresolved work from "
        "this actual, credential-redacted transcript excerpt. Do not infer "
        "completion beyond the excerpt. Return JSON with summary (string), "
        "decisions (array), open_items (array), and uncertainty (string).\n\n"
        + safe
    )


def _check_private_egress(prompt: str) -> None:
    """Conservative last check; never treats redaction as a secrecy proof."""
    from muninn.history.safe_span import _UNRESOLVED_SENSITIVE_VALUE

    if (len(prompt) > 4000 or _UNRESOLVED_SENSITIVE_VALUE.search(prompt)
            or "-----BEGIN " in prompt or "[REDACTED_SENSITIVE_LINE]" in prompt):
        raise RuntimeError("Private excerpt is not eligible for remote model validation")


def _parse_object(value: str) -> dict:
    value = value.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(value[value.find("{"):value.rfind("}") + 1])


async def _one(provider: Provider, prompt: str, *, public: bool) -> dict:
    started = time.monotonic()
    messages = [
        {"role": "system", "content": "Use the supplied evidence only. Reply with one JSON object."},
        {"role": "user", "content": prompt},
    ]
    schema = PUBLIC_SCHEMA if public else PRIVATE_SCHEMA
    body = provider.request_body(messages)
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
        if provider.name == "ollama":
            from muninn.extraction.ollama_slot import async_ollama_slot

            body["format"] = schema
            body["keep_alive"] = 0
            async with async_ollama_slot():
                response = await client.post(
                    f"{provider.base_url.removesuffix('/v1')}/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
            content = (data.get("message") or {}).get("content") or ""
            meta = {"model": data.get("model"), "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"), "cost": None}
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "model_route_check", "strict": True, "schema": schema}}
            response = await client.post(f"{provider.base_url}/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {provider.api_key}"})
            response.raise_for_status()
            data = response.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            meta = {"model": data.get("model"), "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cost": usage.get("cost")}
    answer = _parse_object(content)
    if public:
        checks = {
            "strict_auto_analysis": answer.get("strict_auto_analysis") is False,
            "archive_index_is_cpu_only": answer.get("archive_index_is_cpu_only") is True,
            "ollama_unloads_unknown_when_not_shown": answer.get("ollama_unloads_after_request") is None,
        }
        result = {"checks": checks, "evidence": str(answer.get("evidence", ""))[:250]}
    else:
        result = {
            "decisions": len(answer.get("decisions") or []),
            "open_items": len(answer.get("open_items") or []),
            "summary_chars": len(sanitize_agent_span(str(answer.get("summary", ""))[:1500])),
            "uncertainty_chars": len(sanitize_agent_span(str(answer.get("uncertainty", ""))[:500])),
        }
    return {"model": meta.get("model"), "elapsed_seconds": round(time.monotonic() - started, 1),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cost_usd": meta.get("cost") if provider.name == "openrouter" else None,
            **result}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--archive-query")
    parser.add_argument("--openrouter", action="store_true")
    parser.add_argument("--allow-private-openrouter", action="store_true",
                        help="Explicitly allow one credential-scrubbed archive excerpt to ZDR OpenRouter")
    args = parser.parse_args()
    if args.openrouter and args.archive_query and not args.allow_private_openrouter:
        parser.error("Private history requires --allow-private-openrouter")
    prompt = _private_case(args.archive_query) if args.archive_query else _public_case()
    if args.openrouter and args.archive_query:
        _check_private_egress(prompt)
    base = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    for model in args.models:
        gpu = probe_gpu()
        installed, loaded = probe_ollama(base)
        if gpu is not None:
            gpu = replace(gpu, loaded_models=loaded)
        route = choose_route(gpu, installed, model_hints=(model,))
        if route.provider != "ollama" or route.model != model:
            print(json.dumps({"model": model, "skipped": route.reason}))
            continue
        try:
            provider = Provider("ollama", f"{base}/v1", [model])
            report = await _one(provider, prompt, public=not args.archive_query)
            _, resident = probe_ollama(base)
            report["resident_after"] = model in resident
            print(json.dumps(report, ensure_ascii=False))
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as exc:
            print(json.dumps({"model": model, "error_type": type(exc).__name__}))
    if args.openrouter:
        if llm_settings.key_source() not in {
            "environment (MUNINN_OPENROUTER_API_KEY)",
            "user environment (MUNINN_OPENROUTER_API_KEY)",
        } or not guarded_openrouter_available():
            print(json.dumps({"provider": "openrouter", "skipped": "daily_zdr_cap_unverified"}))
            return 2
        provider = Provider.from_env("openrouter")
        if provider.request_body([]).get("provider") != {
            "zdr": True, "data_collection": "deny", "require_parameters": True,
        }:
            raise RuntimeError("OpenRouter ZDR request policy changed")
        try:
            report = await _one(provider, prompt, public=not args.archive_query)
            report["provider"] = "openrouter"
            print(json.dumps(report, ensure_ascii=False))
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as exc:
            print(json.dumps({"provider": "openrouter", "error_type": type(exc).__name__}))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
