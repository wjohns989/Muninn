"""Check the environment-backed ZDR OpenRouter route without sending inference.

Use validate_real_history_models.py for a separately authorized real-thread
inference check. This preflight never sends conversation content.
"""

from __future__ import annotations

import json
import os
import re

import httpx

from muninn.history import llm_settings
from muninn.history.insights import Provider


def _key_info(client: httpx.Client, key: str) -> dict:
    response = client.get(
        f"{llm_settings.OPENROUTER_API}/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    response.raise_for_status()
    return response.json().get("data") or {}


def main() -> int:
    key = os.environ.get("MUNINN_OPENROUTER_API_KEY", "").strip()
    if not key:
        print(json.dumps({"key_configured": False, "inference_sent": False}))
        return 2
    try:
        provider = Provider("openrouter", llm_settings.OPENROUTER_API, llm_settings.models(), key)
        body = provider.request_body([{"role": "user", "content": "synthetic test"}])
        policy_ok = body.get("provider") == {
            "zdr": True, "data_collection": "deny", "require_parameters": True,
        }
        with httpx.Client(timeout=15.0) as client:
            info = _key_info(client, key)
            cap = info.get("limit")
            remaining = info.get("limit_remaining")
            cap_ok = (
                info.get("limit_reset") == "daily" and cap is not None
                and remaining is not None and 0 < float(cap) <= 1.0
                and float(remaining) > 0 and not info.get("disabled", False)
            )
            catalog_response = client.get(
                f"{llm_settings.OPENROUTER_API}/models",
                params={"zdr": "true"},
                headers={"Authorization": f"Bearer {key}"},
            )
            catalog_response.raise_for_status()
            available = {item.get("id") for item in catalog_response.json().get("data") or []}
        matched = [model for model in provider.models if model in available]
        report = {
            "key_configured": True,
            "key_daily_limit_usd": cap,
            "key_remaining_usd": remaining,
            "key_limit_reset": info.get("limit_reset"),
            "daily_cap_ok": cap_ok,
            "zdr_request_shape_ok": policy_ok,
            "configured_zdr_models": matched,
            "inference_sent": False,
        }
        print(json.dumps(report))
        return 0 if cap_ok and policy_ok and matched else 2
    except Exception as exc:
        code = re.search(r"\b(?:openrouter|HTTP) (\d{3})\b", str(exc))
        print(json.dumps({"inference_sent": False, "error_type": type(exc).__name__,
                          "status_code": int(code.group(1)) if code else None}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
