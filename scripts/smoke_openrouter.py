"""Check the environment-backed ZDR OpenRouter route without sending inference.

Use validate_real_history_models.py for a separately authorized real-thread
inference check. This preflight never sends conversation content.
"""

from __future__ import annotations

import json
import re

import httpx

from muninn.history import llm_settings
from muninn.history.auto_routing import guarded_openrouter_available, openrouter_budget_ceiling
from muninn.history.insights import Provider


def _key_info(client: httpx.Client, key: str) -> dict:
    response = client.get(
        f"{llm_settings.OPENROUTER_API}/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    response.raise_for_status()
    return response.json().get("data") or {}


def main() -> int:
    source = llm_settings.key_source()
    if source not in {"environment (MUNINN_OPENROUTER_API_KEY)",
                      "user environment (MUNINN_OPENROUTER_API_KEY)"}:
        print(json.dumps({"key_configured": False, "inference_sent": False}))
        return 2
    key = llm_settings.api_key()
    if not key:
        print(json.dumps({"key_configured": False, "inference_sent": False}))
        return 2
    try:
        provider = Provider("openrouter", llm_settings.OPENROUTER_API, llm_settings.models(), key)
        body = provider.request_body([{"role": "user", "content": "synthetic test"}])
        policy_ok = body.get("provider") == {
            "zdr": True, "data_collection": "deny", "require_parameters": True,
        }
        # Do not forward the bearer key through proxy settings inherited from
        # the shell; the application uses the same direct-only policy.
        with httpx.Client(timeout=15.0, trust_env=False) as client:
            info = _key_info(client, key)
            cap = info.get("limit")
            remaining = info.get("limit_remaining")
            cap_ok = guarded_openrouter_available()
            daily_ceiling, monthly_ceiling = openrouter_budget_ceiling()
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
            "budget_policy_ok": cap_ok,
            "local_daily_ceiling_usd": daily_ceiling,
            "local_monthly_ceiling_usd": monthly_ceiling,
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
