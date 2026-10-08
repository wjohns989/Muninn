"""List current no-cost ZDR endpoints that support Muninn's output contract.

Read-only discovery. Authentication stays in memory; output contains only public
model IDs, prices, and supported-parameter metadata.
"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from muninn.history import llm_settings
from muninn.history.auto_routing import openrouter_key_status
from muninn.history.remote_accounting import status as accounting_status


def main() -> int:
    key = llm_settings.api_key()
    if not key:
        print(json.dumps({"state": "key_unavailable"}))
        return 2
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        response = client.get("https://openrouter.ai/api/v1/endpoints/zdr",
                              headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        endpoints = response.json().get("data")
    if not isinstance(endpoints, list):
        print(json.dumps({"state": "unexpected_response"}))
        return 2
    free = []
    compatible = []
    paid = []
    inspected = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        pricing = endpoint.get("pricing") or {}
        try:
            prices = (Decimal(str(pricing["prompt"])),
                      Decimal(str(pricing["completion"])),
                      Decimal(str(pricing.get("request", 0))))
        except (KeyError, TypeError, InvalidOperation):
            continue
        no_cost = all(value == 0 for value in prices)
        model = endpoint.get("model_id")
        if not isinstance(model, str):
            continue
        supported = set(endpoint.get("supported_parameters") or [])
        if any(hint in model.lower() for hint in ("gpt-6-luna", "gemini-2.5-flash", "deepseek-chat")):
            inspected.append({"model": model, "supported_parameters": sorted(supported),
                              "input_usd_per_m": float(prices[0] * 1_000_000),
                              "output_usd_per_m": float(prices[1] * 1_000_000),
                              "latency_p50_reported": (endpoint.get("latency_last_30m") or {}).get("p50")})
        contract = ({"response_format", "structured_outputs"} <= supported
                    and bool({"max_tokens", "max_completion_tokens"} & supported))
        if no_cost:
            free.append(model)
            if contract:
                compatible.append(model)
        elif contract and all(value >= 0 for value in prices):
            paid.append({"model": model,
                         "input_usd_per_m": float(prices[0] * 1_000_000),
                         "output_usd_per_m": float(prices[1] * 1_000_000),
                         "latency_p50_reported": (endpoint.get("latency_last_30m") or {}).get("p50")})
    paid.sort(key=lambda item: (item["input_usd_per_m"] + item["output_usd_per_m"], item["model"]))
    print(json.dumps({"state": "ok", "zdr_endpoints": len(endpoints),
                      "free_endpoint_count": len(free), "free_models": sorted(set(free))[:30],
                      "contract_compatible_free_models": sorted(set(compatible))[:30],
                      "contract_compatible_paid_endpoints": len(paid),
                      "least_expensive_paid_endpoints": paid[:15],
                      "inspected_models": inspected[:12],
                      "key_usage": {name: value for name, value in
                                    openrouter_key_status(policy_root=Path(__file__).resolve().parents[1]
                                                           / ".muninn_runtime").items()
                                    if name in {"state", "usage_daily_usd", "usage_monthly_usd"}},
                      "local_accounting": accounting_status(Path(__file__).resolve().parents[1]
                                                             / ".muninn_runtime")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
