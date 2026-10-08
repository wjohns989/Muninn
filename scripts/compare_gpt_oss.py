"""Explicit, small ZDR comparison; no memory publication or route changes.

Inputs and replies stay in encrypted owner-only receipts. No inference retries:
uncertain transport or missing billing preserves the blocking admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_native_cited_comparison import samples
import httpx
from muninn.history import llm_settings
from muninn.history.auto_routing import openrouter_key_status
from muninn.history.capture_journal import CaptureJournal
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.remote_accounting import reserve, status
from muninn.history.remote_policy import read_policy
from muninn.history.secure_analysis import _CITED_SCHEMA, _cited_prompt, _cited_outcome, _request_safe
from muninn.history.secure_archive import SecureHistoryArchive

MODEL = "openai/gpt-oss-120b"
TAG = "deepinfra/bf16"
MAX_TOKENS = 2048


def response_matches(status_code, data):
    return status_code == 200 and data.get("model") == MODEL and "choices" in data


class FrozenSource:
    def __init__(self, window):
        self.window = window

    def reopen(self, _descriptor):
        return self.window

    def validated_proposals(self, _descriptor, proposals):
        for proposal in proposals:
            start, quote = proposal["start"], proposal["quote"]
            if (self.window["text"][start:start + len(quote)] != quote or
                    not any(r["start"] <= start and start + len(quote) <= r["start"] + r["length"]
                            for r in self.window["citation_ranges"])):
                raise ValueError("Unsupported citation")
        return proposals


def save_receipt(journal, directory, name, value):
    path = directory / (name + ".sealed")
    create_private_file(path)
    sealed = journal._seal_search(value, directory.name, name)
    with path.open("wb") as stream:
        stream.write(sealed)
        stream.flush()
        os.fsync(stream.fileno())
    # Verify the durable authenticated preimage before any dependent action.
    verify_private(path)
    if journal._open_search(path.read_bytes(), directory.name, name) != value:
        raise ValueError("Receipt validation failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--send-three-zdr-tests", action="store_true")
    args = parser.parse_args()
    if not args.send_three_zdr_tests:
        parser.error("Explicit --send-three-zdr-tests required")
    root = args.archive_root.resolve()
    selected = samples(root, limit=3)
    if len(selected) != 3:
        raise ValueError("Three screened settled samples required")
    bodies = [{"model": MODEL, "messages": _cited_prompt(item["input"]),
               "provider": {"only": [TAG], "allow_fallbacks": False, "zdr": True,
                            "data_collection": "deny", "require_parameters": True},
               "max_tokens": MAX_TOKENS, "reasoning": {"effort": "low"},
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "cited_memory", "strict": True, "schema": _CITED_SCHEMA}}}
              for item in selected]
    if not all(_request_safe(body) for body in bodies):
        raise ValueError("Request safety gate refused")
    policy = read_policy(root.parent, lambda: (False, 1., 30., False))
    if not policy.enabled:
        raise ValueError("Remote consent disabled")
    with httpx.Client(timeout=180, trust_env=False, follow_redirects=False) as client:
        endpoint = client.get(f"{llm_settings.OPENROUTER_API}/models/{MODEL}/endpoints")
        endpoint.raise_for_status()
        matches = [e for e in endpoint.json()["data"]["endpoints"] if e["tag"] == TAG]
        if len(matches) != 1:
            raise ValueError("Pinned endpoint unavailable")
        prices = matches[0]["pricing"]
        # UTF-8 bytes plus framing reserve is a conservative token upper bound.
        upper = sum((len(json.dumps(b["messages"], ensure_ascii=False).encode()) + 1024)
                    * float(prices["prompt"]) + MAX_TOKENS * float(prices["completion"])
                    for b in bodies)
        if not 0 < upper < .05:
            raise ValueError("Test cost bound exceeded")
        archive = SecureHistoryArchive(root)
        journal = object.__new__(CaptureJournal)
        journal.archive = archive
        directory = root / ("model-comparison-" + uuid.uuid4().hex)
        create_private_directory(directory)
        save_receipt(journal, directory, "inputs", {"samples": selected, "bodies": bodies,
                     "upper_usd": upper, "pricing": prices})
        print(json.dumps({"stage": "prepared", "samples": 3, "upper_usd": upper,
                          "receipt": str(directory)}), flush=True)
        reports, total = [], 0.
        for item, body in zip(selected, bodies):
            key_state = openrouter_key_status(policy_root=root.parent)
            local = status(root.parent)
            headroom = min(policy.daily_usd - max(local["daily_cost_usd"],
                           key_state.get("usage_daily_usd") or 0),
                           policy.monthly_usd - max(local["monthly_cost_usd"],
                           key_state.get("usage_monthly_usd") or 0),
                           key_state.get("key_remaining_usd") or 0)
            if not key_state["admission_ready"] or headroom <= upper:
                raise ValueError("Insufficient verified budget headroom")
            admission = reserve(root.parent, policy.generation, key_state)
            try:
                save_receipt(journal, directory, item["id"] + "-admission",
                             {"id": admission.identifier, "generation": policy.generation})
                admission.mark_unknown()
                started = time.monotonic()
                response = client.post(f"{llm_settings.OPENROUTER_API}/chat/completions",
                    json=body, headers={"Authorization": "Bearer " + llm_settings.api_key()})
                elapsed = time.monotonic() - started
                data = response.json()
                save_receipt(journal, directory, item["id"] + "-response", data)
                if not admission.settle_response(data):
                    raise ValueError("Missing cost; no further requests permitted")
                cost = float(data["usage"]["cost"])
                total += cost
                report = {"sample": item["id"], "http": response.status_code,
                          "seconds": round(elapsed, 2), "cost_usd": cost,
                          "tokens": data["usage"].get("total_tokens"),
                          "returned_model": data.get("model"), "valid": False}
                if response_matches(response.status_code, data):
                    try:
                        content = data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        proposals = parsed.get("proposals", [])
                        report["raw_exact_quotes"] = sum(
                            type(p.get("start")) is int and isinstance(p.get("quote"), str)
                            and item["input"]["text"][p["start"]:p["start"]+len(p["quote"])] == p["quote"]
                            for p in proposals)
                        outcome = _cited_outcome(content, FrozenSource(item["input"]), {}, "openrouter", MODEL)
                        report.update(valid=True, proposals=len(outcome["extraction"]["proposals"]),
                                      finish_reason=data["choices"][0].get("finish_reason"))
                    except (ValueError, TypeError, KeyError) as exc:
                        report["failure"] = getattr(exc, "code", type(exc).__name__)
                reports.append(report)
                print(json.dumps(report), flush=True)
                if not response_matches(response.status_code, data) or total >= .05:
                    break
            finally:
                admission.release_reserved()  # Never clears unknown or settled.
        save_receipt(journal, directory, "report", {"results": reports, "total_usd": total})
        print(json.dumps({"stage": "complete", "success": sum(r["valid"] for r in reports),
                          "attempts": len(reports), "retries": 0, "total_usd": total}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"stage": "stopped", "error_type": type(exc).__name__,
                          "code": getattr(exc, "code", None)}), flush=True)
        raise SystemExit(1)
