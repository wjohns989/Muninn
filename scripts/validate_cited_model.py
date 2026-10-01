"""Probe one real encrypted archive hit through Muninn's cited extraction path.

May build an encrypted structured-source cache, but does not publish ordinary
memories or print transcript/model content. Remote mode requires explicit CLI
acknowledgment and the existing whole-source ZDR and managed-budget gates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.insights import Provider
from muninn.history.remote_accounting import status as remote_accounting_status
from muninn.history.remote_policy import read_policy
from muninn.history.secure_analysis import analyze_cited_window
from muninn.history.secure_archive import SecureHistoryArchive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--provider", choices=("ollama", "openrouter"), default="ollama")
    parser.add_argument("--model", help="Installed Ollama model to test")
    parser.add_argument("--acknowledge-private-zdr", action="store_true",
                        help="Allow one screened private archive window through managed ZDR admission")
    parser.add_argument("--no-think", action="store_true",
                        help="Test Ollama think=false for reasoning-capable local models")
    parser.add_argument("--max-output-tokens", type=int,
                        help="Bound local diagnostic generation (1-2048 tokens)")
    args = parser.parse_args()
    if args.provider == "ollama" and not args.model:
        parser.error("--model is required for Ollama")
    if args.provider == "openrouter" and (not args.acknowledge_private_zdr or args.no_think
                                         or args.max_output_tokens is not None):
        parser.error("OpenRouter requires --acknowledge-private-zdr and no Ollama-only options")
    if args.max_output_tokens is not None and not 1 <= args.max_output_tokens <= 2048:
        parser.error("--max-output-tokens must be between 1 and 2048")
    if args.provider == "ollama":
        os.environ["MUNINN_AUTO_LOCAL_MODEL_HINTS"] = args.model
    if args.no_think or args.max_output_tokens is not None or args.provider == "openrouter":
        original_request_body = Provider.request_body

        def bounded_request(provider, messages):
            body = original_request_body(provider, messages)
            if provider.name == "ollama":
                if args.no_think:
                    body["think"] = False
                if args.max_output_tokens is not None:
                    body.setdefault("options", {})["num_predict"] = args.max_output_tokens
            elif provider.name == "openrouter" and args.provider == "openrouter":
                # Live ZDR endpoint metadata confirms this on the preferred
                # GPT-6 Luna route. require_parameters rejects incompatible
                # fallbacks instead of silently dropping the ceiling.
                body["max_completion_tokens"] = 2048
            return body

        Provider.request_body = bounded_request
    started = time.monotonic()
    try:
        archive = SecureHistoryArchive(args.root)
        index = SecureHistoryBlindIndex(archive)
        matches = index.search(args.query, limit=8, max_candidates=40)["matches"]
        if not matches:
            print(json.dumps({"status": "no_archived_hit", "inference_sent": False}))
            return 2
        source = CitedAnalysisSource(archive)
        descriptor = None
        unsupported = 0
        for match in sorted(matches, key=lambda item: item.get("size_bucket_kib", 0)):
            candidate = source.prepare(match["fetch_capability"])
            if candidate is not None and (args.provider == "ollama" or source.remote_input(candidate) is not None):
                descriptor = candidate
                break
            unsupported += 1
        if descriptor is None:
            print(json.dumps({"status": "no_eligible_cited_source", "inference_sent": False,
                              "candidates_checked": unsupported}))
            return 2
        history = SimpleNamespace(data_dir=args.root.parent)
        if args.provider == "openrouter":
            policy = read_policy(history.data_dir, lambda: (False, 1.0, 30.0, False))
            if not policy.enabled:
                print(json.dumps({"status": "remote_consent_disabled", "inference_sent": False}))
                return 2
        outcome = asyncio.run(analyze_cited_window(history, source, descriptor,
                                                   allow_remote=args.provider == "openrouter",
                                                   prefer_remote=args.provider == "openrouter",
                                                   expected_remote_generation=(policy.generation
                                                       if args.provider == "openrouter" else None)))
        report = {"status": outcome["status"], "provider": outcome.get("provider"),
                  "model": outcome.get("model") if args.provider == "openrouter" else None,
                  "requested_model_used": (outcome.get("model") == args.model
                                           if args.provider == "ollama" else None),
                  "reason": outcome.get("reason"),
                  "output_failure": outcome.get("output_failure"),
                  "proposals": len(outcome.get("extraction", {}).get("proposals", [])),
                  "thinking_disabled": args.no_think,
                  "max_output_tokens": args.max_output_tokens,
                  "candidates_checked": unsupported + 1,
                  "elapsed_seconds": round(time.monotonic() - started, 1)}
        if args.provider == "openrouter":
            accounting = remote_accounting_status(history.data_dir)
            report["accounting_state"] = accounting["state"]
            report["unresolved_admissions"] = accounting["unresolved"]
            report["settled_daily_cost_usd"] = accounting["daily_cost_usd"]
        print(json.dumps(report, sort_keys=True))
        return 0 if (report["status"] == "ok" and (args.provider == "openrouter"
                                                  or report["requested_model_used"])
                     and (args.provider != "openrouter" or
                          (report["accounting_state"] == "ready" and
                           report["unresolved_admissions"] == 0))) else 2
    except Exception as exc:
        report = {"status": "error", "error_type": type(exc).__name__}
        if args.provider == "openrouter":
            try:
                accounting = remote_accounting_status(args.root.parent)
                report.update({"accounting_state": accounting["state"],
                               "unresolved_admissions": accounting["unresolved"],
                               "settled_daily_cost_usd": accounting["daily_cost_usd"]})
            except Exception:
                report["accounting_state"] = "unavailable"
        print(json.dumps(report, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
