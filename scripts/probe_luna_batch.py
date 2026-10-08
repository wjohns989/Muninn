"""Explicit operator-only, two-request synthetic batch; poll never submits.

Uses the existing canonical archive unlock and managed spending settings. No
passphrase/key/configuration is printed, and the existing backlog is untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from muninn.history.auto_routing import _local_setting, openrouter_key_status, remote_policy_snapshot
from muninn.history.batch_activation import read_batch_policy
from muninn.history.batch_diagnostic import DiagnosticStore, request_body, verify_price
from muninn.history.historical_batch import BatchError, BatchOutbox, MODEL
from muninn.history.historical_batch_worker import transport
from muninn.history.secure_archive import SecureHistoryArchive


async def run(args):
    repo = Path(__file__).resolve().parents[1]
    root = Path(_local_setting("MUNINN_DATA_DIR") or repo / ".muninn_runtime").resolve(strict=True)
    archive_root = Path(_local_setting("MUNINN_HISTORY_ARCHIVE_DIR") or root / "history_secure_archive").resolve(strict=True)
    archive = SecureHistoryArchive(archive_root)
    store = DiagnosticStore(archive, root)
    if args.status:
        result = store.summary(store.read(args.status))
    elif args.poll:
        if args.watch and time.time() >= store.read(args.poll)["created_at"] + 86400:
            print(json.dumps({"stage": "diagnostic_deadline_reached", "diagnostic_id": args.poll,
                "no_automatic_retry": True}), flush=True)
            return 2
        result = await store.poll(args.poll, transport)
    else:
        parent = BatchOutbox(archive).read(args.submit_parent)
        if parent["state"] != "submitted" or not parent["provider_id"]:
            raise BatchError("diagnostic_parent_not_submitted")
        live_parent = await transport("GET", provider_id=parent["provider_id"])
        if (live_parent.get("id") != parent["provider_id"] or live_parent.get("model") not in {MODEL, "openai/gpt-6-luna-pro-20260922"}
                or live_parent.get("endpoint") != "/v1/chat/completions"
                or live_parent.get("status") not in {"validating", "in_progress", "finalizing"}
                or live_parent.get("completion_window") != "24h"):
            raise BatchError("diagnostic_parent_no_longer_pending")
        policy, retention = remote_policy_snapshot(root), read_batch_policy(root)
        if not policy.enabled or not retention["enabled"]:
            raise BatchError("diagnostic_consent_revoked")
        body = request_body()
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints")
            response.raise_for_status()
            estimate = verify_price(body, response.json())
        status = openrouter_key_status(policy_root=root)
        ident = store.prepare(parent["id"], policy.generation, retention["generation"], status, body)
        print(json.dumps({"stage": "prepared", "diagnostic_id": ident, "requests": 2,
            "conservative_estimate_usd": float(estimate), "admission_ceiling_usd": .01}), flush=True)
        result = await store.submit(ident, transport,
                                   provider_status=lambda: openrouter_key_status(policy_root=root))
    print(json.dumps({"stage": "diagnostic_status", **result}), flush=True)
    if args.watch:
        if not args.poll:
            raise BatchError("diagnostic_watch_requires_poll")
        record = store.read(args.poll)
        deadline = record["created_at"] + 86400
        previous = (result["provider_state"], result["completed"], result["failed"])
        while result["local_state"] != "terminal_saved" and time.time() < deadline:
            await asyncio.sleep(min(60, max(0, deadline - time.time())))
            if time.time() >= deadline:
                break
            result = await store.poll(args.poll, transport)
            current = (result["provider_state"], result["completed"], result["failed"])
            if current != previous or result["local_state"] == "terminal_saved":
                print(json.dumps({"stage": "diagnostic_status", **result}), flush=True)
            previous = current
        if result["local_state"] != "terminal_saved":
            print(json.dumps({"stage": "diagnostic_deadline_reached", "diagnostic_id": args.poll,
                "no_automatic_retry": True}), flush=True)
            return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--submit-parent", help="Explicit one-shot diagnostic authority for this owned batch ID")
    action.add_argument("--poll", help="Known local diagnostic ID; GET only, never resubmit")
    action.add_argument("--status", help="Read retained local diagnostic status only, no network or mutation")
    parser.add_argument("--watch", action="store_true", help="With --poll, GET every minute until terminal or 24-hour deadline")
    args = parser.parse_args()
    if args.watch and not args.poll:
        parser.error("--watch requires --poll")
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        print(json.dumps({"stage": "diagnostic_error", "category": type(exc).__name__,
            "code": str(exc) if isinstance(exc, BatchError) else "diagnostic_blocked",
            "no_automatic_retry": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
