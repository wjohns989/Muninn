"""Explicit one-shot synthetic streaming test. No retry or backlog publication."""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

from muninn.history.auto_routing import _local_setting, openrouter_key_status, remote_policy_snapshot
from muninn.history.batch_activation import read_batch_policy
from muninn.history.batch_diagnostic import verify_price
from muninn.history.historical_batch import BatchError, BatchOutbox, MODEL, MODEL_IDENTITIES
from muninn.history.historical_batch_worker import transport as batch_transport
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.remote_accounting import _db
from muninn.history.streaming_diagnostic import StreamingStore, request_body


async def run(args):
    root = Path(_local_setting("MUNINN_DATA_DIR") or Path(__file__).resolve().parents[1] / ".muninn_runtime").resolve(strict=True)
    archive = SecureHistoryArchive(Path(_local_setting("MUNINN_HISTORY_ARCHIVE_DIR") or root / "history_secure_archive"))
    store = StreamingStore(archive, root)
    if args.status:
        record = store.read(args.status)
        with _db(root) as (db, _):
            settled = db.execute("SELECT state FROM remote_admissions WHERE id=?", (args.status,)).fetchone() == ("settled",)
        result = store.stream_summary(record, settled=settled)
    elif args.reconcile:
        result = store.reconcile(args.reconcile)  # Saved receipt only; no model request.
    else:
        parent = BatchOutbox(archive).read(args.submit_parent)
        if parent["state"] not in {"submitted", "terminal_saved"} or not parent["provider_id"]:
            raise BatchError("diagnostic_parent_not_submitted")
        live = await batch_transport("GET", provider_id=parent["provider_id"])
        if (live.get("id") != parent["provider_id"] or live.get("model") not in MODEL_IDENTITIES
                or live.get("endpoint") != "/v1/chat/completions" or live.get("completion_window") != "24h"
                or live.get("status") not in {"validating", "in_progress", "finalizing", "completed"}):
            raise BatchError("diagnostic_parent_no_longer_pending")
        policy, retention = remote_policy_snapshot(root), read_batch_policy(root)
        if not policy.enabled or not retention["enabled"]:
            raise BatchError("diagnostic_consent_revoked")
        body = request_body()
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            reply = await client.get(f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints")
            reply.raise_for_status()
            estimate = verify_price(body, reply.json())  # 1024 output allowance > fixed stream max128.
        ident = store.prepare(parent["id"], policy.generation, retention["generation"],
            openrouter_key_status(policy_root=root), body, kind="streaming")
        print(json.dumps({"stage": "prepared", "diagnostic_id": ident,
            "conservative_estimate_usd": float(estimate), "admission_threshold_usd": .01}), flush=True)
        result = await store.submit_stream(ident, provider_status=lambda: openrouter_key_status(policy_root=root))
    print(json.dumps({"stage": "stream_result", **result}), flush=True)
    return 0 if result["validated_output"] and result["billing_settled"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--submit-parent")
    action.add_argument("--status")
    action.add_argument("--reconcile")
    try:
        return asyncio.run(run(parser.parse_args()))
    except Exception as exc:
        print(json.dumps({"stage": "stream_error", "category": type(exc).__name__,
            "code": str(exc) if isinstance(exc, BatchError) else "stream_blocked", "no_automatic_retry": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
