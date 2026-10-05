"""Explicit local retained-batch opt-in/revocation with a verified policy preimage.

No passphrase or API key is printed or saved. This does not restart a service or
submit a model request; the existing serial service worker observes the policy.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muninn.history.batch_activation import configure_batch, read_batch_policy
from muninn.history.private_acl import create_private_directory, create_private_file
from muninn.history.remote_policy import _paths
from scripts.local_runtime_preflight import inspect_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing runtime directory")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--enable", action="store_true")
    action.add_argument("--disable", action="store_true")
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if args.enable or args.disable:
            state = inspect_runtime(Path(__file__).resolve().parents[1], authenticated=True)
            processes = state["muninn_processes"]
            if (len(processes) != 1 or state["listener_owners"] != [processes[0]["pid"]]
                    or state["health_http"] != 200 or state.get("history_http") != 200):
                raise RuntimeError("Shared service identity unavailable")
            source = _paths(root)[2]
            destination = root / ("batch-policy-preimage-" + uuid.uuid4().hex)
            create_private_directory(destination)
            saved = destination / "policy.sqlite3"
            create_private_file(saved)
            with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as before:
                with sqlite3.connect(saved) as copy:
                    before.backup(copy)
                    if copy.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise RuntimeError("Policy preimage invalid")
            result = configure_batch(root, enabled=args.enable, max_batches=args.max_batches)
        else:
            result = read_batch_policy(root)
        print(json.dumps({"stage": "batch_policy", **result}), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"stage": "error", "category": type(exc).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
