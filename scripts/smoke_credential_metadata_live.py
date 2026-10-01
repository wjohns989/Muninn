"""Probe local authenticated credential metadata without printing record details."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from urllib.error import HTTPError, URLError

from scripts.smoke_secure_search_live import _local_auth_token, _request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="openrouter")
    parser.add_argument("--base", default="http://127.0.0.1:42069")
    args = parser.parse_args()
    if not args.base.startswith("http://127.0.0.1:") or not 1 <= len(args.query) <= 64:
        parser.error("Only a bounded query to the local loopback service is permitted")
    token = _local_auth_token()
    if not token:
        print(json.dumps({"status": "auth_unavailable"}), file=sys.stderr)
        return 2
    try:
        response = _request(args.base, token, "/credentials/agent-search",
                            body={"query": args.query, "limit": 20})
        rows = response["data"]
        if not isinstance(rows, list):
            raise ValueError("Invalid metadata response")
        print(json.dumps({"status": "ok", "matches": len(rows),
                          "origins": dict(Counter(row.get("origin", "manual") for row in rows))},
                         sort_keys=True))
        return 0
    except (HTTPError, URLError, ValueError, KeyError, TimeoutError) as exc:
        code = exc.code if isinstance(exc, HTTPError) else type(exc).__name__
        print(json.dumps({"status": "probe_error", "error_type": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
