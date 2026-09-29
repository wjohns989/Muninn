"""Bounded authenticated live search probe; never prints search text or secrets.

The token is read only from MUNINN_AUTH_TOKEN. This command intentionally
prints metadata/timings, not the query, capability, or fetched transcript span.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _local_auth_token() -> str | None:
    token = os.environ.get("MUNINN_AUTH_TOKEN")
    if token or os.name != "nt":
        return token
    # Desktop tools do not always inherit newly added Windows User variables.
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")
            return value if isinstance(value, str) and value else None
    except FileNotFoundError:
        return None


def _request(base: str, token: str, path: str, *, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        base + path,
        data=payload,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=40) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Boxter")
    parser.add_argument("--base", default="http://127.0.0.1:42069")
    parser.add_argument("--deadline-seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.base.startswith("http://127.0.0.1:"):
        parser.error("Only the local loopback Muninn service is permitted")
    token = _local_auth_token()
    if not token:
        print("MUNINN_AUTH_TOKEN is unavailable in this process", file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        queued = _request(args.base, token, "/history/secure/search/jobs", body={"query": args.query, "limit": 3})
        enqueue_ms = round((time.monotonic() - started) * 1000)
        job_id = queued["data"]["job_id"]
        state = None
        while time.monotonic() - started < args.deadline_seconds:
            time.sleep(2)
            status = _request(args.base, token, "/history/secure/search/jobs/" + job_id)["data"]
            state = status["state"]
            if state not in ("pending", "running", "retry"):
                break
        if state != "succeeded":
            print(json.dumps({"state": state or "deadline", "enqueue_ms": enqueue_ms,
                              "elapsed_ms": round((time.monotonic() - started) * 1000),
                              "error_code": status.get("error_code") if state else None}))
            return 1
        result = status["result"]
        details: dict[str, object] = {
            "state": state, "enqueue_ms": enqueue_ms,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "total": result["total"], "match_count": len(result["matches"]),
            "complete": result["complete"],
        }
        if result["matches"]:
            span = _request(
                args.base, token, "/history/secure/fetch",
                body={"capability": result["matches"][0]["fetch_capability"], "max_chars": 500},
            )["data"]
            details["fetch_redaction"] = span["redaction"]
            details["fetch_chars"] = len(span["redacted_text"])
        print(json.dumps(details, sort_keys=True))
        return 0
    except (HTTPError, URLError, ValueError, KeyError, TimeoutError) as exc:
        # Avoid printing response bodies, request headers, query, or capabilities.
        code = exc.code if isinstance(exc, HTTPError) else type(exc).__name__
        print(json.dumps({"state": "probe_error", "error_type": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
