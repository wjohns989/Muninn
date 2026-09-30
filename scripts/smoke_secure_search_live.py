"""Bounded authenticated live search probe; never prints search text or secrets.

The token is read only from MUNINN_AUTH_TOKEN. This command intentionally
prints metadata/timings, not the query, capability, or fetched transcript span.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _safe_reason(value: object) -> str | None:
    """Report a bounded machine code, never an arbitrary server message."""
    if value is None:
        return None
    return value if isinstance(value, str) and re.fullmatch(r"[a-z_]{1,64}", value) else "other"


def _probe_succeeded(analyze: str | None, wait_auto: bool, details: dict) -> bool:
    if details.get("state") != "succeeded":
        return False
    # A short result limit may intentionally truncate an otherwise healthy
    # archive. Do not present that search as exhaustive, but still verify the
    # returned hit and its independently queued analysis.
    if details.get("complete") is not True and not (
        details.get("truncated") is True
        and details.get("missing") == 0
        and details.get("overflow") == 0
        and details.get("match_count", 0) > 0
    ):
        return False
    if analyze:
        expected = "ollama" if analyze == "local" else "openrouter"
        if (details.get("match_count", 0) < 1 or details.get("analysis_status") != "ok"
                or details.get("analysis_provider") != expected):
            return False
    if wait_auto and (not details.get("analysis_queued")
                      or details.get("auto_analysis_state") != "succeeded"):
        return False
    return True


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


def _request(base: str, token: str, path: str, *, body: dict | None = None,
             timeout: int = 40) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        base + path,
        data=payload,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Boxter")
    parser.add_argument("--base", default="http://127.0.0.1:42069")
    parser.add_argument("--deadline-seconds", type=int, default=180)
    parser.add_argument("--analyze", choices=("local", "remote"),
                        help="Optional explicit model-route probe; never prints analysis text")
    parser.add_argument("--wait-auto", action="store_true",
                        help="Wait for the automatically queued analysis job, printing only route and timing")
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
            "ready": result["ready"], "missing": result["missing"],
            "overflow": result["overflow"], "truncated": result["truncated"],
        }
        if args.wait_auto:
            analysis_id = status.get("analysis_job_id")
            details["analysis_queued"] = bool(analysis_id)
            details["analysis_reason"] = _safe_reason(status.get("analysis_reason"))
            if analysis_id:
                model_started = time.monotonic()
                auto_state = None
                while time.monotonic() - started < args.deadline_seconds:
                    time.sleep(2)
                    auto_status = _request(args.base, token, "/history/secure/analysis/jobs/" + analysis_id)["data"]
                    auto_state = auto_status["state"]
                    if auto_state not in ("pending", "running", "retry"):
                        break
                details["auto_analysis_state"] = auto_state or "deadline"
                details["auto_analysis_provider"] = auto_status.get("provider") if auto_state else None
                details["auto_analysis_model"] = auto_status.get("model") if auto_state else None
                details["auto_analysis_ms"] = round((time.monotonic() - model_started) * 1000)
        if result["matches"]:
            capability = result["matches"][0]["fetch_capability"]
            span = _request(
                args.base, token, "/history/secure/fetch",
                body={"capability": capability, "max_chars": 500},
            )["data"]
            details["fetch_redaction"] = span["redaction"]
            details["fetch_chars"] = len(span["redacted_text"])
            if args.analyze:
                model_started = time.monotonic()
                analyzed = _request(
                    args.base, token, "/history/secure/analyze",
                    body={"capability": capability, "allow_remote": args.analyze == "remote",
                          "prefer_remote": args.analyze == "remote"},
                    timeout=240,
                )["data"]
                details["analysis_status"] = analyzed["status"]
                details["analysis_provider"] = analyzed.get("provider")
                details["analysis_model"] = analyzed.get("model")
                if analyzed["status"] != "ok":
                    details["analysis_reason"] = _safe_reason(analyzed.get("reason"))
                details["analysis_ms"] = round((time.monotonic() - model_started) * 1000)
        print(json.dumps(details, sort_keys=True))
        return 0 if _probe_succeeded(args.analyze, args.wait_auto, details) else 2
    except (HTTPError, URLError, ValueError, KeyError, TimeoutError) as exc:
        # Avoid printing response bodies, request headers, query, or capabilities.
        code = exc.code if isinstance(exc, HTTPError) else type(exc).__name__
        print(json.dumps({"state": "probe_error", "error_type": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
