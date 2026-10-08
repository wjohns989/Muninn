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


def _probe_succeeded(analyze: str | None, wait_auto: bool, details: dict,
                     transcript_pages: int = 0) -> bool:
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
    if transcript_pages and (details.get("transcript_state") != "ready"
                             or details.get("transcript_pages_checked", 0) < 1):
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
    parser.add_argument("--existing-job-id", help="Reuse a completed search job without enqueuing another")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Report search result sizes without fetching text")
    parser.add_argument("--smallest-match", action="store_true",
                        help="Use the smallest returned snapshot for fetch and transcript checks")
    parser.add_argument("--transcript-only", action="store_true",
                        help="Project transcript pages without a synchronous span fetch")
    parser.add_argument("--base", default="http://127.0.0.1:42069")
    parser.add_argument("--deadline-seconds", type=int, default=180)
    parser.add_argument("--analyze", choices=("local", "remote"),
                        help="Optional explicit model-route probe; never prints analysis text")
    parser.add_argument("--wait-auto", action="store_true",
                        help="Wait for the automatically queued analysis job, printing only route and timing")
    parser.add_argument("--transcript-pages", type=int, default=0,
                        help="Check up to this many redacted continuation pages without printing their text (1-20)")
    args = parser.parse_args()
    if not 0 <= args.transcript_pages <= 20:
        parser.error("--transcript-pages must be between 0 and 20")
    if args.transcript_only and args.transcript_pages == 0:
        parser.error("--transcript-only requires --transcript-pages")
    if not args.base.startswith("http://127.0.0.1:"):
        parser.error("Only the local loopback Muninn service is permitted")
    if args.existing_job_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.existing_job_id):
        parser.error("--existing-job-id must be one opaque job identifier")
    token = _local_auth_token()
    if not token:
        print("MUNINN_AUTH_TOKEN is unavailable in this process", file=sys.stderr)
        return 2
    started = time.monotonic()
    stage = "enqueue_search"
    try:
        if args.existing_job_id:
            job_id = args.existing_job_id
            enqueue_ms = 0
        else:
            queued = _request(args.base, token, "/history/secure/search/jobs", body={"query": args.query, "limit": 3})
            enqueue_ms = round((time.monotonic() - started) * 1000)
            job_id = queued["data"]["job_id"]
        state = None
        while time.monotonic() - started < args.deadline_seconds:
            time.sleep(2)
            stage = "poll_search"
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
        if result["matches"]:
            details["first_match_size_bucket_kib"] = result["matches"][0]["size_bucket_kib"]
        if args.metadata_only:
            print(json.dumps(details, sort_keys=True))
            return 0
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
            selected = (min(result["matches"], key=lambda match: match["size_bucket_kib"])
                        if args.smallest_match else result["matches"][0])
            details["selected_size_bucket_kib"] = selected["size_bucket_kib"]
            capability = selected["fetch_capability"]
            if not args.transcript_only:
                stage = "fetch"
                span = _request(
                    args.base, token, "/history/secure/fetch",
                    body={"capability": capability, "max_chars": 500},
                    timeout=180,
                )["data"]
                details["fetch_redaction"] = span["redaction"]
                details["fetch_chars"] = len(span["redacted_text"])
            if args.transcript_pages:
                projection_started = time.monotonic()
                stage = "transcript_start"
                projected = _request(
                    args.base, token, "/history/secure/transcript/start",
                    body={"capability": capability},
                )["data"]
                while (projected["state"] == "pending"
                       and time.monotonic() - started < args.deadline_seconds):
                    time.sleep(2)
                    stage = "transcript_poll"
                    projected = _request(
                        args.base, token, "/history/secure/transcript/poll",
                        body={"capability": capability},
                    )["data"]
                details["transcript_state"] = projected["state"]
                details["transcript_build_ms"] = round((time.monotonic() - projection_started) * 1000)
                coverage = projected.get("coverage")
                if isinstance(coverage, dict):
                    details["transcript_coverage"] = {
                        key: coverage.get(key) for key in
                        ("source_units", "conversational_units", "omitted_units")
                    }
                if projected["state"] == "ready":
                    cursor = projected["cursor"]
                    checked = 0
                    characters = 0
                    while cursor and checked < args.transcript_pages:
                        stage = "transcript_page"
                        page = _request(
                            args.base, token, "/history/secure/transcript/page",
                            body={"cursor": cursor},
                        )["data"]
                        if len(page["redacted_text"]) > 4000:
                            raise ValueError("transcript page exceeded size bound")
                        checked += 1
                        characters += len(page["redacted_text"])
                        cursor = page["next_cursor"]
                    details["transcript_pages_checked"] = checked
                    details["transcript_chars_checked"] = characters
                    details["transcript_more"] = cursor is not None
            if args.analyze:
                model_started = time.monotonic()
                stage = "explicit_analyze"
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
        details["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        print(json.dumps(details, sort_keys=True))
        return 0 if _probe_succeeded(args.analyze, args.wait_auto, details,
                                     args.transcript_pages) else 2
    except (HTTPError, URLError, ValueError, KeyError, TimeoutError) as exc:
        # Avoid printing response bodies, request headers, query, or capabilities.
        code = exc.code if isinstance(exc, HTTPError) else type(exc).__name__
        print(json.dumps({"state": "probe_error", "stage": stage, "error_type": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
