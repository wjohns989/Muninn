"""Exercise the authenticated live history-analysis route without printing content.

Only real indexed archive hits are accepted. Remote inference requires an
explicit CLI selection as well as the service's persistent ZDR opt-in and cap.
No transcript, analysis text, key, or capability is written to stdout.
"""

from __future__ import annotations

import argparse
import json
import os

import httpx

from muninn.history.secure_analysis import _remote_eligible

_BASE = "http://127.0.0.1:42069"


def _token() -> str:
    token = os.environ.get("MUNINN_AUTH_TOKEN", "")
    if token:
        return token
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")[0])
    raise RuntimeError("Muninn local authentication is unavailable")


def _post(client: httpx.Client, route: str, payload: dict) -> dict:
    response = client.post(f"{_BASE}{route}", json=payload)
    response.raise_for_status()
    if response.headers.get("cache-control", "").lower() != "no-store":
        raise RuntimeError("Secure history response lacks no-store policy")
    body = response.json()
    if body.get("success") is not True or not isinstance(body.get("data"), dict):
        raise RuntimeError("Secure history response is invalid")
    return body["data"]


def check(query: str, *, remote: bool) -> dict:
    headers = {"Authorization": f"Bearer {_token()}"}
    with httpx.Client(timeout=210.0, trust_env=False, headers=headers) as client:
        found = _post(client, "/history/secure/search", {"query": query, "limit": 5})
        candidates = found.get("matches") or []
        if not isinstance(candidates, list):
            raise RuntimeError("Secure history search returned invalid matches")
        capability = None
        span_chars = 0
        for item in candidates:
            candidate = item.get("fetch_capability") if isinstance(item, dict) else None
            if not isinstance(candidate, str):
                continue
            fetched = _post(client, "/history/secure/fetch",
                            {"capability": candidate, "max_chars": 3000})
            span = fetched.get("redacted_text")
            if (isinstance(span, str) and len(span) >= 100
                    and (not remote or _remote_eligible(span, allow_remote=True))):
                capability = candidate
                span_chars = len(span)
                break
        if capability is None:
            return {"status": "no_eligible_real_hit", "search_matches": len(candidates),
                    "provider": None}
        outcome = _post(client, "/history/secure/analyze", {
            "capability": capability, "allow_remote": remote,
            "prefer_remote": remote,
        })
    provider = outcome.get("provider")
    expected = "openrouter" if remote else "ollama"
    analysis = outcome.get("analysis") or {}
    return {
        "status": outcome.get("status"), "provider": provider,
        "expected_provider": expected, "route_matched": provider == expected,
        "model": outcome.get("model") if provider == expected else None,
        "search_matches": len(candidates), "span_chars": span_chars,
        "summary_chars": len(analysis.get("summary") or ""),
        "decisions": len(analysis.get("decisions") or []),
        "open_items": len(analysis.get("open_items") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True,
                        help="Search term for a real archived transcript; do not use a secret")
    parser.add_argument("--provider", required=True, choices=("ollama", "openrouter"))
    args = parser.parse_args()
    try:
        report = check(args.query, remote=args.provider == "openrouter")
    except httpx.HTTPStatusError as exc:
        report = {"status": "http_error", "http_status": exc.response.status_code,
                  "provider": None}
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
        report = {"status": "error", "error_type": type(exc).__name__, "provider": None}
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "ok" and report.get("route_matched") else 2


if __name__ == "__main__":
    raise SystemExit(main())
