"""Check a local Muninn service without writing or displaying memory contents.

Usage: python -m scripts.smoke_service_readonly --url http://127.0.0.1:42070
"""

from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlparse

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--query", default="Muninn")
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("--url must be a local HTTP endpoint")
    base = args.url.rstrip("/")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=45.0) as client:
            search_response = client.post(
                f"{base}/search", json={"query": args.query, "limit": 3, "rerank": False},
            )
            search_response.raise_for_status()
            search = search_response.json()
            counts_match = False
            for attempt in range(3):
                health_response = client.get(f"{base}/health")
                health_response.raise_for_status()
                health = health_response.json()
                counts = [health.get(key) for key in ("memory_count", "vector_count", "bm25_size")]
                counts_match = len(set(counts)) == 1 and all(isinstance(n, int) for n in counts)
                if counts_match:
                    break
                if attempt < 2:
                    time.sleep(0.25)
        report = {
            "backend": health.get("backend"),
            "health_status": health.get("status"),
            "counts_match": counts_match,
            "memory_count": health.get("memory_count"),
            "search_success": search.get("success") is True,
            "result_count": len(search.get("data") or []),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        print(json.dumps(report))
        return 0 if report["counts_match"] and report["search_success"] else 1
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
