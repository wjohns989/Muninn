"""Exercise one installed Ollama model on real private archive evidence.

The model receives a bounded authenticated span. Neither the span nor its
analysis is printed or persisted. Only a local Ollama route is permitted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_analysis import analyze_secure_hit
from muninn.history.secure_archive import SecureHistoryArchive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    os.environ["MUNINN_AUTO_LOCAL_MODEL_HINTS"] = args.model
    index = SecureHistoryBlindIndex(SecureHistoryArchive(args.root))
    search = index.search(args.query, limit=1, max_candidates=20)
    if not search["matches"]:
        print({"status": "no_archived_hit"})
        return 2
    history = SimpleNamespace(secure_fetch_span=index.fetch_span)
    start = time.perf_counter()
    try:
        result = asyncio.run(analyze_secure_hit(
            history, search["matches"][0]["fetch_capability"], allow_remote=False,
        ))
    except Exception as exc:
        print({"status": "error", "error_type": type(exc).__name__})
        return 1
    print({"status": result["status"], "provider": result.get("provider"),
           "requested_model_used": result.get("model") == args.model,
           "reason": result.get("reason"),
           "seconds": round(time.perf_counter() - start, 2),
           "output_fields": sorted(result.get("analysis", {}))})
    return 0 if result["status"] == "ok" and result.get("model") == args.model else 2


if __name__ == "__main__":
    raise SystemExit(main())
