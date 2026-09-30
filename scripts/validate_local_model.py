"""Exercise one installed Ollama model on real private archive evidence.

The model receives a bounded authenticated span. Neither the span nor its
analysis is printed or persisted. Only a local Ollama route is permitted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import json
import time
import httpx
from pathlib import Path
from types import SimpleNamespace

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_analysis import analyze_secure_hit
from muninn.history import secure_analysis
from muninn.history.secure_archive import SecureHistoryArchive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnose-parser", action="store_true",
                        help="Report only output types/counts if validation fails; never model text")
    args = parser.parse_args()
    os.environ["MUNINN_AUTO_LOCAL_MODEL_HINTS"] = args.model
    index = SecureHistoryBlindIndex(SecureHistoryArchive(args.root))
    search = index.search(args.query, limit=1, max_candidates=20)
    if not search["matches"]:
        print({"status": "no_archived_hit"})
        return 2
    history = SimpleNamespace(_secure_model_window=index._model_window)
    diagnostics = {}
    original_clean = secure_analysis._clean_result
    if args.diagnose_parser:
        def checked_clean(content, **kwargs):
            try:
                return original_clean(content, **kwargs)
            except secure_analysis.ModelOutputInvalid:
                diagnostics["content_chars"] = len(content) if isinstance(content, str) else None
                try:
                    parsed = json.loads(content)
                except (ValueError, TypeError):
                    diagnostics["json_parsed"] = False
                else:
                    diagnostics["json_parsed"] = True
                    diagnostics["root_is_object"] = isinstance(parsed, dict)
                    if isinstance(parsed, dict):
                        diagnostics["exact_fields"] = set(parsed) == {"summary", "decisions", "open_items", "uncertainty"}
                        for key in ("summary", "decisions", "open_items", "uncertainty"):
                            value = parsed.get(key)
                            diagnostics[key + "_is_string"] = isinstance(value, str)
                            diagnostics[key + "_is_list"] = isinstance(value, list)
                            if isinstance(value, list):
                                diagnostics[key + "_count"] = len(value)
                                diagnostics[key + "_strings_only"] = all(isinstance(item, str) for item in value)
                raise
        secure_analysis._clean_result = checked_clean
    start = time.perf_counter()
    try:
        result = asyncio.run(analyze_secure_hit(
            history, search["matches"][0]["fetch_capability"], allow_remote=False,
        ))
    except Exception as exc:
        details = {"status": "error", "error_type": type(exc).__name__}
        if isinstance(exc, httpx.HTTPStatusError):
            details["http_status"] = exc.response.status_code
            body = exc.response.text.lower()
            details["error_flags"] = [word for word in (
                "schema", "grammar", "maxitems", "unsupported", "out of memory", "array",
                "context", "timeout", "loading", "token") if word in body]
        print(details)
        return 1
    print({"status": result["status"], "provider": result.get("provider"),
           "requested_model_used": result.get("model") == args.model,
           "reason": result.get("reason"),
           "parser_diagnostics": diagnostics,
           "seconds": round(time.perf_counter() - start, 2),
           "output_fields": sorted(result.get("analysis", {}))})
    return 0 if result["status"] == "ok" and result.get("model") == args.model else 2


if __name__ == "__main__":
    raise SystemExit(main())
