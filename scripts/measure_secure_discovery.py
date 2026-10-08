"""Read-only timing of strict history source discovery (no transcript reads)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from muninn.history.locations import history_homes, history_sources
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.vault import HistoryVault


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--home", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    signatures = SecureHistoryArchive(args.archive_root).latest_source_signatures()
    manifest_seconds = time.perf_counter() - started
    counts = {"candidates": 0, "unchanged_stat": 0, "changed_stat": 0,
              "changed_bytes": 0, "largest_changed_bytes": 0,
              "missing_roots": 0, "excluded": 0, "errors": 0}
    for home in history_homes(args.home):
        for source in history_sources(home):
            if source.provider not in {"codex", "claude_code", "gemini_cli"}:
                continue
            if not source.exists:
                counts["missing_roots"] += 1
                continue
            for item in HistoryVault._source_files(source):
                if item["kind"] != "transcript":
                    counts["excluded"] += 1
                    continue
                try:
                    path = item["path"].resolve(strict=True)
                    stat = path.stat()
                    counts["candidates"] += 1
                    field = "unchanged_stat" if signatures.get(str(path)) == (
                        stat.st_size, stat.st_mtime_ns
                    ) else "changed_stat"
                    counts[field] += 1
                    if field == "changed_stat":
                        counts["changed_bytes"] += stat.st_size
                        counts["largest_changed_bytes"] = max(counts["largest_changed_bytes"], stat.st_size)
                except OSError:
                    counts["errors"] += 1
    print(json.dumps({"manifest_seconds": round(manifest_seconds, 3),
                      "scan_seconds": round(time.perf_counter() - started - manifest_seconds, 3),
                      **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
