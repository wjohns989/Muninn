"""Profile a real encrypted-history lookup without displaying private results.

This diagnostic reads the local archive/index only. Query terms are supplied by
the operator and should not themselves be secrets because they appear in argv.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    index = SecureHistoryBlindIndex(SecureHistoryArchive(args.root))
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    result = index.search(args.query, limit=3, max_candidates=20)
    profile.disable()
    print({"seconds": round(time.perf_counter() - start, 2),
           "matches": len(result["matches"]), "total": result["total"],
           "ready": result["ready"], "truncated": result["truncated"]})
    output = io.StringIO()
    pstats.Stats(profile, stream=output).sort_stats("cumtime").print_stats(18)
    print(output.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
