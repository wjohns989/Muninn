"""Audit one real local transcript's parsed tool results without showing content.

This is read-only and prints only aggregate counts. It never prints source paths,
tool inputs, output text, call IDs, or credentials.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from muninn.history.parsers import parse_claude_code, parse_codex, parse_gemini

_PARSERS = {
    "claude_code": parse_claude_code,
    "codex": parse_codex,
    "gemini_cli": parse_gemini,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=_PARSERS, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    source = args.path.resolve()
    if not source.is_file() or source.stat().st_size > args.max_bytes:
        print(json.dumps({"ok": False, "reason": "missing_or_oversized"}))
        return 2
    try:
        session = _PARSERS[args.provider](source.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, TypeError):
        print(json.dumps({"ok": False, "reason": "parse_error"}))
        return 1
    if session is None:
        print(json.dumps({"ok": False, "reason": "no_session"}))
        return 2
    events = [event for turn in session.turns for event in turn.tool_events]
    print(json.dumps({
        "ok": True, "provider": args.provider, "turns": len(session.turns),
        "tool_calls": len(events),
        "results_observed": sum(event.result_observed for event in events),
        "unmatched_results": session.unmatched_tool_results,
        "outcomes": dict(Counter(event.outcome for event in events)),
        "actions": dict(Counter(event.action for event in events)),
        "explicit_exit_codes": sum(event.exit_code is not None for event in events),
        "test_counts_observed": sum(event.tests_passed is not None or event.tests_failed is not None
                                    for event in events),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
