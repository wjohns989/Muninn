"""Export the current live state of a legacy Mem0 history database as JSONL.

The source is opened read-only and no deleted memory is revived. The output
path must not already exist; keep exports outside the Muninn repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def live_records(source: Path) -> tuple[list[dict], dict[str, int]]:
    uri = source.resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"Mem0 history integrity check failed: {integrity}")
        events = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        identities = conn.execute("SELECT COUNT(DISTINCT memory_id) FROM history").fetchone()[0]
        state = {}
        for event in conn.execute("SELECT memory_id, event FROM history ORDER BY rowid"):
            memory_id, kind = event
            previous = state.get(memory_id)
            if not memory_id or (
                (kind == "ADD" and previous is not None)
                or (kind in ("UPDATE", "DELETE") and previous not in ("ADD", "UPDATE"))
                or kind not in ("ADD", "UPDATE", "DELETE")
            ):
                raise ValueError("Unexpected Mem0 event order; export requires review")
            state[memory_id] = kind
        rows = conn.execute(
            """WITH latest AS (
                 SELECT rowid AS event_order, *,
                        ROW_NUMBER() OVER (PARTITION BY memory_id ORDER BY rowid DESC) AS position
                 FROM history
               )
               SELECT * FROM latest WHERE position = 1 ORDER BY memory_id"""
        ).fetchall()
        if len(rows) != identities:
            raise ValueError("Mem0 history identity count changed during read")
        deleted = 0
        records = []
        for row in rows:
            if row["event"] == "DELETE" or row["is_deleted"]:
                deleted += 1
                continue
            if row["event"] not in ("ADD", "UPDATE") or not (row["new_memory"] or "").strip():
                raise ValueError("Mem0 live record has an unexpected event or empty text")
            first = conn.execute(
                """SELECT created_at FROM history
                   WHERE memory_id = ? AND event = 'ADD'
                   ORDER BY rowid LIMIT 1""",
                (row["memory_id"],),
            ).fetchone()
            if first is None:
                raise ValueError("Mem0 live record has no ADD event")
            records.append({
                "id": row["memory_id"],
                "content": row["new_memory"],
                "created_at": first["created_at"] or row["created_at"],
                "metadata": {
                    "mem0_actor_id": row["actor_id"],
                    "mem0_role": row["role"],
                    "mem0_updated_at": row["updated_at"],
                },
                "archived": False,
            })
    distinct = {
        hashlib.sha256(" ".join(record["content"].split()).lower().encode("utf-8")).hexdigest()
        for record in records
    }
    return records, {"events": events, "identities": identities,
                     "deleted": deleted, "live": len(records),
                     "distinct_content": len(distinct)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Read-only Mem0 history.db copy")
    parser.add_argument("output", type=Path, help="New JSONL export path")
    parser.add_argument("--expect-events", type=int)
    parser.add_argument("--expect-live", type=int)
    args = parser.parse_args()
    records, report = live_records(args.source)
    if args.expect_events is not None and report["events"] != args.expect_events:
        raise ValueError(f"Event count mismatch: {report['events']} != {args.expect_events}")
    if args.expect_live is not None and report["live"] != args.expect_live:
        raise ValueError(f"Live count mismatch: {report['live']} != {args.expect_live}")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Export must be outside the repository")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
