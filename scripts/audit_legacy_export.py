"""Print aggregate, content-free duplicate statistics for legacy JSONL exports."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from muninn.core.maintenance import content_hash, normalize_legacy_record


def audit(path: Path, source: str, baseline_db: Path | None = None) -> dict[str, int]:
    read = invalid = archived = 0
    by_content = defaultdict(set)
    scoped_keys = set()
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            read += 1
            raw = json.loads(line)
            item = normalize_legacy_record(raw, source=source)
            if item is None:
                invalid += 1
                continue
            archived += int(item["archived"])
            digest = content_hash(item["content"])
            scoped = (digest, item["user_id"], item["namespace"],
                      item["metadata"].get("project"), item["archived"])
            by_content[digest].add(scoped)
            scoped_keys.add(scoped)
            if item["metadata"].get("legacy_id"):
                ids.add(item["metadata"]["legacy_id"])
    report = {"read": read, "invalid": invalid, "archived_rows": archived,
            "unique_legacy_ids": len(ids), "distinct_content": len(by_content),
            "distinct_scoped": len(scoped_keys),
            "content_hashes_across_scopes": sum(len(scopes) > 1 for scopes in by_content.values())}
    if baseline_db is not None:
        uri = baseline_db.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            baseline = conn.execute("SELECT content, archived FROM memories").fetchall()
        live = {content_hash(text) for text, is_archived in baseline if not is_archived}
        archived_baseline = {content_hash(text) for text, is_archived in baseline if is_archived}
        report["baseline_rows"] = len(baseline)
        report["overlap_live"] = len(by_content.keys() & live)
        report["overlap_archived"] = len(by_content.keys() & archived_baseline)
        report["new_content_after_baseline"] = len(by_content.keys() - live - archived_baseline)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline-db", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.path, args.source, args.baseline_db), sort_keys=True))


if __name__ == "__main__":
    main()
