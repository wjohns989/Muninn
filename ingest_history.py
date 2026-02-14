#!/usr/bin/env python3
"""
Legacy history ingestion utility using Muninn's discovery/import APIs.

Usage examples:
  python ingest_history.py --discover-only
  python ingest_history.py --provider codex_cli --all-discovered
  python ingest_history.py --source-id src_abc123 --source-id src_def456
  python ingest_history.py --agent serena --all-discovered
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import requests

DEFAULT_BASE_URL = os.environ.get("MUNINN_SERVER_URL", "http://localhost:42069")

AGENT_PROVIDER_MAP = {
    "codex": ["codex_cli"],
    "claude": ["claude_code"],
    "antigravity": ["antigravity_brain"],
    "serena": ["serena_memory"],
    "all": [],
}


def _request_json(method: str, base_url: str, endpoint: str, payload: Dict) -> Dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    response = requests.request(method=method, url=url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    if not data.get("success", False):
        raise RuntimeError(data.get("detail") or data)
    return data.get("data", {})


def _print_sources(sources: List[Dict]) -> None:
    if not sources:
        print("No legacy sources discovered.")
        return
    print(f"Discovered {len(sources)} sources:")
    print(
        "  source_id                         provider           supported  size_bytes  path"
    )
    for item in sources:
        print(
            f"  {str(item.get('source_id', ''))[:32]:<32} "
            f"{str(item.get('provider', ''))[:16]:<16} "
            f"{str(item.get('parser_supported', False)):<9} "
            f"{int(item.get('size_bytes', 0)):<10} "
            f"{item.get('path', '')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and import legacy assistant/MCP memory sources")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Muninn server base URL")
    parser.add_argument("--discover-only", action="store_true", help="Only run discovery, do not import")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --discover-only")
    parser.add_argument("--provider", action="append", default=[], help="Provider filter (repeatable)")
    parser.add_argument("--root", action="append", default=[], help="Additional root directory to scan")
    parser.add_argument("--include-unsupported", action="store_true", help="Include unsupported files in discovery results")
    parser.add_argument("--max-results-per-provider", type=int, default=100)
    parser.add_argument("--source-id", action="append", default=[], help="Selected discovered source ID (repeatable)")
    parser.add_argument("--path", action="append", default=[], help="Explicit local path to import (repeatable)")
    parser.add_argument("--all-discovered", action="store_true", help="Import all parser-supported discovered sources")
    parser.add_argument(
        "--agent",
        choices=["codex", "claude", "antigravity", "serena", "all"],
        help="Legacy shorthand for provider filter",
    )
    parser.add_argument("--project", default=Path.cwd().name)
    parser.add_argument("--namespace", default="global")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--chronological-order",
        choices=["none", "oldest_first", "newest_first"],
        default="none",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--min-chunk", type=int, default=None)
    parser.add_argument("--max-file-size-bytes", type=int, default=None)
    args = parser.parse_args()

    discover_only = args.discover_only or args.dry_run

    provider_filter = list(args.provider)
    if args.agent:
        provider_filter.extend(AGENT_PROVIDER_MAP[args.agent])
    provider_filter = [item for item in provider_filter if item]
    provider_filter = sorted(set(provider_filter))

    try:
        discover_payload = {
            "roots": args.root,
            "providers": provider_filter,
            "include_unsupported": args.include_unsupported,
            "max_results_per_provider": args.max_results_per_provider,
        }
        discovery = _request_json("POST", args.base_url, "/ingest/legacy/discover", discover_payload)
    except Exception as exc:
        print(f"ERROR: discovery failed: {exc}")
        return 1

    print(
        json.dumps(
            {
                "event": discovery.get("event"),
                "total_discovered": discovery.get("total_discovered"),
                "parser_supported": discovery.get("parser_supported"),
                "parser_unsupported": discovery.get("parser_unsupported"),
                "provider_counts": discovery.get("provider_counts", {}),
            },
            indent=2,
        )
    )
    _print_sources(discovery.get("sources", []))

    if discover_only:
        return 0

    selected_source_ids = list(args.source_id)
    selected_paths = list(args.path)
    if args.all_discovered:
        selected_source_ids.extend(
            item["source_id"]
            for item in discovery.get("sources", [])
            if item.get("parser_supported") is True
        )
    selected_source_ids = sorted(set(selected_source_ids))
    selected_paths = sorted(set(selected_paths))

    if not selected_source_ids and not selected_paths:
        print(
            "ERROR: no import selection provided. Use --source-id/--path or --all-discovered."
        )
        return 2

    import_payload = {
        "selected_source_ids": selected_source_ids,
        "selected_paths": selected_paths,
        "roots": args.root,
        "providers": provider_filter,
        "include_unsupported": args.include_unsupported,
        "max_results_per_provider": args.max_results_per_provider,
        "project": args.project,
        "namespace": args.namespace,
        "recursive": args.recursive,
        "chronological_order": args.chronological_order,
        "max_file_size_bytes": args.max_file_size_bytes,
        "chunk_size_chars": args.chunk_size,
        "chunk_overlap_chars": args.chunk_overlap,
        "min_chunk_chars": args.min_chunk,
        "metadata": {
            "source": "legacy_history_import_script",
            "selection_mode": "all_discovered" if args.all_discovered else "manual",
        },
    }

    try:
        result = _request_json("POST", args.base_url, "/ingest/legacy/import", import_payload)
    except Exception as exc:
        print(f"ERROR: import failed: {exc}")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
