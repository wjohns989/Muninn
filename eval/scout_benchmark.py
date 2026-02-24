"""
Scout Accuracy Evaluation — Benchmark hunt() vs search() (Phase 19).

This script runs the LongMemEval adapter twice:
1. Standard 'search' mode (Hybrid multi-signal)
2. Agentic 'hunt' mode (Multi-hop graph/chain expansion)

It then computes the delta in nDCG and Recall to quantify the Scout ROI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from eval.longmemeval_adapter import run_adapter, AdapterReport

logger = logging.getLogger("Muninn.ScoutEval")


def compare_reports(search_report: AdapterReport, hunt_report: AdapterReport) -> Dict[str, Any]:
    """Compute deltas and comparison stats between search and hunt."""
    ndcg_delta = hunt_report.mean_ndcg_at_k - search_report.mean_ndcg_at_k
    recall_delta = hunt_report.mean_recall_at_k - search_report.mean_recall_at_k
    
    ndcg_pct = (ndcg_delta / search_report.mean_ndcg_at_k * 100) if search_report.mean_ndcg_at_k > 0 else 0
    recall_pct = (recall_delta / search_report.mean_recall_at_k * 100) if search_report.mean_recall_at_k > 0 else 0

    comparison = {
        "metrics": {
            "search": {
                "ndcg": search_report.mean_ndcg_at_k,
                "recall": search_report.mean_recall_at_k,
                "p50_ms": search_report.p50_latency_ms,
            },
            "hunt": {
                "ndcg": hunt_report.mean_ndcg_at_k,
                "recall": hunt_report.mean_recall_at_k,
                "p50_ms": hunt_report.p50_latency_ms,
            },
            "delta": {
                "ndcg": ndcg_delta,
                "recall": recall_delta,
                "ndcg_pct": ndcg_pct,
                "recall_pct": recall_pct,
                "latency_multiplier": hunt_report.p50_latency_ms / search_report.p50_latency_ms if search_report.p50_latency_ms > 0 else 0,
            }
        },
        "by_type_delta": {}
    }

    # Compare by question type
    stypes = set(search_report.by_question_type.keys()) | set(hunt_report.by_question_type.keys())
    for qt in stypes:
        s_stats = search_report.by_question_type.get(qt, {})
        h_stats = hunt_report.by_question_type.get(qt, {})
        
        k = search_report.k
        s_ndcg = s_stats.get(f"mean_ndcg_at_{k}", 0.0)
        h_ndcg = h_stats.get(f"mean_ndcg_at_{k}", 0.0)
        
        comparison["by_type_delta"][qt] = {
            "ndcg_delta": h_ndcg - s_ndcg,
            "count": h_stats.get("count", s_stats.get("count", 0))
        }

    return comparison


def print_comparison(comp: Dict[str, Any], k: int):
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  Scout Evaluation — hunt() vs search() @ k={k}")
    print(bar)
    
    m = comp["metrics"]
    print(f"  Metric        Search      Hunt        Delta       % Change")
    print(f"  ndcg@{k:<3}      {m['search']['ndcg']:.4f}      {m['hunt']['ndcg']:.4f}      {m['delta']['ndcg']:+.4f}      {m['delta']['ndcg_pct']:+.1f}%")
    print(f"  recall@{k:<3}    {m['search']['recall']:.4f}      {m['hunt']['recall']:.4f}      {m['delta']['recall']:+.4f}      {m['delta']['recall_pct']:+.1f}%")
    print(f"  p50 Latency   {m['search']['p50_ms']:>6.1f}ms   {m['hunt']['p50_ms']:>6.1f}ms   x{m['delta']['latency_multiplier']:.1f}")
    
    print(f"\n  ROI Analysis by Question Type (nDCG delta):")
    for qt, delta_info in sorted(comp["by_type_delta"].items(), key=lambda x: x[1]["ndcg_delta"], reverse=True):
        print(f"    {qt:<32}  n={int(delta_info['count']):>3}  {delta_info['ndcg_delta']:+.4f}")
    print(f"{bar}\n")


def main():
    parser = argparse.ArgumentParser(description="Scout hunt() vs search() accuracy comparison")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to LongMemEval dataset")
    parser.add_argument("--server-url", default="http://localhost:42069", help="Muninn server URL")
    parser.add_argument("--auth-token", default="", help="Muninn auth token")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--k", type=int, default=10, help="Recall cutoff")
    parser.add_argument("--depth", type=int, default=2, help="Hunt expansion depth")
    parser.add_argument("--output", type=Path, help="Write comparison JSON to path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    auth_token = args.auth_token or os.environ.get("MUNINN_AUTH_TOKEN", "")

    print(f"[*] Starting 'search' baseline pass...")
    search_report = run_adapter(
        dataset_path=args.dataset,
        server_url=args.server_url,
        auth_token=auth_token,
        k=args.k,
        limit=args.limit,
        method="search",
        namespace="scout_eval_search"
    )

    print(f"[*] Starting 'hunt' evaluation pass (depth={args.depth})...")
    hunt_report = run_adapter(
        dataset_path=args.dataset,
        server_url=args.server_url,
        auth_token=auth_token,
        k=args.k,
        limit=args.limit,
        method="hunt",
        depth=args.depth,
        namespace="scout_eval_hunt"
    )

    comparison = compare_reports(search_report, hunt_report)
    print_comparison(comparison, args.k)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison["search_report"] = search_report.__dict__
        comparison["hunt_report"] = hunt_report.__dict__
        args.output.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        print(f"[*] Comparison report written to {args.output}")


if __name__ == "__main__":
    main()