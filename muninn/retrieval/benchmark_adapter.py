"""
Muninn Benchmark Adapter (v1.0.0)
----------------------------------
Interface for quantitative retrieval evaluation.
Loads Ground Truth (GT) from CI replays and compares against live hybrid search.

Measures:
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Latency (p50, p95)
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import math

logger = logging.getLogger("Muninn.BenchmarkAdapter")

class BenchmarkAdapter:
    def __init__(self, memory_engine):
        self.memory = memory_engine
        self.ground_truth: List[Dict[str, Any]] = []

    def load_replay_log(self, path: str):
        """Load benchmark queries and expected IDs from a JSON replay log."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                # Expected format: [{"query": "...", "expected_ids": ["id1", "id2"], "namespace": "..."}]
                self.ground_truth = data
            logger.info("Loaded %d benchmark queries from %s", len(self.ground_truth), path)
        except Exception as e:
            logger.error("Failed to load replay log: %s", e)
            raise

    async def run_benchmark(self, k: int = 5) -> Dict[str, Any]:
        """Execute benchmark and return aggregate metrics."""
        results = []
        latencies = []

        for entry in self.ground_truth:
            query = entry["query"]
            expected = set(entry.get("expected_ids", []))
            namespace = entry.get("namespace", "global")
            
            start_time = time.perf_counter()
            actual_hits = await self.memory.search(
                query=query,
                limit=k,
                namespace=namespace
            )
            latencies.append(time.perf_counter() - start_time)
            
            actual_ids = [hit["id"] for hit in actual_hits]
            
            # Metrics per query
            precision = self._calculate_precision(expected, actual_ids, k)
            recall = self._calculate_recall(expected, actual_ids)
            ndcg = self._calculate_ndcg(expected, actual_ids)
            
            results.append({
                "query": query,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "hit_count": len(actual_hits)
            })

        avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
        avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
        avg_ndcg = sum(r["ndcg"] for r in results) / len(results) if results else 0
        
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(latencies)//2] if latencies else 0
        p95 = sorted_latencies[int(len(latencies)*0.95)] if latencies else 0

        return {
            "metrics": {
                "precision_at_k": avg_precision,
                "recall_at_k": avg_recall,
                "ndcg_at_k": avg_ndcg,
                "p50_latency_ms": p50 * 1000,
                "p95_latency_ms": p95 * 1000,
            },
            "k": k,
            "total_queries": len(results)
        }

    def _calculate_precision(self, expected: set, actual: List[str], k: int) -> float:
        if not actual: return 0.0
        relevant = sum(1 for hit in actual if hit in expected)
        return relevant / k

    def _calculate_recall(self, expected: set, actual: List[str]) -> float:
        if not expected: return 1.0
        relevant = sum(1 for hit in actual if hit in expected)
        return relevant / len(expected)

    def _calculate_ndcg(self, expected: set, actual: List[str]) -> float:
        if not actual: return 0.0
        
        dcg = 0.0
        for i, hit in enumerate(actual):
            if hit in expected:
                dcg += 1.0 / math.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(len(expected), len(actual))):
            idcg += 1.0 / math.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
