"""
LongMemEval Adapter — Muninn external benchmark grounding (Phase 15).

Maps the LongMemEval benchmark (Wang et al., 2024) single-session QA format
to Muninn's search interface to establish nDCG@10 / Recall@10 baselines.

Dataset: https://github.com/xiaowu0162/LongMemEval
Paper:   "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
         Wang et al., 2024 (EMNLP)

Format — each JSONL line is a question case:
  {
    "question_id":     str,
    "question_type":   str,   # single-session-qa | multi-session-qa | temporal | etc.
    "question":        str,
    "expected_answer": str,
    "question_date":   str,   # ISO 8601
    "sessions": [
      {
        "session_id":    str,
        "date":          str,
        "conversation":  [
          {"role": "user"|"assistant", "content": str, "turn_id": int},
          ...
        ]
      }
    ]
  }

Muninn mapping:
  - Each conversation turn    → one `add_memory` call (content = "[role] text")
  - The question              → `search_memory` query
  - Ground-truth relevance    → heuristic word-overlap against expected_answer
  - Metrics                   → nDCG@10 and Recall@10

Evaluation isolation:
  Each question case uses a unique `user_id` derived from `question_id` so
  memories from one case cannot pollute another case's retrieval results.
  All ingested data is deleted after each case evaluation (default: cleanup=True).

Usage:
  # Evaluate against real LongMemEval dataset
  python -m eval.longmemeval_adapter \\
    --dataset eval/data/longmemeval_oracle.jsonl \\
    --server-url http://localhost:42069 \\
    --auth-token YOUR_TOKEN \\
    --k 10 \\
    --output eval/reports/longmemeval_baseline.json

  # Synthetic self-test (no dataset needed — verifies end-to-end plumbing)
  python -m eval.longmemeval_adapter --selftest --server-url http://localhost:42069

  # Programmatic use
  from eval.longmemeval_adapter import run_adapter, AdapterReport
  report = run_adapter(dataset_path=Path("longmemeval.jsonl"), auth_token="token")
  print(report.mean_ndcg_at_k)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urlrequest

logger = logging.getLogger("Muninn.LongMemEval")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """A single turn from a LongMemEval session conversation."""
    role: str          # "user" or "assistant"
    content: str
    turn_id: int
    session_id: str


@dataclass
class QuestionCase:
    """One LongMemEval evaluation case (question + associated sessions)."""
    question_id: str
    question_type: str
    question: str
    expected_answer: str
    question_date: str
    sessions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CaseResult:
    """Retrieval metrics for a single evaluated question case."""
    question_id: str
    question_type: str
    ndcg_at_k: float
    recall_at_k: float
    latency_ms: float
    ranked_ids: List[str] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)


@dataclass
class AdapterReport:
    """Aggregate LongMemEval evaluation report for Muninn."""
    dataset_path: str
    server_url: str
    k: int
    total_cases: int
    evaluated_cases: int
    skipped_cases: int
    mean_ndcg_at_k: float
    mean_recall_at_k: float
    p50_latency_ms: float
    p95_latency_ms: float
    by_question_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    results: List[CaseResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_dataset(path: Path) -> List[QuestionCase]:
    """
    Parse a LongMemEval JSONL file into a list of QuestionCase objects.

    Malformed lines are logged and skipped; the remainder are returned.
    """
    cases: List[QuestionCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s line %d: %s", path, line_no, exc)
                continue
            if not isinstance(obj, dict):
                logger.warning("Skipping non-object at %s line %d", path, line_no)
                continue
            sessions = obj.get("sessions")
            cases.append(QuestionCase(
                question_id=str(obj.get("question_id", f"q{line_no}")),
                question_type=str(obj.get("question_type", "unknown")),
                question=str(obj.get("question", "")),
                expected_answer=str(obj.get("expected_answer", "")),
                question_date=str(obj.get("question_date", "")),
                sessions=sessions if isinstance(sessions, list) else [],
            ))
    return cases


def extract_turns(case: QuestionCase) -> List[ConversationTurn]:
    """Extract all conversation turns from a QuestionCase's sessions."""
    turns: List[ConversationTurn] = []
    for session in case.sessions:
        session_id = str(session.get("session_id") or uuid.uuid4().hex[:12])
        for turn in session.get("conversation", []):
            role = str(turn.get("role", "user")).strip().lower()
            content = str(turn.get("content", "")).strip()
            turn_id = int(turn.get("turn_id", 0))
            if content:
                turns.append(ConversationTurn(
                    role=role,
                    content=content,
                    turn_id=turn_id,
                    session_id=session_id,
                ))
    return turns


def stable_turn_id(turn: ConversationTurn, ordinal: int) -> str:
    """Derive a stable, unique identifier for a conversation turn."""
    return f"lme_{turn.session_id}_{turn.turn_id:04d}_{ordinal}_{turn.role[:3]}"


def identify_relevant_turns(
    turns: List[ConversationTurn],
    expected_answer: str,
) -> List[str]:
    """
    Identify which turns contain the ground-truth answer using word overlap.

    Heuristic:
    - Short answers (< 3 meaningful words): substring containment check.
    - Longer answers: a turn is relevant when ≥ 40% of the answer's meaningful
      words (length > 2) appear in the turn's content (case-insensitive).

    Returns a list of stable turn IDs (not memory IDs — mapping happens later).
    """
    if not expected_answer.strip():
        return []

    answer_words = [w.lower() for w in expected_answer.split() if len(w) > 2]
    answer_word_set = set(answer_words)

    if len(answer_word_set) < 3:
        # Very short or trivial answer: use substring containment
        return [
            stable_turn_id(t, i)
            for i, t in enumerate(turns)
            if expected_answer.lower() in t.content.lower()
        ]

    threshold = max(2, math.ceil(0.40 * len(answer_word_set)))
    return [
        stable_turn_id(t, i)
        for i, t in enumerate(turns)
        if len(answer_word_set & {w.lower() for w in t.content.split() if len(w) > 2}) >= threshold
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (inline — avoids hard dependency on eval.metrics import)
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_k(relevant: set, ranked: List[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _ndcg_at_k(relevant: set, ranked: List[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0

    def _discount(pos: int) -> float:
        return 1.0 / math.log2(pos + 1)

    dcg = sum(
        _discount(i)
        for i, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    idcg = sum(_discount(i) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def _percentile(values: List[float], pct: float) -> float:
    """Compute a percentile value from a list using the nearest-rank method."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, min(math.ceil(pct / 100.0 * len(sorted_v)) - 1, len(sorted_v) - 1))
    return sorted_v[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Muninn HTTP client (thin, stdlib-only)
# ─────────────────────────────────────────────────────────────────────────────

class MuninnHTTPClient:
    """
    Minimal Muninn REST client using only Python stdlib (urllib).

    Implements the subset of the API needed for ingestion and search:
      POST /add    — ingest one memory
      POST /search — retrieve top-k memories for a query
      POST /delete_all — remove all memories for a user_id (cleanup)
    """

    def __init__(self, server_url: str, auth_token: str, timeout: float = 30.0):
        self.base_url = server_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        self.timeout = int(max(1, timeout))

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers=self._headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Connection error to {url}: {exc.reason}") from exc

    def add(
        self,
        content: str,
        user_id: str,
        metadata: Dict[str, Any],
        namespace: str = "longmemeval",
    ) -> Dict[str, Any]:
        """Ingest one memory and return the server response."""
        return self._post("/add", {
            "content": content,
            "user_id": user_id,
            "namespace": namespace,
            "metadata": metadata,
        })

    def search(
        self,
        query: str,
        user_id: str,
        namespace: str = "longmemeval",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories and return the ranked result list."""
        resp = self._post("/search", {
            "query": query,
            "user_id": user_id,
            "namespace": namespace,
            "limit": limit,
        })
        return resp.get("data", [])  # v3.18.1 fix: server returns list in "data"

    def hunt(
        self,
        query: str,
        user_id: str,
        namespace: str = "longmemeval",
        limit: int = 10,
        depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Perform agentic multi-hop retrieval and return the ranked result list."""
        resp = self._post("/search/hunt", {
            "query": query,
            "user_id": user_id,
            "namespace": namespace,
            "limit": limit,
            "depth": depth,
        })
        return resp.get("data", [])

    def delete_all(self, user_id: str, namespace: str = "longmemeval") -> None:
        """Best-effort cleanup — remove all memories for a given user_id."""
        try:
            self._post("/delete_all", {"user_id": user_id, "namespace": namespace})
        except Exception as exc:
            logger.debug("delete_all cleanup for %s failed (non-fatal): %s", user_id, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter core
# ─────────────────────────────────────────────────────────────────────────────

class LongMemEvalAdapter:
    """
    Evaluates Muninn's retrieval quality against the LongMemEval benchmark.

    Evaluation protocol per question case:
      1. Derive an isolated user_id from question_id — prevents cross-case leakage.
      2. Ingest all conversation turns from the case's sessions via POST /add.
      3. Issue the question as a search query via POST /search or /search/hunt; collect top-k results.
      4. Map ranked search results back to their stable turn IDs.
      5. Compute nDCG@k and Recall@k against heuristically-identified relevant turns.
      6. Optionally delete ingested data (cleanup=True, default).
    """

    def __init__(
        self,
        client: MuninnHTTPClient,
        *,
        k: int = 10,
        limit: Optional[int] = None,
        namespace: str = "longmemeval",
        cleanup: bool = True,
        method: str = "search",
        depth: int = 2,
    ):
        self._client = client
        self.k = k
        self.limit = limit
        self.namespace = namespace
        self.cleanup = cleanup
        self.method = method
        self.depth = depth

    def _user_id(self, question_id: str) -> str:
        """Derive an isolated user_id; capped to prevent oversized identifiers."""
        return f"lme_{question_id[:40]}"

    def _ingest_turns(
        self,
        turns: List[ConversationTurn],
        turn_ids: List[str],
        user_id: str,
    ) -> None:
        """Ingest all conversation turns for a question case."""
        for turn, tid in zip(turns, turn_ids):
            content = f"[{turn.role}] {turn.content}"
            try:
                self._client.add(
                    content=content,
                    user_id=user_id,
                    namespace=self.namespace,
                    metadata={
                        "longmemeval_turn_id": tid,
                        "longmemeval_session_id": turn.session_id,
                        "longmemeval_role": turn.role,
                        "longmemeval_ordinal": str(turn.turn_id),
                    },
                )
            except Exception as exc:
                logger.warning("Failed to ingest turn %s: %s", tid, exc)

    def _map_results_to_turn_ids(
        self,
        results: List[Dict[str, Any]],
        turns: List[ConversationTurn],
        turn_ids: List[str],
    ) -> List[str]:
        """
        Map Muninn search result content back to their stable turn IDs.

        Strategy: match result content against the original turn content using
        an exact substring check on the first 80 characters (avoiding truncation
        artifacts). Unmatched results are omitted from the ranked list.
        """
        ranked: List[str] = []
        for result in results:
            # v3.18.1 fix: server returns memory text in "memory" key, not "content"
            result_content = str(result.get("memory", ""))
            for idx, turn in enumerate(turns):
                # The ingested content was "[role] {turn.content}"
                expected_prefix = f"[{turn.role}] {turn.content[:80]}"
                if result_content.startswith(expected_prefix[:60]):
                    ranked.append(turn_ids[idx])
                    break
        return ranked

    def evaluate_case(self, case: QuestionCase) -> Optional[CaseResult]:
        """
        Evaluate a single LongMemEval question case.

        Returns None when the case has no conversation turns (unevaluable).
        """
        turns = extract_turns(case)
        if not turns:
            logger.debug("Case %s has no turns — skipping", case.question_id)
            return None

        turn_ids = [stable_turn_id(t, i) for i, t in enumerate(turns)]
        relevant_ids = identify_relevant_turns(turns, case.expected_answer)
        user_id = self._user_id(case.question_id)

        try:
            self._ingest_turns(turns, turn_ids, user_id)

            t0 = time.perf_counter()
            try:
                if self.method == "hunt":
                    results = self._client.hunt(
                        query=case.question,
                        user_id=user_id,
                        namespace=self.namespace,
                        limit=self.k,
                        depth=self.depth,
                    )
                else:
                    results = self._client.search(
                        query=case.question,
                        user_id=user_id,
                        namespace=self.namespace,
                        limit=self.k,
                    )
            except Exception as exc:
                logger.error("Search failed for case %s: %s", case.question_id, exc)
                return None
            latency_ms = (time.perf_counter() - t0) * 1000.0

            ranked_ids = self._map_results_to_turn_ids(results, turns, turn_ids)

        finally:
            if self.cleanup:
                self._client.delete_all(user_id, self.namespace)

        relevant_set = set(relevant_ids)
        return CaseResult(
            question_id=case.question_id,
            question_type=case.question_type,
            ndcg_at_k=_ndcg_at_k(relevant_set, ranked_ids, self.k),
            recall_at_k=_recall_at_k(relevant_set, ranked_ids, self.k),
            latency_ms=latency_ms,
            ranked_ids=ranked_ids,
            relevant_ids=relevant_ids,
        )

    def run(self, cases: List[QuestionCase]) -> AdapterReport:
        """Run the full evaluation and return a consolidated AdapterReport."""
        if self.limit is not None:
            cases = cases[: self.limit]

        results: List[CaseResult] = []
        skipped = 0

        for i, case in enumerate(cases, start=1):
            logger.info(
                "[%d/%d] %s  type=%s",
                i, len(cases), case.question_id, case.question_type,
            )
            result = self.evaluate_case(case)
            if result is None:
                skipped += 1
            else:
                results.append(result)

        # ── Aggregate metrics ──────────────────────────────────────────────
        ndcgs = [r.ndcg_at_k for r in results]
        recalls = [r.recall_at_k for r in results]
        latencies = [r.latency_ms for r in results]

        mean_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0
        mean_recall = statistics.mean(recalls) if recalls else 0.0

        # ── Per-question-type breakdown ────────────────────────────────────
        by_type: Dict[str, Dict[str, Any]] = {}
        for r in results:
            qt = r.question_type
            acc = by_type.setdefault(qt, {"count": 0, "_ndcg_sum": 0.0, "_recall_sum": 0.0})
            acc["count"] += 1
            acc["_ndcg_sum"] += r.ndcg_at_k
            acc["_recall_sum"] += r.recall_at_k

        by_type_summary: Dict[str, Dict[str, float]] = {}
        for qt, acc in by_type.items():
            n = acc["count"]
            by_type_summary[qt] = {
                "count": float(n),
                f"mean_ndcg_at_{self.k}": acc["_ndcg_sum"] / n if n else 0.0,
                f"mean_recall_at_{self.k}": acc["_recall_sum"] / n if n else 0.0,
            }

        return AdapterReport(
            dataset_path="(in-memory)",
            server_url=self._client.base_url,
            k=self.k,
            total_cases=len(cases),
            evaluated_cases=len(results),
            skipped_cases=skipped,
            mean_ndcg_at_k=mean_ndcg,
            mean_recall_at_k=mean_recall,
            p50_latency_ms=_percentile(latencies, 50.0),
            p95_latency_ms=_percentile(latencies, 95.0),
            by_question_type=by_type_summary,
            results=results,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic self-test cases (no external dataset required)
# ─────────────────────────────────────────────────────────────────────────────

_SELFTEST_DATASET: List[Dict[str, Any]] = [
    {
        "question_id": "selftest-001",
        "question_type": "single-session-qa",
        "question": "What is the user's favourite programming language?",
        "expected_answer": "Python",
        "question_date": "2025-01-10T10:00:00Z",
        "sessions": [
            {
                "session_id": "sess-st-A",
                "date": "2025-01-09T09:00:00Z",
                "conversation": [
                    {
                        "role": "user",
                        "content": "My favourite programming language is Python.",
                        "turn_id": 1,
                    },
                    {
                        "role": "assistant",
                        "content": "Great choice! Python is very versatile.",
                        "turn_id": 1,
                    },
                    {
                        "role": "user",
                        "content": "I also enjoy TypeScript for frontend work.",
                        "turn_id": 2,
                    },
                    {
                        "role": "assistant",
                        "content": "Both are excellent for their respective domains.",
                        "turn_id": 2,
                    },
                ],
            }
        ],
    },
    {
        "question_id": "selftest-002",
        "question_type": "single-session-qa",
        "question": "What city does the user live in?",
        "expected_answer": "Berlin",
        "question_date": "2025-01-10T10:00:00Z",
        "sessions": [
            {
                "session_id": "sess-st-B",
                "date": "2025-01-09T08:00:00Z",
                "conversation": [
                    {
                        "role": "user",
                        "content": "I live in Berlin, Germany. I love it here.",
                        "turn_id": 1,
                    },
                    {
                        "role": "assistant",
                        "content": "Berlin is a wonderful city with a rich history!",
                        "turn_id": 1,
                    },
                ],
            }
        ],
    },
    {
        "question_id": "selftest-003",
        "question_type": "temporal-reasoning",
        "question": "What database issue did the user investigate?",
        "expected_answer": "Redis queue backlog",
        "question_date": "2025-01-12T10:00:00Z",
        "sessions": [
            {
                "session_id": "sess-st-C",
                "date": "2025-01-11T14:00:00Z",
                "conversation": [
                    {
                        "role": "user",
                        "content": "Spent today investigating a Redis queue backlog affecting production.",
                        "turn_id": 1,
                    },
                    {
                        "role": "assistant",
                        "content": "Redis queue backlogs can have several root causes. "
                                   "Have you checked the consumer lag metrics?",
                        "turn_id": 1,
                    },
                    {
                        "role": "user",
                        "content": "Yes, the consumer lag spiked 10x after the deployment.",
                        "turn_id": 2,
                    },
                ],
            }
        ],
    },
]


def run_selftest(server_url: str, auth_token: str) -> AdapterReport:
    """
    Run a synthetic self-test against a live Muninn server.

    Uses three synthetic question cases with known-good ground truth so the
    adapter pipeline can be validated without the full LongMemEval dataset.
    """
    client = MuninnHTTPClient(server_url, auth_token)
    adapter = LongMemEvalAdapter(client, k=5, cleanup=True, namespace="lme_selftest")
    cases = [
        QuestionCase(
            question_id=c["question_id"],
            question_type=c["question_type"],
            question=c["question"],
            expected_answer=c["expected_answer"],
            question_date=c["question_date"],
            sessions=c["sessions"],
        )
        for c in _SELFTEST_DATASET
    ]
    report = adapter.run(cases)
    report.dataset_path = "(synthetic self-test)"
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Public programmatic entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_adapter(
    dataset_path: Path,
    server_url: str = "http://localhost:42069",
    auth_token: str = "",
    k: int = 10,
    limit: Optional[int] = None,
    cleanup: bool = True,
    namespace: str = "longmemeval",
    method: str = "search",
    depth: int = 2,
) -> AdapterReport:
    """
    Evaluate a LongMemEval JSONL dataset against a running Muninn server.

    Args:
        dataset_path: Path to the LongMemEval JSONL file.
        server_url:   Muninn server base URL.
        auth_token:   Muninn auth token (falls back to MUNINN_AUTH_TOKEN env var).
        k:            Cutoff for nDCG@k and Recall@k.
        limit:        Maximum number of cases to evaluate (None = all).
        cleanup:      Delete ingested data after each case (default: True).
        namespace:    Muninn namespace for evaluation isolation.
        method:       Retrieval method — 'search' or 'hunt'.
        depth:        Expansion depth for 'hunt' method.

    Returns:
        AdapterReport with aggregate and per-case metrics.
    """
    effective_token = auth_token or os.environ.get("MUNINN_AUTH_TOKEN", "")
    cases = parse_dataset(dataset_path)
    client = MuninnHTTPClient(server_url, effective_token)
    adapter = LongMemEvalAdapter(
        client, k=k, limit=limit, cleanup=cleanup, namespace=namespace, method=method, depth=depth
    )
    report = adapter.run(cases)
    report.dataset_path = str(dataset_path)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.longmemeval_adapter",
        description=(
            "Evaluate Muninn retrieval quality against the LongMemEval benchmark. "
            "Reports nDCG@k and Recall@k baselines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", type=Path, default=None,
        metavar="PATH",
        help="Path to LongMemEval JSONL file.",
    )
    p.add_argument(
        "--server-url", default="http://localhost:42069",
        metavar="URL",
        help="Muninn server base URL (default: %(default)s).",
    )
    p.add_argument(
        "--auth-token", default="",
        metavar="TOKEN",
        help="Muninn auth token. Falls back to MUNINN_AUTH_TOKEN env var.",
    )
    p.add_argument(
        "--k", type=int, default=10,
        metavar="K",
        help="Retrieval cutoff for nDCG@k and Recall@k (default: %(default)s).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        metavar="N",
        help="Maximum number of cases to evaluate (default: all).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        metavar="PATH",
        help="Write JSON report to this file path.",
    )
    p.add_argument(
        "--namespace", default="longmemeval",
        metavar="NS",
        help="Muninn namespace for evaluation isolation (default: %(default)s).",
    )
    p.add_argument(
        "--method", default="search", choices=["search", "hunt"],
        help="Retrieval method to evaluate (default: %(default)s).",
    )
    p.add_argument(
        "--depth", type=int, default=2,
        help="Expansion depth for 'hunt' method (default: %(default)s).",
    )
    p.add_argument(
        "--no-cleanup", dest="cleanup", action="store_false", default=True,
        help="Retain ingested data after evaluation (default: delete after each case).",
    )
    p.add_argument(
        "--selftest", action="store_true",
        help="Run a synthetic self-test against the server (no --dataset required).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return p


def _muninn_version() -> str:
    try:
        from muninn.version import __version__
        return __version__
    except Exception:
        return "unknown"


def _print_report(report: AdapterReport) -> None:
    bar = "=" * 62
    k = report.k
    print(f"\n{bar}")
    print(f"  LongMemEval Baseline Report — Muninn v{_muninn_version()}")
    print(bar)
    print(f"  Dataset       : {report.dataset_path}")
    print(f"  Server        : {report.server_url}")
    print(f"  Cases         : {report.evaluated_cases}/{report.total_cases} evaluated"
          f"  ({report.skipped_cases} skipped)")
    print(f"  nDCG@{k:<3}      : {report.mean_ndcg_at_k:.4f}")
    print(f"  Recall@{k:<3}    : {report.mean_recall_at_k:.4f}")
    print(f"  Latency p50   : {report.p50_latency_ms:.1f} ms")
    print(f"  Latency p95   : {report.p95_latency_ms:.1f} ms")
    if report.by_question_type:
        print(f"\n  Breakdown by question type:")
        for qt in sorted(report.by_question_type):
            m = report.by_question_type[qt]
            n = int(m.get("count", 0))
            ndcg = m.get(f"mean_ndcg_at_{k}", 0.0)
            recall = m.get(f"mean_recall_at_{k}", 0.0)
            print(f"    {qt:<32}  n={n:>4}  nDCG={ndcg:.3f}  Recall={recall:.3f}")
    print(f"{bar}\n")


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    auth_token = args.auth_token or os.environ.get("MUNINN_AUTH_TOKEN", "")
    if not auth_token:
        logger.warning(
            "No auth token provided (--auth-token or MUNINN_AUTH_TOKEN). "
            "Requests to Muninn will fail with 401."
        )

    if args.selftest:
        logger.info("Running synthetic self-test against %s …", args.server_url)
        report = run_selftest(args.server_url, auth_token)
    elif args.dataset is not None:
        if not args.dataset.exists():
            logger.error("Dataset file not found: %s", args.dataset)
            raise SystemExit(1)
        report = run_adapter(
            dataset_path=args.dataset,
            server_url=args.server_url,
            auth_token=auth_token,
            k=args.k,
            limit=args.limit,
            cleanup=args.cleanup,
            namespace=args.namespace,
            method=args.method,
            depth=args.depth,
        )
    else:
        logger.error("Provide --dataset <path> or --selftest.")
        raise SystemExit(1)

    _print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report_dict = asdict(report)
        args.output.write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
        logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()