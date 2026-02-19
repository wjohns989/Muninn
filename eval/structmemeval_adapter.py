"""
StructMemEval Adapter — structured/factoid memory recall benchmark for Muninn.

StructMemEval evaluates a memory system's ability to store and retrieve
*specific structured facts* — names, dates, numbers, entities, and short
enumerable lists — as opposed to LongMemEval's conversational QA focus.
This makes the two benchmarks complementary: a system that scores well on
both has demonstrated recall across *both* prose-context and fact-structured
memory retrieval modes.

Dataset format (JSONL, one JSON object per line):
    {
        "case_id":              str,      # unique identifier
        "question":             str,      # natural-language question
        "expected_answer":      str,      # ground-truth answer string
        "answer_type":          str,      # "string" | "number" | "entity" | "list"
        "memories":             [str],    # list of memory strings to ingest
        "relevant_memory_index": int      # 0-based index into memories[] of
                                          # the memory that contains the answer
    }

Metrics:
    - Exact Match (EM): 1 if top-1 retrieved memory contains the exact answer token
    - Token F1: harmonic mean of token-precision and token-recall between the
                answer string and the concatenated retrieved memory content
    - MRR@k: Mean Reciprocal Rank at k (position of first relevant memory)

Usage:
    # Run selftest (no server required):
    python eval/structmemeval_adapter.py --selftest

    # Run against a real dataset with a live Muninn server:
    python eval/structmemeval_adapter.py --dataset path/to/dataset.jsonl \\
        --server-url http://localhost:42069 --auth-token <token>

    # Write JSON report:
    python eval/structmemeval_adapter.py --dataset dataset.jsonl \\
        --output report.json --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("structmemeval")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class StructCase:
    """A single StructMemEval benchmark case."""

    case_id: str
    question: str
    expected_answer: str
    answer_type: str  # "string" | "number" | "entity" | "list"
    memories: list[str]
    relevant_memory_index: int


@dataclass
class CaseResult:
    """Evaluation result for one StructMemEval case."""

    case_id: str
    answer_type: str
    exact_match: float       # 0.0 or 1.0
    token_f1: float          # 0.0–1.0
    mrr_at_k: float          # 0.0–1.0 based on rank of relevant memory
    rank_of_relevant: int    # 1-based rank; k+1 if not in top-k
    latency_ms: float | None = None
    error: str | None = None


@dataclass
class AdapterReport:
    """Aggregate report from a StructMemEval run."""

    total_cases: int
    evaluated_cases: int
    skipped_cases: int
    mean_exact_match: float
    mean_token_f1: float
    mean_mrr_at_k: float
    k: int
    by_answer_type: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    case_results: list[CaseResult]
    errors: list[str]


# ---------------------------------------------------------------------------
# Metric functions (inline — no external dependencies)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace/punctuation into tokens."""
    import re
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def exact_match(expected: str, retrieved_content: str) -> float:
    """
    Return 1.0 if *expected* appears verbatim (case-insensitive) in
    *retrieved_content*, else 0.0.

    For ``answer_type="list"``, the expected string is treated as a
    comma-separated list; EM requires every item to appear in the content.
    """
    exp_lower = expected.strip().lower()
    content_lower = retrieved_content.lower()
    if "," in exp_lower:
        # List answer — all items must appear
        items = [item.strip() for item in exp_lower.split(",") if item.strip()]
        return 1.0 if all(item in content_lower for item in items) else 0.0
    return 1.0 if exp_lower in content_lower else 0.0


def token_f1(expected: str, retrieved_content: str) -> float:
    """
    Compute token-level F1 between *expected* and *retrieved_content*.

    Tokens are lower-cased alphanumeric sequences.  F1 = 2·P·R/(P+R).
    """
    exp_tokens = _tokenize(expected)
    ret_tokens = _tokenize(retrieved_content)
    if not exp_tokens:
        return 1.0 if not ret_tokens else 0.0
    if not ret_tokens:
        return 0.0
    exp_set = set(exp_tokens)
    ret_set = set(ret_tokens)
    common = exp_set & ret_set
    if not common:
        return 0.0
    precision = len(common) / len(ret_set)
    recall = len(common) / len(exp_set)
    return 2.0 * precision * recall / (precision + recall)


def mrr_at_k(rank: int, k: int) -> float:
    """Return 1/rank if rank <= k, else 0.0."""
    if rank < 1 or rank > k:
        return 0.0
    return 1.0 / rank


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile (0–100) over *values* via nearest-rank."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, math.ceil(pct / 100.0 * len(sorted_v)) - 1)
    return sorted_v[idx]


# ---------------------------------------------------------------------------
# Dataset parser
# ---------------------------------------------------------------------------


def parse_dataset(path: str | Path) -> list[StructCase]:
    """
    Parse a StructMemEval JSONL file into a list of :class:`StructCase`.

    Malformed lines are logged and skipped rather than raising an exception
    so that partial datasets remain usable.
    """
    cases: list[StructCase] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s:%d — %s", p.name, lineno, exc)
                continue
            if not isinstance(obj, dict):
                logger.warning("Skipping non-object at %s:%d", p.name, lineno)
                continue
            try:
                case = StructCase(
                    case_id=str(obj["case_id"]),
                    question=str(obj["question"]),
                    expected_answer=str(obj["expected_answer"]),
                    answer_type=str(obj.get("answer_type", "string")),
                    memories=[str(m) for m in obj["memories"]],
                    relevant_memory_index=int(obj["relevant_memory_index"]),
                )
                if not (0 <= case.relevant_memory_index < len(case.memories)):
                    logger.warning(
                        "Case %s: relevant_memory_index %d out of range (len=%d) — skipping",
                        case.case_id,
                        case.relevant_memory_index,
                        len(case.memories),
                    )
                    continue
                cases.append(case)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid case at %s:%d — %s", p.name, lineno, exc)
    return cases


# ---------------------------------------------------------------------------
# HTTP client (stdlib only)
# ---------------------------------------------------------------------------


class MuninnHTTPClient:
    """
    Minimal Muninn server HTTP client using only the Python standard library.

    Methods mirror the MCP tool surface (add_memory, search_memory,
    delete_all_memories) via the REST API exposed by ``server.py``.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:42069",
        auth_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def add(
        self,
        content: str,
        *,
        user_id: str = "structmemeval",
        namespace: str = "structmemeval",
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "content": content,
            "user_id": user_id,
            "namespace": namespace,
        }
        if memory_id:
            body["memory_id"] = memory_id
        return self._post("/memories", body)

    def search(
        self,
        query: str,
        *,
        user_id: str = "structmemeval",
        namespace: str = "structmemeval",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "query": query,
            "user_id": user_id,
            "namespace": namespace,
            "limit": limit,
        }
        result = self._post("/memories/search", body)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("results", result.get("memories", []))
        return []

    def delete_all(
        self,
        *,
        user_id: str = "structmemeval",
        namespace: str = "structmemeval",
    ) -> None:
        try:
            self._post("/memories/delete_all", {"user_id": user_id, "namespace": namespace})
        except Exception as exc:
            logger.warning("delete_all failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Adapter core
# ---------------------------------------------------------------------------


class StructMemEvalAdapter:
    """
    Evaluates Muninn against the StructMemEval benchmark.

    For each case:
    1. Ingest all ``memories`` into an isolated user/namespace scope.
    2. Search for the ``question`` and retrieve top-k results.
    3. Compute Exact Match and Token F1 against ``expected_answer`` using
       the *best* (highest-scoring) retrieved memory text.
    4. Compute MRR@k from the rank of the ``relevant_memory_index`` memory.
    5. Delete all memories to isolate cases.
    """

    def __init__(
        self,
        client: MuninnHTTPClient,
        k: int = 10,
        cleanup: bool = True,
    ) -> None:
        self.client = client
        self.k = k
        self.cleanup = cleanup

    def _stable_memory_id(self, case_id: str, memory_index: int) -> str:
        """Deterministic memory ID for reproducible evaluation."""
        token = f"sme_{case_id}_{memory_index:04d}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, token))

    def _user_ns(self, case_id: str) -> tuple[str, str]:
        """Isolated user/namespace for a case, capped at 40 chars each."""
        safe_id = case_id[:36].replace("/", "_").replace("\\", "_")
        return f"sme_{safe_id}", "structmemeval"

    def evaluate_case(self, case: StructCase) -> CaseResult:
        """Ingest memories, search, compute metrics, clean up, return result."""
        user_id, namespace = self._user_ns(case.case_id)
        ingested_ids: list[str] = []

        try:
            t0 = time.perf_counter()

            # Ingest phase
            for idx, mem_text in enumerate(case.memories):
                mid = self._stable_memory_id(case.case_id, idx)
                self.client.add(
                    mem_text,
                    user_id=user_id,
                    namespace=namespace,
                    memory_id=mid,
                )
                ingested_ids.append(mid)

            # Search phase
            results = self.client.search(
                case.question,
                user_id=user_id,
                namespace=namespace,
                limit=self.k,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Identify the relevant memory ID
            relevant_mid = self._stable_memory_id(case.case_id, case.relevant_memory_index)
            relevant_memory_text = case.memories[case.relevant_memory_index]

            # Extract returned memory IDs and content
            returned_ids: list[str] = []
            returned_texts: list[str] = []
            for r in results:
                rid = str(r.get("id") or r.get("memory_id") or "")
                text = str(r.get("content") or r.get("text") or r.get("memory") or "")
                returned_ids.append(rid)
                returned_texts.append(text)

            # Exact Match & Token F1 — use best match across all returned texts
            all_content = " ".join(returned_texts)
            em = exact_match(case.expected_answer, all_content)
            tf1 = max(
                (token_f1(case.expected_answer, t) for t in returned_texts),
                default=0.0,
            )

            # MRR@k — find rank of relevant memory
            rank = self.k + 1  # assume not found
            for i, rid in enumerate(returned_ids):
                if rid == relevant_mid:
                    rank = i + 1
                    break
            # Fallback: check text similarity for cases where ID may differ
            if rank > self.k:
                for i, text in enumerate(returned_texts):
                    if exact_match(relevant_memory_text, text) > 0:
                        rank = i + 1
                        break

            mrr = mrr_at_k(rank, self.k)

            return CaseResult(
                case_id=case.case_id,
                answer_type=case.answer_type,
                exact_match=em,
                token_f1=tf1,
                mrr_at_k=mrr,
                rank_of_relevant=rank,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            logger.error("Error evaluating case %s: %s", case.case_id, exc)
            return CaseResult(
                case_id=case.case_id,
                answer_type=case.answer_type,
                exact_match=0.0,
                token_f1=0.0,
                mrr_at_k=0.0,
                rank_of_relevant=self.k + 1,
                latency_ms=None,
                error=str(exc),
            )
        finally:
            if self.cleanup:
                user_id, namespace = self._user_ns(case.case_id)
                self.client.delete_all(user_id=user_id, namespace=namespace)

    def run(
        self,
        cases: list[StructCase],
        *,
        verbose: bool = False,
    ) -> AdapterReport:
        """Evaluate all cases and return an :class:`AdapterReport`."""
        results: list[CaseResult] = []
        errors: list[str] = []
        skipped = 0

        for i, case in enumerate(cases, start=1):
            if not case.memories:
                logger.info("Skipping case %s — no memories", case.case_id)
                skipped += 1
                continue

            if verbose:
                print(f"[{i}/{len(cases)}] {case.case_id} ({case.answer_type})", flush=True)

            result = self.evaluate_case(case)
            results.append(result)
            if result.error:
                errors.append(f"{result.case_id}: {result.error}")

        # Aggregate metrics
        def _safe_mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        em_scores = [r.exact_match for r in results]
        f1_scores = [r.token_f1 for r in results]
        mrr_scores = [r.mrr_at_k for r in results]
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]

        # Per-answer-type breakdown
        by_type: dict[str, dict[str, float]] = {}
        for atype in sorted({r.answer_type for r in results}):
            group = [r for r in results if r.answer_type == atype]
            by_type[atype] = {
                "cases": len(group),
                "mean_exact_match": _safe_mean([r.exact_match for r in group]),
                "mean_token_f1": _safe_mean([r.token_f1 for r in group]),
                "mean_mrr_at_k": _safe_mean([r.mrr_at_k for r in group]),
            }

        return AdapterReport(
            total_cases=len(cases),
            evaluated_cases=len(results),
            skipped_cases=skipped,
            mean_exact_match=_safe_mean(em_scores),
            mean_token_f1=_safe_mean(f1_scores),
            mean_mrr_at_k=_safe_mean(mrr_scores),
            k=self.k,
            by_answer_type=by_type,
            latency_ms={
                "mean": _safe_mean(latencies),
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "count": len(latencies),
            },
            case_results=results,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Selftest dataset (3 synthetic cases — no server required for unit tests)
# ---------------------------------------------------------------------------


_SELFTEST_DATASET: list[StructCase] = [
    StructCase(
        case_id="sme-st-001",
        question="What is the auth token rotation interval?",
        expected_answer="90 days",
        answer_type="string",
        memories=[
            "The project uses FastAPI with uvicorn on port 42069.",
            "Auth tokens should be rotated every 90 days per the security policy.",
            "The vector store uses Qdrant with fastembed for embedding generation.",
        ],
        relevant_memory_index=1,
    ),
    StructCase(
        case_id="sme-st-002",
        question="How many tests passed in Phase 14?",
        expected_answer="694",
        answer_type="number",
        memories=[
            "Phase 13 delivered ColBERT multi-vector MaxSim and temporal query expansion.",
            "Phase 15 achieved 727 tests passing with 33 new tests added.",
            "Phase 14 completed with 694 tests passing at 100% pass rate.",
        ],
        relevant_memory_index=2,
    ),
    StructCase(
        case_id="sme-st-003",
        question="Which memory scopes are supported?",
        expected_answer="project, global",
        answer_type="list",
        memories=[
            "KuzuDB graph store uses PRECEDES and CAUSES relationship types.",
            "Memory scope can be either project or global; project memories are "
            "namespace-isolated while global memories are visible across all projects.",
            "OTel spans use gen_ai.operation.name and gen_ai.system attributes.",
        ],
        relevant_memory_index=1,
    ),
]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_selftest(*, k: int = 3, verbose: bool = False) -> AdapterReport:
    """
    Evaluate the selftest dataset using in-process metric computation only.

    No Muninn server is required — memories are searched by simple
    keyword overlap against the question, simulating the retrieval oracle
    without network calls.
    """

    class _OracleClient(MuninnHTTPClient):
        """In-process oracle client: returns memories sorted by keyword overlap."""

        def __init__(self) -> None:
            self._store: dict[str, list[tuple[str, str]]] = {}  # (user, ns) → [(id, text)]

        def add(self, content: str, *, user_id: str, namespace: str, memory_id: str | None = None) -> dict:
            key = (user_id, namespace)
            mid = memory_id or str(uuid.uuid4())
            self._store.setdefault(key, []).append((mid, content))
            return {"memory_id": mid, "status": "ok"}

        def search(self, query: str, *, user_id: str, namespace: str, limit: int = 10) -> list[dict]:
            key = (user_id, namespace)
            entries = self._store.get(key, [])
            import re

            qtokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))

            def _score(text: str) -> int:
                ttokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
                return len(qtokens & ttokens)

            ranked = sorted(entries, key=lambda e: _score(e[1]), reverse=True)
            return [{"id": mid, "content": text} for mid, text in ranked[:limit]]

        def delete_all(self, *, user_id: str, namespace: str) -> None:
            self._store.pop((user_id, namespace), None)

    adapter = StructMemEvalAdapter(_OracleClient(), k=k, cleanup=True)
    return adapter.run(_SELFTEST_DATASET, verbose=verbose)


def run_adapter(
    dataset_path: str | Path,
    *,
    server_url: str = "http://localhost:42069",
    auth_token: str | None = None,
    k: int = 10,
    limit: int | None = None,
    cleanup: bool = True,
    verbose: bool = False,
) -> AdapterReport:
    """Run the adapter against a real Muninn server with a JSONL dataset."""
    cases = parse_dataset(dataset_path)
    if limit is not None and limit > 0:
        cases = cases[:limit]
    client = MuninnHTTPClient(server_url=server_url, auth_token=auth_token)
    adapter = StructMemEvalAdapter(client, k=k, cleanup=cleanup)
    return adapter.run(cases, verbose=verbose)


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def _report_to_dict(report: AdapterReport) -> dict[str, Any]:
    return {
        "total_cases": report.total_cases,
        "evaluated_cases": report.evaluated_cases,
        "skipped_cases": report.skipped_cases,
        "mean_exact_match": report.mean_exact_match,
        "mean_token_f1": report.mean_token_f1,
        "mean_mrr_at_k": report.mean_mrr_at_k,
        "k": report.k,
        "by_answer_type": report.by_answer_type,
        "latency_ms": report.latency_ms,
        "errors": report.errors,
        "case_results": [
            {
                "case_id": r.case_id,
                "answer_type": r.answer_type,
                "exact_match": r.exact_match,
                "token_f1": r.token_f1,
                "mrr_at_k": r.mrr_at_k,
                "rank_of_relevant": r.rank_of_relevant,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in report.case_results
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="StructMemEval adapter — structured/factoid memory recall benchmark for Muninn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dataset",
        metavar="PATH",
        help="Path to StructMemEval JSONL dataset file.",
    )
    mode.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Run the built-in selftest dataset using an in-process oracle client. "
            "No Muninn server required. Useful for CI validation."
        ),
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("MUNINN_SERVER_URL", "http://localhost:42069"),
        metavar="URL",
        help="Muninn server base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("MUNINN_AUTH_TOKEN"),
        metavar="TOKEN",
        help="Muninn auth token (default: $MUNINN_AUTH_TOKEN).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        metavar="K",
        help="Retrieval cutoff for MRR@k (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N cases (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON report to PATH (default: stdout).",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=True,
        help="Do not delete memories after each case (for debugging).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-case progress to stdout.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.selftest:
        report = run_selftest(k=args.k, verbose=args.verbose)
    else:
        report = run_adapter(
            args.dataset,
            server_url=args.server_url,
            auth_token=args.auth_token,
            k=args.k,
            limit=args.limit,
            cleanup=args.cleanup,
            verbose=args.verbose,
        )

    doc = _report_to_dict(report)

    if args.output:
        Path(args.output).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"StructMemEval report written to: {args.output}")
    else:
        print(json.dumps(doc, indent=2))

    # Print summary line
    print(
        f"\n[structmemeval] cases={report.evaluated_cases} "
        f"EM={report.mean_exact_match:.3f} "
        f"F1={report.mean_token_f1:.3f} "
        f"MRR@{report.k}={report.mean_mrr_at_k:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
