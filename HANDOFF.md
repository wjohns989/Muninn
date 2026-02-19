# Muninn Development Handoff

> **Updated**: 2026-02-19
> **Branch**: `feature/v3.13.0-sota-verdict-v1`
> **Version**: v3.13.0 (Phase 16 COMPLETE)
> **Status**: Phase 16 done. 788 tests pass. PR #45 ready for merge.

---

## Current State

### What's Working
- **788 tests pass** (100% pass rate — 727 Phase 15 + 61 Phase 16)
- **Server**: FastAPI on `http://localhost:42069`, auth token via `MUNINN_AUTH_TOKEN`
- **MCP**: Registered as "muninn" (tools: `mcp__muninn__*`) in Claude Code user config with auth token baked in
- **Claude Desktop**: Already correctly registered as "muninn"
- **Phase 14 (v3.11.0)**: Project-scoped memory — **MERGED** (PR #43, 2026-02-19)
- **Phase 15 (v3.12.0)**: Operational hardening — **MERGED** (PR #44, 2026-02-19)
- **Phase 16 (v3.13.0)**: SOTA+ signed verdict v1 — **COMPLETE**, PR #45 ready for merge

### Server Quick Start

```bash
# Start server (Windows — run from repo root)
$env:MUNINN_AUTH_TOKEN = (Get-Content .muninn_token)
python server.py

# Or from bash
MUNINN_AUTH_TOKEN=$(cat .muninn_token) python server.py
```

**Auth token** is stored in `.muninn_token` (gitignored). Also set permanently via `setx`:
```
Token: ij0w9VmdPH5dxnE7vG-lZCXPPWhX9uU7HpBJODg0zoQ
```

### Data Directory
```
C:\Users\user\AppData\Local\AntigravityLabs\muninn\
├── metadata.db          # SQLite — 74 memories
├── qdrant_v8/           # Vector store — 73 vectors
└── kuzu_v12/            # Graph store — activated in Phase 15
```

---

## Phase 16 (v3.13.0) Summary — 2026-02-19

### Changes Delivered

#### 1. `cmd_sota_verdict` — Provenance Block
**File**: `eval/ollama_local_benchmark.py`
**Change**: Verdict JSON now includes `provenance` object:
```json
{
  "provenance": {
    "commit_sha": "<40-char hex or null>",
    "input_file_hashes": { "bench_report": "<sha256>", ... },
    "promotion_signature": "hmac-sha256=<hex> | null",
    "verdict_schema_version": "1.0"
  }
}
```
`commit_sha` from `git rev-parse HEAD` (5s timeout; `null` if git unavailable).
`input_file_hashes`: SHA256 per `--*-report` CLI arg actually provided.

#### 2. HMAC-SHA256 Promotion Signature
**Helpers added** (before `cmd_sota_verdict` in `eval/ollama_local_benchmark.py`):
- `_get_commit_sha(repo_root)` — git rev-parse HEAD with timeout
- `_sha256_file(path)` — 1 MiB chunk streaming SHA256
- `_compute_hmac_signature(canonical_data, signing_key)` — canonical JSON (`sort_keys=True`) → HMAC-SHA256 → `"hmac-sha256=<64hex>"`
- `_evaluate_longmemeval_gate(report, *, min_ndcg_at_10, min_recall_at_10)` — normalizes 3 LME report formats

**Canonical payload** (for signing):
```python
{"run_id": ..., "passed": ..., "commit_sha": ..., "input_file_hashes": {...}}
```
Signature is `null` when `--signing-key` not provided.

#### 3. LongMemEval Hard Gate
**New CLI args** (`verdict` subcommand):
- `--longmemeval-report PATH` — path to LME adapter JSON output
- `--min-longmemeval-ndcg FLOAT` — default `0.60`
- `--min-longmemeval-recall FLOAT` — default `0.65`
- `--require-longmemeval` / `--no-require-longmemeval` — default `False`
- `--signing-key STR` — HMAC secret

Gate is a **hard gate** in `overall_passed`:
```python
overall_passed = all([..., longmemeval_gate_evaluation["passed"]])
```
Accepts three LME report formats: `{ndcg_at_10, recall_at_10}`, `{mean_ndcg_at_10, mean_recall_at_10}`, `{cutoffs.@10.{ndcg, recall}}`.

**Backward compatibility**: All Phase 16 args use `getattr(args, 'field', default)` — pre-existing `SimpleNamespace`-based tests are unaffected.

#### 4. `eval/structmemeval_adapter.py` (new)
**Architecture**:
- Data models: `StructCase`, `CaseResult`, `AdapterReport` (dataclasses)
- `exact_match()`, `token_f1()`, `mrr_at_k()`, `_percentile()` — inline, no external deps
- `parse_dataset(path)` — JSONL parser with malformed-line skipping and out-of-range index rejection
- `MuninnHTTPClient` — stdlib urllib: `add()`, `search()`, `delete_all()`
- `StructMemEvalAdapter` — `evaluate_case()` (ingest → search → EM/F1/MRR → cleanup) + `run()`
- `_SELFTEST_DATASET` — 3 synthetic cases (string/number/list answer types)
- `run_selftest()` with inner `_OracleClient` (keyword-overlap, no server needed)
- Full CLI: `--dataset`/`--selftest`, `--server-url`, `--auth-token`, `--k`, `--limit`, `--output`, `--no-cleanup`, `--verbose`
- **Selftest result**: EM=1.000, MRR@10=1.000

#### 5. `tests/test_v3_13_0_sota_verdict_v1.py` (new, 61 tests)
8 test classes:
- `TestProvenanceHelpers` (11): SHA256 file hashing, commit SHA retrieval, HMAC signature correctness
- `TestLongMemEvalGate` (8): pass/fail thresholds, missing keys, three report formats, boundary values
- `TestSotaVerdictPayload` (10): full integration via `build_parser()` + `cmd_sota_verdict`; patches `_get_commit_sha`
- `TestStructMemEvalMetrics` (12): unit tests for EM, token-F1, MRR@k
- `TestStructMemEvalDatasetParser` (4): JSONL parsing, malformed lines, out-of-range index
- `TestStructMemEvalAdapter` (8): evaluate_case, run, cleanup, error capture
- `TestStructMemEvalSelftest` (5): oracle correctness for all 3 synthetic cases
- `TestVersionBump313` (2): version == 3.13.0, pyproject.toml match

---

## Phase 15 (v3.12.0) Summary — 2026-02-19

### Changes Delivered

#### 1. Auth Token Propagation Fix (`lifecycle.py`)
**File**: `muninn/mcp/lifecycle.py`
**Symptom**: `start_server()` spawned `server.py` without `MUNINN_AUTH_TOKEN` → each process generated a random token → all MCP tool calls returned HTTP 401 in clean environments
**Fix**: `token = os.environ.get("MUNINN_AUTH_TOKEN") or get_token()` passed via `spawn_detached_process(..., env={"MUNINN_AUTH_TOKEN": token})`
**Tests**: 5 unit tests in `TestStartServerAuthPropagation`

#### 2. Graph Memory Chains Smoke Tests
**File**: `tests/test_v3_12_0_operational_hardening.py` — `TestGraphChainsSmoke`
**Coverage**: 7 tests using real KuzuDB (`tmp_path`): CAUSES/PRECEDES edge creation, retrieval, `get_entity_count()`, invalid relation rejection, self-loop rejection, `MemoryChainDetector.detect_links()`

#### 3. OTel GenAI Attribute Validation
**File**: `tests/test_v3_12_0_operational_hardening.py` — `TestOTelGenAIAttributes`
**Coverage**: 8 tests: required keys, operation name mapping, no-op when disabled, dot-namespaced keys, `maybe_content()` privacy default, `None`-value attribute skip

#### 4. LongMemEval Adapter
**File**: `eval/longmemeval_adapter.py` (new, production-grade)
**Tests**: 13 tests in `TestLongMemEvalAdapter`

#### 5. Stale Version Assertion Fix
**File**: `tests/test_v3_11_0_project_scope.py` → `>= (3, 11, 0)` tuple comparison

---

## Architecture Notes

### Auth Flow
```
Claude Code → spawns mcp_wrapper.py (with MUNINN_AUTH_TOKEN from MCP env config)
           → mcp_wrapper.py uses token in Authorization: Bearer header
           → server.py validates token from MUNINN_AUTH_TOKEN env var
```

Both processes MUST share the same token. The MCP registration `-e` flag is the reliable cross-session mechanism.

### SOTA+ Verdict Payload Shape (v3.13.0)
```json
{
  "muninn_version": "3.13.0",
  "run_id": "...",
  "overall_passed": true,
  "provenance": {
    "commit_sha": "abc123...",
    "input_file_hashes": { "bench_report": "sha256:..." },
    "promotion_signature": "hmac-sha256:...",
    "verdict_schema_version": "1.0"
  },
  "gates": {
    "quality": {...},
    "reliability": {...},
    "statistical": {...},
    "reproducibility": {...},
    "profile": {...},
    "external_benchmarks": {
      "longmemeval": {
        "passed": true,
        "ndcg_at_10": 0.72,
        "recall_at_10": 0.68,
        "thresholds": {"min_ndcg_at_10": 0.60, "min_recall_at_10": 0.65}
      }
    }
  }
}
```

### MCP Server Registration (User-Scope)
```json
{
  "muninn": {
    "type": "stdio",
    "command": "C:/Users/user/miniconda3/python.exe",
    "args": ["C:/Users/user/muninn_mcp/mcp_wrapper.py"],
    "env": {
      "MUNINN_AUTH_TOKEN": "ij0w9VmdPH5dxnE7vG-lZCXPPWhX9uU7HpBJODg0zoQ"
    }
  }
}
```

---

## Test Suite
```bash
# Run full suite
pytest tests/ -q

# Run Phase 16 tests only
pytest tests/test_v3_13_0_sota_verdict_v1.py -v

# Run Phase 15 tests only
pytest tests/test_v3_12_0_operational_hardening.py -v

# Run just sota-verdict tests
pytest tests/test_ollama_local_benchmark.py -k "sota_verdict" -v
```

**Expected**: 788 pass, 2 skipped, 0 fail

---

## Open Items / Next Steps

### Phase 17 Candidates
- [ ] **LongMemEval real-dataset run**: Obtain public LongMemEval JSONL and establish nDCG@10 ≥ 0.60 baseline with signed verdict artifact committed to repo
- [ ] **Parser sandbox**: Security hardening for optional pdf/docx binary parsers
- [ ] **Token rotation docs**: Periodic `.muninn_token` rotation procedures
- [ ] **StructMemEval real dataset**: Publish/obtain structured factoid QA JSONL for live server evaluation

### Known Remaining Gap
**SOTA+ real-dataset evidence**: Both adapters (`eval/longmemeval_adapter.py`, `eval/structmemeval_adapter.py`) are production-ready and selftests pass. A real-dataset run requires obtaining the public LongMemEval JSONL dataset and running against a live Muninn server. The signed verdict artifact has not been produced against real data yet — that's Phase 17.

---

## Files Changed (Phase 16)

| File | Change |
|------|--------|
| `eval/ollama_local_benchmark.py` | Added `_get_commit_sha`, `_sha256_file`, `_compute_hmac_signature`, `_evaluate_longmemeval_gate` helpers; extended `cmd_sota_verdict` with provenance block, HMAC signing, LongMemEval hard gate; 5 new CLI args; all Phase 16 args use `getattr` defaults |
| `eval/structmemeval_adapter.py` | New — production StructMemEval adapter with inline metrics, selftest oracle, MuninnHTTPClient, full CLI |
| `tests/test_v3_13_0_sota_verdict_v1.py` | New — 61 tests across 8 classes (all pass) |
| `muninn/version.py` | `3.12.0` → `3.13.0` |
| `pyproject.toml` | `version = "3.12.0"` → `version = "3.13.0"` |
| `SOTA_PLUS_PLAN.md` | Phase 16 checklist items marked complete, validation history updated |
| `HANDOFF.md` | Updated to Phase 16 complete state |
