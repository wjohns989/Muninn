# Muninn Development Handoff

> **Updated**: 2026-02-19
> **Branch**: `feature/v3.14.0-benchmark-suite-parser-sandbox`
> **Version**: v3.14.0 (Phase 17 COMPLETE)
> **Status**: Phase 17 done. 848 tests pass. PR #46 ready for merge.

---

## Current State

### What's Working
- **848 tests pass** (100% pass rate — 788 Phase 16 + 60 Phase 17)
- **Server**: FastAPI on `http://localhost:42069`, auth token via `MUNINN_AUTH_TOKEN`
- **MCP**: Registered as "muninn" (tools: `mcp__muninn__*`) in Claude Code user config with auth token baked in
- **Claude Desktop**: Already correctly registered as "muninn"
- **Phase 14 (v3.11.0)**: Project-scoped memory — **MERGED** (PR #43, 2026-02-19)
- **Phase 15 (v3.12.0)**: Operational hardening — **MERGED** (PR #44, 2026-02-19)
- **Phase 16 (v3.13.0)**: SOTA+ signed verdict v1 — **MERGED** (PR #45, 2026-02-19)
- **Phase 17 (v3.14.0)**: Synthetic benchmark suite + parser security sandbox — **COMPLETE**, PR #46 ready for merge

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
C:\Users\wjohn\AppData\Local\AntigravityLabs\muninn\
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
    "command": "C:/Users/wjohn/miniconda3/python.exe",
    "args": ["C:/Users/wjohn/muninn_mcp/mcp_wrapper.py"],
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

**Expected**: 848 pass, 2 skipped, 0 fail

```bash
# Run Phase 17 tests only
pytest tests/test_v3_14_0_benchmark_suite.py -v
```

---

## Open Items / Next Steps

### Phase 18 Candidates
- [ ] **Live benchmark run + signed verdict artifact**: Run `eval/run_benchmark.py --production` against live server with synthetic datasets and commit the signed verdict artifact to `eval/reports/`
- [ ] **Public LongMemEval JSONL**: Obtain `longmemeval_oracle.jsonl` from the paper authors (https://github.com/xiaowu0162/LongMemEval) and establish real nDCG@10 baseline
- [ ] **GitHub Actions CI**: Add `.github/workflows/benchmark.yml` running `run_benchmark.py --dry-run` on every PR to prevent adapter regression
- [ ] **Token rotation utility**: CLI command `python -m muninn.cli rotate-token` to replace `.muninn_token` and update MCP registrations automatically

### Known Remaining Gap
**SOTA+ production-run evidence**: The synthetic benchmark datasets (`eval/data/longmemeval_synthetic_v1.jsonl`, `eval/data/structmemeval_suite_v1.jsonl`) and pipeline (`eval/run_benchmark.py`) are production-ready. The signed verdict artifact against a live Muninn server with the synthetic data has not been committed yet — that's Phase 18 P1. This requires a running Muninn server and takes ~5 minutes.

---

## Phase 17 (v3.14.0) Summary — 2026-02-19

### Changes Delivered

#### 1. Synthetic LongMemEval Dataset (`eval/data/longmemeval_synthetic_v1.jsonl`)
30 LongMemEval-format benchmark cases covering all question types:
- `single-session-qa` (10): Factual recall from single-session conversations
- `multi-session-qa` (8): Cross-session memory retrieval
- `temporal` (6): Time-anchored questions about events and versions
- `adversarial` (3): Cases with distractor information to test retrieval precision
- `entity-centric` (3): Entity-focused factoid questions
All content is Muninn-domain realistic (architecture decisions, version history, feature flags).

#### 2. Synthetic StructMemEval Suite (`eval/data/structmemeval_suite_v1.jsonl`)
30 StructMemEval-format benchmark cases covering all answer types:
- `string` (10): Text-valued facts (project names, license, framework names)
- `number` (8): Numeric facts (port numbers, test counts, version numbers, thresholds)
- `entity` (7): Named entities (company names, library names, database names)
- `list` (5): Multi-value facts (scope values, question types, supported types)
Each case has valid `relevant_memory_index` and non-empty `memories[]`.

#### 3. Automated Benchmark Runner (`eval/run_benchmark.py`)
Full CI benchmark pipeline:
- **Dry-run mode**: Runs adapter selftests, no server required — CI-safe
- **Production mode**: Health-checks server, runs adapters against datasets, gates results
- **Gate evaluation**: LongMemEval (nDCG@10, Recall@10) + StructMemEval (Exact Match)
- **Mandatory gates**: `--require-longmemeval` / `--require-structmemeval` flip gates to hard requirements
- **JSON report**: Structured output with run_id, timestamps, commit_sha, per-adapter results, gate decisions
- **Skip flags**: `--skip-lme` / `--skip-sme` for partial runs
- Data classes: `AdapterResult`, `BenchmarkRunReport`

**CLI usage**:
```bash
# Dry-run (no server needed):
python -m eval.run_benchmark --dry-run

# Production run with signed verdict:
python -m eval.run_benchmark --production \
  --server-url http://localhost:42069 \
  --auth-token $(cat .muninn_token) \
  --require-longmemeval \
  --output eval/reports/sota_evidence_$(date +%Y%m%d).json
```

#### 4. Parser Security Sandbox (`muninn/ingestion/sandbox.py` + `_parser_subprocess.py`)
**Architecture**: PDF and DOCX parsing now runs in a subprocess:
```
server.py → parser.py → sandbox.py → subprocess → _parser_subprocess.py
                                    ↑ JSON protocol ↑
```
**`muninn/ingestion/_parser_subprocess.py`** (subprocess worker):
- Entry point: `python -m muninn.ingestion._parser_subprocess <type> <path>`
- Parses PDF (pypdf) or DOCX (python-docx) and writes `{"text": "..."}` JSON to stdout
- Output capped at 2 MB (`MAX_OUTPUT_CHARS`)
- Exception-safe: all errors produce `{"error": "..."}` + exit 1
- Exit codes: 0=success, 1=parse/runtime error, 2=usage error

**`muninn/ingestion/sandbox.py`** (sandbox executor):
- `sandboxed_parse_binary(path, source_type, timeout=30.0)`
- 4 MB stdout cap (`MAX_STDOUT_BYTES`) on parent side
- Hard timeout enforcement via `subprocess.run(timeout=...)`
- JSON protocol validation with structured error propagation
- Optional `fallback_in_process=True` for constrained environments

**`muninn/ingestion/parser.py`** (caller):
```python
def _parse_pdf(path):
    from muninn.ingestion.sandbox import sandboxed_parse_binary
    return sandboxed_parse_binary(path, "pdf", timeout=30.0)
```

#### 5. `tests/test_v3_14_0_benchmark_suite.py` (new, 60 tests)
8 test classes:
- `TestSyntheticLongMemEvalDataset` (8): dataset existence, 30-case count, field completeness, question type coverage, session structure, conversation turn validity
- `TestSyntheticStructMemEvalDataset` (8): dataset existence, 30-case count, field completeness, answer type coverage, memory validity, index bounds
- `TestBenchmarkRunnerImport` (5): module imports, dataclass construction, commit SHA helper
- `TestBenchmarkRunnerDryRun` (10): report structure, overall_passed logic, JSON output schema, skip flags, production health-check failure, UUID uniqueness, ISO timestamp
- `TestBenchmarkRunnerGates` (8): LME gate pass/fail thresholds, SME gate thresholds, null-report handling, mandatory gate impact on overall_passed
- `TestParserSandbox` (10): import, ValueError on bad type, RuntimeError on missing file, subprocess success, error response, timeout, invalid JSON, empty stdout, disabled fallback, stdout size cap
- `TestParserSubprocessWorker` (8): import, wrong-argc exit, unsupported-type exit, missing-file exit, PDF success JSON, DOCX success JSON, exception JSON, truncation
- `TestVersionBump314` (2): version == 3.14.0, pyproject.toml match

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

## Files Changed (Phase 17)

| File | Change |
|------|--------|
| `eval/data/longmemeval_synthetic_v1.jsonl` | New — 30 synthetic LongMemEval-format benchmark cases (5 question types) |
| `eval/data/structmemeval_suite_v1.jsonl` | New — 30 synthetic StructMemEval-format benchmark cases (4 answer types) |
| `eval/run_benchmark.py` | New — automated CI benchmark pipeline; dry-run + production modes; LME+SME gate evaluation; JSON report output; argparse CLI |
| `muninn/ingestion/_parser_subprocess.py` | New — subprocess worker for sandboxed PDF/DOCX parsing; JSON protocol; 2 MB output cap |
| `muninn/ingestion/sandbox.py` | New — sandbox executor; 30s timeout; 4 MB stdout cap; structured error propagation |
| `muninn/ingestion/parser.py` | `_parse_pdf()` and `_parse_docx()` now route through `sandboxed_parse_binary()` |
| `tests/test_v3_14_0_benchmark_suite.py` | New — 60 tests across 8 classes (all pass) |
| `tests/test_v3_13_0_sota_verdict_v1.py` | Version assertion updated to `>= (3, 13, 0)` tuple comparison |
| `muninn/version.py` | `3.13.0` → `3.14.0` |
| `pyproject.toml` | `version = "3.13.0"` → `version = "3.14.0"` |
| `SOTA_PLUS_PLAN.md` | Phase 17 section added; validation history updated |
| `HANDOFF.md` | Updated to Phase 17 complete state |
