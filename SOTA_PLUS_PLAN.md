# Muninn SOTA+ Implementation Plan

> **Version**: v3.1.1 → v3.3.0 Roadmap
> **Status**: Active implementation (ROI-first tranche in progress)
> **Estimated Effort**: 22–32 developer-days across 3 phases
> **License Constraint**: Apache-2.0 — all dependencies verified compatible
> **Backward Compatibility**: 100% — all enhancements are additive & optional

---

## Accuracy + Execution Update (2026-02-14)

### ROI-first execution order (current)
1. Goal Compass + drift guardrail + goal-aware retrieval signal.
2. Cross-assistant handoff export/import with checksum + idempotency ledger.
3. Eval gate hardening (nDCG/Recall/MRR + latency budgets).
4. MCP 2025-11 compatibility and OpenTelemetry GenAI instrumentation.

### Completed in this implementation slice
1. Instructor config is now fully wired from `MuninnConfig` into `ExtractionPipeline`.
2. Docker path contract fixed: `get_data_dir/get_config_dir/get_log_dir` now honor runtime Docker detection, not only `MUNINN_DOCKER=1`.
3. Recall trace fidelity improved: traces now record native per-signal raw scores (vector cosine, BM25, graph/temporal signal scores), not rank proxies.
4. Adaptive weight entropy now uses score distributions instead of rank-derived pseudo-scores.
5. Version consistency fixed with single source of truth (`muninn/version.py`) and synchronized package/server/MCP versions.
6. Additional retrieval correctness fix: graph signal now passes entity lists correctly and uses deterministic scoring.
7. Additional scope-safety fix: final retrieval result filtering now enforces `user_id`/namespace constraints.
8. Goal Compass is now implemented end-to-end (goal persistence, drift checks, retrieval signal, API + MCP tools).
9. Cross-assistant handoff export/import is implemented with deterministic SHA-256 checksum verification and idempotent replay ledger.
10. Eval gate upgraded with latency summaries (`avg/p50/p95`) plus regression/budget checks (`--baseline-report`, `--max-metric-regression`, `--max-p95-latency-ms`).
11. MCP lifecycle hardening added: explicit protocol negotiation (`2025-11-25`, `2025-06-18`, `2024-11-05`), initialization gating, JSON Schema 2020-12 tags in tool schemas, read-only tool annotations.
12. Optional OpenTelemetry GenAI instrumentation added behind feature flag `MUNINN_OTEL_GENAI=1` with privacy-safe default (`MUNINN_OTEL_CAPTURE_CONTENT=0`).
13. SQLite user scoping now uses exact JSON filtering (`json_extract`) with fallback to LIKE for non-JSON1 builds; this removes false misses on quoted IDs and improves query correctness/perf.
14. Retrieval feedback persistence loop implemented end-to-end:
    - `retrieval_feedback` SQLite table + scoped index,
    - `record_retrieval_feedback` API/MCP surface,
    - adaptive weight calibration path via bounded per-signal multipliers,
    - short-TTL scoped cache with invalidation on new feedback.
15. Retrieval feedback calibration upgraded with a counterfactual path:
    - optional `rank` + `sampling_prob` persisted per feedback event,
    - SNIPS-style estimator option (`weighted_mean`/`snips`) with propensity clipping + effective-sample safeguards,
    - config-driven policy (`estimator`, propensity floor, minimum effective samples, default sampling probability).
16. Eval harness now supports benchmark competency tracks:
    - optional dataset `track` labels,
    - per-track `Recall/MRR/nDCG` and latency summaries,
    - compatible with MemoryAgentBench-style slices (accurate retrieval, selective forgetting, etc.).
17. Eval release gating now supports preset policy profiles + track coverage enforcement:
    - new preset catalog (`vibecoder_memoryagentbench_v1`) with default regression/latency budgets,
    - per-track regression gates against baseline reports,
    - required track case-count gates (`--required-track TRACK:MIN_CASES`),
    - auditable gate configuration emitted in reports (`gate_config`).
18. Eval now supports paired statistical rigor against baseline predictions:
    - permutation-test p-values and bootstrap CIs on per-query paired deltas,
    - effect-size signal (`cohens_d`) for practical impact,
    - optional significant-regression gate (`--gate-significant-regressions`) across global and track metrics.
19. Canonical benchmark artifact discipline is now implemented for the vibecoder preset:
    - committed dataset/predictions/baseline report artifacts (`146` cases across MemoryAgentBench-aligned tracks),
    - SHA-256 manifest integrity contract,
    - reproducibility verifier CLI (`python -m eval.artifacts verify --preset vibecoder_memoryagentbench_v1`).
20. Eval significance gates now include configurable multiple-comparison correction:
    - supported methods: `none`, `bonferroni`, `holm`, `bh`,
    - configurable family scope: `all` or `by_track`,
    - gate decisions now consume corrected p-values (`p_value_adjusted`) with raw/adjusted signals both preserved for auditability.
21. MCP conformance hardening tranche completed:
    - strict JSON-RPC method handling now returns `-32601` for unknown request methods (notification methods remain no-op),
    - `notifications/initialized` is now accepted only after successful `initialize` negotiation,
    - `tools/call` and `initialize` param shape validation now returns explicit `-32602` for invalid parameter contracts,
    - schema contract coverage expanded with protocol tests for tool-name uniqueness, object schema shape, and required-property consistency.
22. Canonical artifact coverage is expanded beyond a single bundle:
    - new robustness slice preset `vibecoder_memoryagentbench_stress_v1` with committed `dataset/predictions/baseline_report/manifest`,
    - deterministic 60-case cross-track stress corpus (`16/6/30/8`) with hard-negative ranking pressure and elevated latency profile,
    - aggregate verifier command added: `python -m eval.artifacts verify --all` for multi-bundle CI integrity checks.
23. OTel operationalization tranche is now implemented:
    - production runbook added (`docs/OTEL_GENAI_OBSERVABILITY.md`) with collector setup, smoke-test workflow, and environment profile examples,
    - collector config example added (`examples/otel/collector-config.yaml`) for local OTLP bring-up,
    - privacy policy controls documented for default-safe deployment and bounded diagnostic capture.
24. OTel content-capture guardrail improved in code:
    - added `MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS` with bounded parsing and safe fallback,
    - tracer now uses dynamic package version (`muninn.version.__version__`) instead of hardcoded instrumentation version.
25. Phase 3C Python SDK tranche is now implemented:
    - new `muninn/sdk` package with `MuninnClient` + `AsyncMuninnClient`,
    - mem0-style aliases exported at package root (`Memory`, `AsyncMemory`),
    - typed SDK exceptions (`MuninnConnectionError`, `MuninnAPIError`),
    - method parity across health/add/search/goal/handoff/feedback/admin endpoints.
26. Phase 3B multi-source ingestion tranche is now implemented:
    - new feature-gated `muninn/ingestion` package with fail-open parser pipeline for `txt/md/json/csv/tsv/html` and optional `pdf/docx`,
    - chunk-level provenance metadata (`source_sha256`, byte size, chunk offsets, source path/type),
    - `MuninnMemory.ingest_sources(...)` orchestration with per-source/per-chunk failure isolation,
    - REST `/ingest`, MCP `ingest_sources` tool wiring, and SDK `ingest_sources(...)` sync/async coverage.
27. Legacy assistant/MCP memory migration tranche is now implemented:
    - discovery catalog for local assistant artifacts and MCP memory stores (`codex_cli`, `claude_code`, `serena_memory`, `cursor`, `vscode`, `copilot`, `antigravity`, plus custom roots),
    - selection-based legacy import (`selected_source_ids`/`selected_paths`) with contextual metadata injection per source,
    - parser/contextualization upgrades for chat-heavy sources (`.jsonl/.ndjson`) and sqlite-backed stores (`.vscdb/.db/.sqlite*`),
    - REST (`/ingest/legacy/discover`, `/ingest/legacy/import`) + MCP (`discover_legacy_sources`, `ingest_legacy_sources`) + SDK parity.
28. Browser control-center tranche is now implemented:
    - rebuilt first-party UI served at `/` from `dashboard.html` with direct operational controls,
    - checkbox-based legacy discovery/reingestion workflow for assistant/MCP artifacts,
    - project-folder contextual ingestion flow with chronological ordering (`none|oldest_first|newest_first`) and tunable chunking controls,
    - health/search/consolidation actions consolidated in one interface for practical end-user operation.
29. Open-PR security/correctness remediation tranche is now implemented:
    - ingestion allow-list enforcement added (`allowed_roots`) to block arbitrary file reads from untrusted tool inputs,
    - runtime chunk/file bounds validation added to prevent pathological chunking and oversized-ingest DoS vectors,
    - legacy discovery/import now validates user-provided roots and selected paths against the ingestion allow-list,
    - `/ingest` endpoint now preserves upstream `HTTPException` status codes (no blanket 500 remap).
30. Eval/SDK reliability corrections from review threads are now implemented:
    - `Recall@k`/`nDCG@k` now ignore duplicate relevant IDs to prevent inflated metrics,
    - SDK `delete` methods now URL-encode `memory_id` path segments,
    - SDK success payload unwrap now preserves non-`data` success payloads instead of discarding them,
    - parser/discovery robustness improved (`sqlite` URI escaping, safer glob derivation, custom-root sqlite artifact discovery).
31. Phase 3A memory chains tranche is now implemented:
    - new `muninn/chains` package with deterministic temporal/causal chain detector and retrieval expansion helper,
    - graph-store support for first-class memory-to-memory `PRECEDES` / `CAUSES` edges with confidence + provenance fields,
    - chain-link persistence wired into `add` and `update` paths with scoped candidate scans and entity-overlap reasoning,
    - hybrid retrieval fusion now includes optional chain signal (`memory_chains` feature flag) with explainable trace attribution.
32. Post-restart stability + quality hygiene tranche completed:
    - repository integrity verified after crash/restart (`git fsck --full` clean),
    - conflict-resolver test warning source removed,
    - branch workflow moved to one-open-PR-per-phase policy.
33. Phase 4A operator adaptation baseline started:
    - `dashboard.html` now persists control-center preferences in browser storage,
    - model profile selection is displayed in UI and carried into ingestion metadata for traceability.
34. Phase 4B extraction routing baseline implemented:
    - profile-based Instructor routing now supports `low_latency` / `balanced` / `high_reasoning` paths,
    - deterministic xLAM/Ollama fallback chains are now constructed from config/env policy,
    - add/update extraction calls now accept operator model-profile hints with backward-compatible fallback behavior for legacy mocks/tests.
35. Phase 4C startup/session adaptation baseline implemented:
    - MCP initialize now performs startup readiness checks for Muninn/Ollama and attempts autostart when enabled,
    - initialize instructions now provide explicit startup prompts when dependencies are unavailable,
    - assistant-specific profile override now works via `MUNINN_OPERATOR_MODEL_PROFILE` with metadata injection defaults.
36. Phase 4D VRAM-aware profile policy baseline implemented:
    - extraction config now supports `MUNINN_VRAM_BUDGET_GB` to auto-select profile models by GPU budget tier,
    - 16GB-class defaults are now explicitly right-sized (`low=llama3.2:3b`, `balanced=qwen3:8b`, `high=qwen3:14b`),
    - 30B/32B routes are now reserved for explicit high-VRAM budgets rather than default high-reasoning baseline.
37. Phase 4E helper-first runtime scheduling baseline implemented:
    - extraction config now supports independent runtime/ingestion/legacy-ingestion profile defaults,
    - runtime add/update extraction now defaults to `runtime_model_profile` while ingestion paths default to ingestion-specific profiles,
    - MCP wrapper now supports operation-scoped overrides (`MUNINN_OPERATOR_RUNTIME_MODEL_PROFILE`, `MUNINN_OPERATOR_INGESTION_MODEL_PROFILE`, `MUNINN_OPERATOR_LEGACY_INGESTION_MODEL_PROFILE`) with generic fallback to `MUNINN_OPERATOR_MODEL_PROFILE`.
38. Phase 4F runtime profile-control tranche implemented:
    - design documented in `docs/plans/2026-02-14-phase4f-profile-control-design.md`,
    - runtime get/set profile APIs now shipped in memory core + REST + MCP + SDK,
    - profile mutations now work without server restarts for active assistant/IDE workflows.
39. Phase 4G profile-policy audit visibility baseline implemented:
    - runtime profile mutations are now persisted in metadata audit events,
    - profile mutation history is now queryable via memory core + REST + MCP + SDK.
40. Phase 4H local model-matrix benchmarking baseline implemented:
    - versioned local model matrix shipped (`eval/ollama_model_matrix.json`),
    - benchmark prompt pack shipped (`eval/ollama_benchmark_prompts.jsonl`),
    - local sync/benchmark CLI shipped (`python -m eval.ollama_local_benchmark ...`),
    - phase plan documented (`docs/plans/2026-02-14-phase4h-local-ollama-benchmarking.md`).
41. Phase 4I model ability/resource benchmarking baseline implemented:
    - rubric-based ability scoring added to live benchmark runs,
    - model summaries now include `ability_per_second` and `ability_per_vram_gb`,
    - `legacy-benchmark` mode added for deterministic old-project ingestion-like evaluation,
    - targeted unit tests added (`tests/test_ollama_local_benchmark.py`).
42. Phase 4J profile-promotion gate baseline implemented in-branch:
    - policy file added (`eval/ollama_profile_promotion_policy.json`),
    - `profile-gate` command added to evaluate live/legacy benchmark reports against profile thresholds,
    - deterministic recommendation output added for `low_latency`, `balanced`, and `high_reasoning` promotion decisions.
43. Phase 4K phase-boundary hygiene gate baseline implemented:
    - new gate utility added (`python -m eval.phase_hygiene`) for deterministic phase/PR hygiene checks,
    - report output added at `eval/reports/hygiene/phase_hygiene_<timestamp>.json`,
    - test budget checks now include skipped/warning thresholds in addition to PR/review/check constraints,
    - command execution hardening applied (`shell=False` tokenized execution + JUnit-first pytest summary parsing).

### Verification evidence
- Full-suite verification now green in-session: `418 passed, 2 skipped, 0 warnings`.
- Targeted tests for this tranche now pass:
  - `29 passed` (`tests/test_eval_artifacts.py`, `tests/test_eval_presets.py`, `tests/test_eval_run.py`, `tests/test_eval_metrics.py`, `tests/test_eval_gates.py`, `tests/test_eval_statistics.py`)
  - `12 passed` (`tests/test_mcp_wrapper_protocol.py`)
  - `23 passed` (`tests/test_eval_artifacts.py`, `tests/test_eval_statistics.py`, `tests/test_eval_presets.py`, `tests/test_eval_run.py`, `tests/test_eval_gates.py`, `tests/test_eval_metrics.py`)
  - `21 passed` (`tests/test_eval_statistics.py`, `tests/test_eval_presets.py`, `tests/test_eval_run.py`, `tests/test_eval_gates.py`, `tests/test_eval_metrics.py`)
  - `15 passed` (`tests/test_eval_gates.py`, `tests/test_eval_metrics.py`, `tests/test_eval_run.py`)
  - `48 passed` (`tests/test_sqlite_feedback.py`, `tests/test_eval_metrics.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_weight_adapter.py`, `tests/test_eval_gates.py`)
  - `27 passed` (`tests/test_memory_feedback.py`, `tests/test_config.py`)
- Compile verification passed on all touched modules and tests.
- SDK tranche verification: `7 passed` (`tests/test_sdk_client.py`).
- Ingestion tranche verification: `51 passed` (`tests/test_ingestion_pipeline.py`, `tests/test_memory_ingestion.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_sdk_client.py`, `tests/test_config.py`).
- Legacy migration tranche verification: `32 passed` (`tests/test_ingestion_parser.py`, `tests/test_memory_ingestion.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_sdk_client.py`).
- UI + chronological ingestion verification: `34 passed` (`tests/test_ingestion_pipeline.py`, `tests/test_memory_ingestion.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_sdk_client.py`).
- PR-remediation tranche verification: `83 passed` (`tests/test_eval_metrics.py`, `tests/test_sdk_client.py`, `tests/test_ingestion_pipeline.py`, `tests/test_ingestion_parser.py`, `tests/test_ingestion_discovery.py`, `tests/test_memory_ingestion.py`, `tests/test_config.py`, `tests/test_mcp_wrapper_protocol.py`).
- Memory-chains tranche verification: `40 passed` (`tests/test_memory_chains.py`, `tests/test_hybrid_retriever.py`, `tests/test_memory_update_path.py`, `tests/test_config.py`, `tests/test_memory_feedback.py`) + `40 passed` (`tests/test_recall_trace.py`, `tests/test_feature_flags.py`).
- Phase 4D VRAM-policy verification: `36 passed` (`tests/test_config.py`, `tests/test_extraction_pipeline.py`).
- Phase 4E runtime-vs-ingestion profile scheduling verification: `69 passed` (`tests/test_config.py`, `tests/test_memory_ingestion.py`, `tests/test_memory_update_path.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_extraction_pipeline.py`).
- Phase 4F runtime profile-control verification: `45 passed` (`tests/test_memory_profiles.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_sdk_client.py`).
- Phase 4G profile-policy audit visibility verification: `49 passed` (`tests/test_memory_profiles.py`, `tests/test_sqlite_profile_policy_events.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_sdk_client.py`).
- Phase 4H local benchmark tooling smoke checks: `python -m eval.ollama_local_benchmark list` and `python -m eval.ollama_local_benchmark sync --dry-run`.
- Phase 4H initial five-model quick-pass benchmark snapshot captured and documented (`docs/plans/2026-02-14-phase4h-local-ollama-benchmarking.md`).
- Phase 4I/4J benchmark helper verification: `8 passed` (`tests/test_ollama_local_benchmark.py`).
- Phase 4K hygiene helper verification: `5 passed` (`tests/test_phase_hygiene.py`) and `13 passed` (`tests/test_ollama_local_benchmark.py`, `tests/test_phase_hygiene.py`).

### Newly discovered ROI optimizations (implemented)
1. **Tenant filter correctness + performance**: replaced fragile `metadata LIKE` user matching with JSON1 exact-match where available.
2. **Release gate enforceability**: eval CLI now supports fail-fast gate semantics for metric regressions and latency budget breaches.
3. **Interop resilience**: MCP version negotiation now explicitly handles protocol mismatches and avoids silent drift.
4. **Adaptive relevance tuning loop**: persisted implicit feedback now calibrates retrieval signal weights per `(user, namespace, project)` with bounded online multipliers.
5. **Counterfactual calibration option**: SNIPS-style inverse-propensity normalization reduces policy-exposure bias when rank/sampling probabilities are captured.
6. **Competency-aware eval reporting**: per-track metrics expose where retrieval quality improves or regresses instead of masking with only global averages.
7. **Reproducible gate policy profiles**: preset-based defaults reduce evaluation drift across environments and make release criteria machine-auditable.
8. **Statistical gate hardening**: paired significance + CI/effect-size reduces false promotion/rejection caused by topic-sample noise.
9. **Artifact integrity hardening**: checksum + reproducibility verification catches silent benchmark drift before release gating.
10. **Multiple-testing control for gate trust**: corrected p-values now reduce false positives from large track/cutoff/metric hypothesis families.
11. **Protocol diagnosability + standards alignment**: explicit JSON-RPC method/param errors remove silent MCP integration failure modes and improve interoperability debugging ROI.
12. **Artifact ops scalability**: one-shot `verify --all` preserves integrity/reproducibility guarantees as benchmark bundles grow, reducing CI and release-maintenance overhead.
13. **Telemetry privacy hardening**: bounded capture length and explicit runbook policy reduce sensitive-data spill risk while preserving incident-debug capability.
14. **SDK integration throughput + reliability**: reusable sync/async transports with typed error channels reduce connection churn and improve deterministic handling in agent runtime loops.
15. **Ingestion blast-radius reduction**: per-source fail-open parsing with strict chunking invariants prevents single bad files from halting batch ingestion while preserving auditability.
16. **Cross-assistant historical continuity at scale**: source discovery + selection-based import closes manual migration gaps and creates measurable ROI by preserving prior project context across IDE/assistant switches.
17. **Operational adoption ROI via browser UX**: consolidating discovery/import/project-ingest/search controls into one UI lowers operator friction and reduces CLI-only dependency for memory maintenance workflows.
18. **Causal-context continuity ROI**: memory-chain edge persistence + retrieval expansion improves multi-step incident/debug recall, reducing repeated root-cause rediscovery across sessions.
19. **Helper-first VRAM budget control ROI**: decoupling runtime extraction profile from ingestion profiles preserves low-latency memory assist during active coding while still allowing higher-caliber offline ingest/import passes.
20. **Live policy adaptation ROI**: runtime profile control API removes restart overhead and enables assistant/IDE orchestration to adjust helper vs ingest compute posture in-session.
21. **Operational traceability ROI**: profile-policy mutation events make dynamic runtime tuning auditable across sessions and assistants, reducing silent drift risk.
22. **Local model-selection evidence ROI**: versioned model matrix plus prompt-stable benchmark harness enables repeatable 16GB-class profiling decisions instead of ad-hoc model swaps.

### High-ROI SOTA additions from web research now required in roadmap
1. MCP 2025-11-25 compatibility tranche (tasks, elicitation schema/defaults, JSON Schema 2020-12 assumptions, tool metadata improvements).
2. Memory-specific benchmark gate using MemoryAgentBench competencies (accurate retrieval, test-time learning, long-range understanding, selective forgetting).
3. GenAI observability tranche using OpenTelemetry GenAI semantic conventions (opt-in content capture + privacy-aware controls).
4. Adaptive model-caliber routing: keep xLAM as optional provider, maintain profile-based fallback chains (low-latency/balanced/high-reasoning), and expose assistant-session profile selection independent of think-level toggles.

## Executive Summary

This plan advances Muninn from v3.0 (the most technically complete local-first MCP memory server) to v3.3.0 (definitively SOTA+ in the MCP memory category) by closing high-ROI gaps first and sequencing remaining Phase 3 work.

**Gaps addressed to date:**
1. LLM extraction less nuanced than cloud competitors
2. Python SDK for programmatic use (now implemented: sync + async + mem0-style aliases)
3. Windows-centric deployment (excludes 70%+ of developers)
4. Retrieval quality gating/observability enforceability for release integrity

**Still open gaps:**
1. Ingestion hardening follow-ups (parser sandbox/process isolation for optional binary backends and broader enterprise corpus adapters)
2. Benchmark breadth expansion for additional adversarial/noise slices and domain diversity
3. Profile auto-promotion operationalization still needs implementation (operator-triggered orchestration, controlled policy-apply pipeline, rollback checkpoints, and alerting)

**Advancements implemented to date:**
5. Explainable recall traces (UNIQUE — no competitor has this)
6. Adaptive retrieval weights (entropy-based dynamic fusion)
7. NLI-based conflict detection (UNIQUE — no competitor has this)
8. Semantic deduplication at ingestion + consolidation
9. Python SDK sync/async interoperability layer
10. Multi-source ingestion with provenance-rich fail-open parsing
11. Memory chains with temporal/causal linking and retrieval expansion

**After all features, Muninn will be the ONLY memory system that combines:**
- Local-first architecture (no cloud dependency)
- 6-signal hybrid retrieval with adaptive weights (vector, graph, bm25, temporal, goal, chain)
- Explainable recall traces with per-signal attribution
- NLI-based conflict detection for memory integrity
- Structured extraction via Instructor (matches Mem0 quality)
- Memory chains with temporal/causal linking
- Semantic deduplication at ingestion + consolidation
- Cross-platform (Windows/Linux/macOS/Docker)
- Multi-source ingestion (files, conversations, APIs)
- Python SDK + REST API + MCP protocol
- Background consolidation daemon (neuroscience-inspired)
- 4-tier memory hierarchy with promotion
- Bi-temporal provenance tracking

---

## Phase 1: Foundation (v3.1.0)

> **Priority**: 🔴 HIGHEST
> **Risk**: LOW
> **Effort**: 6–9 days
> **Theme**: Unblock users, improve quality, create unique differentiator

### 1A. Platform Abstraction

**Problem**: Muninn is Windows-centric. pywin32/pystray optional deps, some hardcoded paths, subprocess flags are Windows-only. This excludes Linux, macOS, and Docker users (70%+ of developer market).

**Solution**: Abstract all platform-specific code behind `muninn/platform.py` using `platformdirs` library.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/platform.py` | Cross-platform path resolution, process flags, detection utilities |
| `Dockerfile` | Container deployment image |
| `docker-compose.yml` | Full stack: Muninn + Ollama |
| `tests/test_platform.py` | Platform detection and path tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/config.py` | Use `platform.get_data_dir()` for default paths |
| `mcp_wrapper.py` | Use `platform.get_process_creation_flags()` instead of Windows-specific flags |
| `pyproject.toml` | Add `platformdirs` dependency (MIT license) |

**Technical Design:**

```python
# muninn/platform.py
import os, sys
from pathlib import Path

try:
    from platformdirs import user_data_dir, user_config_dir, user_log_dir
except ImportError:
    def user_data_dir(appname, appauthor=None):
        if sys.platform == "win32":
            return str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / appname)
        elif sys.platform == "darwin":
            return str(Path.home() / "Library" / "Application Support" / appname)
        else:
            return str(Path(os.environ.get("XDG_DATA_HOME",
                           Path.home() / ".local" / "share")) / appname)

def get_data_dir() -> Path:
    return Path(os.environ.get("MUNINN_DATA_DIR", user_data_dir("muninn")))

def get_process_creation_flags() -> int:
    if sys.platform == "win32":
        import subprocess
        return subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    return 0

def is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("MUNINN_DOCKER") == "1"
```

**Docker Support:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[all]"
ENV MUNINN_DATA_DIR=/data MUNINN_HOST=0.0.0.0
EXPOSE 42069
VOLUME /data
CMD ["python", "-m", "muninn.server"]
```

**Dependency**: `platformdirs` — MIT License ✅

---

### 1B. Extraction Enhancement (Instructor Integration)

**Problem**: Current xLAM extraction uses raw prompt → JSON parse, which is fragile and unvalidated. Ollama extraction is unimplemented. No structured output guarantees.

**Solution**: Replace fragile JSON parsing with the `instructor` library, which uses Pydantic models to guarantee structured output with automatic retry. Works with ANY OpenAI-compatible endpoint (xLAM, Ollama, vLLM, LM Studio, cloud).

**New Files:**
| File | Purpose |
|---|---|
| `muninn/extraction/models.py` | Pydantic schemas for structured extraction |
| `muninn/extraction/instructor_extractor.py` | Instructor-based extractor |
| `tests/test_instructor_extractor.py` | Extraction quality tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/extraction/pipeline.py` | Integrate Instructor as Tier 2, improve routing |
| `muninn/core/config.py` | Add `extraction.provider` config (instructor/xlam/ollama) |
| `pyproject.toml` | Add `instructor` dependency (MIT license) |

**Technical Design:**

```python
# muninn/extraction/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name, properly capitalized")
    entity_type: str = Field(description="One of: person|org|tech|concept|project|file|preference|location")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

class ExtractedRelation(BaseModel):
    subject: str = Field(description="Source entity name")
    predicate: str = Field(description="Relationship: uses|prefers|created|depends_on|knows|part_of|works_with")
    object: str = Field(description="Target entity name")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    temporal_context: Optional[str] = Field(default=None)

class ExtractedMemoryFacts(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list, description="Atomic factual statements")
    summary: Optional[str] = Field(default=None, description="One-sentence summary")
    temporal_context: Optional[str] = Field(default=None)
```

```python
# muninn/extraction/instructor_extractor.py
import instructor
from openai import OpenAI
from .models import ExtractedMemoryFacts

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine. Given text from a conversation
or document, extract structured facts including entities (people, technologies, organizations,
concepts, projects, preferences), relationships between entities, key factual statements,
and a one-sentence summary. Be precise and factual. Only extract what is explicitly stated."""

class InstructorExtractor:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self.client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=api_key),
            mode=instructor.Mode.JSON
        )
        self.model = model

    def extract(self, text: str) -> ExtractedMemoryFacts:
        return self.client.chat.completions.create(
            model=self.model,
            response_model=ExtractedMemoryFacts,
            max_retries=2,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract structured facts:\n\n{text}"}
            ],
        )
```

**Routing Logic (Enhanced Pipeline):**
```
Tier 1: Rules (always, ~0ms) — guaranteed baseline
Tier 2: Instructor (if endpoint configured, ~200ms–2s) — structured, validated
Merge: Union entities/relations, deduplicate by lowercase name
```

**Key Insight**: Instructor eliminates the need for SEPARATE xLAM and Ollama extractors. One extractor handles ALL backends because they all speak the OpenAI API format. This simplifies the codebase while dramatically improving quality.

**Dependency**: `instructor` — MIT License ✅

---

### 1C. Explainable Recall Traces

**Problem**: search() returns opaque combined scores. Users/agents have no idea WHY a particular memory was recalled, making debugging and trust impossible.

**Solution**: Track per-signal contributions through RRF fusion and return detailed `RecallTrace` objects explaining each retrieval decision.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/core/recall_trace.py` | RecallTrace, SignalContribution dataclasses |
| `tests/test_recall_trace.py` | Trace generation and explanation tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/types.py` | Import and re-export RecallTrace types |
| `muninn/retrieval/hybrid.py` | Track per-signal scores, generate traces |
| `server.py` | Include trace in search response when requested |
| `mcp_wrapper.py` | Format trace in search results |

**Technical Design:**

```python
# muninn/core/recall_trace.py
from pydantic import BaseModel
from typing import List, Optional

class SignalContribution(BaseModel):
    signal: str                  # "vector"|"bm25"|"graph"|"temporal"
    raw_score: float             # Original score from that signal
    rank: int                    # Rank position in that signal's results
    rrf_contribution: float      # Actual RRF score contribution
    weight: float                # Weight applied to this signal
    explanation: str             # Human-readable explanation

class RecallTrace(BaseModel):
    memory_id: str
    final_score: float
    signals: List[SignalContribution]
    rerank_score: Optional[float] = None
    importance_boost: float = 0.0
    dominant_signal: str         # Which signal contributed most
    explanation: str             # Full human-readable explanation
```

**Implementation in HybridRetriever:**

```python
def _rrf_fusion_with_traces(self, signal_results):
    """Modified RRF that tracks attribution per signal per document"""
    traces = {}
    for signal_name, results in signal_results.items():
        weight = self.weights[signal_name]
        for rank, (memory_id, raw_score) in enumerate(results):
            if memory_id not in traces:
                traces[memory_id] = RecallTrace(memory_id=memory_id, signals=[], ...)
            rrf_contrib = weight / (RRF_K + rank + 1)
            traces[memory_id].signals.append(SignalContribution(
                signal=signal_name, raw_score=raw_score, rank=rank,
                rrf_contribution=rrf_contrib, weight=weight,
                explanation=self._explain_signal(signal_name, raw_score, rank)
            ))
    for trace in traces.values():
        trace.final_score = sum(s.rrf_contribution for s in trace.signals)
        trace.dominant_signal = max(trace.signals, key=lambda s: s.rrf_contribution).signal
        trace.explanation = self._generate_explanation(trace)
    return traces
```

**Example Output:**
```json
{
  "memory": {"id": "abc123", "content": "User prefers Python for backend..."},
  "score": 0.847,
  "trace": {
    "dominant_signal": "vector",
    "explanation": "Recalled primarily due to semantic similarity (0.91) to query, with keyword match on 'Python' (BM25 rank #1) and graph connection via 'Python' entity",
    "signals": [
      {"signal": "vector", "raw_score": 0.91, "rank": 0, "rrf_contribution": 0.016, "explanation": "High semantic similarity (0.91) to query"},
      {"signal": "bm25", "raw_score": 4.2, "rank": 0, "rrf_contribution": 0.013, "explanation": "Keyword match: 'Python' (BM25 score 4.2)"},
      {"signal": "graph", "raw_score": 1.0, "rank": 2, "rrf_contribution": 0.016, "explanation": "Connected via entity 'Python' (2-hop graph traversal)"},
      {"signal": "temporal", "raw_score": 0.72, "rank": 5, "rrf_contribution": 0.008, "explanation": "Moderately recent (3 days old, importance 0.72)"}
    ]
  }
}
```

**Competitive Advantage**: NO competitor (Mem0, Graphiti, Memento, MemoryGraph) provides per-signal retrieval explanations. This is a unique differentiator.

**Dependencies**: None (uses existing Pydantic, existing retrieval infrastructure)

---

### 1D. Feature Flags System

**Purpose**: Centralize all feature toggles to prevent configuration sprawl.

**New File:**
| File | Purpose |
|---|---|
| `muninn/core/feature_flags.py` | Centralized feature flag management |

```python
from dataclasses import dataclass, field

@dataclass
class FeatureFlags:
    # Phase 1 (default ON — low cost, high value)
    explainable_recall: bool = True
    instructor_extraction: bool = True  # When endpoint available

    # Phase 2 (default OFF — higher cost, opt-in)
    conflict_detection: bool = False
    semantic_dedup: bool = False
    adaptive_weights: bool = False

    # Phase 3 (default OFF — require additional deps)
    memory_chains: bool = False
    multi_source_ingestion: bool = False

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        import os
        return cls(
            explainable_recall=os.environ.get("MUNINN_EXPLAIN_RECALL", "1") == "1",
            conflict_detection=os.environ.get("MUNINN_CONFLICT_DETECTION", "0") == "1",
            # ... etc
        )
```

---


## Phase 1.1: Stabilization & Measurement Gate (v3.1.1)

> **Priority**: 🔴 HIGHEST (BLOCKER before v3.2 claims)
> **Risk**: LOW
> **Effort**: 3–5 days
> **Theme**: Correctness, trustworthiness, measurable retrieval quality

### 1.1A. Correctness Fix Bundle

**Scope (must ship together):**
1. **Instructor wiring fix** — pass `instructor_base_url`, `instructor_model`, and `instructor_api_key` from config into `ExtractionPipeline` initialization.
2. **Docker path contract fix** — make `get_data_dir()` use `is_running_in_docker()` (not only `MUNINN_DOCKER`) to align behavior with tests and container reality.
3. **Recall trace fidelity fix** — store true signal-native raw scores where available (vector cosine, BM25 score, graph traversal confidence, temporal decay score), not rank proxies.
4. **Version consistency fix** — unify `pyproject.toml`, `muninn.__version__`, MCP wrapper `serverInfo.version`, and docs.

**Acceptance Criteria:**
- Zero known correctness mismatches from v3.1 gap audit.
- All platform and extraction tests pass on Linux/macOS/Windows CI.

### 1.1B. Retrieval Evaluation Gate (vibecoder-focused)

**Why:** SOTA claims require reproducible metrics, especially for adaptive weighting and explainability.

**Deliverables:**
- `eval/` harness with reproducible query sets from real coding workflows:
  - project goal recall
  - dependency decisions
  - architecture constraints
  - recent task continuity across assistants
- Metrics: `nDCG@k`, `Recall@k`, `MRR`, p50/p95 latency, and contradiction false-positive rate.
- A/B mode: fixed weights vs adaptive weights; explain off vs explain on.

**Research Basis:** Hybrid retrieval and RRF evaluation practices from Elastic/Qdrant/Pinecone and BEIR-style benchmarking.

### 1.1C. Goal-Drift Guardrail (new, high ROI)

**Problem:** For vibecoders hopping between assistants/IDEs, sessions drift and lose the primary project goal.

**Solution:** Add a lightweight "Goal Compass" memory primitive:
- `ProjectGoal` record (north-star objective, constraints, definition-of-done).
- `GoalDriftCheck` at query/add time:
  - compute semantic distance between current intent and `ProjectGoal`.
  - if drift > threshold, prepend a concise steering reminder in responses.
- `goal_relevance` score becomes a fifth retrieval signal (low cost, high UX value).

**New Files (proposed):**
- `muninn/goal/compass.py`
- `muninn/goal/drift.py`
- `tests/test_goal_compass.py`

**Backward compatibility:** Fully optional via feature flag (`MUNINN_GOAL_COMPASS=1`).

---


## Phase 2: Intelligence (v3.2.0)

> **Priority**: 🟡 HIGH
> **Risk**: MEDIUM
> **Effort**: 7–10 days
> **Theme**: Memory integrity, retrieval quality, intelligent adaptation

### 2A. NLI-Based Conflict Detection

**Problem**: Contradictory memories coexist silently. "User prefers Python" and "User switched to Rust" both retrieved, confusing agents.

**Solution**: Use Natural Language Inference (NLI) cross-encoder model to detect contradiction between new and existing memories during add().

**New Files:**
| File | Purpose |
|---|---|
| `muninn/conflict/__init__.py` | Conflict detection package |
| `muninn/conflict/detector.py` | NLI-based contradiction detection |
| `muninn/conflict/resolver.py` | Resolution strategies (supersede, merge, flag) |
| `tests/test_conflict_detector.py` | Detection accuracy tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/memory.py` | Add conflict check in `add()` method |
| `muninn/core/types.py` | Add `ConflictResult`, `ConflictResolution` |
| `muninn/core/config.py` | Conflict detection config options |
| `server.py` | Return conflicts in add response |

**Technical Design:**

```python
# muninn/conflict/detector.py
class ConflictDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        # DeBERTa-v3-small: 44MB, Apache-2.0 license, 91.65% accuracy on SNLI
        # Outputs: [contradiction_score, entailment_score, neutral_score]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def detect_conflicts(self, new_content: str,
                         existing_memories: List[MemoryRecord],
                         threshold: float = 0.7) -> List[ConflictResult]:
        conflicts = []
        for existing in existing_memories:
            features = self.tokenizer(new_content, existing.content,
                                      padding=True, truncation=True, return_tensors="pt")
            scores = self.model(**features).logits.softmax(dim=1)[0]
            # scores = [contradiction, entailment, neutral]
            if scores[0].item() > threshold:
                conflicts.append(ConflictResult(
                    new_content=new_content,
                    existing_memory=existing,
                    contradiction_score=scores[0].item(),
                    suggested_resolution=self._suggest_resolution(existing, scores)
                ))
        return conflicts
```

**Resolution Strategies:**
| Strategy | When Applied | Action |
|---|---|---|
| `SUPERSEDE` | New memory is clearly more recent, same topic | Archive old, store new |
| `MERGE` | Both partially true, complementary | Combine into unified memory |
| `FLAG_FOR_REVIEW` | High contradiction, both seem valid | Return to user/agent for decision |
| `KEEP_EXISTING` | New memory low confidence, existing high | Discard new |

**Performance Mitigation:**
- Only runs when `feature_flags.conflict_detection == True`
- Only checks against memories with similarity > 0.8 (pre-filtered by vector search)
- DeBERTa-v3-small is fast (~10ms per pair on CPU)
- Model loaded once, cached in memory

**Model**: `cross-encoder/nli-deberta-v3-small` — Apache-2.0 License ✅
**Dependency**: `transformers` + `torch` (only when conflict detection enabled) — Apache-2.0 ✅

---

### 2B. Semantic Deduplication

**Problem**: Same fact expressed differently creates redundant memories. "I use VS Code" and "My editor is Visual Studio Code" both stored separately.

**Solution**: Embedding-based near-duplicate detection at ingestion time using existing vector infrastructure.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/dedup/__init__.py` | Deduplication package |
| `muninn/dedup/semantic_dedup.py` | Embedding-based near-duplicate detection |
| `tests/test_semantic_dedup.py` | Dedup accuracy and false-positive tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/memory.py` | Add dedup check in `add()` before storage |
| `muninn/consolidation/daemon.py` | Enhanced merge with dedup |

**Technical Design:**

```python
# muninn/dedup/semantic_dedup.py
class SemanticDedup:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold  # Higher than consolidation merge (0.92)

    def check_duplicate(self, embedding, vector_store, content, metadata_store):
        matches = vector_store.search(embedding, limit=3, score_threshold=self.threshold)
        for memory_id, score in matches:
            existing = metadata_store.get(memory_id)
            if existing and score >= self.threshold:
                if self._content_overlap(content, existing.content) > 0.8:
                    return DedupResult(
                        is_duplicate=True,
                        existing_memory_id=memory_id,
                        similarity=score,
                        strategy=DedupStrategy.UPDATE_EXISTING
                    )
        return None
```

**Strategies:**
| Strategy | Action | When |
|---|---|---|
| `UPDATE_EXISTING` | Merge new info into existing memory | High similarity, new has additional detail |
| `SKIP` | Discard new memory entirely | Near-identical content |
| `LINK` | Store both, mark as related | Similar but distinct perspectives |

**Dependencies**: None — uses existing fastembed embeddings and Qdrant search ✅

---

### 2C. Adaptive Retrieval Weights

**Problem**: Fixed signal weights (`vector: 1.0, graph: 1.0, bm25: 0.8, temporal: 0.5`) are suboptimal for different query types. Short keyword queries benefit from BM25; temporal queries need temporal boost.

**Solution**: Entropy-based dynamic weighting that adapts per-query based on signal confidence and query characteristics.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/retrieval/weight_adapter.py` | Entropy-based + feedback-based weight adaptation |
| `tests/test_weight_adapter.py` | Weight adaptation correctness tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/retrieval/hybrid.py` | Use `WeightAdapter` instead of fixed `SIGNAL_WEIGHTS` |
| `muninn/store/sqlite_metadata.py` | Add `retrieval_feedback` table |

**Implementation status update (2026-02-14):**
- Retrieval feedback loop is now wired in:
  - persistence in SQLite (`retrieval_feedback`),
  - scoped multiplier computation,
  - `MuninnMemory.record_retrieval_feedback(...)`,
  - server endpoint `POST /feedback/retrieval`,
  - MCP tool `record_retrieval_feedback`,
  - adaptive retrieval path consumes feedback multipliers when enabled.

**Technical Design:**

```python
# muninn/retrieval/weight_adapter.py
import math

class WeightAdapter:
    DEFAULT_WEIGHTS = {"vector": 1.0, "graph": 1.0, "bm25": 0.8, "temporal": 0.5}

    def compute_weights(self, query: str, signal_results: dict) -> dict:
        weights = dict(self.DEFAULT_WEIGHTS)

        # 1. Query-based adaptation
        tokens = query.lower().split()
        if len(tokens) <= 3:
            weights["bm25"] *= 1.3           # Short queries → keyword match
        if any(t in tokens for t in ["recent", "latest", "today", "yesterday", "new"]):
            weights["temporal"] *= 2.0        # Temporal queries → recency
        if any(t in tokens for t in ["related", "connected", "about", "who"]):
            weights["graph"] *= 1.5           # Relational queries → graph

        # 2. Entropy-based confidence per signal
        for signal, results in signal_results.items():
            if results and len(results) > 1:
                scores = [r[1] for r in results if r[1] > 0]
                entropy = self._normalized_entropy(scores)
                confidence = 1.0 - entropy
                weights[signal] *= (0.5 + confidence)  # Scale [0.5, 1.5]

        return weights

    def _normalized_entropy(self, scores: list) -> float:
        if not scores or all(s == 0 for s in scores):
            return 1.0
        total = sum(scores)
        probs = [s / total for s in scores if s > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
```

**Academic Basis**: Inspired by "Entropy-Based Dynamic Hybrid Retrieval for Adaptive Query Weighting in RAG Pipelines" (ICML 2025 Workshop) and "Multi-Field Adaptive Retrieval" (ICLR 2025 Spotlight).

**Dependencies**: None — pure Python math ✅

---

## Phase 3: Ecosystem (v3.3.0)

> **Priority**: 🟢 MEDIUM
> **Risk**: MEDIUM
> **Effort**: 9–13 days
> **Theme**: Advanced capabilities, broader adoption, growth features

### 3A. Memory Chains

**Problem**: Memories are isolated nodes. Sequential events ("Started project X" → "Encountered bug" → "Fixed bug" → "Completed project") have no causal/temporal linking.

**Solution**: Detect temporal and causal chain relationships between memories, store as directed graph edges, enable chain retrieval.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/chains/__init__.py` | Chain detection package |
| `muninn/chains/detector.py` | Temporal/causal chain link detection |
| `muninn/chains/retriever.py` | Chain traversal and reconstruction |
| `tests/test_chain_detector.py` | Chain detection accuracy tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/store/graph_store.py` | New `PRECEDES`/`CAUSES` edge types, chain traversal queries |
| `muninn/retrieval/hybrid.py` | Chain expansion in graph search results |
| `muninn/core/types.py` | `ChainLink`, `MemoryChain` types |

**Chain Detection Algorithm:**
```
Score = temporal_proximity + entity_overlap + causal_markers + project_match
- temporal_proximity: 1.0 - (hours_apart / 24.0), capped at [0, 1]
- entity_overlap: shared_entities * 0.3
- causal_markers: +0.5 if causal language detected ("because", "therefore", "led to")
- project_match: +0.2 if same project context
- Threshold: score >= 0.6 → create chain link
```

**Graph Schema Extensions:**
```cypher
CREATE REL TABLE PRECEDES(FROM Memory TO Memory, confidence DOUBLE, shared_entities STRING[])
CREATE REL TABLE CAUSES(FROM Memory TO Memory, confidence DOUBLE, evidence STRING)
```

**Dependencies**: None — uses existing Kuzu graph store ✅

---

### 3B. Multi-Source Ingestion

**Problem**: Muninn only accepts memory via `add()` API call with string content. Users can't import existing knowledge bases, conversation histories, or documents.

**Solution**: Pluggable ingestion pipeline: Source → Parse → Chunk → Extract → Store.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/ingestion/__init__.py` | Ingestion package |
| `muninn/ingestion/pipeline.py` | Main ingestion orchestrator |
| `muninn/ingestion/chunker.py` | Semantic chunking with overlap |
| `muninn/ingestion/parsers/text.py` | .txt and .md parsing |
| `muninn/ingestion/parsers/pdf.py` | .pdf parsing (pypdf) |
| `muninn/ingestion/parsers/docx.py` | .docx parsing (python-docx) |
| `muninn/ingestion/parsers/html.py` | .html parsing (beautifulsoup4) |
| `muninn/ingestion/parsers/json_parser.py` | .json conversation import |
| `muninn/ingestion/parsers/csv_parser.py` | .csv data import |
| `muninn/ingestion/sources/chatgpt.py` | ChatGPT export format adapter |
| `muninn/ingestion/sources/obsidian.py` | Obsidian vault adapter |
| `tests/test_ingestion_pipeline.py` | Pipeline integration tests |
| `tests/test_chunker.py` | Chunking quality tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/memory.py` | New `ingest()` method |
| `server.py` | New `/ingest` endpoint |
| `mcp_wrapper.py` | New `ingest_file` MCP tool |
| `pyproject.toml` | New `[ingestion]` extras: pypdf, python-docx, beautifulsoup4 |

**Supported Formats:**
| Format | Library | License |
|---|---|---|
| .txt, .md | Built-in | — |
| .pdf | pypdf | BSD-3 ✅ |
| .docx | python-docx | MIT ✅ |
| .html | beautifulsoup4 | MIT ✅ |
| .json | Built-in | — |
| .csv | Built-in | — |

**⚠️ CRITICAL**: pymupdf/PyMuPDF is AGPL-3.0 — INCOMPATIBLE with Apache-2.0. Use `pypdf` (BSD) instead.

**Chunking Strategy:**
```
1. Parse → typed elements (paragraph, heading, code_block, table, list)
2. Keep code blocks and tables as single chunks (preserve context)
3. Merge small paragraphs up to max_tokens (512 default)
4. Split large paragraphs at sentence boundaries
5. Add overlap from previous chunk (50 tokens default) for context
```

---

### 3C. Python SDK

**Problem**: Muninn can only be used via REST API (server.py) or MCP protocol (mcp_wrapper.py). No programmatic Python API for direct integration.

**Solution**: Sync + Async Python SDK wrapping MuninnMemory directly.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/sdk/__init__.py` | Public SDK API |
| `muninn/sdk/client.py` | `MuninnClient` (sync) |
| `muninn/sdk/async_client.py` | `AsyncMuninnClient` (async) |
| `muninn/sdk/models.py` | SDK-specific Pydantic models |
| `tests/test_sdk_client.py` | Sync client tests |
| `tests/test_sdk_async.py` | Async client tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/__init__.py` | Export `Memory = MuninnClient` for convenience |
| `pyproject.toml` | New `[sdk]` extras group |

**API Design (Mem0-Compatible):**

```python
from muninn import Memory

# Simple usage
m = Memory()
m.add("User prefers Python for backend", user_id="alice")
results = m.search("programming language preference", user_id="alice")
m.close()

# Context manager
with Memory(data_dir="~/.myapp/memory") as m:
    m.add("Important fact", metadata={"project": "myapp"})
    results = m.search("fact", explain=True)  # With recall traces

# Async
from muninn import AsyncMemory
async with AsyncMemory() as m:
    await m.add("Async fact")
    results = await m.search("fact")
```

**Dependencies**: None — wraps existing MuninnMemory engine ✅

---


### 3D. Cross-Assistant Handoff + Interop Pack (Vibecoder Priority)

**Problem**: Users switch constantly between Claude, ChatGPT, Cursor, Windsurf, Copilot, CLI agents, and IDE workflows. State continuity is fragile.

**Solution**: Add a portable "handoff bundle" and interop contract.

**Key capabilities:**
- `export_handoff(project_id)` creates signed JSON bundle containing:
  - current goal + constraints
  - recent decisions and unresolved questions
  - top memories by importance/recency
  - provenance and checksum
- `import_handoff(bundle)` performs idempotent merge with conflict-aware reconciliation.
- MCP tool + REST endpoint parity for handoff operations.

**High-performance implementation notes:**
- Use append-only event IDs + idempotent receiver semantics for safe repeated imports.
- Keep bundle small via summarization + top-k memory selection.
- Store `source_assistant`, `source_workspace`, and `source_session` for traceability.

**Dependencies**: None required (standard library signing + hashing); optional compression via `zstd`.

---


## Dependency Summary

### New Dependencies by Phase

| Phase | Dependency | License | Required/Optional | Size |
|---|---|---|---|---|
| 1 | `platformdirs` | MIT | Required | ~50KB |
| 1 | `instructor` | MIT | Optional (extraction) | ~200KB |
| 1 | `openai` | Apache-2.0 | Optional (via instructor) | ~500KB |
| 2 | `transformers` | Apache-2.0 | Optional (conflict detection) | ~8MB |
| 2 | `torch` | BSD-3 | Optional (conflict detection) | ~200MB |
| 3 | `pypdf` | BSD-3 | Optional (ingestion) | ~2MB |
| 3 | `python-docx` | MIT | Optional (ingestion) | ~1MB |
| 3 | `beautifulsoup4` | MIT | Optional (ingestion) | ~500KB |

**pyproject.toml Extras Groups:**
```toml
[project.optional-dependencies]
extraction = ["instructor>=1.0", "openai>=1.0"]
conflict = ["transformers>=4.30", "torch>=2.0"]
ingestion = ["pypdf>=4.0", "python-docx>=1.0", "beautifulsoup4>=4.12"]
sdk = []  # No additional deps, uses core
all = ["muninn-mcp[extraction,conflict,ingestion]"]
```

---

## Risk Matrix

| # | Feature | Risk | Impact if Fails | Mitigation |
|---|---|---|---|---|
| 1A | Platform Abstraction | LOW | Path errors on new platforms | Comprehensive platform tests, CI matrix |
| 1B | Extraction Enhancement | LOW-MED | Extraction quality regression | Rules tier always works, Instructor is additive |
| 1C | Explainable Recall | LOW | Slightly slower search response | Trace construction is O(n) lightweight |
| 2A | Conflict Detection | MEDIUM | Slower add() operations | Optional flag, async option, small model |
| 2B | Semantic Dedup | LOW-MED | False positive dedup (loses memory) | High threshold (0.95), secondary text check |
| 2C | Adaptive Weights | MEDIUM | Retrieval quality regression | Falls back to fixed weights, A/B testable |
| 3A | Memory Chains | LOW-MED | False chain links | Confidence threshold, only causal markers |
| 3B | Multi-Source Ingestion | MEDIUM | Parser failures on edge-case docs | Per-parser error handling, skip-and-continue |
| 3C | Python SDK | LOW | Event loop conflicts | Detect async context, clear error messages |

---

## ROI Prioritization

| Rank | Feature | Effort | Value | Unique? |
|---|---|---|---|---|
| 1 | Explainable Recall Traces | 2 days | 🔴 HIGHEST | ✅ UNIQUE |
| 2 | Platform Abstraction | 2-3 days | 🔴 HIGH | Growing expectation |
| 3 | Extraction Enhancement | 2-3 days | 🔴 HIGH | Parity + quality |
| 4 | Conflict Detection | 3-4 days | 🟡 HIGH | ✅ UNIQUE |
| 5 | Semantic Dedup | 2-3 days | 🟡 MEDIUM-HIGH | Common need |
| 6 | Python SDK | 3-4 days | 🟡 MEDIUM-HIGH | Distribution req. |
| 7 | Adaptive Weights | 2-3 days | 🟢 MEDIUM | Research-backed |
| 8 | Memory Chains | 2-3 days | 🟢 MEDIUM | Advanced graph |
| 9 | Multi-Source Ingestion | 4-5 days | 🟢 MEDIUM | Growth feature |

---

## Validation Checklist

- [ ] ALL existing APIs remain unchanged (backward compatible)
- [ ] ALL new features are optional and disabled by default where appropriate
- [ ] ALL new dependencies are Apache-2.0/MIT/BSD compatible
- [ ] NO code copied from competitors (100% original implementation)
- [ ] NO AGPL dependencies (pymupdf explicitly avoided)
- [ ] Every new module has corresponding unit tests
- [ ] Feature flags centralize all toggles
- [ ] Docker deployment works with stdio MCP and REST modes
- [ ] Cross-platform CI validates Windows + Linux + macOS
- [ ] Performance impact measured for each hot-path addition (add, search)

---

## Implementation Timeline

```
Phase 1 (v3.1.0): Weeks 1-2
├── 1A. Platform Abstraction (Days 1-3)
├── 1B. Extraction Enhancement (Days 3-5)
├── 1C. Explainable Recall (Days 5-7)
├── 1D. Feature Flags (Day 7)
└── Testing + Validation (Days 8-9)

Phase 1.1 (v3.1.1): Week 3
├── 1.1A. Correctness Fix Bundle (Days 1-2)
├── 1.1B. Retrieval Evaluation Gate (Days 2-4)
├── 1.1C. Goal-Drift Guardrail (Days 4-5)
└── Testing + Validation (Day 5)

Phase 2 (v3.2.0): Weeks 4-5
├── 2A. Conflict Detection (Days 1-4)
├── 2B. Semantic Dedup (Days 4-6)
├── 2C. Adaptive Weights (Days 6-8)
└── Testing + Validation (Days 9-10)

Phase 3 (v3.3.0): Weeks 6-8
├── 3A. Memory Chains (Days 1-3)
├── 3B. Multi-Source Ingestion (Days 3-8)
├── 3C. Python SDK (Days 8-11)
├── 3D. Cross-Assistant Handoff + Interop Pack (Days 11-12)
└── Testing + Validation (Days 12-13)
```

---

## Appendix: Competitive Positioning After SOTA+

| Feature | Muninn v3.3 | Mem0 | Graphiti/Zep | Memento | MemoryGraph |
|---|---|---|---|---|---|
| Local-first | ✅ | ❌ Cloud | ❌ Neo4j | ✅ | ✅ |
| Explainable recall | ✅ UNIQUE | ❌ | ❌ | ❌ | ❌ |
| Conflict detection | ✅ UNIQUE | Partial | ❌ | ❌ | ❌ |
| Hybrid retrieval (4-signal) | ✅ | Vector only | Graph + Vector | Vector | Graph |
| Adaptive weights | ✅ | ❌ | ❌ | ❌ | ❌ |
| Memory chains | ✅ | ❌ | ✅ Temporal KG | ❌ | ❌ |
| Structured extraction | ✅ Instructor | ✅ LLM | ✅ LLM | ❌ | ❌ |
| Multi-source ingestion | ✅ | ❌ | ✅ Episodes | ❌ | ❌ |
| Python SDK | ✅ | ✅ | ✅ | ❌ | ❌ |
| MCP protocol | ✅ | ❌ | ❌ | ✅ | ✅ |
| Cross-platform | ✅ | ✅ | ✅ | ✅ | ✅ |
| Background consolidation | ✅ UNIQUE | ❌ | ❌ | ❌ | ❌ |
| Memory hierarchy (4-tier) | ✅ UNIQUE | ❌ | ❌ | ❌ | ❌ |
| Bi-temporal provenance | ✅ | ❌ | ✅ | ❌ | ❌ |
| Semantic dedup | ✅ | ❌ | ❌ | ❌ | ❌ |
