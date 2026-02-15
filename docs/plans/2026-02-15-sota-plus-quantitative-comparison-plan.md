# SOTA+ Quantitative Comparison Plan

Date: 2026-02-15  
Status: In progress (Phase 4AF baseline + Phase 5A continuation hardening applied)

## Objective

Define a deterministic, evidence-grade quantitative comparison protocol that decides whether Muninn meets SOTA+ standards at release time.

## Implemented Baseline (2026-02-15)

- Unified gate runner now exists: `python -m eval.ollama_local_benchmark sota-verdict`.
- Verdict output now includes:
  - normalized candidate/baseline eval evidence,
  - optional normalized profile-gate + transport + auxiliary benchmark evidence,
  - deterministic gate-family outcomes for quality/reliability/statistical/reproducibility/profile-policy.
- Implementation details are captured in `docs/plans/2026-02-15-phase4af-unified-sota-verdict-command.md`.

## Cadence Decision (2026-02-15)

For active implementation/enhancement phases, use deferred benchmark cadence:

1. Run fast deterministic tranche checks continuously (unit/protocol/hygiene + targeted soak checks).
2. Reuse bounded-age benchmark reports for interim policy evaluation when needed (`dev-cycle --defer-benchmarks`).
3. Run full benchmark matrix replay at release-readiness boundaries and scheduled CI windows.

This keeps implementation throughput high without weakening final SOTA+ evidence requirements.

## Cadence Reaffirmation (2026-02-15, Phase 5A)

Decision remains enhancement-first:

1. Continue implementing remaining architecture/performance improvements first.
2. Keep running fast deterministic regression gates continuously.
3. Delay full benchmark matrix replay until release-readiness boundaries, not after every tranche.

Rationale:

- Reduces wasted benchmark spend while core capabilities are still moving.
- Avoids noisy intermediate benchmark churn from partially implemented features.
- Preserves final evidentiary rigor by requiring full matrix + gate-family pass before SOTA+ labeling.

## Interim Mitigation Update (2026-02-15, continuation tranche)

Implemented to reduce external host-side 120s transport timeout risk while blocker remains open:

1. MCP tool response payloads are now bounded in wrapper transport (`MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS`), with deterministic truncation metadata.
2. Search text responses now use the same bounded transport limiter.
3. MCP public error messages now redact connection/timeout/internal details while preserving actionable validation errors.
4. Browser dashboard rendering now avoids dynamic `innerHTML` injection paths for operator-visible result/table payloads.
5. Long-running MCP tools now support automatic task-mode deferral (`tools/call` auto-task) so heavy ingest calls can return immediate task handles instead of consuming the host synchronous timeout window.
6. Transport closure campaign automation now exists via `python -m eval.mcp_transport_closure`, producing deterministic closure-evidence artifacts with explicit criterion booleans.

Current assessment:

- Risk is reduced for large-response and reflected-error classes.
- Blocker remains open until closure criteria are met across rolling soak windows in host runtime.

## Decision Rule

A release is "SOTA+" only if all four gates pass:

1. Quality Superiority Gate
2. Reliability and Latency Gate
3. Statistical Validity Gate
4. Reproducibility and Integrity Gate

Any gate failure blocks SOTA+ labeling.

## Comparison Set

### Candidate

- Current release candidate (RC)

### Baselines

- Baseline A: last stable release tag
- Baseline B: fixed-weight retrieval ablation
- Baseline C: no-chain/no-goal signal ablation
- Baseline D: no-rerank ablation

## Benchmark Matrix

### External/Published Benchmarks

- MemoryAgentBench: https://arxiv.org/abs/2507.05257
- LongMemEval: https://arxiv.org/abs/2410.10813
- StructMemEval: https://arxiv.org/abs/2602.11243
- Mem2Act: https://arxiv.org/abs/2505.08200

### Internal Benchmarks

- vibecoder_memoryagentbench_v1 (canonical)
- vibecoder_memoryagentbench_stress_v1 (robustness slice)
- transport soak matrix (line/framed, outage simulation)

## Metric Families

### Retrieval Quality

- nDCG@5, nDCG@10
- Recall@5, Recall@20
- MRR
- top-1 accuracy (task-specific where available)

### Memory Competency

- goal continuity recovery rate
- unresolved-thread recovery rate
- structured relation recall/precision
- selective forgetting correctness (when benchmark supports it)

### Reliability

- MCP request success rate
- transport-closed incidence per 1k calls
- timeout incidence per 1k calls
- deterministic error semantics coverage (`-3260x`, `-32001` classes)

### Performance

- p50/p95 latency for add/search/tool-call paths
- throughput under bounded concurrency
- memory/VRAM footprint snapshots by profile

## Statistical Protocol

- Minimum 5 independent seeds/run shards per benchmark slice
- Paired comparison where matched samples exist
- Bootstrap confidence intervals for primary metrics
- Permutation test p-values for paired deltas
- Multiple-testing correction: Holm or Benjamini-Hochberg (pre-declared per family)
- Effect-size reporting required (`Cohen's d` or equivalent)

## SOTA+ Thresholds (Initial)

### Quality Superiority Gate

Must satisfy both:

- Primary family mean improvement >= 8% relative vs Baseline A
- At least 70% of primary tracks show non-negative delta

### Reliability and Latency Gate

Must satisfy all:

- transport-closed incidence <= 0.1 per 1k calls
- timeout incidence <= 0.25 per 1k calls
- p95 latency regression <= 10% vs Baseline A on primary service paths

### Statistical Validity Gate

Must satisfy all:

- corrected p-value < 0.05 on primary quality gain claim
- lower CI bound > 0 for primary gain
- no statistically significant regressions on mandatory safety/reliability tracks

### Reproducibility and Integrity Gate

Must satisfy all:

- artifact checksum verification passes
- rerun reproducibility within predefined tolerance bands
- manifest ties decision to commit SHA + artifact hashes + config preset

## Execution Workflow

1. During enhancement phases: run deferred cadence with freshness-bounded reused reports.
2. At release-readiness: generate full benchmark artifacts for RC and baselines.
3. Run eval gates and statistical tests.
4. Run transport soak matrix.
5. Build promotion manifest with all evidence.
6. Apply go/no-go decision.

## Required Artifacts

- `eval/reports/**` benchmark outputs
- `eval/reports/mcp_transport/**` soak results
- gate summary JSON with per-family outcomes
- promotion manifest JSON containing:
  - commit SHA
  - benchmark artifact hashes
  - gate outcomes
  - timestamp and operator context

## Blocker Closure Criteria for Transport Intermittency

The transport intermittency blocker is closed only when:

- 30 consecutive soak runs pass,
- no unresolved transport-closed regressions occur in the same observation window,
- p95 remains below configured cap in >= 95% of runs,
- failures (if any) are classified to host-environment causes with no unresolved wrapper defects.

## Open Tasks

1. Add benchmark adapters for LongMemEval/StructMemEval/Mem2Act result normalization.
2. Add continuous-interaction memory benchmark slice (EMemBench-style) adapter for long-session retention/consistency scoring.
3. Wire scheduled CI benchmark replay and drift-alert policy to `sota-verdict` (release-boundary trigger + scheduled cadence only).
4. Add signed promotion-manifest emission bound to verdict artifact SHA + commit SHA.
5. Add dashboard/report template for leadership-facing release evidence.
6. Add host-runtime transport diagnostics bundle capture for timeout regressions (wrapper log snapshot + response-size distribution + per-tool p95 wall time).
7. Wire closure-campaign artifact summary into scheduled CI and release checks.
