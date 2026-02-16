# SOTA+ Quantitative Comparison Plan

Date: 2026-02-15  
Status: In progress (Phase 4AF baseline + Phase 5A internal implementation complete + Phase 5B external host-runtime validation active; replay/diagnostics gate stack + release-profile strict provenance mode + blocker-decision provenance policy hardening applied + in-window closure-readiness decision passed)

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

## Phase Boundary Update (2026-02-16)

Phase 5A internal implementation scope is complete (see `docs/plans/2026-02-16-phase5a-completion-and-phase5b-launch.md`).

Phase 5B now tracks external host-runtime closure scope:

1. Validate strict replay profile behavior in host-captured environments.
2. Collect extended host-runtime closure-window evidence.
3. Execute evidence-bound transport blocker closure decision.

## Interim Mitigation Update (2026-02-15, continuation tranche)

Implemented to reduce external host-side 120s transport timeout risk (initially while blocker remained open):

1. MCP tool response payloads are now bounded in wrapper transport (`MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS`), with deterministic truncation metadata.
2. Search text responses now use the same bounded transport limiter.
3. MCP public error messages now redact connection/timeout/internal details while preserving actionable validation errors.
4. Browser dashboard rendering now avoids dynamic `innerHTML` injection paths for operator-visible result/table payloads.
5. Long-running MCP tools now support automatic task-mode deferral (`tools/call` auto-task) so heavy ingest calls can return immediate task handles instead of consuming the host synchronous timeout window.
6. Transport closure campaign automation now exists via `python -m eval.mcp_transport_closure`, producing deterministic closure-evidence artifacts with explicit criterion booleans.
7. MCP wrapper now logs per-tool transport telemetry (`elapsed_ms`, response byte totals/max, budget/remaining budget) with near-timeout warning thresholds to accelerate root-cause diagnosis in host-runtime incidents.
   - Implementation detail: `docs/plans/2026-02-15-phase5a3-mcp-tool-call-telemetry-hardening.md`.
8. `tasks/result` now has a host-safe max-wait budget (`MUNINN_MCP_TASK_RESULT_MAX_WAIT_SEC`) with deterministic retryable error (`-32002`) when non-terminal waits exceed budget, preventing indefinite blocking from overrunning host timeout windows.
   - Implementation detail: `docs/plans/2026-02-15-phase5a4-mcp-task-result-host-safe-wait-budget.md`.
9. `tasks/result` compatibility mode now exists to handle spec/client drift:
   - `MUNINN_MCP_TASK_RESULT_MODE=auto|blocking|immediate_retry`,
   - `MUNINN_MCP_TASK_RESULT_AUTO_RETRY_CLIENTS` for deterministic client-profile auto-selection,
   - optional per-request `params.wait` override.
   - Implementation detail: `docs/plans/2026-02-15-phase5a5-mcp-task-result-compatibility-mode.md`.
10. Transport closure artifacts now include compatibility telemetry rollups:
   - error-code totals,
   - task-result mode/profile distributions,
   - retryable-task-result error incidence ratio.
   - Implementation detail: `docs/plans/2026-02-15-phase5a6-closure-telemetry-huginn-branding.md`.
11. Browser-first standalone UX is now explicitly branded as Huginn while preserving Muninn naming for MCP-attached assistant mode.
   - Implementation detail: `docs/plans/2026-02-15-phase5a6-closure-telemetry-huginn-branding.md`.
12. Transport closure campaign now supports deterministic non-terminal `tasks/result` probe criteria:
   - soak probe controls: `--probe-nonterminal-task-result`, `--task-worker-start-delay-ms`,
   - closure criterion: `nonterminal_task_result_probe_met`,
   - closure telemetry adds probe enabled/success/failure counts and success ratio.
   - Implementation detail: `docs/plans/2026-02-15-phase5a7-task-result-nonterminal-probe.md`.
13. Wrapper tool-response formatting now applies bounded pre-serialization payload compaction:
   - item/depth/string preview limits are enforced before JSON serialization,
   - reduces oversized-response formatting spikes that can consume host timeout windows,
   - final text truncation guardrail remains enforced.
   - Implementation detail: `docs/plans/2026-02-16-phase5a8-tool-response-pre-serialization-compaction.md`.
14. Transport diagnostics bundle utility now provides deterministic wrapper/soak/closure triage evidence:
   - `python -m eval.mcp_transport_diagnostics`,
   - includes per-tool latency/size summaries + incident counters + recent transport artifact rollups.
   - Implementation detail: `docs/plans/2026-02-16-phase5a9-transport-diagnostics-bundle.md`.
15. Transport diagnostics now support deterministic gate enforcement and hygiene-policy integration:
   - diagnostics now emits explicit gate verdicts (`results.gate`) and can fail on threshold violations (`--enforce-gate`),
   - phase hygiene can now consume diagnostics summaries and fail on incident budgets in PR/release checks.
   - Implementation detail: `docs/plans/2026-02-16-phase5a10-transport-diagnostics-hygiene-gating.md`.
16. Transport incident replay automation utility now exists:
   - `python -m eval.mcp_transport_incident_replay`,
   - scans bounded lookback windows for transport signatures and auto-triggers diagnostics capture when incident thresholds are met,
   - emits deterministic replay artifacts with signature evidence + diagnostics execution metadata.
   - Implementation detail: `docs/plans/2026-02-16-phase5a11-transport-incident-replay-automation.md`.
17. Replay automation is now wired into PR/release checks:
   - workflow: `.github/workflows/transport-incident-replay-gate.yml`,
   - artifact upload + replay summary output are now included in check runs,
   - strict host-log mode support added via replay utility guardrail (`--require-log-path-exists`).
   - Implementation detail: `docs/plans/2026-02-16-phase5a12-pr-release-replay-gate-wiring.md`.
18. Replay gate now includes release-profile strict provenance mode:
   - workflow profile options: `pr_safe`, `release_host_captured`,
   - release profile forces strict log requirements and digest capture (`--require-log-path-exists`, `--include-log-sha256`),
   - release trigger support added (`release.published`) for boundary checks.
   - Implementation detail: `docs/plans/2026-02-16-phase5a13-release-profile-replay-strictness-provenance.md`.
19. Blocker decision utility now exists for deterministic closure readiness:
   - `python -m eval.mcp_transport_blocker_decision`,
   - evaluates replay/closure criteria and emits explicit closure-ready verdict + violations.
   - Implementation detail: `docs/plans/2026-02-16-phase5b1-transport-blocker-decision-utility.md`.
20. Blocker decision provenance-scope hardening now exists:
   - `python -m eval.mcp_transport_blocker_decision --replay-provenance-policy latest_min`,
   - closure decisions can now enforce provenance on latest required replay evidence set (`min_replay_runs`) instead of entire historical lookback,
   - decision reports now emit provenance evidence counts (`required/evaluated/passing`) + selected replay paths for auditability.
   - Implementation detail: `docs/plans/2026-02-16-phase5b2-replay-provenance-policy-hardening.md`.
21. Strict replay evidence campaign now meets closure criteria in-window:
   - additional strict replay artifacts captured with host-log SHA-256 provenance,
   - enforced blocker decision gate now passes with no violations under `latest_min`.
   - Implementation detail: `docs/plans/2026-02-16-phase5b3-strict-replay-evidence-closure-readiness.md`.
22. Release-boundary blocker decision gate wiring now exists:
   - `.github/workflows/transport-incident-replay-gate.yml` now executes blocker decision enforcement in `release_host_captured` profile,
   - release events default to strict profile when no manual input profile is provided,
   - decision artifact + verdict summary now publish alongside replay/diagnostics artifacts.
   - Implementation detail: `docs/plans/2026-02-16-phase5b4-release-boundary-blocker-decision-gate-wiring.md`.

Current assessment:

- Risk is reduced for large-response and reflected-error classes.
- Blocker remains open until closure criteria are met across rolling soak windows in host runtime.
- Closure campaign automation now has both smoke and full-window deterministic evidence:
  - 5-run smoke: `eval/reports/mcp_transport/mcp_transport_closure_20260215_212349.json`
  - 30-run closure window: `eval/reports/mcp_transport/mcp_transport_closure_20260215_213858.json` (`closure_ready=true`, streak `30`, p95 ratio `1.0`)
- Post-hardening regression evidence:
  - soak pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_220359.json`
  - closure mini-campaign pass: `eval/reports/mcp_transport/mcp_transport_closure_20260215_220419.json` (`closure_ready=true`, streak `5`, p95 ratio `1.0`)
- Post-compatibility-mode regression evidence:
  - soak pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_221650.json`
  - closure mini-campaign pass: `eval/reports/mcp_transport/mcp_transport_closure_20260215_221709.json` (`closure_ready=true`, streak `5`, p95 ratio `1.0`)
- Post-telemetry/branding regression evidence:
  - soak pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_224206.json`
  - closure mini-campaign pass with telemetry: `eval/reports/mcp_transport/mcp_transport_closure_20260215_224225.json` (`closure_ready=true`, streak `5`, p95 ratio `1.0`)
- Post-nonterminal-probe criterion evidence:
  - soak probe pass: `eval/reports/mcp_transport/mcp_transport_soak_20260215_235614.json` (observed retryable `-32002`)
  - closure mini-campaign pass with probe criterion: `eval/reports/mcp_transport/mcp_transport_closure_20260215_235635.json` (`closure_ready=true`, `nonterminal_task_result_probe_met=true`, probe success ratio `1.0`)
- Post-pre-serialization-compaction verification:
  - targeted compile + protocol/transport tests pass: `108 passed` (`tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
- Post-diagnostics-bundle verification:
  - targeted compile + protocol/transport/diagnostics tests pass: `110 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - diagnostics artifact: `eval/reports/mcp_transport/mcp_transport_diagnostics_20260216_001515.json` (no wrapper-level failure signals in 24h window)
- Post-diagnostics-gate/hygiene integration verification:
  - targeted compile + protocol/transport/diagnostics/hygiene tests pass: `119 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - gate-mode diagnostics artifact: `eval/reports/mcp_transport/mcp_transport_diagnostics_20260216_005047.json` (`results.gate.passed=true`)
  - hygiene checks now support transport incident budgets for deterministic policy enforcement.
- Post-incident-replay automation verification:
  - compile + replay tests pass: `python -m py_compile eval/mcp_transport_incident_replay.py tests/test_mcp_transport_incident_replay.py`
  - replay suite result: `3 passed` (`tests/test_mcp_transport_incident_replay.py`)
  - live replay artifact: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_010414.json` (`results.triggered=false`)
- Post-replay-gate wiring verification:
  - expanded targeted suite: `123 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_transport_incident_replay.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - workflow wiring present: `.github/workflows/transport-incident-replay-gate.yml`
- Post-release-profile strictness verification:
  - expanded targeted suite: `124 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_transport_incident_replay.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - release-capable strict profile wiring present: `.github/workflows/transport-incident-replay-gate.yml`
  - strict-profile replay artifact: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_012731.json` (provenance fields populated, `results.triggered=false`)
- Post-blocker-decision utility verification:
  - expanded targeted suite: `127 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_transport_incident_replay.py`, `tests/test_mcp_transport_blocker_decision.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - decision artifact: `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_013548.json` (`blocker_closure_ready=false`, violations: `replay_run_count_meets_min`, `replay_provenance_met`)
- Post-provenance-policy hardening verification:
  - blocker decision suite: `4 passed` (`tests/test_mcp_transport_blocker_decision.py`)
  - expanded targeted suite: `128 passed` (`tests/test_mcp_transport_diagnostics.py`, `tests/test_phase_hygiene.py`, `tests/test_mcp_transport_incident_replay.py`, `tests/test_mcp_transport_blocker_decision.py`, `tests/test_mcp_wrapper_protocol.py`, `tests/test_mcp_transport_soak.py`, `tests/test_mcp_transport_closure.py`)
  - live decision artifact: `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_014909.json` (`replay_provenance.policy=latest_min`, blocker still open due strict replay evidence shortage: required `3`, evaluated `2`, passing `1`)
- Post-strict-replay evidence campaign:
  - strict replay artifacts: `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_015345.json`, `eval/reports/mcp_transport/mcp_transport_incident_replay_20260216_015355.json`
  - enforced decision artifact: `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_015409.json` (`blocker_closure_ready=true`, violations: none, latest-min provenance evidence `required=3`, `evaluated=3`, `passing=3`)
- Post-release-boundary gate wiring:
  - workflow updated: `.github/workflows/transport-incident-replay-gate.yml`
  - release profile now runs enforced blocker-decision gate and uploads decision artifact for auditable release-boundary checks.
- Remaining operational risk is primarily external host-runtime intermittency; wrapper-side transport regressions are now diagnosable and gate-enforceable.

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
6. Sustain Phase 5B host-captured strict replay evidence cadence so the latest-min closure window remains populated (>=3 strict provenance artifacts in-window).
7. Wire closure-campaign artifact summary into scheduled CI and release checks.
8. Wire `nonterminal_task_result_probe_met` and probe-success telemetry thresholds into scheduled CI/release gates so closure evidence includes explicit probe consistency requirements.
9. Validate end-to-end release-profile workflow execution in CI (`release_host_captured`) and capture first CI-generated blocker decision artifact for governance baseline.
