# SOTA+ Quantitative Comparison Plan

Date: 2026-02-15  
Status: Planned (pre-completion gate design)

## Objective

Define a deterministic, evidence-grade quantitative comparison protocol that decides whether Muninn meets SOTA+ standards at release time.

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
- StructMemEval: https://arxiv.org/abs/2509.09090
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

1. Generate benchmark artifacts for RC and baselines.
2. Run eval gates and statistical tests.
3. Run transport soak matrix.
4. Build promotion manifest with all evidence.
5. Apply go/no-go decision.

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
2. Add unified gate runner that emits one authoritative SOTA+ verdict object.
3. Add CI workflow for scheduled benchmark replay and drift alerting.
4. Add dashboard/report template for leadership-facing release evidence.
