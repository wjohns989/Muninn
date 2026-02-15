# Phase 4I Plan: Ability + Resource Benchmarking (Live + Legacy)

Date: 2026-02-15
Owner: Codex
Status: Implemented baseline, pending CI promotion gates

## Why this phase exists

Phase 4H proved local model sync and latency/throughput comparisons. It did not yet provide:
1. Repeatable ability/accuracy signals.
2. Resource-normalized comparison signals.
3. Legacy-ingestion-like evaluation from real old project roots.

Phase 4I closes that gap.

## Crash impact check (post-restart)

1. Git integrity re-check remained clean (`git fsck --full`, dangling tree only).
2. Open PR check remained clean (no open PR/comment backlog before changes).
3. Stale pytest process was identified during this phase and treated as an operational artifact, not repository corruption.

## Implemented outputs

1. Extended `eval/ollama_local_benchmark.py` live benchmark scoring:
   - rubric checks per prompt case,
   - `avg_ability_score`,
   - `ability_pass_rate`,
   - resource-efficiency metrics:
     - `ability_per_second`
     - `ability_per_vram_gb`.
2. Added `legacy-benchmark` command:
   - builds deterministic benchmark cases from old project roots,
   - runs extraction-style prompts against selected local models,
   - scores legacy-ingestion-oriented ability per run/model.
3. Updated prompt pack with rubric metadata:
   - `eval/ollama_benchmark_prompts.jsonl`.
4. Added benchmark helper tests:
   - `tests/test_ollama_local_benchmark.py` (expanded to `8 passed` by Phase 4J).
5. Updated docs:
   - `eval/README.md`,
   - `docs/PLAN_GAP_EVALUATION.md`,
   - `docs/MUNINN_COMPREHENSIVE_ROADMAP.md`,
   - `SOTA_PLUS_PLAN.md`.

## How to run

```bash
# Live suite: ability + resource efficiency
python -m eval.ollama_local_benchmark benchmark --repeats 2 --num-predict 192

# Legacy suite from old projects
python -m eval.ollama_local_benchmark legacy-benchmark \
  --legacy-roots "C:/path/to/old_project_1,C:/path/to/old_project_2" \
  --repeats 1 \
  --dump-cases eval/reports/ollama/legacy_cases.jsonl
```

## Browser UI brainstorming backlog (practical, non-overengineered)

1. Benchmark launch panel:
   - pick model set + roots + repeats,
   - start run and open report artifact.
2. Profile templates with intent labels:
   - `coding_light` (runtime low-latency),
   - `balanced_default`,
   - `offline_planning` (heavier ingest/planning only).
3. Safe-mode policy cards:
   - stricter defaults for broad ingest / legacy import / profile mutation.
4. Evidence panel:
   - surface last run `avg_ability_score`, `ability_per_vram_gb`, and p95 latency before allowing default profile promotion.

## Model-policy view (16GB helper-first)

Decision:
1. Keep xLAM available as low-VRAM helper/specialist path; do not remove it.
2. Keep balanced/reasoning alternatives selectable by session/profile.
3. Use benchmark evidence to decide profile defaults, not static preference.

Reasoning:
1. xLAM is intentionally tiny and task-specialized, so it is a strong fit for runtime helper constraints.
2. Larger models can outperform on deep reasoning, but consume more VRAM and can cannibalize active development workloads.
3. The right control plane is model caliber selection by workload phase (runtime helper vs ingestion/planning), not one permanent global model.

### Practical resource envelope (current matrix baseline)

1. `xlam:latest` is sub-1GB in local pull footprint and remains the lowest-cost helper path.
2. `qwen2.5-coder:7b`, `qwen3:8b`, `deepseek-r1:8b`, and `llama3.1:8b` are 4.7GB–5.2GB class pulls and fit the 16GB helper+IDE target when not over-parallelized.
3. `qwen3:14b` is the explicit high-reasoning optional tier and should be reserved for offline planning/ingestion windows on 16GB systems.

## External research pointers used in this phase

1. Ollama model library landing and model pages (`qwen3`, `deepseek-r1`, `llama3.2`, `devstral`): https://ollama.com/library
2. Qwen3 technical report (reasoning/thinking-budget context): https://arxiv.org/abs/2505.09388
3. xLAM model card (tool-calling orientation): https://huggingface.co/Salesforce/xLAM-7b-r

## Open items for Phase 4J

1. Promote ability/resource thresholds into CI/nightly gate policy.
2. Add profile-level promotion rules driven by benchmark deltas + telemetry.
3. Add benchmark run controls to browser UI with strict safe defaults.
