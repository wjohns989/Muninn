# Phase 4H Plan: Local Ollama Model Matrix and Benchmarking

Date: 2026-02-14  
Owner: Codex

## Why this phase exists

Phase 4 already implemented profile routing and runtime policy mutation. The remaining operational gap is reproducible local evidence for model selection under active-development constraints (16GB VRAM class, helper-first runtime behavior).

## Objectives

1. Build a versioned local model matrix for runtime/ingestion profile decisions.
2. Pull and cache a practical benchmark set locally via Ollama.
3. Add repeatable local benchmarking that compares latency and throughput across curated prompts.
4. Keep generated artifacts out of git while retaining versioned benchmark inputs.

## Completed in this tranche

1. Added a versioned model matrix:
   - `eval/ollama_model_matrix.json`
2. Added a curated benchmark prompt pack:
   - `eval/ollama_benchmark_prompts.jsonl`
3. Added local sync/benchmark CLI:
   - `eval/ollama_local_benchmark.py`
4. Added documentation for local matrix sync + benchmark workflow:
   - `eval/README.md`
5. Hardened ignore rules for local cache/report artifacts:
   - `.ollama/`
   - `ollama-local-cache/`
   - `eval/reports/ollama/`
   - `eval/reports/model_benchmarks/`

## Target local model set (default pull set)

1. `xlam:latest` (low-latency helper, structured extraction)
2. `qwen3:8b` (balanced quality/latency baseline)
3. `deepseek-r1:8b` (reasoning-focused contender)
4. `qwen2.5-coder:7b` (implementation-heavy coding contender)
5. `llama3.1:8b` (stability/control baseline)

Optional:
1. `qwen3:14b` (high-reasoning offline ingestion/planning where VRAM budget permits)

## Current local cache status (2026-02-14)

Installed:
1. `xlam:latest` (`986 MB`)
2. `qwen3:8b` (`5.2 GB`)
3. `deepseek-r1:8b` (`5.2 GB`)
4. `qwen2.5-coder:7b` (`4.7 GB`)
5. `llama3.1:8b` (`4.9 GB`)

Not yet pulled:
1. `qwen3:14b` (optional high-reasoning tier)

## Initial quick-pass benchmark snapshot (2026-02-14)

Method:
1. One bounded generation prompt per model.
2. `temperature=0`, `num_predict=96`.
3. Metrics collected: wall-clock latency and Ollama-reported eval tokens/sec.

Results:
1. `xlam:latest`: `6.682s`, `242.54 tok/s` (fastest helper response in this pass).
2. `qwen3:8b`: `16.642s`, `7.50 tok/s` (higher latency in this pass; quality candidate remains in balanced tier).
3. `deepseek-r1:8b`: `53.913s`, `19.48 tok/s` (slowest latency, reasoning-oriented candidate).
4. `qwen2.5-coder:7b`: `13.448s`, `111.93 tok/s` (strong latency profile for implementation-focused candidate).
5. `llama3.1:8b`: `14.305s`, `14.89 tok/s` (middle-latency fallback baseline).

Interpretation for helper-first policy:
1. Keep runtime helper profile anchored to low-latency models (`xlam` / `qwen2.5-coder`) during active coding.
2. Reserve `qwen3:8b` and `deepseek-r1:8b` for targeted balanced/reasoning passes where latency budget is less strict.
3. Use `llama3.1:8b` as a compatibility fallback baseline in comparative gates.

## Operational commands

```bash
python -m eval.ollama_local_benchmark list
python -m eval.ollama_local_benchmark sync
python -m eval.ollama_local_benchmark benchmark --repeats 2
```

## Acceptance criteria

1. Default matrix models are cached locally and visible in `ollama list`.
2. Benchmark script emits JSON report with per-model success rate, average latency, p95 latency, and eval tokens/sec.
3. Report output path is gitignored and not committed.
4. Roadmap and gap docs reflect Phase 4H completion status and next actions.

## Next-phase dependency (Phase 4I)

Use generated benchmark outputs to define profile-promotion gates:
1. profile-aware latency thresholds (runtime vs ingestion),
2. profile-aware retrieval quality thresholds,
3. automatic default-policy promotion only when thresholds pass.
