# Phase 4P Plan: Apply-Checkpoint Provenance Enforcement Flags

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4p-apply-checkpoint-provenance-enforcement`

## Objective

Move provenance from passive audit metadata to active policy controls by allowing `apply-checkpoint` to require explicit provenance fields before runtime policy mutation.

## Implemented

1. Added enforcement flags to `apply-checkpoint`:
   - `--require-change-context`
   - `--require-pr-number`
   - `--require-commit-sha`
   - `--require-branch-name`
2. Added manifest provenance validator:
   - validates `change_context` object shape,
   - normalizes/validates `pr_number`, `pr_url`, `commit_sha`, `branch_name`,
   - enforces required-field flags before apply proceeds.
3. Added guardrails:
   - missing `change_context` now blocks apply when provenance enforcement flags are enabled,
   - malformed `change_context` continues to be rejected deterministically.
4. Added tests:
   - require-change-context failure path,
   - require-pr/commit/branch failure paths,
   - success path when all required provenance fields are present.

## Validation

1. `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py`
3. Result: `23 passed`
4. CLI sanity:
   - `python -m eval.ollama_local_benchmark apply-checkpoint --help`

## ROI / Impact

1. Enables policy mutation controls that can be aligned with branch/review governance requirements.
2. Reduces risk of applying checkpoint artifacts that lack review provenance.
3. Creates a clean bridge toward stronger governance policies (for example, require PR + commit linkage in production workflows).

## Follow-up opportunities

1. Add optional git verification mode:
   - verify required commit exists locally and is reachable from configured protected branch.
2. Add policy presets for enforcement profiles:
   - dev: optional provenance,
   - protected: require PR + commit + branch.
3. Surface enforcement toggles in Browser UI checkpoint-apply controls.
