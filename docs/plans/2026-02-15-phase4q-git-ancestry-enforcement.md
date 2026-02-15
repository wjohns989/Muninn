# Phase 4Q Plan: Git Ancestry Enforcement for Checkpoint Apply

Date: 2026-02-15  
Owner: Codex  
Status: Implemented + security-hardening follow-up in branch `feat/phase4q-git-ancestry-enforcement`

## Objective

Extend provenance enforcement so `apply-checkpoint` can optionally verify that the approved commit SHA is reachable from a designated git ref/branch before policy mutation.

## Implemented

1. Added `apply-checkpoint` flag:
   - `--require-commit-reachable-from <ref>`
2. Added git ancestry verification helper:
   - verifies ref resolvability (`git rev-parse --verify -- <ref>`),
   - resolves ref to commit SHA before ancestry evaluation,
   - checks commit ancestry (`git merge-base --is-ancestor <commit> <resolved_ref_sha>`),
   - returns deterministic validation outcomes and error messages.
3. Added apply-path enforcement logic:
   - requires manifest `change_context.commit_sha` when ancestry enforcement is enabled,
   - blocks apply if commit is not reachable from required ref,
   - blocks apply if git verification fails.
4. Added tests:
   - missing commit SHA with ancestry flag,
   - unreachable commit failure path,
   - git verification error path,
   - reachable commit success path,
   - dash-prefixed ref hardening path (`--` option separator + resolved SHA usage),
   - invalid ref-resolution output rejection path.

## Validation

1. `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py`
3. Result: `29 passed`
4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py tests/test_phase_hygiene.py tests/test_mcp_wrapper_protocol.py`
5. Result: `64 passed`
6. CLI sanity:
   - `python -m eval.ollama_local_benchmark apply-checkpoint --help`

## ROI / Impact

1. Adds branch-aligned commit lineage guardrail to policy mutation workflow.
2. Reduces risk of applying checkpoints tied to commits outside intended integration branch.
3. Enables stronger governance modes without forcing external CI dependencies.

## Follow-up opportunities

1. Add branch allow-list enforcement profiles (for example `main`, release branches).
2. Add optional strict remote verification (`origin/<branch>`) to reduce local-branch drift risk.
3. Surface ancestry verification status in Browser UI before apply confirmation.
