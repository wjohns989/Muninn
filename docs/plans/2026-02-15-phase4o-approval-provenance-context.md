# Phase 4O Plan: Approval Provenance Context for Policy Governance

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4o-approval-provenance-context`

## Objective

Strengthen Phase 4N policy-approval governance by binding approval decisions to concrete code-review provenance (PR/commit/branch), so policy-apply artifacts can be audited against repository history deterministically.

## Implemented

1. Added approval provenance context support in `eval/ollama_local_benchmark.py`:
   - new optional `approval-manifest` flags:
     - `--pr-number`
     - `--pr-url`
     - `--commit-sha`
     - `--branch-name`
2. Added context normalization/validation:
   - commit SHA format validation (`7-40` lowercase hex),
   - PR URL scheme validation (`http(s)://`),
   - PR number positivity check (`> 0`),
   - automatic git commit/branch discovery when explicit values are not provided.
3. Added apply-report provenance propagation:
   - `apply-checkpoint` now carries manifest `change_context` into output report.
4. Added guardrail:
   - `apply-checkpoint` now rejects non-object `change_context` values in manifests.
5. Added test coverage:
   - manifest includes provenance context + git fallback behavior,
   - invalid commit SHA rejection,
   - apply-report includes provenance context,
   - non-object `change_context` rejection.

## Validation

1. `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py`
3. Result: `18 passed`
4. CLI sanity:
   - `python -m eval.ollama_local_benchmark approval-manifest --help`
   - `python -m eval.ollama_local_benchmark apply-checkpoint --help`

## ROI / Impact

1. Creates deterministic linkage between policy approvals and repository review context.
2. Reduces audit ambiguity during rollback/incident forensics.
3. Enables future policy automation to enforce review provenance predicates (for example: required PR association, protected-branch commit lineage).

## Follow-up opportunities

1. Add optional enforcement flags on `apply-checkpoint`:
   - require non-null PR number and commit SHA before apply.
2. Add signed-approval support (artifact signature + verifier) for tamper-resilient policy governance.
3. Add Browser UI controls for selecting approval manifest and displaying provenance integrity summary before apply.
