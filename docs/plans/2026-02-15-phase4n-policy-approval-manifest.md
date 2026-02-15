# Phase 4N Plan: Policy Approval Manifest + Checkpoint Apply Controls

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4n-policy-approval-manifest`

## Objective

Close the remaining governance gap after Phase 4M by requiring explicit approval artifacts before a saved profile-policy checkpoint can be applied.

## Implemented

1. Added `approval-manifest` command to `eval/ollama_local_benchmark.py`:
   - inputs: `--checkpoint`, `--decision`, `--approved-by`, optional notes/source/output.
   - validates checkpoint target policy structure.
   - writes deterministic approval artifact including checkpoint path + SHA-256 digest.
2. Added `apply-checkpoint` command:
   - requires `--checkpoint` and `--approval-manifest`.
   - blocks apply unless decision is `approved`.
   - blocks apply on checkpoint hash mismatch.
   - blocks apply on checkpoint path mismatch when manifest path is present.
   - applies checkpoint policy through `POST /profiles/model` (or dry-run), then writes deterministic apply report.
3. Added strict profile guardrails:
   - checkpoint profile fields validated against supported profile set (`low_latency`, `balanced`, `high_reasoning`).
4. Added test coverage:
   - approval artifact generation + hash checks.
   - approved manifest apply success path.
   - rejection path and hash-mismatch path.

## Validation

1. `python -m py_compile eval/ollama_local_benchmark.py tests/test_ollama_local_benchmark.py`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ollama_local_benchmark.py`
3. Result: `15 passed`

## ROI / Risk Reduction

1. Prevents accidental or silent policy flips by requiring explicit operator approval records.
2. Adds tamper evidence for checkpoint artifacts via SHA-256 binding.
3. Improves auditability by attaching reviewer identity + notes to policy apply events.
4. Enables cleaner phase-boundary review workflows (evidence → approval → apply) while preserving rollback capability from Phase 4M.

## Follow-up opportunities

1. Add branch/PR metadata (`pr_number`, `commit_sha`, `review_url`) into approval manifests for tighter review traceability.
2. Add optional dual-approval threshold for high-risk profile transitions.
3. Expose approval/apply controls in Browser UI with checkpoint picker and hash verification preview.
