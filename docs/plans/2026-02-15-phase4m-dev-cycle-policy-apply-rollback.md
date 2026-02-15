# Phase 4M Plan: Dev-Cycle Policy Apply + Rollback Checkpoints

Date: 2026-02-15  
Owner: Codex  
Status: Implemented baseline in branch `feat/phase4k-roadmap-closure`

## Objective

Close the remaining operational gap between benchmark evidence and runtime policy mutation by:
1. binding successful `dev-cycle` benchmark outputs to controlled profile-policy apply actions, and
2. providing deterministic rollback from a saved checkpoint.

## Implemented

1. Extended `eval/ollama_local_benchmark.py` `dev-cycle` command:
   - optional `--apply-policy` path added,
   - validates gate pass status (or explicit override),
   - validates required profile recommendations exist for target defaults,
   - fetches current profile policy from `GET /profiles/model`,
   - writes checkpoint artifact before apply (`profile_policy_checkpoint_<run_id>.json`),
   - applies new defaults via `POST /profiles/model` (or dry-run via `--apply-dry-run`).
2. Added rollback command:
   - `python -m eval.ollama_local_benchmark rollback-policy --checkpoint ...`,
   - restores previous active profile defaults captured in checkpoint,
   - writes rollback report artifact (`policy_rollback_<run_id>.json`).
3. Added guardrails and validation:
   - strict profile-name validation against supported profile set,
   - recommendation coverage checks to avoid evidence-free policy flips,
   - explicit `--allow-apply-when-gate-fails` escape hatch for advanced operators.
4. Added test coverage:
   - `test_cmd_dev_cycle_apply_policy_writes_checkpoint`,
   - `test_cmd_rollback_policy_applies_previous_checkpoint`.

## Why this improves ROI

1. Turns benchmark outputs into immediately actionable runtime policy updates with auditability.
2. Reduces blast radius of bad profile promotions via deterministic rollback.
3. Preserves your no-auto-merge/no-nightly workflow while still enforcing evidence-backed policy decisions.

## Operational commands

```bash
# Evidence run + policy apply
python -m eval.ollama_local_benchmark dev-cycle \
  --legacy-roots "C:/path/to/old_project_1,C:/path/to/old_project_2" \
  --repeats 1 \
  --apply-policy \
  --muninn-url http://127.0.0.1:42069

# Rollback from checkpoint
python -m eval.ollama_local_benchmark rollback-policy \
  --checkpoint eval/reports/ollama/profile_policy_checkpoint_<run_id>.json \
  --muninn-url http://127.0.0.1:42069
```

## Follow-up opportunities

1. Add browser UI controls for checkpoint select/apply/rollback workflows.
2. Attach profile-policy apply/rollback events to a dedicated alerting rule in ops telemetry.
3. Add policy-apply provenance link from checkpoint artifact to PR/comment metadata for review traceability.
