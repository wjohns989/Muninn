# Agent Continuation Runbook

Date: 2026-02-16  
Scope: Fast handoff to another coding agent when session budget is exhausted.

## Purpose

Provide one deterministic entrypoint for resuming Muninn development without context loss.

## Current Development Snapshot

- Branch: `feat/phase5a-user-profile-standalone-foundation`
- PR: `#37` (`https://github.com/wjohns989/Muninn/pull/37`)
- Head commit at handoff creation: `101ae42`
- Active phase: Phase 5B external host-runtime validation/governance
- Latest transport decision artifact: `eval/reports/mcp_transport/mcp_transport_blocker_decision_20260216_015409.json` (`blocker_closure_ready=true`)

## 60-Second Resume Steps

1. Sync and open the active branch:
```bash
git fetch origin
git checkout feat/phase5a-user-profile-standalone-foundation
git pull --ff-only
```
2. Install runtime + dev extras:
```bash
pip install -e .[dev]
```
3. Open the phase trackers first:
- `SOTA_PLUS_PLAN.md`
- `docs/plans/2026-02-15-sota-plus-quantitative-comparison-plan.md`
- `docs/plans/2026-02-16-phase5a-completion-and-phase5b-launch.md`

## Start/Access Development Surfaces

### API Service
```bash
python server.py
```

### Standalone Browser App (Huginn mode)
```bash
python muninn_standalone.py
```

### MCP Wrapper (Muninn mode for assistants/IDEs)
```bash
python mcp_wrapper.py
```

## Minimum Verification Gate Before New Changes

```bash
python -m pytest -q \
  tests/test_mcp_transport_blocker_decision.py \
  tests/test_mcp_transport_incident_replay.py \
  tests/test_mcp_transport_diagnostics.py \
  tests/test_phase_hygiene.py
```

## Phase 5B Operational Commands

### Strict replay evidence capture
```bash
python -m eval.mcp_transport_incident_replay \
  --lookback-hours 24 \
  --require-log-path-exists \
  --include-log-sha256
```

### Blocker decision (strict latest evidence policy)
```bash
python -m eval.mcp_transport_blocker_decision \
  --lookback-hours 48 \
  --min-replay-runs 3 \
  --max-replay-signature-count 0 \
  --require-replay-provenance \
  --replay-provenance-policy latest_min \
  --min-closure-runs 1 \
  --require-latest-closure-ready \
  --require-latest-probe-criterion \
  --enforce-gate
```

### Release-boundary workflow trigger (manual CI run)
```bash
gh workflow run transport-incident-replay-gate.yml \
  -f gate_profile=release_host_captured \
  -f log_path=mcp_wrapper.log \
  -f lookback_hours=24
```

## Next Work Items (Ordered)

1. Capture first CI-generated strict-profile blocker decision artifact from workflow execution.
2. Sustain strict replay cadence so latest-min window remains populated with >=3 provenance-complete artifacts.
3. Continue quantitative-plan open tasks (benchmark adapters, signed promotion manifest, scheduled verdict cadence).

## Working Rules for Successor Agent

1. Keep one open PR policy intact.
2. Commit incrementally with verifiable evidence in docs/plans.
3. Do not merge in the same response where PR changes are created.
4. Update `SOTA_PLUS_PLAN.md` and quantitative plan whenever status/criteria change.
