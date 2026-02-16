# Phase 5A.7: Deterministic Non-Terminal `tasks/result` Probe Hardening

Date: 2026-02-15  
Status: Implemented

## Objective

Close a remaining diagnostic gap in transport closure evidence by intentionally exercising the non-terminal `tasks/result` retryable path (`-32002`) under deterministic soak conditions.

## Implemented

1. `eval.mcp_transport_soak` now supports deterministic non-terminal probe controls:
   - `--probe-nonterminal-task-result`
   - `--task-worker-start-delay-ms`
2. Soak harness now:
   - performs task-backed `tools/call`,
   - immediately requests `tasks/result` with `wait=false`,
   - validates and records observed retryable non-terminal error semantics.
3. Wrapper task-worker deterministic probe control added:
   - `MUNINN_MCP_TASK_WORKER_START_DELAY_MS` (bounded/clamped parser).
4. `eval.mcp_transport_closure` now forwards probe controls and evaluates probe success in closure criteria:
   - criterion: `nonterminal_task_result_probe_met`
5. Closure telemetry now includes non-terminal probe rollout quality:
   - enabled count,
   - success count,
   - failure count,
   - success ratio.

## Verification

1. Compile checks:
   - `python -m py_compile eval/mcp_transport_soak.py eval/mcp_transport_closure.py mcp_wrapper.py tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py tests/test_mcp_wrapper_protocol.py`
2. Targeted tests:
   - `python -m pytest -q tests/test_mcp_transport_soak.py tests/test_mcp_transport_closure.py tests/test_mcp_wrapper_protocol.py`
   - Result: `105 passed`.
3. Deterministic soak probe evidence:
   - `python -m eval.mcp_transport_soak --iterations 10 --warmup-requests 2 --timeout-sec 15 --transport framed --server-url http://127.0.0.1:1 --failure-threshold 1 --cooldown-sec 30 --max-p95-ms 5000 --task-result-mode auto --task-result-auto-retry-clients "claude desktop,claude code,cursor,windsurf,continue" --probe-nonterminal-task-result --task-worker-start-delay-ms 350 --inject-malformed-frame`
   - Result: `outcome=pass`, probe observed `-32002`.
   - Artifact: `eval/reports/mcp_transport/mcp_transport_soak_20260215_235614.json`.
4. Closure campaign with probe criterion:
   - `python -m eval.mcp_transport_closure --streak-target 5 --max-campaign-runs 5 --transports framed,line --soak-iterations 10 --soak-warmup-requests 2 --soak-timeout-sec 15 --soak-max-p95-ms 5000 --soak-server-url http://127.0.0.1:1 --soak-task-result-mode auto --soak-task-result-auto-retry-clients "claude desktop,claude code,cursor,windsurf,continue" --soak-probe-nonterminal-task-result --soak-task-worker-start-delay-ms 350`
   - Result: `closure_ready=true`, `nonterminal_task_result_probe_met=true`, probe success ratio `1.0`.
   - Artifact: `eval/reports/mcp_transport/mcp_transport_closure_20260215_235635.json`.

## ROI / Blocker Impact

1. Converts non-terminal compatibility semantics into explicit closure evidence instead of inferred behavior.
2. Reduces false confidence risk by requiring observed retryable semantics in campaign windows.
3. Improves root-cause triage for host-side transport intermittency by separating wrapper semantics from external runtime transport behavior.
