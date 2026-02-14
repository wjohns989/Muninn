# Phase 4F Design: Runtime Profile Control API

Date: 2026-02-14  
Status: Approved for implementation

## Brainstormed options

1. Env-only profile control
- Keep all profile policy in env vars (`MUNINN_*PROFILE*`) and require process restart.
- Pros: smallest code change, low risk.
- Cons: high friction during active coding sessions; cannot adapt profiles dynamically from assistant/UI workflows.

2. Runtime profile control API (recommended)
- Add explicit runtime get/set endpoints for extraction profile policy and expose parity through MCP + SDK.
- Pros: enables dynamic operation without restarts, keeps one source of truth in server process, improves cross-assistant operability.
- Cons: requires validation/guardrails to avoid invalid profile mutations.

3. Auto-tuning controller first
- Build adaptive profile switching from telemetry (latency/success) immediately.
- Pros: highest potential automation.
- Cons: premature without stable manual control plane and promotion gates; high overengineering risk for current phase.

## Selected approach

Implement option 2 now, then layer option 3 later behind evaluation gates.

## Scope for this phase

1. Add runtime profile policy control in memory core:
- Read current active profile policy.
- Update allowed profile fields with strict validation.

2. Add REST endpoints:
- `GET /profiles/model` for current profile policy.
- `POST /profiles/model` for controlled mutation of profile fields.

3. Add MCP tools:
- `get_model_profiles` (read-only).
- `set_model_profiles` (write).

4. Add SDK parity:
- Sync + async methods for get/set profile policy.

5. Add targeted tests:
- Memory profile mutation behavior.
- MCP tool schema + payload behavior.
- SDK request shape and async path.

## Out of scope (explicit)

- Automated profile switching from telemetry.
- Persistence of runtime mutations to env/.toml files.
- Per-session policy namespaces beyond existing metadata + operator override model.

## Success criteria

1. Profile policy can be inspected and changed without server restart.
2. Invalid profile values are rejected deterministically.
3. MCP and SDK surfaces are feature-parity with REST.
4. Existing extraction and ingestion behavior remains backward-compatible.
