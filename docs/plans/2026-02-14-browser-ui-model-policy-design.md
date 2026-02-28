# Browser UI + Model Policy Brainstorm Design

Date: 2026-02-14
Context: Phase-4 planning after Phase-3 completion

## Problem Statement
The control-center UI is operational for ingestion/search/consolidation, but it does not let users persist workflow preferences or choose quality-vs-latency model behavior. Extraction routing is also static (`xLAM` + fixed Ollama fallback), which is misaligned with current model variability and operator needs.

## Approaches Considered

### Option A (Recommended): Profile-Based Policy + Persistent UI Preferences
- Add three model profiles: `low_latency`, `balanced`, `high_reasoning`.
- Add deterministic fallback chains per profile (provider/model ordered list).
- Persist user UI preferences in browser storage first, with optional server profile endpoint for shared environments.
- Expose profile selector and advanced preference panel in control center.

Tradeoffs:
- Pros: High ROI, low cognitive load, avoids provider lock-in, supports current/future models.
- Cons: Requires profile governance and small increase in config surface.

### Option B: Keep xLAM Primary, add single Ollama fallback only
- Keep current behavior but make fallback model configurable.

Tradeoffs:
- Pros: Minimal implementation effort.
- Cons: Weak adaptability, no explicit thinking controls, higher risk of stale defaults.

### Option C: Full per-request model picker with arbitrary knobs
- Let user set provider/model/params each action.

Tradeoffs:
- Pros: Maximum flexibility.
- Cons: High UX and support burden; likely overengineering at current maturity.

## Architecture Recommendation
1. `ModelPolicyConfig` in core config:
   - `default_profile`
   - `profiles.<name>.routes[]` with health-check + capability flags (`tool_calling`, `thinking`).
2. `ProfileRouter` in extraction layer:
   - selects first healthy compatible route;
   - emits decision telemetry (`profile`, `provider`, `model`, `fallback_depth`).
3. Browser UI preferences:
   - persisted settings (`namespace`, ingestion defaults, log verbosity, profile).
   - explicit "safe mode" toggles for broad ingest operations.

## Error Handling + Safety
- Route unavailable -> deterministic fallback.
- No route healthy -> structured error with remediation hints.
- License warning for non-commercial models (xLAM family) in startup diagnostics and UI warning ribbon.

## Validation Plan
1. Unit tests: profile selection, fallback order, unsupported capability filtering.
2. Integration tests: API/UI round-trip for preference persistence.
3. Eval extensions: compare profile latency/quality deltas (nDCG/Recall/MRR + p95 latency).
4. Ops checks: OTel dimensions for profile usage and failure rates.

## Overengineering Check
This is not overengineering if constrained to three profiles and deterministic routing. It becomes overengineering only when exposing arbitrary low-level model knobs before evidence of need.
