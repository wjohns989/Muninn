# Local control center overhaul (implementation plan)

Status (2026-09-29): partially implemented. The local installation has proven
bounded secure search and both local/Ollama and ZDR/OpenRouter analysis routes.
The dashboard at port 42069 has an Encrypted History tab with durable search-job
status, generation-bound archive/index coverage, capture-intent queue and source-scan
status, accepted hook invocation counts,
on-click bounded excerpts, persistent local ZDR consent, adjustable daily/monthly
admission thresholds, and provider-side key-cap status. Hook counts do not prove
unique host events or completed capture. This is not yet a full spending or
resource control center: actual spend, vault scan coverage, model residency,
host-origin hook proof, and backup receipts are not shown there. The full
overhaul remains lower priority than the working local capture/recovery and
credential-discovery path.
Some legacy-oriented controls and copy remain outside the History tab and need
strict-mode review. The root page no longer embeds an API bearer; retain that
property throughout the overhaul.

## Product contract

One loopback UI should answer four questions without confusing denominators:

1. Is the single local service healthy, which code/configuration is running, and
   are Claude Code, Codex, and Gemini CLI events arriving?
2. For each source/version, is its text discovered, durably archived, indexed,
   searchable, credential-scanned, interpreted, and backed up? If not, why?
3. What is using CPU, RAM, disk, GPU, or OpenRouter now, and what is queued or
   deferred? Idle must not keep a local completion model in VRAM.
4. What private data or credential metadata can an agent retrieve, and what
   local consent, cap, passphrase, and audit boundary governs a sensitive action?

The UI is an operator view over authoritative backend state, never its own
source of truth. Every count has a denominator and source generation. Every
operation has a stable id, state, progress, cancellation or retry behavior,
and an explicit distinction between request accepted and work completed.

## Navigation and screen behavior

| Screen | Required content and actions | Backend dependency |
| --- | --- | --- |
| Home | Service owner, health, code/config identity, archive/index/vault/backup generations, pending/errors, resource residency, last real hook by client. | Read-only consolidated status with component timestamps and stale indicators. |
| History | Project/time/source filters, bounded redacted search results, related transcript windows with provenance and cursor, exact archive-versus-index coverage; no raw credential in normal view. | Authenticated secure search/fetch jobs, project filtering, source event time, pagination. |
| Work queue | Capture, index, retrieval, analysis, vault-scan and backup jobs; stage, byte/window cursor, retry cause, next attempt, cancel/retry where safe. | Versioned reconciliation ledger and idempotent job APIs. |
| Models | Installed Ollama models, fresh VRAM/headroom, selected route and reason, active inference, unload confirmation, bounded local test on approved real source. | Resource telemetry and route-decision endpoints; never treat a model `loaded` flag as VRAM proof. |
| Privacy and spend | Local ZDR enable/revoke, daily/monthly app ceilings, provider-enforced key cap and reset period, spent/remaining, pending remote jobs, one-time first-run budget. | Authenticated persistent policy API with audit, provider-key status without value, per-dispatch recheck and cancellation on revoke. |
| Credentials | Metadata-only service/project/source-location search, scan coverage/ambiguity/errors, hidden local unlock for explicit use/reveal, reveal/use audit. | Separate credential authorization; never send a value through ordinary search/MCP or a browser URL. |
| Backups and recovery | Archive/vault backup generation, authenticated validation result, destination, restore drill and warnings about absent passphrases or plaintext originals. | Durable backup receipts and read-only restore verification before any destructive restore. |
| Setup | Portable data/model paths, one-service topology, hook install/check status, Claude retention, permissions, source roots, and diagnostics. | Validated local configuration API; changing a path never silently relocates/deletes data. |

## Security and interaction rules

- Serve on loopback. Keep the bearer only in tab memory after explicit entry;
  never embed it in HTML, a URL, localStorage, logs, or a repository file.
- Normal history retrieval shows bounded, credential-redacted evidence. A model
  may receive a bounded authenticated raw window only under its route/privacy
  policy. Credential values require the separate local vault-use/reveal flow.
- Do not use a generic browser bearer as authority to reveal credentials. Use
  dedicated local authorization, a hidden passphrase prompt, explicit action,
  and a non-secret audit receipt. A missing or declined prompt changes no data.
- Consent and budget controls must show effective provider-side and local caps;
  increasing a UI number cannot raise the OpenRouter key's own hard limit.
  Revocation must block new dispatch and clearly classify in-flight outcomes.
- Escape all displayed source/model text; do not render transcript, project, or
  model output as trusted HTML. Use content security policy and narrow origins.
- In strict mode, hide or explicitly label legacy plaintext import/consolidation
  controls rather than offering a button that will fail or imply auto promotion.

## Dependency-ordered delivery

1. Correct misleading strict-mode copy and disable unsafe legacy affordances.
   Verify anonymous root contains no bearer and a strict-mode operator cannot
   trigger the legacy import path from the UI.
2. Expose read-only consolidated status/coverage with exact generation, last
   success, pending/error reason, and hook delivery evidence. Check it against
   archive manifest, index, journal, and a real recent transcript.
3. Add secure history search/fetch/job views with bounded cursors and explicit
   timeout/expiry/retry states; manually inspect a real large-source tail hit.
4. Add resource/model view and ZDR consent/spending API/UI. Test local GPU busy,
   idle unload, remote denial, cap exhaustion, and revoke-during-queue.
5. Add vault metadata/scan progress and local authenticated use flow, then
   backup/recovery and portable setup screens. Test on encrypted real data
   without displaying values in ordinary search or logs.
6. Run browser accessibility, keyboard, narrow-window, auth-expiry, origin,
   script-injection, and restart/recovery checks against the exact installed
   candidate. Compare every displayed count to its backend generation; record
   any unsupported state as unknown, not zero or complete.

The on-demand local GPU/Ollama resource view and its authenticated API are
implemented in source but await a controlled service restart and live validation.
The existing dashboard can serve the new HTML before that restart, but the new
endpoint is unavailable until the running server loads the updated code.
An isolated headless Edge check verified authenticated History rendering at
1280px and 390px against the running loopback service; the narrow check exposed
and drove a responsive-layout fix. This is not yet keyboard/accessibility or
resource-endpoint live acceptance.
The metadata-only Credentials tab uses the existing opt-in authenticated endpoint
and was exercised in isolated Edge sessions at both widths with a nonsecret
query; it does not expose values or report complete vault-scan coverage. The
separate passphrase-gated use flow and scan progress/coverage UI remain pending.

Current delivery boundary: step 3's bounded search/fetch/job view, part of
step 2's archive/index, capture-intent/source-scan, and accepted-hook status, and part of step 4's
consent/admission and provider-key-cap status UI are implemented. The rest of steps 1-6 are not
accepted as complete; in particular, a source-only UI check is not a rendered
browser or accessibility acceptance test. Runtime counts, provider spending,
and backups must come from authoritative, generation-labeled backend receipts
before the UI displays them as current.

Implementation should reuse the existing FastAPI service and authenticated
routes. New write APIs require independent review of privacy/authority and a
rollback path before the UI invokes them. A polished page is not proof of data
coverage, model quality, or a current backup.
