# Muninn local control center

## Outcome and approach

Make the existing localhost interface useful for operating the actual local
installation. A user must be able to distinguish working, waiting, blocked,
unverified and complete without interpreting logs or comparing incompatible
counts. This is the full UI roadmap, not a claim that the installation is ready.

Keep the current vanilla HTML/CSS/JS and authenticated API. An incremental
control-center rebuild preserves working ingestion, search, transcript paging
and policy forms. A framework rewrite adds migration and dependency cost with
no missing capability it alone solves. Styling isolated counters is cheaper
initially but leaves operational states and actions fragmented.

## Information architecture and delivery order

1. **Overview:** local connection/authentication; encrypted capture and index
   coverage; interpretation/retained-checkpoint state; actionable blockers;
   explicit unknown recovery/client/readiness evidence. Remove decorative live-
   looking telemetry. Storage presence never means interpreted coverage.
2. **Backlog & costs:** transcript windows are the primary processing unit;
   source versions have a separate denominator. Show all capture-lane window
   states, reuse, privacy-parked work, original/repair checkpoint sizes and actual
   provider request counts, drain expiry and enrollment of latest/all versions.
   Keep provider waiting, gather time, job claims, accepted API requests and
   validated publication separate. Do not invent an overall completion percent.
3. **History & evidence:** retain bounded, redacted transcript search/paging;
   expose cited-memory search, look-up and source following with provenance and
   provisional status. Distinguish event time, file time and archive time. No
   unlimited payload or raw-original/credential value in ordinary agent views.
4. **Review:** source/project/time-grounded ambiguity groups for all memory
   types, not just credentials; distinguish pending, rejected and deferred.
   Evidence-gated filing and user questions for contradictory/missing evidence.
   Review must not silently discard, bulk accept or move records on page load.
5. **Credential metadata:** existing service/project/source search stays value-
   free. Add local-authorized use/reveal and audit status only after exact
   capability boundaries are proven. Passphrases stay in local hidden prompts,
   never browser storage, chat, argv or environment. Surface worker input needs
   only through authenticated, authoritative worker state, not stale lock files.
6. **Models & controls:** persist/revoke ZDR and temporary nontraining batch
   consent independently; display live routing (including remote-only), actual
   limits, admission thresholds versus provider hard caps, and existing ceiling
   overrides. Never launch inference or save consent automatically. Keep local
   resource checks on demand. Retained paid recovery is not cancelled on revoke.
7. **Clients & recovery:** per-host configuration, actual native event delivery
   and archived result proof are separate. Show code identity and current backup
   generation/currency/restore proof only from authoritative authenticated
   evidence. Include desktop launcher health/start behavior and recovery actions
   without automatic service start, deletion or redundant workers.

## First implementation boundary

Implement the shared accessible shell, Overview, Backlog & costs and an explicit
readiness checklist using existing read-only status endpoints. Preserve all
existing operational forms. Keep later consumers visible as acceptance gaps,
not green features or disabled controls masquerading as implementation. No new
backend schema, provider dispatch, credential access or spending-policy change.

Fetch authenticated history status for Overview and Backlog; use the existing
generation/index, capture queue, enrichment, historical enrollment and batch
fields. Dedicated-key costs are provider daily/monthly usage, not run total or
local ledger. Run-total settled admissions including repairs/failed requests,
baseline reconciliation, retained aggregate bills and request-level success
rates require their own read-only accounting consumer before those labels appear.
Unknown is never zero. Null/negative/nonfinite/incompatible values stay unknown.
Invalid/failed responses clear prior counts. Every load carries a token/session
and sequence guard; logout invalidates pending completions and clears private
DOM, capabilities, forms and timers. Static public UI never embeds a bearer.

## Data flow and failure behavior

Browser-memory bearer -> existing authenticated GET -> allowlisted fields ->
textContent/DOM nodes. No raw status dump, HTML from sources or browser token
persistence. No remote fonts/assets. Explicit refresh; bounded polling only of
the local service while the relevant page is visible and authenticated. Provider
key-status is on demand, not every local poll. Show last successful sample time
and explain that counts are samples, not streaming telemetry. Failed sampling
invalidates evidence; 401 clears session and relocks. Batch in-progress inside
its provider allowance is waiting, not failed. Drain expiry is not completion.
Do not delete/cancel/resubmit retained work or weaken serial checkpoint gates.

## Acceptance and proof

- Focused red-first JS harnesses exercise actual page functions: latest/all-
  version enrollment separation, windows versus source versions, repair parent
  counts, unknown/error/reset states, HTML-like metadata rendered literally,
  logout and token-change races; no hidden API POST on initial view/refresh.
- Reuse intact transcript/privacy tests; run all affected dashboard tests and
  JS syntax checks. No full-suite or whole-installation claim from those tests.
- Independent review of the auth/data-flow diff before serving the changed UI.
- Inspect actual localhost rendered locked state and responsive layout. Use
  authenticated read-only status proof; positive browser authenticated workflow
  requires safe local token entry, never exposing the token to this chat.
- Keep later roadmap gates explicitly incomplete until representative rendered
  behavior and authoritative records prove them. Existing successful backups,
  hook receipts and model results are scoped evidence, not blanket readiness.

## Remaining UI gates after the first boundary

### Cited-memory read boundary

Reuse installed search/get/source/review-queue POST readers, without a new
backend or bulk loading. Search shows at most ten records; review pages six,
and represent only the safe noncredential provisional/needs-user subset.
No page count is the total ambiguity queue and no empty page proves resolution.
Source identity must match the requested opaque memory reference. Display
bounded context only for `context_state=available`; otherwise explicitly
withhold it, never substitute the record quote. Per-record state, truth status,
proposal origin and time/project basis remain visible as provenance, not truth.

Tokens, review cursors and transcript capabilities stay in session memory;
source capabilities are held only by removable button closures. Lock/new reads
clear all results, detail, cursors and transcript state. Independent session,
sequence and transcript guards reject stale completions/clicks. Busy controls
prevent parallel UI reads; 30-second observation timeouts do not cancel backend
readers or imply their CPU slot is released. No automatic retry, inference,
filing, deletion, consent or budget mutation. Existing redacted transcript paging
provides whole-transcript access incrementally, never a raw-original fallback.

Focused synthetic tests exercise the actual extracted JavaScript and literal
DOM sinks, missing/withheld text, invalid identity and bounds, same-token session
changes, old transcript clicks and review subset paging. Independent review
requires these context and invalidation rules before activation.

Cited-memory signed-in manual browser QA; generic review actions beyond the
eligible noncredential browse-only view; local-only
credential-use approval/audit; independent batch opt-in/revoke form; run-aware
accounting/reconciliation; authoritative backup/restore currency and installed
revision; natural-host client evidence; authenticated manual browser QA. Each
needs a focused consumer/proof, not another application rewrite.
