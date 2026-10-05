# Local opportunity and real agent access

## Changed dependency and proof

Candidate `66a89ec` separates the local maximum-wait timer from the shared
capture-attempt cooldown. Remote capture traffic cannot perpetually restart the
local wait during ongoing chats. A resource-gated local-primary path or safe
pre-send local fallback records an opportunity; GPU contention/cancellation can
still defer actual inference. No budget, privacy, lease, provider, queue capacity
or temporary-drain authority was changed.

The new real worker-plumbing regression failed against the prior implementation:
a remote-only refusal reset the local timer to 30 instead of leaving its expired
maximum wait at zero. The local-primary and fallback cases already passed.
After the repair, 111 affected cadence, scheduler, automatic-service and window
queue checks passed. All four GitHub workflows passed, including the full locked
dependency suite. Independent actual-diff review was CLEAR with GPU-contention
and cancellation limitations retained explicitly.

## Installed runtime

The verified-owner reload waited for durable workers to become idle, validated
six encrypted database preimages and restarted only Muninn. One listener,
strict encrypted-history readiness and automatic local/remote flags were
preserved. Protected access returned 200 with authentication and 401 without;
no token appeared in anonymous HTML. Existing untracked files were preserved.

Two live samples 57.4 seconds apart showed local max-wait decreasing from
1,715.23 to 1,657.82 seconds while the shared attempt cooldown changed from
13.59 to 26.50 seconds. Four post-reload private refusals and one running job
were observed, not four model successes. This proves that capture attempts no
longer restart the local timer; it does not yet prove a private local inference
after its full maximum wait. The expired historical drain was not extended.

## Actual agent and browser behavior

The current chat's Muninn tool performed a real cited-memory search, returning
three of sixteen matches. Following one Claude Code memory returned 344 bounded
redacted source characters and an expiring transcript capability. A real CPU-only
projection completed with 192 pages, 1,487 conversational units and 18,102 omitted
units. Two sequential pages each returned 2,026 redacted characters with more
pages available. Raw text, capability values and source identifiers were not
printed or written to this receipt. This is live agent-tool behavior, not a
stubbed MCP response or a claim of reading all 192 pages.

The installed dashboard passed an isolated headless Edge check at 390px:
authenticated Home health/archive/capture status, on-demand resource status,
16 installed model rows and all seven sidebar keyboard actions. No policy write,
credential reveal or model inference was performed by that browser check.

Current configuration reports four installed hooks each for Codex and Claude
Code, with actual acceptance receipts for all four event types. Gemini has
SessionStart/SessionEnd receipts; its installed AfterAgent/PreCompress hooks lack
real-event receipt proof. An acceptance receipt is not unique-host-event or full
capture proof; Codex SessionEnd's last outcome is `no_transcript`.

## Remaining whole-goal boundaries

Historical interpretation and ambiguity resolution remain incomplete. Source
enrollment completion is not interpreted coverage. Unknown remote outcomes are
not reset. Batch dispatch is not implemented or activated, and the existing
account-filtered catalog's lack of batch models needs a scoped eligibility and
non-training-retention resolution without weakening ordinary ZDR.

Portable vault/archive recovery has separate retained evidence; this slice did
not unlock or rescan credentials. The remaining vault-use/scan and backup/setup
UI work and missing Gemini real-event proof are not represented as complete by
this focused installation checkpoint.
