# Installed independent batch-permission control

Backend candidate: `2dd00efb6b2ada0ec1db56f7ffee04e6ab7b37ac`.
This is a scoped control-center acceptance gap, not whole-installation completion.

## Proof

- 101 affected API, remote-policy and dashboard checks passed in 63.94 seconds.
- Independent review identified a revoke/re-enable race. The added real isolated
  policy interleaving test failed on the prior candidate (`DID NOT RAISE`).
- The writer-fenced enable check now binds the observed remote generation as well
  as enabled state. 50 focused API/activation/capacity checks passed in 58.92
  seconds after that fix. These scopes overlap; do not report 151 unique tests.
- Independent changed-issue review: CLEAR before activation. Disable remains
  permitted; backup and DDL still follow the fenced generation checks.
- Existing-service reload validated seven database preimages with zero in-flight
  jobs. PID 75468 retired; exactly one PID 83388 owns the same loopback port 42069.
  Strict archive readiness, automatic capture and automatic remote processing
  were preserved. No user-setting persistence or consent write was performed.
- Live GET-only proof: anonymous batch policy 401; main authenticated policy 200
  with `Cache-Control: no-store`. Returned policy equals independent read-only
  local policy both before and after GET: enabled, generation 2, quota 10,000,
  remaining 9,981 at observation. Quota may decrease as authorized work proceeds.
- Anonymous/authenticated existing protected route remains 401/200. Served HTML
  bytes exactly match the candidate; the refreshed browser shows the locked auth
  modal, the new card is present, and Save starts disabled with unknown policy.
- Retained paid record `70cf6bf913e94bd3a0c90dfa177ab57c` remains submitted to
  `batch-1791277299-KCYWFGxWgKneY0AzyGC8`: 33 windows / 25 provider requests.
  Item digest `e9797bc10923af6b94f1b58ba34d8e735066d1cd9c450d07d22094283ff82ebb`
  and exact wire digest
  `b1f5f8d8eb162041caa2e93061355eecfc739a4d1f4ac40681eb6e0248f9bb74`
  match the pre-reload record. No batch deletion, cancellation, repacking or
  operator-side model request occurred.
- Independent integration examination: CLEAR, reusing the unchanged reviewed
  diff and supplied live results rather than repeating operational effects.

## Context-size efficiency check

A read-only authenticated-ciphertext sample of at most 32 latest retained rows
contained 20 provider-submitted original/repair records: 421 windows in 326
provider requests. Request pack sizes were 1:274, 2:31, 3:13, 4:4, 5:2, 10:2.
Only two requests reached the ten-window cap. This is a retained sample, not an
authorized-run success denominator or a measured savings/latency result. It does
not justify broadly increasing context size as the next catch-up repair.
Existing managed remote limits remain $5/day and $50/month.

Fresh post-installation status shows 33 windows / 25 provider requests awaiting
the provider, 4,348 privacy-parked windows, 6,083 pending snapshot source versions,
and one live credential-triage worker. Source versions are not windows, and
enrollment complete is not interpretation complete. Normal quiet-time cadence
is active; the temporary backlog-drain deadline is not enabled.

## Behavior and limits

Batch permission is separate from ZDR permission and dollar budgets. Explicit
refresh precedes Save. Every enabled save confirms temporary nontraining
retention, a fresh permission generation/quota, and possible charges within
existing limits. Main local authentication, actual loopback peer, same browser
origin, strict fields and generation CAS protect writes. A verified private
SQLite preimage precedes mutation. Stale or uncertain saves require rereading;
there is no automatic retry. Revocation preserves already-admitted paid recovery.

Positive signed-in manual browser save/revoke remains unrun; actual extracted JS
and isolated authenticated API tests cover it without changing live permission.
Full pytest was not run and no merge was performed. Whole-goal gaps remain:
historical interpretation/ambiguity, hidden local vault authorization, natural
host-client proof, recovery currency, generic review actions, and run-aware UI
accounting. This receipt does not convert those gaps into completion.
