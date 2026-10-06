# Installed capture recovery fairness

Source candidate: `9173930` on `codex/muninn-adaptive-memory-rehearsal`.

## Defect sensitivity and bounded verification

- The prior source failed the old-unsent-failure capacity regression: new
  planning consumed the only slot needed for retry.
- The prior source left a private local failure in `failed/local_output_quote`
  instead of making it reachable by the existing parked-private ZDR lane.
- New private-only tests initially failed against the absent transition.
- The prior reload helper failed the exact paid-outbox preimage regression.
- A first affected run passed 80 tests; its owned-checkpoint test incorrectly
  let batch selection readmit the chosen failed job. The corrected fixture keeps
  it running until another batch owns the checkpoint, then fails it locally.
- Expanded affected recovery/capacity/activation/private claim/service/cadence/
  drain checks: 190 passed in 125.44 seconds. One test loaded its old erroneous
  monkeypatch import while that test was corrected; the corrected exact
  serialized-screening test separately passed in 1.94 seconds.
- Historical latest/all-version enrollment, packing and worker checks: 74
  passed in 49.54 seconds.
- Paid stop-fence and reload checks: 99 passed in 13.86 seconds, including
  copying the encrypted paid outbox while fenced, integrity checking and exact
  copied-row equality. Total: 364 affected checks passed; not a full suite.
- Independent native actual-diff review cleared fairness, fresh private
  screening, and the exact paid-outbox preimage inclusion. No model or provider
  requests were made by these isolated checks. No merge was performed.

## Actual existing-service installation

The reviewed reload preserved the existing launch environment and capture
settings. No remote/retention flag change or drain extension was requested.
Seven encrypted database preimages validated before stopping the owned process;
zero in-flight jobs were observed. This includes `historical-batches.db`, formerly
omitted by the `.sqlite3` selection. No paid batch deletion, cancellation,
repacking or operator resubmission occurred.

After reload, one verified installation process/listener owns port 42069:
PID 75468 (retired PID 50096). Health 200; anonymous protected request 401;
authenticated protected request 200; strict archive ready; anonymous HTML does
not contain the auth token. Automatic capture, analysis and remote-only routing
remain enabled. The separate existing credential triage worker count remains one.

Before and after authenticated read-only identity evidence matches:

- Owner `91f9e75b42cd4bceb96a6e4ae13b700d`, phase `sent`.
- Provider ID `batch-1791275902-8j95FCfwGXi2KCRKPazK`, state `submitted`.
- 50 transcript windows / 44 actual root provider requests.
- Item digest
  `15328b5aa5c8565e55daefe3fc2345d5cdf248f05f986f1b5516ecd1ae8f1875`.
- Exact wire digest
  `52fc625a64adf271b7317a5fd87cce258eae1dfda52d4462e89e7e8babdad4a1`.

A concurrent additional pre-stop identity observation was unavailable; it was
not used as preservation proof. The earlier successful authenticated observation
and completed post-reload observation above are the comparison evidence.

## Exact limits

Live status still awaits the submitted provider batch. These changes do not
prove that all old failures have been retried successfully: the next selection
must wait for the owned checkpoint's bill, citations and durable ACKs. They do
not acknowledge empty-context failures or retry dispatched unknown outcomes.
The ordinary quiet-time cadence remains in effect; expiry of temporary drain is
not completion. No increased packing speed, savings, exact remaining-window
count, whole-backlog completion, full vault review or whole-goal readiness is
claimed. No credential or transcript value is recorded here.
