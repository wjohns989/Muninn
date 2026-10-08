# Independent local batch-permission control

This implements the existing control-center plan's explicit retained-batch
opt-in/revoke gap. Keep vanilla UI and the existing managed policy/serial worker.
CLI-only control does not fulfill the requested UI revocation; a framework or
new consent store adds no missing capability. Credential unlocking/reveal is
not part of this change and remains separately local-authenticated.

GET reads enabled state, generation, configured quota and remaining checkpoints
without creating policy or dispatching providers. POST requires the main local
bearer, actual loopback peer, exact browser origin when present, strict bounded
fields and the observed generation. No key, budget or model field is accepted.

Under the existing policy writer lock, verify generation and current remote
consent before any DDL/mutation. Privately create a unique policy preimage,
copy via a separate read-only SQLite connection while fenced, check integrity
and ACL, then use the existing audited generation update. Backup/CAS failure
must not mutate consent. The existing atomic paid authorization fence controls
new dispatches; admitted recovery and encrypted outboxes are never cancelled,
deleted, repacked or resent by this control.

The UI explicitly reads before enabling Save. Preserve the actual existing quota,
not a default one-batch value. Every enabled save confirms temporary nontraining
retention, not ZDR, a fresh generation/quota and possible background charges under
unchanged spending limits. Token, epoch and sequence guard all completions and
finally blocks; lock clears state/forms. An uncertain/stale save requires explicit
reread; no auto retry, POST on page load, browser credential storage or remote key
probe. Unknown never appears as saved or disabled.

Independent native design review cleared this scope and required preimages before
DDL plus guarded finally/catch paths. Actual diff review, isolated real-policy/API
tests and actual-JS race/confirmation tests precede backend activation. Live
verification must GET only, preserve existing consent/quota/paid identities and
verify one strict authenticated local service. Positive signed-in browser saves
use synthetic tests; no live consent mutation is authorized merely for QA.
