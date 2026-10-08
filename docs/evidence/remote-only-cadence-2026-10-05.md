# Remote-only backlog cadence correction

Observed before installation: two newly successful jobs and fourteen pre-send
privacy refusals in ten minutes. Each refusal incorrectly started the same
30-second cooldown as a real model dispatch. Fourteen such cooldowns account for
approximately seven minutes of potential waiting, not provider request latency.

Remote-only capture now starts model cooldown at the durable pre-HTTP dispatch
boundary. A proved unsent privacy refusal parks its window and yields one second;
other unsent outcomes retain backoff. Privacy checks, single-consumer ownership,
spending admission, and non-remote-only cadence are unchanged.

The regression failed against the previous behavior. Focused cadence, automatic
capture, backlog-drain, and secure-analysis tests passed: 81 tests in 22.63 seconds.
Independent review cleared the actual service diff. This is component proof, not
a measured live throughput improvement or chronological batch completion.

The candidate also includes passphrase-authenticated local memory review with
immutable provenance/truth, append-only state-bound transitions, credential-risk
exclusion, rejected-result filtering, and verified encrypted ledger preimages.
Retained focused proof: 69 ledger tests, one additional rollback regression, and
seven CLI tests passed; independent integration review cleared the actual diff.

## Follow-up: network cadence and batch recovery

Current read-only sample: seven settled provider admissions in ten minutes,
mean request duration 5.27 seconds, mean charge $0.000646. This is a small live
sample, not a full-backlog price forecast or model-quality benchmark.

The remote-only fenced dispatch now uses a five-second start interval; actual
provider/billing failures reset the original thirty-second backoff. Unsent
budget/policy outcomes and local inference retain their original interval.
One serial consumer and shared spending admission remain in force. No change
to the historical deadline, $5/day/$50/month policy, or local GPU admission.

The same inference consumer now supports retained batch submission/recovery and
publication. Its production callback DEFAULTS TO DENY NEW SUBMISSIONS: separate
retention-consent binding and chronological selection remain pending. Already
sent owners can poll their exact encrypted provider ID, retain the full terminal,
settle a non-BYOK bill, publish validated siblings, and advance only when all
members pass. Unknown POST responses are not retried; no DELETE is implemented.
Polling, saving, settlement and local publication may recover after revocation.
Transport uses a fixed HTTPS origin, no redirects/proxies/retries, a sixty-second
overall request timeout, and a four-MiB decoded-response bound before strict JSON
parsing. This transport bound is not a source/transcript-size cutoff.

Focused five-suite run passed 96 tests in 34.36 seconds before the final added
service failure-backoff assertion; its affected automatic-service rerun passed
22 tests in 12.92 seconds. Initial batch-worker proof passed 14 tests in 10.07
seconds, including service shutdown draining. The affected worker/service run
including four transport checks passed 40 tests in 23.69 seconds. Independent
review cleared cadence and publication/shutdown fencing, but found recovery
startup depended on ordinary automation flags. Startup now also admits recovery
of an existing durable owner without enabling new work. The final affected
worker/automatic-service rerun including its startup regression passed 41 tests
in 22.82 seconds; Ruff and whitespace checks passed. Independent review of this
startup correction is required before installation.

Independent follow-up review cleared the startup correction and the default-deny
recovery integration. New-retention activation remains outside this installed
candidate's claim. The final empty-window regression run passed 23 automatic
service tests in 15.20 seconds; independent review cleared its narrow pre-send
allowlist. All changed Python files passed Ruff and whitespace checks.

Further live diagnosis found fifteen pre-send `insufficient_context` results in
the latest ten-minute journal slice. `_analyze_window` returns this only for an
empty span, before routing/reservation/POST. These also now yield one second
without a model-attempt interval. Their failed outcome remains visible; they
are not counted as interpreted coverage or model successes. Budget denials and
uncertain/sent calls retain backoff. The parameterized regression distinguishes
private, empty, budget-denied and dispatched/uncertain paths.
These isolated tests do not establish live batch activation or portable recovery
of the separate remote-accounting store.
