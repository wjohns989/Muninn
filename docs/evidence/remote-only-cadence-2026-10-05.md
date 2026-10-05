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
