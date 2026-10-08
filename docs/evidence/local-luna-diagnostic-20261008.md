# One authorized two-request diagnostic, not a second backlog checkpoint

W explicitly requested another smaller batch to distinguish submission problems
from a delayed provider. The known 60-request parent remains retained and is not
cancelled, deleted, resent or acknowledged by this test.

The prior accounting guard prohibited any second unresolved admission. The
operator-only probe adds a nullable diagnostic parent binding to the existing
admission ledger, with separate unique fences for ordinary unresolved work,
diagnostics, and one lifetime diagnostic attempt per parent. Normal admission
still checks ALL unresolved rows, including diagnostics. No automatic selector,
HTTP/MCP endpoint, retention flag, spending cap, model route or journal queue
gains parallel submission authority. This is not an adaptive production lane.

A consistent private SQLite policy preimage is verified before the additive
migration. Inputs and bounded received replies are encrypted with a distinct
canonical history-key domain in the same accounting database. The uncertainty
phase and admission transition commit together before HTTP; timeouts never
resubmit. Even a mismatched received receipt is preserved before validation.
Recovery keeps the encrypted diagnostic and all three admission index fences;
restored remote/batch policies remain disabled. Diagnostics have separate bills
and zero backlog publications; aggregate run accounting includes their actual
settled costs once, with a diagnostic subset identifying them.

The fixed synthetic envelope uses the existing Luna/OpenAI batch transport:
one POST, two random custom IDs, one JSON control and one original Muninn strict
citation schema. Each caps output at 512 tokens. Highest published OpenAI input
and completion prices, including overrides, are used without assuming a batch
discount; serialized UTF-8 bytes plus 1024 input overhead tokens bound the
conservative estimate. A fresh provider-usage sample and managed headroom check
precede the durable pre-POST fence. The admission threshold is $0.01, not a
provider-enforced invoice cap. No private transcript or credential is input.

Validation: 54 diagnostic/accounting/interval tests passed initially; final nine
diagnostic tests passed in 4.19s, including portable restore, changed-index refusal,
fixed-input restriction, fresh-budget refusal before POST, preserved mismatched
receipt, no resubmission and exact result/quote validation. The 74 existing
accounting/interval/activation checks passed in 34.98s; 23 diagnostic/paid recovery
checks passed in 29.78s before the last safety fixes. These overlapping runs are
not summed as unique tests or a full suite. Independent source review identified
fresh usage, receipt-retention and output-boundary gaps; corrected paths received
CLEAR. Parent observed test exits. No merge/full-suite claim.

Actual live submission succeeded: local diagnostic
`ae796bc8563044e7b3a26e60fa649965`, two requests, conservative estimate
$0.0029752, admission threshold $0.01. Initial provider `validating`, subsequent
GET `in_progress`, zero completed/failed at age 95.2 seconds. This proves receipt
acceptance and reachable polling only, NOT inference success, validated output,
actual billing, or a size-dependent cause of the parent's delay.

The bounded local watcher (tool session 78475) polls the exact stored ID every
minute, stopping on terminal evidence/error or the 24-hour allowance. It never
POSTs, switches providers or deletes retained evidence. `--status` is a local-only
read for the existing ten-minute monitor. A lost POST receipt remains unknown
and requires exact-ID recovery, not another submission. Until the diagnostic
bill settles, the ordinary paid guard and owned-reload fence may continue to
block dependent actions; no hold is falsely released to make a check green.
