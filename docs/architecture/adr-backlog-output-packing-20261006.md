# Evidence-bound backlog packing at 80 percent of output capacity

Status: installed as candidate 035b863; focused checks and protected bridge reads passed.

## Context and decision

The installed Luna Pro batch packer limits one request to ten adjacent transcript
windows, reserving 20,480 output tokens despite the endpoint's advertised 128,000
completion-token ceiling. W explicitly requested an 80 percent allowance.
Use at most 50 contiguous windows from the same authenticated source, project and
plan: 50 times 2,048 equals 102,400 output tokens. Each stored/cited window remains
at most 3,000 characters; this is request packing, not retrieval chunk enlargement.
The input/context and exact serialized 8 MiB batch bound remain separately enforced.

## Alternatives and trade-offs

Ten retains the current small failure domain but underuses output capacity.
Thirty was the initial bounded proposal, now superseded by W's fifty-window choice.
Packing unrelated sources or filling the model's entire output ceiling would
weaken provenance isolation or remove the requested headroom and is not selected.
GLM has not been benchmarked against this backlog; no provider or model switch is
included. Luna Pro and its exact accepted identities remain pinned.

The 25,600-token difference is unused ceiling, not guaranteed useful output:
reasoning tokens count within the requested allowance, and model behavior can
still truncate or omit slots. A whole-frame failure can require repair of every
slot. Existing failed-only unpacked children, two-round ceiling, durable sibling
publication, aggregate billing and serial paid ownership remain mandatory.
Measured throughput and cost improvements are unknown until live accepted replies.

## Compatibility and acceptance

Only newly selected requests use the wider ceiling. Retained cohort sizes/digests
must reproduce byte-identical ten-window wire bodies; prepared/sent/completed
batches are never repacked, deleted or cancelled. Restore and active recovery
readers must accept larger records before selection starts. Old readers with a
ten-window decoder cannot recover new fifty-window cohorts; preserve the compatible
candidate until its paid checkpoints settle rather than rolling back blindly.

Acceptance: red-first capacity proof; 50/51 boundary and exact slot/citation
validation; unchanged project/source/ordinal screening; retained wire equality;
portable encrypted restore; actual worker restart/publication with a late bad slot
and a truncated whole reply; successful siblings excluded from repairs; one
aggregate charge per original/child batch; bounded overrides; independent diff
review, guarded installed reload and protected MCP read. Paid quality/latency remains
a distinct live acceptance claim. Spending limits stay $5/day and $50/month.

Focused candidate verification: 115 tests passed across packing, streaming JSONL,
source evidence/append and cited source/window tests. Two fifty-window worker
fixtures initially exceeded the separate 32-window planning-transaction limit;
they now queue 32 plus 18 without enlarging production planning transactions.
Both corrected worker restart/publication/repair tests passed in 73.14 seconds.
The escaped-character boundary regression failed against the old lexer and passes
with the writer correction; existing malformed complete caches are not silently
rebound or repaired. Independent source/fixture review was CLEAR. No full-suite,
live fifty-window model-quality or measured savings claim is made.

Cold-start and retained-input preservation evidence is recorded in
`docs/evidence/local-resume-20261007.md`. Existing paid inputs are unchanged.
