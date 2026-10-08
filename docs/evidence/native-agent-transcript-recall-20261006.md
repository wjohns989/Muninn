# Native agent transcript recall — 2026-10-06

## Outcome and current evidence

The current Codex chat searched the real cited ledger, followed one provisional
candidate to 284 characters of bounded source context, and started that source's
CPU-only transcript projection. The initial response was `pending`. Source
inspection confirmed that polling intentionally uses the original capability,
not a job ID; a missing job ID is not a lost job.

Following the same candidate again obtained a fresh expiring capability for the
same snapshot. Start reused its completed projection and returned seven pages.
The actual native MCP page tool read all seven: **14,182 redacted characters**,
ending with `next_cursor=null`. Every returned page respected its 4,000-character
bound, `more`/cursor agreement and `strict-best-effort` redaction marker. No model
call or raw-original request was made. Text and capabilities remained in the
active task's private tool context; they are not included in this receipt.

This proves full continuation for that supported conversational projection,
not all historical coverage, perfect credential detection, or a live
pending-to-ready poll transition. Unsupported/nonconversational units are not
silently described as retrieved conversation.

The current chat's cached 20-tool catalog still lacks transcript polling and
has older instructions. The freshly installed bridge passed four-client profile
agreement, project context, and two distinct review pages on an anchored ledger
snapshot (5,950 events), in 21.136 seconds. Its compact catalog is independently
checked for the complete cited-source/start/poll/page workflow by the updated
installed-profile probe; a missing polling tool now fails that probe explicitly.
Client MCP reconnection is distinct from restarting the shared server. This
work does not restart or change the parent client.

The final actual installed-profile probe passed in **751 ms**, listed 20 tools,
and reported `transcript_workflow_exposed=true`. All four configured profiles
matched. This is current live catalog proof, not an inference from source names.

## Changes and checks

- Start-tool and shared recall guidance explicitly say to start once, poll with
  the original capability (not a job ID), and read cursor/next_cursor as needed.
- A red-first regression reproduced the missing continuation instructions.
- Final focused workflow, API, projection-access, user-bridge and probe checks:
  **83 passed in 5.56 seconds**, one unrelated Hugging Face environment warning.
- Independent source review returned CLEAR for the actual scoped guidance/probe diff, not a
  fabricated host refresh or whole-installation completion claim.

## Preserved live operation and remaining dependencies

The same verified Miniconda-owned Muninn PID 85372 serves port 42069; anonymous
protected access is 401, authenticated access is 200, and public HTML contains
no bearer. Strict archive readiness is true, with 7,061 snapshots at generation
3,244. Capture jobs are archived except two explicitly missing Claude sources.

The retained paid checkpoint is unchanged: 33 windows / 18 requests,
`4b84b05c96044cea9a40bf120061f007`, provider
`batch-1791278985-LekmzqQB5r5zaQ9NPMuw`, locally submitted/awaiting provider.
That is not a fresh provider-terminal proof. No cancellation, deletion, resend,
repacking or model dispatch occurred.

Exactly one credential worker, PID 81412, is alive and awaiting the local hidden
passphrase. No replacement worker or passphrase-in-chat request was created.
Historical interpretation, 4,740 privacy-parked windows, six unknown outcomes,
and ambiguity resolution remain incomplete; disabled backlog-drain mode is not
completion. This is a recall acceptance gap closed, not the full goal achieved.
