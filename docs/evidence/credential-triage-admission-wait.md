# Credential triage: monitored admission wait

The existing shared admission gate correctly blocks credential ZDR triage while
a paid historical checkpoint is unresolved. Previously the interactive worker
exited before asking for a passphrase, requiring another manual launch later.

## Operational change

The existing Python worker and both PowerShell entry points support an explicit
finite wait, without a new daemon, secret store, service reload or policy change:

```powershell
# Add to the existing local triage launcher with its root/log/backup arguments:
-Provider openrouter -WaitForRemoteSeconds 3600
```

`--wait-for-readiness SECONDS` is the Python equivalent. Omitted/zero retains
immediate behavior. Positive values require an interactive console, OpenRouter,
its existing managed policy, and model review; the maximum is 86,400 seconds.
It cannot be combined with the existing one-shot `--check-readiness`.

While the local admission is busy, the worker reports static nonsecret status
once per minute and does not query the provider key, unlock the vault, create
backups, reserve spending or send a model request. The monotonic deadline is
anchored once. Revocation, malformed status, provider failure and budget denial
stop rather than being retried. Ctrl+C cancels. Only ready admission reaches the
existing hidden local passphrase prompt, validated pre/post backups and bounded
review. Existing per-call admission/revocation checks still handle races.

## Evidence and limits

The new regression failed against the preimage's absent wait capability.
Mock-clock tests exercise busy-to-ready, deadline anchoring, late readiness,
revocation, malformed/contradictory status, cancellation, static exception
reporting, one-shot compatibility and interactive prompt ordering. Both actual
PowerShell files parse without syntax errors.

Independent design review cleared this bounded workflow. A live first launch
proved its Python process remained alive but PowerShell's transcript did not
expose native output while the command ran. That owned waiter was stopped only
after exact PID/interpreter/module/wait-argument verification, one unresolved
managed admission, and absent pre/post backup destinations. The old console/log
were preserved; no service or batch was touched.

The worker now writes a separate, immediately flushed progress JSONL through one
serializer. Only fixed stage/state codes, finite integer counts, booleans and
known queue-status counts are permitted; candidate text, credentials, IDs and
cursors are omitted. `awaiting_passphrase` precedes the hidden prompt. The
OpenRouter PowerShell worker uses `remote_policy/triage-progress` under the
existing owner-only managed policy folder. Python creates
that directory owner-only only when absent, never repairs existing ACLs, and
exclusively creates a new file before waiting/unlocking. Existing files/links,
nonprivate parents, vault/archive paths and backup destinations fail closed.
The CLI default creates no progress file. All original path components are
checked before resolution/creation and every append; linked/reparse ancestors
and parent traversal are rejected. A real Windows junction regression proves
no file or private child is created through the link.

The runtime root itself is not owner-only and has three effective nonowner,
nonadministrative delete-child grants. It was not repermissioned. A second owned
waiting process was stopped before unlock/backup work after this operational
check. The corrected placement's existing policy parent passed owner-only ACL
verification, had zero other delete-child grants, and had no linked ancestors.
That is the boundary for progress integrity, not a claim that all runtime paths
are private. Preserved first/second console logs contain no credential values.

Final focused Python checks: **84 passed in 18.70 seconds** across wait/progress,
managed credential ZDR and ambiguity triage. Both final PowerShell scripts parsed
without syntax errors. These are affected-workflow proofs, not a full test suite.

Independent final source review cleared the linked-ancestor fix and protected
placement. Exact corrected launch: PowerShell PID 68288, Python PID 34420 using
the existing Miniconda interpreter and checkout. Live progress was read while
that Python process remained alive: `waiting_for_remote_admission`,
`remote_admission_busy`, one unresolved admission, `passphrase_needed=false`,
remaining 3,539 seconds. The one-hour deadline is counting down rather than
resetting. The live status file is under the verified policy parent:
`.muninn_runtime/remote_policy/triage-progress/credential-triage-wait-20261005-235139.log.progress.jsonl`.
No credential classification or passphrase entry is proven by this launch.
This is not proof of resolved credential ambiguity, a free spending slot,
successful model classification or complete backlog coverage. A competing
admission after readiness can still stop the existing review safely. No batch
is cancelled, deleted, resubmitted or repacked by this change.
