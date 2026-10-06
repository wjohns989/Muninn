# Installed Gemini hook/capture proof

Changed dependency: a reusable explicit probe of the installed native Gemini
hook runner, rather than settings-only proof or Python handler tests.
No application/service/client settings were changed and no service was restarted.

## Current evidence

- Gemini CLI `0.61.0` is installed. Its Muninn SessionStart, SessionEnd,
  AfterAgent and PreCompress settings match the portable installer exactly.
- The installed native core module used by the runner has SHA-256
  `570a6666c70b379fc22983848b3032d3329db57e5382c068597d2e1751359b22`.
- Local source inspection confirms the installed CLI has AfterAgent and
  PreCompress firing sites and a command runner with per-hook timeouts.
- The probe executes only the matching user-level Muninn commands, with the
  native runner's actual Windows shell, stdin handling and timeout enforcement.
  It selects an existing already archived Gemini transcript under the Gemini
  home. It does not read or print transcript content as model input.
- Both explicit operator-triggered handoffs succeeded: AfterAgent 642 ms,
  PreCompress 658 ms, exit 0, no child diagnostic output. New authenticated
  endpoint receipts reported capture intent for both events. The journal then
  reached archived status after the forced capture, with a newer source revision.
- A separate final check fully authenticated the committed encrypted blob and
  compared its digest with the unchanged source bytes: match. Queue status alone
  was not accepted as content-integrity proof. No raw text, path, token or
  passphrase is retained in this evidence.
- The probe launched no Gemini, Ollama or OpenRouter model call. Existing
  independently authorized background backlog processing was not stopped or
  reconfigured. Fetch from the diagnostic Node process is disabled.
- 43 focused hook/secure-ack/probe checks passed before adding the final archive
  comparison helper; its final three helper tests passed, including detection
  of a changed source. Unchanged handler/installer checks are reused.

The first native probe returned `native_runner_failed` without detailed timing
evidence. Its cause remains unknown. One bounded diagnostic recheck supplied the
successful timings and durable capture proof above; this is not evidence that a
specific intermittent problem was diagnosed or fixed.

## Limits and reuse

This verifies the installed runner-to-local-Muninn capture handoff with real
encrypted storage. It is **operator-triggered**, not a new full Gemini model
reply/automatic compaction cycle. Aggregate endpoint counters count accepted
invocations, not unique or naturally generated host events. Do not relabel these
two probe receipts as natural chat/compaction receipts.

Reuse requires matching CLI package/core identity, installed hook commands,
Python/client code, authenticated endpoint and encrypted capture configuration.
The live command changes capture telemetry and can requeue the selected existing
transcript. It therefore defaults to settings inspection; actual capture requires
`--execute-existing-capture`. Do not run it as a read-only heartbeat.

Portable entry point: `scripts/probe_installed_gemini_hooks.py`, accompanied by
the native runner adapter and isolated helper tests. It reports only fixed
nonsecret stage summaries, and a missing durable receipt fails even when the
hook child itself exits 0.
