# Native cheap-agent comparison: bounded real-data trial

User requested a trial with the cheapest native chat subagent rather than another
OpenRouter model invocation. The native model was `gpt-6-luna`, low effort.
This is a comparison, not new publication or a bulk provider change.

## Inputs and boundary

Ten text-distinct, real, whole-unit-screened archive windows with already
acknowledged OpenRouter `openai/gpt-6-luna-pro` outputs; 9,947 input characters.
These are predominantly repeated historical macro/ticker research requests.
Selection is biased toward nonempty proposals and is not representative of the
whole corpus. Actual transcript text and model replies are not stored here.

The helper uses query-only SQLite and bypasses constructors that initialize
schemas. Sealed input/stage bindings and settlement are authenticated; source
windows are reopened and screened before emission. Default sample emission
withholds the saved OpenRouter answers. The child received only IDs and input
windows, without the reference answers. Original outputs require a separate
explicit parent-comparison flag. No model request, queue write, memory publication,
credential reveal or service restart belongs to the helper.

## Actual results

- First native turn returned 3/10 requested items. One bounded repair turn
  returned the remaining seven: 10/10 final item completeness, not 100% first-pass
  completeness.
- All ten proposed quotes were exact, uniquely occurring source substrings,
  but all ten numerical offsets needed the existing deterministic correction.
  Accepted saved OpenRouter offsets have already passed its correction path,
  so they are not evidence of better raw coordinate-generation accuracy.
- Native returned zero current open items for these expired historical requests;
  the saved OpenRouter outputs each had an open-item entry. This is a narrow
  temporal-handling observation, not a judgment that every historical task is
  resolved or that all OpenRouter entries are incorrect.
- Worker turn elapsed times: 8.716 and 18.179 seconds (26.895 seconds combined).
  This excludes parent orchestration, sample preparation and time between turns.
- Trial input tokens: 91,178 total, of which 83,456 cached and 7,722 uncached;
  output tokens: 1,211. Earlier code-audit/setup work is not included in these two
  processing turns. A cold new worker has additional context overhead.
- Additional OpenRouter requests for this trial: zero. Native Codex dollar cost
  is unavailable; token/quota usage is not an OpenRouter dollar estimate.

Independent examination of four actual pairs found supported historical
summaries, conservative completion claims and sound temporal treatment after
coordinate correction. It explicitly did not certify global provider equivalence,
market-research accuracy or whole-history throughput.

## Decision and checks

Native chat agents are viable for bounded assisted review on the observed scope,
but this trial does not justify replacing the unattended backlog route. The local
service cannot call a native chat subagent directly as though it were an Ollama
or OpenRouter API. Exact local deduplication/reuse remains a separate, valuable
optimization; existing parent-window reuse currently excludes cloud results.
No spending policy or live provider setting changed for this trial.

Four isolated helper-boundary tests passed: database write refusal, withheld
reference answers by default, invalid-limit refusal before archive/key access,
and explicit-original-output opt-in. Those tests do not substitute for the real
trial or the semantic review above.
