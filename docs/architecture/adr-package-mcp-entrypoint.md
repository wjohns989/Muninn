# ADR: One executable stdio MCP implementation

## Context

The checkout contained untracked package entrypoints with a duplicate transport
loop and plaintext diagnostic payload files. The configured bridge already uses
the tracked `mcp_wrapper` implementation. Both routes must preserve authentication,
protocol negotiation, task/deadline behavior and the shared-service boundary.

## Decision

Preserve hashed preimages of the existing package files, retain their compatibility
helpers, and make `python -m muninn.mcp` delegate executable startup to
`mcp_wrapper.main`. Initialization checks readiness without spawning Muninn or
Ollama. Explicit bootstrap opt-ins remain separate and default off. Diagnostics
contain fixed event names or error classes, never supplied payloads or exception
values; no new plaintext trace file is written.

## Alternatives and trade-offs

Maintaining two executable transport loops would require two implementations of
every dispatch, cancellation and timeout fix. Replacing all existing compatibility
helpers would unnecessarily discard user work. Delegation removes that executable
divergence while retaining the private helper surface for compatibility. That
surface still needs targeted tests when directly used; it is not a second service
owner. Existing trace files are not automatically deleted or declared sanitized.

## Proof scope

Red-first isolated tests reproduce lifecycle calls during initialization,
plaintext payload traces and exception-value leakage. A subprocess runs the
actual package entrypoint with fresh fixture-only settings and an ephemeral HTTP
fixture: initialization, tool enumeration and ping complete, only readiness GETs
occur, no model request or live service is used, and no trace payload is persisted.
This establishes bridge plumbing/security boundaries, not model quality or live
capture activation. Existing live services and user settings remain untouched.
