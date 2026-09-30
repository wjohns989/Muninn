# Secure transcript continuation (implementation contract)

## Outcome and current gap

An agent must be able to search encrypted history, fetch an authenticated,
credential-redacted conversational excerpt, and continue through the related
transcript in bounded pages, including transcripts larger than memory. Ordinary
MCP/HTTP results never contain raw credentials, tool payloads, or archive paths.
The explicitly authorized local credential-use/reveal path remains separate.

Today `search_secure_history` returns a signed hit capability and
`fetch_secure_history` returns one redacted span of at most 4,000 characters.
It cannot continue. The current structured parser also skips physical JSONL
lines above 256 KiB. Neither behavior meets the continuation requirement.
An archive-size limit, a raw-JSON fallback, or an unbounded in-memory parse is
not an acceptable way to close it.

## Selected design

Build a versioned, encrypted conversational projection for each immutable
archive snapshot. The CPU-only background builder streams and authenticates
the *entire* encrypted source, extracts only known provider user/assistant
message fields, applies credential redaction before publication, and writes
AEAD-sealed, bounded pages to owner-only staging. It atomically publishes a
completion record after source authentication and projection verification.
Partial or stale projections are never served. Normal capture and search do
not wait for projection; an unprojected hit reports `projection_pending` and
queues bounded work. It never falls back to returning raw JSONL.

The parser must consume JSON tokens and long string values incrementally.
`json.loads` per line and `ijson` scalar events can materialize an arbitrarily
large string, so neither alone establishes bounded memory. For fields whose
role is known only later in a record, stage candidate text in encrypted
bounded frames and publish it only after the complete record validates as a
supported conversational message. Unknown schemas, malformed JSON, tool
results, metadata, attachments, and injected host content are omitted with
explicit counts, never treated as conversation text.

Redaction must run over logical message content before slicing it into pages.
It needs streaming state for secrets that cross input/page boundaries;
unclassifiable material fails closed instead of leaking partial values.
Changing parser or redactor semantics invalidates old projections and triggers
rebuild. The ciphertext is bound by AEAD associated data to vault identity,
blob identity and hash, source size, snapshot version, projection format,
parser/redactor versions, page ordinal, and plaintext length. No plaintext
spool file or model call is part of projection building.

`fetch_secure_history` retains its existing short-span behavior for compatibility.
A new MCP/HTTP context-page operation starts from an existing search hit and
returns at most 4,000 redacted characters plus a signed continuation token.
Continuation binds the original hit's vault/blob/hash/version/term, next page,
parser/redactor generation, expiry, and a page/session budget. The endpoint
still requires the main local token, no-store response, one-at-a-time execution,
and rate limiting. A client cannot choose an arbitrary offset or another
snapshot. A renewed search can begin another bounded session, so quotas are
resource/privacy controls rather than a permanent transcript-size cutoff.

For a large source, page retrieval reads only authenticated projection pages,
not the entire archive again. A hit in tool-only content or a redacted secret
may yield `no_conversational_match`, never a raw fallback. The response says
whether more pages exist and whether any source records were deliberately
omitted, so agents cannot mistake partial context for the full transcript.

## Required tests before live rollout

1. Search hit -> first page -> subsequent pages -> final page reproduces the
   expected supported user/assistant conversation in order, with no duplicates
   or missing characters, for a multi-chunk source and a long single message.
2. Memory use remains bounded as source size and a single JSON string grow;
   private staging and committed database contain no plaintext canary.
3. A secret spanning input chunks or page boundaries is absent in *every*
   response. Tool results, metadata, injected text, and raw JSON syntax are
   absent even if those fields contain the query term.
4. Forged, replayed, wrong-vault/blob/version/term/page, expired, and exhausted
   cursors fail closed. A missing, stale, interrupted, or tampered projection
   releases no context.
5. Corrupt late archive chunks, bad UTF-8, malformed/unknown provider rows,
   cancellation, process crash, and insufficient disk space leave no completed
   projection and do not damage the existing archive or blind index.
6. Representative 1 GiB transcript build and multiple page fetches establish
   actual elapsed time, peak memory, disk overhead, and contention while
   capture/search remain available. No fixed ETA is claimed before measurement.
7. Live MCP and HTTP checks use a real indexed transcript and disclose only
   result metadata in test logs; a controlled service restart requires separate
   approval after code review and green CI.

## Rollout order and gates

Implement streaming structured extraction and redaction with red-first fixtures;
then encrypted projection staging/publication and recovery; then capability-
bound page API/MCP; then focused security, integration, large-source, and live
checks. Independently review the persistence/authority boundary and final
diff. Keep the current verified live service running until that gate is clear.
Commit and push portable code/docs/tests to the connected repository; runtime
paths, PIDs, tokens, and user-specific sources stay out of Git.

The existing credential scan is a separate remaining gate: its last project
summary had 29 UTF-8 errors and one walk error before parser/walk fixes. It
requires a new interactive, passphrase-held rescan and validated encrypted
backup; a successful archive-only scan does not close the project-file gap.
