# Graph credential boundary

The SQLite hydration boundary does not protect independent legacy graph text.
Actual graph search returns summaries and entity-match strings directly; entity
listing returns names/type/namespace directly. `/graph` also previously ignored
its requested user filter and echoed driver exceptions.

Reuse the existing credential-specific pure projection at the two text-producing
GraphStore reads. Preserve normal IDs, scores, rank, mention counts and useful
credential service/location metadata. These views are ephemeral copies, not
stored replacements. A credential-shaped legacy memory ID is omitted from text
search, not released raw or replaced by a fabricated alias that could identify
another memory. Numeric/ID-only graph consumers are not descriptive text exports.

Screen complete inputs to the five ordinary graph write paths before connection,
upsert, nested entity creation or 500-character truncation. Apply the same guard
to caller-supplied label queries. Driver diagnostics contain a static operation
and exception category, never query labels or exception contents. `/graph`
forwards its requested user filter, projects defensively for alternate graph
implementations, uses `no-store`, and reports a static error.

| Consumer | Boundary |
| --- | --- |
| Graph search / entity listing | Shared ephemeral credential projection |
| Direct ordinary graph writes | Shared refusal before any side effect |
| Label query diagnostics | Input guard and category-only errors |
| Authenticated `/graph` | User filter, defensive view, no-store, static failure |

No graph schema/migration, deletion, index rewrite, federation/network policy or
vault unlock is part of this change. Federation bundles currently hydrate through
the SQLite boundary; this is not a federation authorization review. Stored legacy
plaintext/index copies still require separately authenticated coordinated migration.
Best-effort detection does not certify arbitrary unlabeled or novel secrets safe.
The existing graph migration branch is unchanged; no whole-graph restore or legacy
schema-upgrade claim is made by these getter/write guards.

Synthetic checks must exercise the actual getters and compiled HTTP handler,
normal metadata preservation, refusal before connection/partial relation writes,
full-input-before-truncation and exception/log non-disclosure. No live credential
marker may be persisted. Live proof may print only aggregate/auth/cache/detection
results, never entities, summaries, keys or passphrases. Installation remains gated
on independent actual-diff review and the existing exact-candidate guarded reload.
