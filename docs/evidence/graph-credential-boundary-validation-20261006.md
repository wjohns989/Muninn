# Graph credential boundary: isolated proof

Changed dependency: `muninn/store/graph_store.py` text reads/write inputs and the
actual authenticated `/graph` handler, using the existing ordinary credential
projection. Legacy SQLite hydration alone left these graph consumers uncovered.

Twelve initial synthetic-only regression checks failed before implementation:
legacy entity/search values were released, writes were not refused before a
connection, exception text reached logs/HTTP and the user filter was ignored.
After implementation and review corrections, the final affected set passed 72
checks in 2.92 seconds:
`test_graph_credential_boundary.py`, `test_ordinary_credential_boundary.py`,
`test_federation.py`, `test_federation_broadcast.py`, `test_memory_shutdown.py`.
The earlier 58/68-pass runs overlap and are not summed. One unrelated installed
Hugging Face environment-variable deprecation warning was emitted.

Graph fixtures use synthetic rows and a mock driver, not a live Kuzu store. They
exercise actual public method bodies and a compiled actual HTTP handler without
importing server configuration, opening user data, model inference or peer egress.
Coverage includes exact normal metadata/IDs/scores/counts, original row-list
nonmutation, nested relation refusal before partial writes, whole-input screening
before truncation, label guards before querying and category-only diagnostics
for reads, writes, schema errors, traversal and reference operations. The HTTP
fixture proves user filter forwarding, defensive projection, no-store and static
failure. No real credential value is used as a fixture or printed.

Independent design review flagged two label consumers (`find_related_memories`
and `get_entity_centrality`); both are guarded before database access in the
actual change and covered by focused tests. Actual-diff review and live exact-
candidate installation are separate dependent proof, not assumed by these tests.

Actual-diff review then flagged whole-record ID projection and omitted numeric-
annotated inputs. Search now projects only descriptive fields, preserves safe
canonical IDs, and omits credential-shaped legacy IDs instead of releasing values
or inventing masked aliases. Confidence/hours-apart are guarded before connection,
despite annotations. Four added regression cases prove mixed identity handling
and numeric-argument refusal. Compilation of the handler strips the decorator:
route-level authentication still requires a live HTTP check after installation.

This source change does not migrate or delete original graph/SQLite/vector
records, alter schema, unlock the credential vault or complete the historical
backlog. Arbitrary unlabeled credentials, legacy vault-only persistence, complete
federation isolation and full data recovery remain outside this component proof.
