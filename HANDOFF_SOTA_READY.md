# Muninn Handoff: SOTA+ Production Ready (v3.24.0)

## 🎯 Current Status & Intent
Muninn is now at **v3.24.0 (Phase 26 COMPLETE)**. The system has been hardened against CI deadlocks and all 1422 core tests are passing. Experimental worktrees have been consolidated into `feature/sota-plus-archive` to preserve research while maintaining a clean `main` branch.

The primary intent is to move from **Stable Infrastructure** to **Proactive Intelligence** by finalizing the SOTA+ logic identified in the experimental review.

## ✅ Accomplishments in this Session
1.  **CI Remediation**: Resolved the 10-minute hang in `Transport Incident Replay Gate` by fixing `StoreLock` (added `LOCK_NB` for non-blocking timeouts on Linux) and correcting test mocks for `ensure_server_running`.
2.  **Modularization Merge**: Merged PR #56, completing the migration to the `muninn/mcp/` modular package and deprecating the legacy monolithic `mcp_wrapper.py` (now a facade).
3.  **Experimental Consolidation**:
    - Recovered 50+ planning documents and research logs from orphan worktrees.
    - Archived state in `feature/sota-plus-archive`.
    - Created `docs/SOTA_EXPERIMENTAL_REVIEW.md` summarizing ROI opportunities.
4.  **Version Alignment**: Bumped project to `v3.24.0` across all metadata and updated `README.md` with the Cognitive Architecture roadmap.

## ⚠️ Known Issues: Local UI Functionality
The local Muninn service (`muninn_standalone.py` / `server.py`) currently has a regression in **UI-based ingestion**:
- **Symptoms**: Memories added via the "Direct Ingestion" or "File Discovery" tabs in the Huginn dashboard (localhost:42069) are not appearing in search or the activity log.
- **Hypothesis**: Likely a failure in the extraction pipeline (Ollama connectivity) or a mismatch in `user_id`/`namespace` scoping between the UI (hardcoded `global_user`) and the server initialization.
- **Recent Change**: Security was recently clarified to support both `MUNINN_API_KEY` and `MUNINN_AUTH_TOKEN`. Ensure `server.py` is correctly passing the injected token to the UI.

## 🚀 SOTA+ Roadmap (Next Steps)
The following tasks are prioritized based on the `SOTA_EXPERIMENTAL_REVIEW.md`:

1.  **[P0] SNIPS -> Importance Loop**: Modify `muninn/scoring/importance.py` to accept retrieval feedback multipliers. Memories that are frequently "Helpful" should have their decay slowed.
2.  **[P0] Centrality Normalization**: Fix the "Centrality Discontinuity" where memories without graph entities are penalized. Implement a baseline centrality for entity-free nodes to ensure fair RRF fusion.
3.  **[P1] CoALA Session Inhibition**: Implement a short-term inhibition filter in `HybridRetriever` to prevent context rot (repeating the same memories in a single session).
4.  **[Cleanup] Worktree Finalization**: All orphan worktrees have been removed. Future work should branch from `main`.

---

## 🛠 Developer Prompt
"Resume Muninn development. The system is at v3.24.0 with all tests passing, but the local UI ingestion is currently failing. First, diagnose and fix the UI ingestion failure (check Ollama connectivity and scoping in `server.py`). Then, implement the P0 ROI items from `docs/SOTA_EXPERIMENTAL_REVIEW.md`: integrate SNIPS multipliers into the importance scoring model and fix the centrality discontinuity for entity-free memories. Ensure all changes maintain the 100% test pass rate."
