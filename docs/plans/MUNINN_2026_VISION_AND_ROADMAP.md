# Muninn 2026 Vision & Capability Roadmap (Revised)

> **Document Status:** SOTA+ APPROVED  
> **Last Updated:** 2026-02-20  
> **Objective:** Define the next-generation architecture for Muninn. This revised roadmap explicitly favors high-ROI, pragmatic, and deterministic engineering patterns over high-complexity theoretical algorithms.

## 1. Executive Vision

Building on the successful extraction, retrieval, and consolidation architectures established in v3.3 of the SOTA+ plan, Muninn's 2026 trajectory focuses on robust, uncompromising local-first execution.

The core ambition is to evolve Muninn into a deeply integrated, self-optimizing engine using deterministic heuristics and strict isolation, definitively replacing Mem0 without sacrificing local hardware resources to background neural processing.

---

## 2. Identified Gaps & Pragmatic Solutions

Based on the v3.3 gap analysis and a rigorous feasibility audit, these are the critical areas needing attention:

| Domain | Current State (v3.3) | 2026 Pragmatic Target |
|--------|----------------------|-----------------------|
| **Governance** | Rule-based decay/promotion | **Elo-Rated Retention**: Leveraging the newly integrated SNIPS retrieval feedback. Memories act as "players" in an Elo system; useful retrievals increase ratings (slowing decay), useless ones drop ratings. |
| **Ingestion Security** | Standard subprocess parsing | **Strict Subprocess Isolation**: External parsers run in heavily restricted OS subprocesses (timeouts, memory caps, stripped permissions) communicating only via stdout, neutralizing malicious documents. |
| **Conflict Resolution**| Periodic cleanup daemon | **Temporal Shadowing & LLM Merge**: During the 6h consolidation cycle, overlapping contradictory entity graphs are synthesized using the existing xLAM/Ollama extraction models and deterministic timestamp precedence. |
| **Integrity** | Graph cycles possible | **Episodic Context Reconstruction**: Strict bi-temporal Kuzu queries allowing agents to seamlessly "time-travel" to a project's state at time `T` for pure, untainted context. |

*(Note: Multi-device federation has been explicitly abandoned to preserve the ultra-fast, local-first tri-store architecture without the catastrophic complexity of distributed P2P consensus.)*

---

## 3. Preemptive Debugging & Risk Mitigation

| Anticipated Risk/Bug | Mitigation Strategy | ROI / Impact |
|----------------------|-----------------------|--------------|
| **SQLite Write Locking** | Enable `WAL` (Write-Ahead Logging) mode on metadata stores. Implement a `ConsolidationWriteQueue` to batch-write graph and metadata updates instead of iterative locking. | **High:** Prevents the MCP server from freezing during agent ingestion if the background daemon is running a heavy merge cycle. |
| **Memory Bloat (OOM)** | Implement strict L1 Hash/Semantic filtering *before* ingestion. Drop any new episodic memory with >0.98 cosine similarity to an active working memory node. | **High:** Conserves Qdrant memory constraints and limits background consolidation processing time. |
| **LLM Schema Instability** | Enforce structured generation via `Outlines` or `Instructor` for local xLAM calls. | **Critical:** Guarantees 0% JSON parse failures, ending the cycle of retry loops that consume tokens and time. |

---

## 4. Phased Delivery Roadmap & Checkpoints

### Phase 21: Zero-Trust Parser Isolation & Ingestion Safety

* **Objective:** Bullet-proof ingestion against binary exploits without over-engineering Wasm.
* **Checkpoints:**
  * [ ] Encapsulate Tika/PDFPlumber/binary parsers in `subprocess.run` wrappers.
  * [ ] Apply strict OS-level resource limits (rlimits, timeouts).
  * [ ] Implement robust `subprocess` crash recovery mapping to a "Fallback Extraction" (raw text only).
  * [ ] Integrate `Outlines` or `Instructor` for guaranteed xLAM `get_entities` JSON schema compliance.
* **Success Parameters:** A payload designed to infinite-loop or OOM the parser gracefully crashes the subprocess in <5s, logs a warning, and allows the MCP server to continue operating normally.

### Phase 22: Temporal Knowledge Graph (TKG) & Shadowing

* **Objective:** Evolve Kuzu into a fully realized Temporal Knowledge Graph (TKG) to resolve semantic hallucination traversing time.
* **Checkpoints:**
  * [ ] Formalize `VALID_FROM` and `VALID_TO` edges in the Kuzu schema to support pure chronological reasoning.
  * [ ] Add contradiction-detection heuristic during Phase 2 (MERGE) of the consolidation cycle (e.g., entity conflict scoring).
  * [ ] Dispatch an LLM synthesis prompt to evaluate the contradiction, establishing "Shadow Edges" where outdated facts are preserved but bypassed in default retrieval.
* **Success Parameters:** Longitudinal queries (e.g., "What was the architecture before we switched to SQLite?") successfully traverse the TKG, and changing a known configuration fact deterministically shadows the older memory.

### Phase 23: Elo-Rated SNIPS Governance

* **Objective:** Dynamic retention without the overhead of neural RL agents.
* **Checkpoints:**
  * [ ] Establish a baseline Elo rating (e.g., 1200) for all newly ingested episodic memories.
  * [ ] Hook into the `record_retrieval_feedback` endpoint.
  * [ ] When a memory is successfully used (outcome=1.0), calculate Elo gain against the task complexity; if penalized, calculate drop.
  * [ ] Map the Elo rating directly into the standard exponential decay curve as a powerful multiplier.
* **Success Parameters:** Unused conversational logs cleanly age out of the system 4x faster than highly referenced architectural guidelines, purely driven by usage statistics.

---

## 5. Execution Protocol

1. Feature Branching (`feature/v3.4.0-pragmatic-roadmap`)
2. Use deterministic, statically typed Python logic.
3. Validate using `eval.phase_hygiene` and standard pytest suites.
4. **Never sacrifice quality for speed. Code > documentation. Evidence > assumptions.**
