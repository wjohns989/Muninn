# Muninn 2026: Critical Feasibility Audit & Course Correction

> **Date:** 2026-02-20
> **Status:** AUDITED / COURSE-CORRECTED
> **Objective:** Evaluate the initial 2026 Vision and Roadmap for over-engineering, hidden assumptions, and ROI inefficiencies.

## 1. Executive Summary of Audit

The initial 2026 roadmap proposed highly conceptual, research-grade features (RL-driven policies, NLI continuous resolution, P2P CRDT sync). Upon rigorous evaluation under the "Efficiency > Verbosity" and "Production-Grade Only" constraints, these proposals were deemed **over-engineered, excessively complex, and risky** for a local-first Python MCP server.

The strategy has been formally pivoted to achieve the same functional outcomes using simpler, more robust heuristics and existing architectural components.

---

## 2. Component Audits & Resolutions

### 2.1. P2P Memory Federation (Phase 24)

* **Initial Proposal:** Cross-device CRDT Sync via Merkle-DAGs.
* **Critical Audit:** Achieving atomic synchronization across a heterogeneous tri-store (SQLite, Kuzu, Qdrant) over a P2P network in Python is mathematically complex and fragile. It distracts from Muninn's core value proposition: being an ultra-fast, local-first memory store.
* **Resolution:** **ABORTED**. The "Multi-device sync" objective has been completely removed from the 2026 scope. Muninn will remain fiercely local. Users requiring sync can leverage standard Git-backed dotfile syncing for the `~/.muninn/data` directory offline, rather than requiring the MCP server to handle live P2P networking.

### 2.2. Policy-Aware Memory Governance (Phase 23)

* **Initial Proposal:** Q-learning/Reinforcement Learning agents for dynamic decay rates.
* **Critical Audit:** Training a local neural value network for memory retention introduces non-determinism and massive risk of catastrophic forgetting or "reward hacking" (e.g., deleting everything).
* **Resolution:** **PIVOTED**. We will implement an **Elo-Rated SNIPS Governance** system. We *already* built SNIPS retrieval feedback integration (PR #50). We will map retrieval success to a standard Elo rating algorithm. Memories that help solve tasks gain Elo; unused/penalized memories lose Elo. This provides dynamic decay without the over-engineering of an RL model.

### 2.3. Live Integrity & NLI Conflict Resolution (Phase 22)

* **Initial Proposal:** Continuous background NLI (DeBERTa) inference to detect semantic contradictions.
* **Critical Audit:** Running DeBERTa continuously in the background will cripple local hardware resources and cause latency spikes, violating SOTA+ performance requirements.
* **Resolution:** **PIVOTED**. We will use **Temporal Shadowing & LLM-Assisted Merge**. Conflict resolution is pushed to the existing 6-hour consolidation `MERGE` phase. If two memories highly correlate on an entity but contradict, we use the *existing* local LLM (xLAM or Ollama) to synthesize the update via a simple prompt schema, relying heavily on `ingested_at` timestamps to let newer facts gracefully "shadow" older ones logically.

### 2.4. Zero-Trust Parser Sandboxing (Phase 21)

* **Initial Proposal:** WebAssembly (Wasm) or complex containerized isolation.
* **Critical Audit:** Wasm runtimes for Python parsing libraries (like `pdfplumber` or `tika`) are not natively mature and require massive maintenance overhead.
* **Resolution:** **PIVOTED**. Implement **Strict OS-Level Subprocess Isolation**. Run parsing tasks via `subprocess.run` with aggressive OS timeouts, memory cgroup limits (Linux) or Job Objects (Windows), and pipe the text output back. If the subprocess crashes from a malicious PDF, the main MCP server remains unaffected. This achieves 90% of the safety with 10% of the complexity.

---

## 3. High-ROI Ecosystem Impact

By abandoning these over-engineered concepts, we reclaim thousands of development hours. The ROI impact is massive:

1. **Dependency Reduction:** We avoid bloating the package with heavy RL libraries (`stable-baselines3`, `transformers` for NLI, P2P networking libs).
2. **Performance Maintenance:** The daemon remains lightweight, running entirely within SQLite/Kuzu limits without saturating the GPU in the background.
3. **Deterministic Testing:** Elo ratings and Subprocess isolation are 100% deterministic and can be easily mocked and tested in CI (`eval.phase_hygiene`), whereas RL and continuous NLI are notoriously difficult to test reliably.
