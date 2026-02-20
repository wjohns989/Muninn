# Muninn 2026 Vision & Capability Roadmap

> **Document Status:** DRAFT  
> **Last Updated:** 2026-02-20  
> **Objective:** Define the next-generation architecture, feature expansions, and optimization parameters for Muninn over the 2026 timeframe, filling the gaps from the v3.3 SOTA+ implementation.

## 1. Executive Vision

Building on the successful extraction, retrieval, and consolidation architectures established in v3.3 of the SOTA+ plan, Muninn's 2026 trajectory focuses on moving from a high-performance **Reactive Memory Store** to a **Proactive Cognitive Architecture**.

The next milestones demand integration of RL-driven memory retention policies, bulletproof parse-time isolation for untrusted documents, cross-device P2P synchronization, and real-time temporal reconstruction.

## 2. Identified Gaps & Optimization Opportunities (Post-v3.3)

Based on our recent implementation reviews, these are the critical areas needing attention:

| Domain | Current State (v3.3) | 2026 Target State |
|--------|----------------------|-------------------|
| **Governance** | Rule-based decay/promotion (fixed thresholds) | **RL-Driven Policy Governance**: Q-learning agents that adjust importance/decay rates based on retrieval success and user feedback. |
| **Ingestion Security** | Standard subprocess parsing (tika/pdfplumber) | **Parser Sandboxing**: Strict process isolation (seccomp/containerized) for handling untrusted binary formats (DOCX, PDF) to prevent exploitation. |
| **Federation** | Mentioned in architecture, no formal protocol | **Cross-Device CRDT Sync**: Merkle-DAG based P2P synchronization to share Procedural/Semantic memories safely between instances. |
| **Cognitive Arch** | Simple retrieval-augmented generation (RAG) | **Full CoALA Integration**: Reasoning, Acting, and Memory (Working, Episodic, Semantic) operating in a continuous, reflective loop. |
| **Data Integrity** | Periodic cleanup daemon | **Live Conflict Resolution**: Background LLM verification passes using NLI (Natural Language Inference) to detect and resolve contradictory facts in Semantic memory. |

---

## 3. New Feature Brainstorming

### 3.1. Policy-Aware Memory Governance (RL-Driven)

* **Concept:** Instead of static `DECAY_THRESHOLD = 0.1`, use a multi-armed bandit or Q-learning approach to adjust the decay multiplier per memory *category*. Memories that successfully answer questions get a "reward" signal, strengthening their weights; unhelpful memories decay faster.
* **Implementation:** Introduce a `Feedback` table. Modify the consolidation daemon to update weights dynamically.

### 3.2. E-mem Episodic Context Reconstruction

* **Concept:** Allow agents to "time-travel" to a specific point in a project's history. Reconstruct the graph strictly from nodes and edges timestamped prior to `T`.
* **Implementation:** Implement strictly bi-temporal queries in Kuzu. Ensure all node/edge updates are append-only (or soft-delete).

### 3.3. Zero-Trust Parser Sandboxing

* **Concept:** File ingestion parsing (especially complex binaries like PDF/DOCX) is a massive attack vector.
* **Implementation:** Isolate the parsing engine. Use an empty chroot or WebAssembly (Wasm) runtime for document text extraction to guarantee no host access during unstructured data ingestion.

---

## 4. Preemptive Debugging & Risk Mitigation

Before executing the above phases, we must solve these predictable issues:

| Anticipated Risk/Bug | Root Cause Prediction | Preemptive Mitigation Strategy |
|----------------------|-----------------------|--------------------------------|
| **SQLite Write Locking** | The consolidation daemon (especially during graph re-indexing) will hold write locks, blocking concurrent agent ingestion calls. | Enable `WAL` (Write-Ahead Logging) mode on all SQLite metadata stores. Implement a distinct `ConsolidationWriteQueue` buffer to batch writes rather than locking iteratively. |
| **Memory Bloat (OOM)** | Out-of-control Episodic memory ingestion from verbose or looping LLM outputs, exceeding the 100k+ node scale limits. | Implement strict L1 Hash/Semantic filtering *before* ingestion. Drop any new episodic memory with >0.98 cosine similarity to a node created in the last 1-hour window. |
| **xLAM JSON Instability** | Extraction models occasionally fail to close JSON tags or follow the PA-Tool output schema exactly. | Implement strict enforcement using `Outlines` or `Instructor` for structured generation, guaranteeing schema compliance at the sampling level. |
| **Graph Edge Explosion** | "Connects-everything" syndrome where a common entity (e.g., "Python") creates thousands of dense edges, slowing traversal algorithms. | Cap global edge degree per entity. Entities approaching the limit mutate into "Supernodes" requiring explicit pathing intent to traverse, effectively pruning passive search over-expansion. |

---

## 5. Phased Delivery Roadmap & Checkpoints

### Phase 21: Zero-Trust Parser Isolation & Performance Tuning

* **Objective:** Bullet-proof ingestion and resolve performance bottlenecks.
* **Checkpoints:**
  * [ ] Introduce standalone `.wasm` or restricted-subprocess environment for document parsers.
  * [ ] Apply WAL mode to SQLite dbs; implement async write-batching queue.
  * [ ] Deploy Outlines/Instructor over local xLAM calls for guaranteed JSON schema conformance.
* **Success Parameters:** 0% JSON parse failures in CI; parsing a malicious PDF results in graceful crash with no host leakage; SQLite concurrent write benchmark improves by 5x to handle parallel agent ingestion.

### Phase 22: Live Integrity & NLI Conflict Resolution

* **Objective:** Prevent semantic hallucination inside the memory store.
* **Checkpoints:**
  * [ ] Build NLI pipeline using a lightweight BERT/DeBERTa model.
  * [ ] Add background task to the Consolidation Engine: Sample Semantic memories about the same entity.
  * [ ] Apply NLI: If (`M1` contradicts `M2`), flag for resolution logic (e.g., trust newer by timestamp, or ask user).
* **Success Parameters:** Known contradictory facts are identified and resolved within the 6-hour consolidation cycle with >95% accuracy.

### Phase 23: CoALA Cognitive Architecture & RL Policy

* **Objective:** Adaptive retention.
* **Checkpoints:**
  * [ ] Introduce RL agent tracking reward signals from retrieval successes.
  * [ ] Dynamically update `importance` decay multipliers per memory type.
  * [ ] Complete Episodic "Time-Travel" bi-temporal query interface for Kuzu.
* **Success Parameters:** System automatically prunes useless logs 3x faster than vital project code configurations, confirmed by A/B testing against baseline static decay.

### Phase 24: P2P Memory Federation

* **Objective:** Securely sync memories between local machines.
* **Checkpoints:**
  * [ ] Implement Merkle-DAG state representation of Procedural/Semantic memory layers.
  * [ ] Define protocol for sharing via generated `sync_bundle.json`.
  * [ ] Enable conflict-free resolution using timestamped state vectors.
* **Success Parameters:** Two distinct Muninn instances correctly sync to an identical Semantic memory state without losing explicitly confirmed data on either side.

---

## 6. Execution Protocol

As with all Muninn development, we proceed via the established SOTA+ guidelines:

1. Feature Branching (`feature/v3.4.0-parser-sandbox`)
2. Rigorous isolated test creation.
3. Code construction using the most efficient tooling and avoiding LLM-heavy dependencies where static logic suffices.
4. Validation using `eval.phase_hygiene` and standard pytest suites.
