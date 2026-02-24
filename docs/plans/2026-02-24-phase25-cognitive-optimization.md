# Phase 25: Cognitive Optimization & Knowledge Distillation

**Date:** 2026-02-24
**Status:** DRAFT
**Objective:** Move beyond simple storage/retrieval to active "Cognitive Optimization" — compressing messy episodic logs into pristine semantic knowledge, proactively foraging for missing context, and surgically correcting errors based on feedback.

## 1. Vision

Current Muninn is a "perfect recorder" — it remembers everything you throw at it. Over time, this leads to **Context Rot**: the accumulation of trivial, outdated, or fragmented facts that clutter retrieval.

Phase 25 introduces **Knowledge Distillation**: a biological "sleep cycle" for AI memory that rewrites history into a clean "textbook" format. It also adds **Epistemic Foraging** (active curiosity) and **Self-Correction** (memory surgery).

---

## 2. The Three Pillars

### Pillar A: Sleep-Time Knowledge Distillation (Consolidation 2.0)
*   **Concept:** Episodic memory is messy. Semantic memory should be clean.
*   **Mechanism:** A background `DistillationDaemon` runs during idle periods.
    1.  **Cluster:** Identify clusters of related episodic memories (e.g., "50 facts about 'Project Phoenix'").
    2.  **Synthesize:** Feed these 50 fragments to a local LLM (e.g., `qwen2.5`) with a "Librarian Prompt."
    3.  **Rewrite:** The LLM produces a single, structured "Semantic Manual" entry (e.g., "Project Phoenix Architecture Guide").
    4.  **Archive:** The original 50 episodes are marked `archived` (removed from hot vector index, kept in SQLite for audit).
*   **ROI:** Massive reduction in retrieval noise and token costs; higher quality "ground truth."

### Pillar B: Epistemic Foraging (Active Inference)
*   **Concept:** Don't just fail; explore.
*   **Mechanism:** When retrieval confidence is low or ambiguous (high entropy in RRF scores):
    1.  **Detect Ambiguity:** `OmissionDetector` (from Phase 24) flags "low confidence".
    2.  **Forage:** The system triggers a secondary `MuninnScout` "foraging" run using *related* entities from the graph, checking neighbors of neighbors.
    3.  **Hypothesize:** If still ambiguous, generate a "Clarifying Question" to the agent/user (e.g., "I found references to both 'v1' and 'v2' databases; which context is relevant?").

### Pillar C: Self-Correcting Memory (Generative Feedback)
*   **Concept:** "No, that's wrong" should fix the database.
*   **Mechanism:**
    1.  **Trigger:** Explicit negative feedback signal (`outcome=0.0`) or conversational "correction" pattern.
    2.  **Trace:** Identify the specific memory ID used in the erroneous response (via `grounding_memory_ids` from Phase 24).
    3.  **Surgery:** Dispatch a "Mutation Job":
        *   Retrieve the faulty memory.
        *   Apply the user's correction.
        *   **Rewrite** the content in-place (or tombstone + replace).
    4.  **Reinforce:** Boost the `importance` of the corrected memory.

---

## 3. Architecture & Implementation

### New Module: `muninn.optimization`

#### 3.1 `DistillationDaemon`
*   **Extends:** `muninn/consolidation/daemon.py`
*   **Dependencies:** `ExtractionPipeline` (for synthesis)
*   **Logic:**
    *   Query `metadata` for `memory_type="episodic"` & `archived=False`.
    *   Cluster by `entity_names` intersection (Jaccard similarity).
    *   Process clusters > 5 items.

#### 3.2 `ForagingEngine`
*   **Extends:** `muninn/retrieval/scout.py`
*   **Logic:**
    *   Implements `active_inference_search(query)`
    *   Computes "Information Gain" (IG) potential of unvisited graph nodes.

#### 3.3 `MemorySurgeon`
*   **New Class:** `muninn/optimization/surgeon.py`
*   **Methods:**
    *   `correct_memory(target_id: str, correction: str)`
    *   `refactor_cluster(cluster_ids: List[str], new_concept: str)`

---

## 4. MCP Tools

1.  `trigger_distillation`: Force-run the sleep-cycle process (manual).
2.  `correct_fact`: Explicitly patch a memory with new truth (agent-driven surgery).
3.  `forage_knowledge`: Active exploration tool for agents stuck in ambiguity.

## 5. Success Metrics

*   **Compression Ratio:** `(Original Episodic Tokens) / (Distilled Semantic Tokens)` > 5x.
*   **Retrieval Precision:** nDCG improvement on "complex multi-hop" benchmarks.
*   **Correction Retention:** % of times the *corrected* fact is retrieved in the next turn (should be 100%).
