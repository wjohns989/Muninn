# Phase 26: Epistemic Foraging & Cognitive Optimization

**Date:** 2026-02-24
**Status:** DRAFT
**Objective:** Implement the final pillar of the "Cognitive Optimization" triad: **Epistemic Foraging**. This moves the agent from "passive retrieval" to "active exploration" when it encounters ambiguity or uncertainty.

## 1. Vision

Standard RAG fails when the answer isn't in the top-K chunks.
**Epistemic Foraging** applies "Information Foraging Theory" (Pirolli & Card) to AI memory:
1.  **Ambiguity Detection:** Measure entropy in retrieval scores. If high entropy (flat distribution), the agent is "confused".
2.  **Scent Following:** Use the Knowledge Graph (Kuzu) to find "scent trails" — entities related to the query but not in the immediate top-K.
3.  **Active Querying:** Automatically generate secondary queries to "forage" along these trails.

## 2. Architecture

### New Module: `muninn/optimization/foraging.py`

#### 2.1 `EntropyDetector`
*   **Metric:** Shannon Entropy of normalized retrieval scores.
*   **Logic:**
    *   Normalize scores to prob distribution $P$.
    *   $H(X) = -\sum P(x) \log P(x)$
    *   High H -> Ambiguity (many weak matches). Low H -> Certainty (one strong match).

#### 2.2 `ForagingEngine` (extends `MuninnScout`)
*   **Trigger:** When $H(X) > Threshold$.
*   **Action:**
    1.  Take top-N ambiguous entities.
    2.  Query Graph for `(Entity)-[RELATED]->(Neighbor)`.
    3.  Compute "Information Gain Potential" of neighbors (e.g., node degree or importance score).
    4.  Execute secondary vector search using Neighbor terms.

### 2.3 MCP Integration
*   **Tool:** `forage_knowledge(query, ambiguity_threshold=0.7)`
*   **Auto-Trigger:** Can be enabled in `search_memory` via `auto_forage=True` flag.

## 3. Implementation Plan

### Step 1: Entropy Logic
- [ ] Implement `calculate_retrieval_entropy(scores)` in `muninn/scoring/entropy.py`.

### Step 2: Foraging Engine
- [ ] Implement `ForagingEngine` class in `muninn/optimization/foraging.py`.
- [ ] Implement graph traversal for "scent" detection.

### Step 3: MCP Wiring
- [ ] Add `forage_knowledge` tool.
- [ ] Update `search_memory` to return entropy metrics in `_meta`.

### Step 4: Validation
- [ ] Test case: "Apple" (ambiguous: Fruit vs Tech company).
- [ ] Verify foraging expands context to "iPhone" or "Pie" depending on graph scent.

## 4. ROI
*   **Solves "Zero-Recall"**: Finds answers that keyword search misses.
*   **Reduces Hallucination**: Agent admits "I am exploring" instead of guessing.
