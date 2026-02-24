# Phase 24: Cognitive Architecture (CoALA) Integration Plan

**Date:** 2026-02-24
**Status:** DRAFT
**Objective:** Bridge the gap between Muninn's passive memory store and an agent's active decision-making process by introducing "Proactive Reasoning" capabilities inspired by the CoALA (Cognitive Architectures for Language Agents) framework.

## 1. Vision

Muninn currently excels at **Episodic** and **Semantic** memory (Long-term). However, it lacks a dedicated **Reasoning** layer that can evaluate the *completeness* and *validity* of retrieved information relative to a user's goal.

Phase 24 introduces `muninn.reasoning`, a module dedicated to **Omission Filtering** and **Native Grounding**.

### The "Omission" Problem
Agents often hallucinate details when they lack information (e.g., guessing a deployment script path).
**Solution:** A "Gap Detector" that retrieves relevant context and explicitly identifies what is *missing* before the agent attempts to act.

## 2. Architecture

### New Module: `muninn.reasoning`

#### 2.1 `OmissionDetector`
*   **Input:** `query` (User intent), `context` (Current conversation state/working memory).
*   **Process:**
    1.  **Retrieval:** Use `MuninnScout` to hunt for memories related to the query and context.
    2.  **Analysis:** Use `ExtractionPipeline` (LLM) with a specialized "Gap Analysis" prompt.
        *   *Prompt logic:* "User wants to [Action]. We know [Memories]. Do we have all necessary information (e.g., credentials, paths, IPs)? If not, list exactly what is missing."
    3.  **Output:** `ReasoningResult` object containing:
        *   `verdict`: "sufficient" | "insufficient"
        *   `missing_info`: List[str] (e.g., ["production IP address", "AWS credentials"])
        *   `grounding_memories`: List[str] (IDs of memories used to verify)

### 2.2 API / MCP Integration
*   **REST Endpoint:** `POST /reasoning/detect-gaps`
*   **MCP Tool:** `detect_information_gaps(query, context)`

## 3. Implementation Steps

### Step 1: Core Reasoning Engine
- [ ] Scaffold `muninn/reasoning/` package.
- [ ] Implement `OmissionDetector` class using `MuninnScout`.
- [ ] Define `GapAnalysis` pydantic model.

### Step 2: Extraction Prompting
- [ ] Add `GAP_ANALYSIS` prompt template to `muninn/extraction/prompts.py` (or similar).
- [ ] Wire `ExtractionPipeline` to support ad-hoc reasoning tasks (or simple LLM call if pipeline is too rigid).

### Step 3: Server & MCP Wiring
- [ ] Add `/reasoning/detect-gaps` endpoint to `server.py`.
- [ ] Add `detect_information_gaps` tool to `mcp_wrapper.py`.

### Step 4: Validation
- [ ] Create `tests/test_v3_24_0_coala_reasoning.py`.
- [ ] Verify detection of missing info in known scenarios (e.g., "Deploy to prod" with no prod IP memory).

## 4. ROI / Value
*   **Reliability:** Prevents agents from guessing parameters.
*   **Efficiency:** Fails fast before expensive multi-step execution.
*   **Safety:** explicitly flags missing credentials or permissions.
