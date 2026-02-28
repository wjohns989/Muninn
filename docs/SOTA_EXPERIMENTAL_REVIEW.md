# Muninn SOTA+ Experimental Review & Consolidation Report

> **Date**: February 28, 2026
> **Scope**: Consolidated review of experimental branches (`frosty-meitner`, `modest-pascal`, `parser-sandbox-hardening`, `mcp-timeout`)
> **Verdict**: High-value theoretical roadmap items identified; implementation logic mostly superseded by v3.24.0 modular architecture.

---

## 1. Overview of Experimental Worktrees

Four experimental worktrees were reviewed and consolidated into the `feature/sota-plus-archive` branch. These branches primarily represented the development transition towards the modular MCP architecture now present in `main`.

### 1.1 `claude/frosty-meitner` & `claude/modest-pascal`
- **Focus**: UI/UX refactor of `dashboard.html` and the initial modularization of `mcp_wrapper.py`.
- **Status**: Mostly redundant. The modular components (`muninn/mcp/`) are already live on `main`. The UI changes in these branches were incomplete and contained regressions (deleting existing diagnostic tools).
- **Consolidated Value**: The `docs/plans/` directory in these branches contains a rich history of Phase 4 and 5 planning that was previously "lost" in individual worktrees.

### 1.2 `feature/parser-sandbox-hardening` & `chore/mcp-timeout`
- **Focus**: Process sandboxing for ingestion parsers and transport timeout hardening.
- **Status**: Valuable hardening logic was identified. While some implementation details were older (pre-multimodal), the core "timeout-budgeting" philosophy is SOTA+.
- **Consolidated Value**: `docs/Refining MCP Verification.md` provides a comprehensive log of the reasoning behind the current transport incident replay logic.

---

## 2. SOTA+ Advancement Analysis: ROI Opportunities

The review of `docs/roi_and_architecture_analysis.md` (recovered from `parser-sandbox-hardening`) identified three critical advancement opportunities that are NOT yet in `main` but provide high SOTA+ ROI.

### 2.1 P0: SNIPS Multipliers -> Importance Scoring
- **Gap**: SNIPS retrieval feedback (Phase 2) adjusts retrieval weights but NOT memory decay.
- **ROI**: High. Prevents the "silent decay" of memories that are highly useful but old/low-centrality.
- **Action**: Pass signal-specific SNIPS multipliers into `calculate_importance()` during the consolidation decay phase.

### 2.2 P0: Centrality Discontinuity Fix
- **Gap**: Memories without graph entities receive `centrality=0`, causing a systematic 20% score disadvantage.
- **ROI**: High. Improves the fairness of the importance model for plain-text vs. entity-linked memories.
- **Action**: Implement a centrality baseline/normalization factor for entity-free memories.

### 2.3 P1: CoALA-Style Session Inhibition
- **Gap**: Long sessions suffer from "context rot" (repeatedly injecting the same memories).
- **ROI**: High. Reduces token waste by 30-45% and window pollution by penalizing recently-retrieved items in the same session.
- **Action**: Implement session-aware inhibition in `hybrid.py` using a simple LRU cache of retrieved memory IDs.

---

## 3. SOTA+ Verdict

While the **code** in the experimental branches was largely an earlier version of the current modular architecture, the **theoretical and planning assets** are foundational for the next era of Muninn (v3.25.0+). 

The consolidation into `feature/sota-plus-archive` successfully preserves:
1. The **ROI Roadmap**: A clear path to fixing logical inaccuracies in the importance model.
2. The **Planning Provenance**: Detailed Phase 4/5 logs that justify current security and transport decisions.
3. The **Vision Document**: `docs/plans/MUNINN_2026_VISION_AND_ROADMAP.md` which defines the leap from "Persistent Memory" to "Cognitive Infrastructure".

**Next Steps**:
- Implement the P0 SNIPS integration on `main`.
- Deprecate and remove the old worktrees now that their value is archived.
