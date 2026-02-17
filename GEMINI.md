# Muninn MCP — SOTA+ Development Context

**Role:** Muninn is a local-first, assistant-agnostic persistent memory infrastructure designed to outstrip existing systems (like Mem0 or Serena) through deterministic retrieval, neuroscience-inspired consolidation, and production-grade operational controls.

**Core Mandate:** This project is developed *using Muninn to develop itself*. Every contribution must meet SOTA+ (State of the Art Plus) standards of precision, quality, and logic.

---

## 🚀 SOTA+ Development Philosophy

1.  **Production-Grade Only:** NEVER use placeholders, stubs, or "samples" for core logic. All code must be production-ready, typed, and robust.
2.  **Evidence-Driven Logic:** Every decision must be grounded in the codebase's existing architecture (`docs/ARCHITECTURE.md`) or the SOTA+ implementation plan (`SOTA_PLUS_PLAN.md`).
3.  **Optimal Reasoning:** Question every assumption. Understand the implications of every choice. Avoid illogical or unsound reasoning at all costs.
4.  **Tool-First Execution:** Always leverage the most powerful tools available (MCPs, research tools, `context7`) to ensure a comprehensive grasp of the project before acting.
5.  **Quality Over Speed:** Never sacrifice precision for speed. A correctly implemented, well-tested feature is the only acceptable outcome.

---

## 🛠 Operational Guidelines (Always-On)

When working on Muninn, the following modes are strictly enforced:

*   **--orchestrate:** Manage multi-step implementations with clear planning and checkpoints.
*   **--ultrathink:** Perform deep, reflective reasoning on architectural impacts before modification.
*   **--uc (Ultra-Compressed):** Maintain maximum token efficiency in communication without losing clarity.
*   **--delegate:** Use sub-agents for parallel analysis or specialized investigation when appropriate.
*   **--all-mcp:** Utilize all available Model Context Protocol tools to verify state and context.
*   **--task-manage:** Track progress through the `SOTA_PLUS_PLAN.md` phases rigorously.

---

## 🏗 System Architecture

*   **Retrieval Engine:** 6-signal hybrid (Vector, Graph, BM25, Temporal, Goal, Chain) with RRF fusion and adaptive weighting.
*   **Extraction:** Tiered pipeline (Rules → Instructor/xLAM → Ollama) for validated structured output.
*   **Memory Integrity:** NLI-based conflict detection and semantic deduplication.
*   **Consolidation:** Background daemon performing decay, merge, and promotion (Episodic → Semantic → Procedural).
*   **Transport:** Hardened MCP stdio wrapper with deadline budgeting and recovery guardrails.

---

## 📂 Project Structure

- **Core Engine:** `muninn/` (native engine, stores, extraction, retrieval).
- **Control Center:** `dashboard.html` (Huginn UI) and `muninn_standalone.py`.
- **Integrations:** `mcp_wrapper.py` (stdio server), `muninn/sdk` (Sync/Async).
- **Evaluation:** `eval/` harness with MemoryAgentBench-style competency tracks.
- **Plans:** `docs/plans/` and `SOTA_PLUS_PLAN.md`.

---

## 🚦 Getting Started & Verification

**Installation:** `pip install -e .[all]`

**Standard Verification Suite:**
1.  **Lint:** `ruff check .`
2.  **Test:** `pytest` (ensure `500+ passed`)
3.  **Benchmark:** `python -m eval.ollama_local_benchmark sota-verdict`
4.  **Hygiene:** `python -m eval.phase_hygiene`

**Documentation Reference:**
- `docs/ARCHITECTURE.md`: Technical deep-dive.
- `SOTA_PLUS_PLAN.md`: Active implementation tranches and progress.
- `docs/AGENT_CONTINUATION_RUNBOOK.md`: Instructions for agent handoffs.