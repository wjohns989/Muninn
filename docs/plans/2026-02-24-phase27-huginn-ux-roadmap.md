# Phase 27: Huginn UX — The Cognitive Control Center

**Date:** 2026-02-24
**Status:** DRAFT
**Objective:** Transform Muninn from a developer-focused MCP server into a consumer-grade "External Brain" interface. Focus on interactive knowledge mapping, natural language operations, and high-visibility ROI metrics.

## 1. The Vision: "From Logs to Landscapes"

Non-technical users struggle with "Memory IDs" and "JSON Filters". Huginn replaces these with a **Spatial Knowledge Map**. Users should *see* their knowledge as a landscape of interconnected ideas, not a list of text chunks.

## 2. Core UI Components (The Huginn Suite)

### 2.1 "The Weaver" (Interactive Knowledge Graph)
*   **Tech:** D3.js / Force-Directed Graph.
*   **Feature:** 
    *   **Context Zoom:** High-importance semantic nodes are large; noisy episodic nodes are small/hidden until zoomed.
    *   **Temporal Shadows:** Visualize "shadowed" (outdated) facts in a faded red style to show history.
    *   **Active Scent:** When a search is performed, the "Scent Trail" (path through the graph) glows.

### 2.2 "The Oracle" (Conversational Command Center)
*   **Feature:** A persistent chat-like input that accepts NL commands.
    *   *"Forget everything I said about the old API."* -> Triggers `delete_batch` via `MemorySurgeon`.
    *   *"Summarize our progress on Phase 26."* -> Triggers `hunt` + `synthesis`.
    *   *"Is there a conflict in my meeting notes?"* -> Triggers `GapAnalysis`.

### 2.3 "The Insight Panel" (ROI Visualizations)
*   **Metrics:**
    *   **Cognitive Load Saved:** Counter showing total facts retrieved by agents.
    *   **Distillation Efficiency:** "Last night, 400 logs were compressed into 12 pages of knowledge."
    *   **Knowledge Growth:** Graph of entities discovered over time.

## 3. Advanced ROI Opportunities

| Item | Impact | Complexity | Requirement |
|------|--------|------------|-------------|
| **Knowledge Inheritence** | High | Med | Export/Import handoff bundles for new team members. |
| **Hallucination Prevention** | Critical | High | `OmissionDetector` dashboard alerts when an agent's confidence is low. |
| **Fact-Check Loop** | Med | Low | UI for "Confirming" ad-hoc extractions to boost importance. |
| **Proactive Foraging** | Med | High | "Suggested Research" based on gaps in the Knowledge Graph. |

## 4. Architectural Shift: `Dashboard v3`

We will transition from a single `dashboard.html` to a modular **React-based** architecture.

### New Directory Structure:
```
C:\Users\user\muninn_mcp\ui
├── src/
│   ├── components/ (Graph, Chat, Stats)
│   ├── hooks/ (useMuninn, useAuth)
│   ├── services/ (api.ts, socket.ts)
│   └── App.tsx
├── public/
├── package.json
└── vite.config.ts
```

### Server Changes:
*   `server.py` will serve the built static assets from `ui/dist`.
*   Support for **WebSockets** for real-time "Thought" streams (seeing the agent search in real-time).

## 5. Implementation Roadmap (Phase 27)

### Step 1: UI Scaffolding
- [ ] Initialize Vite/React project in `ui/`.
- [ ] Implement D3 graph foundation.

### Step 2: Real-time Streams
- [ ] Add `ThoughtStream` hook to `MuninnMemory` to broadcast search steps.
- [ ] Implement WebSocket endpoint in `server.py`.

### Step 3: ROI Dashboard
- [ ] Implement "Knowledge Growth" and "Hallucination Saved" widgets.

### Step 4: Refined Search & Forage
- [ ] Unified search bar with "Hunt Mode" and "Forage" toggle.
- [ ] Visual scent-trail rendering.
