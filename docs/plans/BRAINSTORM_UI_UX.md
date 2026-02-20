# Muninn Huginn UI/UX Brainstorm & Refactor Plan

## 1. Research Synthesis: Design for AI & Novices

### Core Principles
*   **Transparency & Trust**: AI systems can be opaque. The UI must visualize "thinking" states and provide clear feedback on *what* data is being processed.
*   **Progressive Disclosure**: Novices get simple, high-level controls (e.g., "Add Project", "Search"). Experts get the raw JSON editors and log streams via a "Advanced Mode" toggle.
*   **Visualizing the Invisible**: Memory is abstract. We need to make it concrete:
    *   **Network Graph**: Show memories as nodes and links.
    *   **Timeline**: Show ingestion events over time.
    *   **Stats Cards**: Big numbers for "Total Memories", "Total Connections".
*   **Guidance, Not Just Controls**: Replace raw form fields with "Wizards" or guided inputs.
    *   *Bad*: Textarea for "Ingest Paths".
    *   *Good*: "Add Source" button -> File Picker/Path Input -> Validation -> Confirmation.

### Branding & Aesthetic (Dark Mode "CodeOps")
*   **Palette**: Deep, rich backgrounds (Navy/Slate) instead of pure black.
    *   *Background*: `#0f172a` (Slate 900)
    *   *Panel*: `#1e293b` (Slate 800)
    *   *Border*: `#334155` (Slate 700)
    *   *Text*: `#f1f5f9` (Slate 100)
    *   *Muted*: `#94a3b8` (Slate 400)
    *   *Accent*: `#38bdf8` (Sky 400) - energetic, tech-forward.
    *   *Success*: `#4ade80` (Green 400)
    *   *Warning*: `#facc15` (Yellow 400)
    *   *Error*: `#f87171` (Red 400)
*   **Typography**: `Inter` or `Segoe UI`. Clean, sans-serif, high readability. Monospace for logs/IDs.
*   **Shape Language**: Rounded corners (12px-16px). Subtle borders (1px solid). Soft, diffuse shadows for depth.

---

## 2. Refactoring Plan: `dashboard.html`

### A. Layout Restructuring
**Current**: Dense grid of 6+ panels.
**Proposed**:
1.  **Header**: Simplified. Logo/Title on left. Status indicators (Health, Auth) as subtle badges on right.
2.  **Hero/Stats Row**: 3-4 key metrics (Memories, Projects, Uptime/Status) displayed prominently.
3.  **Main Content Area (Tabs/Sections)**:
    *   **Overview**: Search bar (central), recent activity log.
    *   **Ingestion**: A focused "Add Data" section. Split "Project" vs "Legacy" into clear choices.
    *   **Management**: Preferences, User Profile, Federation (Advanced).
4.  **Footer**: Version info, links to docs.

### B. Feature Enhancements for Novices
1.  **"Magic" Search Bar**: Centralized search that feels like a conversation starter.
    *   *Placeholder*: "Ask your memory..." instead of "Query".
2.  **Visual Status**: pulsating green dot for "Online".
3.  **Human-Readable Logs**:
    *   *Current*: `[2026-02-19 10:00:00] Ingestion started...`
    *   *Proposed*: Styled timeline entries. Icons for event types (Download, Parse, Save).

### C. Technical Implementation (Single File)
To maintain the "standalone" nature of `dashboard.html`, we will keep it as a single file but use:
*   **CSS Variables**: For easy theming.
*   **Modern CSS**: Flexbox/Grid, backdrop-filter for "glass" effects.
*   **Vanilla JS**: No build step required.
*   **SVG Icons**: Inline SVGs for visual polish without external assets.

## 3. Action Items
1.  **Refactor HTML Structure**: Semantic tags (`header`, `main`, `section`, `footer`).
2.  **Update CSS**: Apply the "Slate/Sky" palette and rounded card aesthetic.
3.  **Enhance JS**: Add "Tab" switching logic to declutter the view.
4.  **Add Visuals**: Inline SVG icons for the sidebar/tabs.

---
*Created by Loki-Mode (Opus) - Phase: Design*
