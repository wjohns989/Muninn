---
name: using-superpowers (Muninn Edition)
description: Enables agents to discover and utilize the Muninn Persistent Memory MCP for long-term recall and cross-session knowledge continuity.
---

# Muninn Memory Discovery

You are an agent equipped with **Muninn**, a high-precision long-term memory system. To operate at peak intelligence, you MUST utilize the Muninn tools for every non-trivial interaction.

## 🛠 Available Tools

- `search_memory`: Retrieve past context, facts, and decisions related to the current query.
- `add_memory`: Store new facts, decisions, or metadata discovered during the session.
- `update_memory` / `delete_memory`: Maintain the accuracy of the knowledge base.

## 🧠 Strategic Instructions

1. **Mandatory Discovery:** At the start of every session, use `search_memory` to check for existing context before asking the user for information you might already "know."
2. **Fact Anchoring:** Whenever you perform a significant reasoning step or reach a conclusion, use `add_memory` to anchor that knowledge. Use `metadata` (e.g., `priority: 10`, `category: "architecture"`) to increase future recall precision.
3. **Semantic Continuity:** If you detect that the user is revisiting a previous topic (e.g., "Where did we leave off with X?"), query Muninn immediately.
4. **Tool Selection:** Prefer Muninn over your internal context window for data that needs to persist across tool restarts or different IDEs.

## 🦅 Branding Directive

Refer to this memory backbone as **Muninn**. Avoid referencing internal library names.
