"""
Mimir Interop Relay — Codex CLI Adapter
========================================
Adapter for the OpenAI `codex` CLI (codex-cli).

CLI invocation pattern:
  codex exec "<prompt>" [--json] [--sandbox <level>] [--quiet]

Output format:
  Codex CLI outputs JSONL (newline-delimited JSON) when --json is set.
  The final summary line has the shape:
    {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "..."}]}

  Or a completion summary:
    {"type": "completion", "usage": {"input_tokens": N, "output_tokens": N, "total_tokens": N}}

  Additionally there is a stats line emitted to stderr (not stdout).

Sandbox levels:
  read-only    → tools.readonly policy
  workspace-only → default / tools.allowed policy
  none         → forbidden (network blocked, no tools)

Environment variables required:
  OPENAI_API_KEY — API key for OpenAI models used by Codex CLI
"""

from __future__ import annotations

import json
import logging
import os
import shutil

from ..models import IRPEnvelope, IRPMode, ProviderName
from .base import BaseAdapter

logger = logging.getLogger("Muninn.Mimir.adapters.codex_cli")

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CodexAdapter(BaseAdapter):
    """
    Executes relay calls via the `codex` CLI.

    Codex CLI emits JSONL on stdout; we collect all lines, find the
    assistant message, and the usage/completion summary.
    """

    provider = ProviderName.CODEX_CLI
    _binary_name = "codex"

    # IRP tool policy → codex sandbox level (v0.80.0)
    _SANDBOX_MAP: dict[str, str] = {
        "forbidden": "read-only",
        "readonly": "read-only",
        "allowed": "workspace-write",
    }

    def env_vars(self) -> list[str]:
        return ["OPENAI_API_KEY"]

    def _build_command(self, envelope: IRPEnvelope) -> list[str]:
        """Build the `codex exec` command from an IRP envelope."""
        prompt = _build_prompt_text(envelope)
        sandbox = self._SANDBOX_MAP.get(envelope.policy.tools, "read-only")

        cmd: list[str] = [
            self._resolved_binary_path,
            "exec",
            prompt,
            "--json",
            "--sandbox", sandbox,
        ]

        logger.debug(
            "codex command: codex exec <prompt[%d chars]> --sandbox %s",
            len(prompt),
            sandbox,
        )
        return cmd

    def _parse_output(self, raw: str, returncode: int) -> dict:
        """
        Parse JSONL output from codex exec.

        Extracts the last assistant message content and any usage stats.
        """
        if not raw:
            return {}

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        parsed: dict = {
            "messages": [],
            "usage": {},
            "stats": {},
        }

        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            if msg_type == "message" and obj.get("role") == "assistant":
                content = obj.get("content", [])
                texts = []
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                if texts:
                    parsed["messages"].append("\n".join(texts))

            elif msg_type == "completion":
                parsed["usage"] = obj.get("usage", {})

            elif msg_type == "stats":
                parsed["stats"] = obj

        return parsed

    def _extract_text(self, parsed: dict, raw: str) -> str:
        """Extract the last assistant message from parsed Codex output."""
        if not parsed:
            return raw
        messages = parsed.get("messages", [])
        if messages:
            return messages[-1]
        return raw

    def _extract_tokens(self, parsed: dict) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from parsed Codex output."""
        if not parsed:
            return 0, 0
        usage = parsed.get("usage", {})
        if isinstance(usage, dict):
            return (
                int(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
                int(usage.get("output_tokens", usage.get("completion_tokens", 0))),
            )
        return 0, 0

    async def _do_availability_check(self) -> bool:
        """Codex is available if binary is on PATH."""
        import shutil
        has_binary = shutil.which(self._binary_name) is not None

        if not has_binary:
            logger.debug("%s: binary '%s' not found on PATH", self.provider.value, self._binary_name)
            return False

        available = await self._check_binary_version()
        if available:
            has_key = bool(os.environ.get("OPENAI_API_KEY"))
            if not has_key:
                logger.debug(
                    "%s: binary present but OPENAI_API_KEY not set — "
                    "relying on local CLI session.",
                    self.provider.value
                )
        return available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt_text(envelope: IRPEnvelope) -> str:
    """
    Assemble a final prompt string from an IRP envelope for Codex.

    Format mirrors the Claude adapter for consistency.
    """
    parts: list[str] = []

    if envelope.context:
        ctx_lines = [f"{k}: {v}" for k, v in envelope.context.items()]
        if ctx_lines:
            parts.append("## Context\n" + "\n".join(ctx_lines))

    if envelope.request.inputs:
        inp_lines = [f"[{inp.name}]\n{inp.value}" for inp in envelope.request.inputs]
        parts.append("## Inputs\n" + "\n\n".join(inp_lines))

    parts.append(envelope.request.instruction)
    return "\n\n".join(parts)
