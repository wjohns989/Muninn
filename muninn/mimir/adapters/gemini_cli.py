"""
Mimir Interop Relay — Gemini CLI Adapter
==========================================
Adapter for the Google `gemini` CLI (gemini-cli).

CLI invocation pattern:
  gemini -p <prompt> [--output-format json] [--no-tools] [--yolo]

Output format (JSON with --output-format json):
  {
    "response": "<text>",
    "stats": {
      "promptTokenCount": N,
      "candidatesTokenCount": N,
      "totalTokenCount": N,
      "tools": {
        "totalCalls": N,
        "successful": N,
        "failed": N
      }
    }
  }

Note: When the policy forbids tool usage, we pass `--no-tools` and
additionally validate the output stats.tools.totalCalls == 0 in the
PolicyEngine (check_no_tools_result).

Environment variables required:
  GEMINI_API_KEY  — API key for Gemini models
  (or GOOGLE_API_KEY as fallback — Gemini CLI accepts either)
"""

from __future__ import annotations

import json
import logging
import os
import shutil

from ..models import IRPEnvelope, IRPMode, ProviderName
from .base import BaseAdapter

logger = logging.getLogger("Muninn.Mimir.adapters.gemini_cli")

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GeminiAdapter(BaseAdapter):
    """
    Executes relay calls via the `gemini` CLI.

    Tool usage validation is the responsibility of PolicyEngine
    (check_no_tools_result), but this adapter also passes --no-tools
    when the policy forbids it to prevent tool calls at source.
    """

    provider = ProviderName.GEMINI_CLI
    _binary_name = "gemini"

    def env_vars(self) -> list[str]:
        return ["GEMINI_API_KEY"]

    def _build_command(self, envelope: IRPEnvelope) -> list[str]:
        """Build the `gemini` CLI command from an IRP envelope."""
        prompt = _build_prompt_text(envelope)

        cmd: list[str] = [
            self._resolved_binary_path,
            "-p", prompt,
            "-o", "json",
        ]

        # Approval mode: --yolo and --approval-mode are mutually exclusive.
        # Use --approval-mode yolo for Mimir. Read-only 'plan' mode is
        # currently experimental in Gemini CLI and requires extra config.
        # For now, we use restricted allowed-tools for forbidden/readonly.
        if envelope.policy.tools in ("forbidden", "readonly"):
            cmd += ["--approval-mode", "yolo", "--allowed-tools", "none"]
        else:
            cmd += ["--approval-mode", "yolo"]

        # Mode B: request structured JSON
        if envelope.mode == IRPMode.STRUCTURED:
            cmd += [
                "--system-instruction",
                "Respond only with valid JSON. Do not include markdown fences.",
            ]

        # Network policy (advisory — Gemini CLI doesn't have a network flag)
        if (
            envelope.policy.network.value == "deny_all"
            and envelope.policy.tools == "allowed"
        ):
            logger.warning(
                "gemini: policy.network=deny_all but policy.tools=allowed — "
                "Gemini tools may still make network calls. "
                "Consider setting policy.tools=forbidden for strict isolation."
            )

        logger.debug(
            "gemini command: gemini -p <prompt[%d chars]>",
            len(prompt),
        )
        return cmd

    def _parse_output(self, raw: str, returncode: int) -> dict:
        """
        Parse Gemini JSON output.

        Gemini emits a single JSON object with `response` and `stats` keys.
        Falls back gracefully if output is not valid JSON.
        """
        if not raw:
            return {}

        raw = raw.strip().lstrip("\ufeff")

        # Gemini may emit multiple lines; take the last JSON object
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for candidate in reversed(lines):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

        # Last resort: try full raw
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            logger.debug("gemini: JSON parse failed, returning raw text")
            return {"raw_text": raw}

    def _extract_text(self, parsed: dict, raw: str) -> str:
        """Extract the primary response text from Gemini output."""
        if not parsed:
            return raw

        # Standard Gemini JSON output: {"response": "..."}
        if "response" in parsed:
            return str(parsed["response"])

        # Fallback fields
        for key in ("text", "content", "output", "message", "result"):
            if key in parsed:
                return str(parsed[key])

        return raw

    def _extract_tokens(self, parsed: dict) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from Gemini stats."""
        if not parsed:
            return 0, 0

        stats = parsed.get("stats", {})
        if isinstance(stats, dict):
            return (
                int(stats.get("promptTokenCount", 0)),
                int(stats.get("candidatesTokenCount", 0)),
            )
        return 0, 0

    async def _do_availability_check(self) -> bool:
        """Gemini is available if binary is on PATH."""
        import shutil
        has_binary = shutil.which(self._binary_name) is not None

        if not has_binary:
            logger.debug("%s: binary '%s' not found on PATH", self.provider.value, self._binary_name)
            return False

        available = await self._check_binary_version()
        if available:
            has_gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
            has_google_key = bool(os.environ.get("GOOGLE_API_KEY"))
            if not (has_gemini_key or has_google_key):
                logger.debug(
                    "%s: binary present but no API key set — "
                    "relying on local CLI session.",
                    self.provider.value
                )
        return available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt_text(envelope: IRPEnvelope) -> str:
    """
    Assemble a final prompt string from an IRP envelope for Gemini.

    Format mirrors the other adapters for consistency.
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
