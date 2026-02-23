"""
Mimir Interop Relay — Claude Code Adapter
==========================================
Adapter for the `claude` CLI (Claude Code / claude-code).

CLI invocation pattern:
  claude -p <prompt> --output-format json [--max-turns 1] [--allowedTools none]

Output format (JSON):
  {
    "type": "result",
    "subtype": "success",
    "result": "<text>",
    "session_id": "...",
    "usage": {"input_tokens": N, "output_tokens": N},
    "cost_usd": 0.0
  }

On error the CLI may return a JSON object with "type": "error" or exit non-zero
with a plain-text error on stderr.

Environment variables required:
  ANTHROPIC_API_KEY — API key for the Anthropic models
  (alternatively, Claude Code's auth cookie may be used if already logged in)
"""

from __future__ import annotations

import json
import logging
import os
import shlex

from ..models import IRPEnvelope, IRPMode, IRPNetworkPolicy, ProviderName, ProviderResult
from .base import BaseAdapter

logger = logging.getLogger("Muninn.Mimir.adapters.claude_code")

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ClaudeCodeAdapter(BaseAdapter):
    """
    Executes relay calls via the `claude` CLI (Claude Code).

    The adapter maps IRP/1 envelope fields to Claude Code CLI flags:
      - instruction   → `-p <text>`
      - mode A/B      → `--output-format json --max-turns 1`
      - policy.tools  → `--allowedTools none` (forbidden) or default
      - policy.network→ no-network flag is not natively supported; advisory only
    """

    provider = ProviderName.CLAUDE_CODE
    _binary_name = "claude"

    # IRP mode → max-turns hint
    _MODE_MAX_TURNS: dict[IRPMode, int] = {
        IRPMode.ADVISORY: 1,
        IRPMode.STRUCTURED: 1,
        IRPMode.RECONCILE: 1,
    }

    def env_vars(self) -> list[str]:
        return ["ANTHROPIC_API_KEY"]

    def _build_command(self, envelope: IRPEnvelope) -> list[str]:
        """
        Build the `claude` CLI command from an IRP envelope.

        Full prompt is constructed as:
          [system context from envelope.context]
          [input values]
          [instruction]
        """
        prompt = _build_prompt_text(envelope)
        max_turns = self._MODE_MAX_TURNS.get(envelope.mode, 1)

        cmd: list[str] = [
            self._resolved_binary_path,
            "-p", prompt,
            "--output-format", "json",
            "--max-turns", str(max_turns),
            "--permission-mode", "dontAsk",
        ]

        # Tool restrictions
        if envelope.policy.tools == "forbidden":
            cmd += ["--allowedTools", ""]
        elif envelope.policy.tools == "readonly":
            # read-only subset: list files, read files, grep, search
            cmd += ["--allowedTools", "Bash(read),Read,Glob,Grep"]

        # Response format hint (mode B → ask for JSON)
        if envelope.mode == IRPMode.STRUCTURED:
            cmd += ["--system-prompt",
                    "Respond only with valid JSON. No markdown fences."]

        logger.debug(
            "claude command: claude -p <prompt[%d chars]> --max-turns %d",
            len(prompt),
            max_turns,
        )
        return cmd

    async def call(self, envelope: IRPEnvelope) -> ProviderResult:
        """
        Execute a relay call to Claude Code via the programmatic Agent SDK.

        This bypasses the Node-based CLI's terminal rendering issues on Windows.
        """
        import time
        from claude_agent_sdk import query, ClaudeAgentOptions

        t_start = time.monotonic()
        prompt = _build_prompt_text(envelope)
        max_turns = self._MODE_MAX_TURNS.get(envelope.mode, 1)

        # Map tools
        allowed = []
        if envelope.policy.tools == "forbidden":
            allowed = []
        elif envelope.policy.tools == "readonly":
            allowed = ["Bash(read)", "Read", "Glob", "Grep"]
        else:
            allowed = ["*"]

        options = ClaudeAgentOptions(
            max_turns=max_turns,
            allowed_tools=allowed,
            permission_mode="dontAsk",
            output_format="json",
        )

        # Handle Mode B (Structured) system prompt
        if envelope.mode == IRPMode.STRUCTURED:
            options.system_prompt = "Respond only with valid JSON. No markdown fences."

        raw_text = ""
        input_tokens = 0
        output_tokens = 0
        parsed = {}

        try:
            logger.debug("claude-sdk: sending query (max_turns=%d)", max_turns)
            # Stateless query returns an async generator of Message objects
            try:
                async for message in query(prompt=prompt, options=options):
                    # Handle message content (TextBlocks)
                    if hasattr(message, "content") and message.content:
                        for block in message.content:
                            if hasattr(block, "text"):
                                raw_text += block.text

                    # Handle usage/stats events
                    if hasattr(message, "type") and message.type == "usage_event":
                        if hasattr(message, "usage"):
                            input_tokens += getattr(message.usage, "input_tokens", 0)
                            output_tokens += getattr(message.usage, "output_tokens", 0)
            except Exception as sdk_iter_exc:
                # The SDK might raise MessageParseError for unknown types (e.g. rate_limit_event)
                # We log this as a warning and proceed with whatever text we captured.
                logger.warning("claude-sdk: internal iteration error (partial text len=%d): %s", 
                               len(raw_text), sdk_iter_exc)

            # Rehydrate 'parsed' for Mimir downstream
            if raw_text:
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    parsed = {"result": raw_text}
            else:
                parsed = {}

        except Exception as exc:
            latency_ms = int((time.monotonic() - t_start) * 1000)
            logger.exception("claude-sdk: unexpected error: %s", exc)
            return ProviderResult(
                provider=self.provider,
                raw_output="",
                error=f"SDK Error: {exc}",
                latency_ms=latency_ms,
                available=True,
            )

        latency_ms = int((time.monotonic() - t_start) * 1000)
        return ProviderResult(
            provider=self.provider,
            raw_output=raw_text,
            parsed=parsed,
            latency_ms=latency_ms,
            input_tokens=input_tokens,  # Token extraction from SDK messages can be added
            output_tokens=output_tokens,
            available=True,
        )

    def _parse_output(self, raw: str, returncode: int) -> dict:
        """
        Parse Claude Code JSON output.

        The CLI emits a single JSON object on stdout when --output-format json
        is used. Handles both success and error objects.
        """
        if not raw:
            return {}

        # Strip any leading/trailing whitespace and BOM
        raw = raw.strip().lstrip("\ufeff")

        # Claude Code sometimes emits NDJSON (multiple JSON lines);
        # take the last non-empty line as the result envelope.
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        target = lines[-1] if lines else raw

        try:
            obj = json.loads(target)
        except json.JSONDecodeError:
            logger.debug("claude: JSON parse failed, returning raw text")
            return {"raw_text": raw}

        return obj

    def _extract_text(self, parsed: dict, raw: str) -> str:
        """Extract the primary text from a parsed Claude output dict."""
        if not parsed:
            return raw

        # Success envelope: {"type": "result", "result": "..."}
        if parsed.get("type") == "result" and "result" in parsed:
            return str(parsed["result"])

        # Error envelope: {"type": "error", "error": {...}}
        if parsed.get("type") == "error":
            err = parsed.get("error", {})
            if isinstance(err, dict):
                return err.get("message", raw)
            return str(err)

        # Fallback: look for common text fields
        for key in ("content", "text", "message", "output"):
            if key in parsed:
                return str(parsed[key])

        return raw

    def _extract_tokens(self, parsed: dict) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from parsed output."""
        usage = parsed.get("usage", {}) if parsed else {}
        if isinstance(usage, dict):
            return (
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )
        return 0, 0

    async def _do_availability_check(self) -> bool:
        """
        Claude Code is available if:
        1. `claude` binary is on PATH
        2. The binary responds to --version
        """
        import shutil

        # Binary check (mandatory)
        has_binary = shutil.which(self._binary_name) is not None
        if not has_binary:
            logger.debug("%s: binary '%s' not found on PATH", self.provider.value, self._binary_name)
            return False

        # Version check (mandatory to verify execution)
        available = await self._check_binary_version()

        # Auth status logging (advisory)
        if available:
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            if not has_key:
                logger.debug(
                    "%s: binary present but ANTHROPIC_API_KEY not set — "
                    "relying on local CLI session/cookies.",
                    self.provider.value
                )
        return available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prompt_text(envelope: IRPEnvelope) -> str:
    """
    Assemble a final prompt string from an IRP envelope.

    Format:
      [Context key=value pairs]
      [Named inputs]
      ---
      <instruction>
    """
    parts: list[str] = []

    # Embed structured context
    if envelope.context:
        ctx_lines = []
        for k, v in envelope.context.items():
            ctx_lines.append(f"{k}: {v}")
        if ctx_lines:
            parts.append("## Context\n" + "\n".join(ctx_lines))

    # Named inputs
    if envelope.request.inputs:
        inp_lines = []
        for inp in envelope.request.inputs:
            inp_lines.append(f"[{inp.name}]\n{inp.value}")
        parts.append("## Inputs\n" + "\n\n".join(inp_lines))

    # Instruction
    parts.append(envelope.request.instruction)

    return "\n\n".join(parts)
