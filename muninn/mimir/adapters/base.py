"""
Mimir Interop Relay — Base Provider Adapter
============================================
Abstract base class for all Mimir provider adapters.

Each adapter wraps a single CLI tool (claude, codex, gemini) and provides:
  - async call(envelope) → ProviderResult
  - async check_available() → bool
  - env_vars() → list[str]   (required env-var names)

Adapters are responsible for:
  1. Building the subprocess command from an IRPEnvelope
  2. Executing the subprocess with timeout
  3. Parsing raw output into a ProviderResult
  4. Capturing latency and token counts where available
  5. Handling subprocess errors gracefully (never raising)

Platform note:
  On Windows, all subprocess calls use CREATE_NO_WINDOW to suppress console
  pop-ups, and STARTUPINFO to detach from the parent terminal.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..models import IRPEnvelope, ProviderName, ProviderResult, RunStatus

logger = logging.getLogger("Muninn.Mimir.adapters.base")

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"


def _get_subprocess_kwargs() -> dict:
    """
    Return platform-specific kwargs for asyncio subprocess creation.
    On Windows, suppresses the console window that would otherwise flash.
    """
    if _IS_WINDOWS:
        import subprocess

        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        return {
            "creationflags": subprocess.CREATE_NO_WINDOW,
            "startupinfo": si,
        }
    return {}


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """
    Abstract base class for Mimir provider adapters.

    Subclasses implement `_build_command`, `_parse_output`, and `env_vars`.
    The `call` and `check_available` methods provide generic orchestration.
    """

    #: Provider identity — set by each subclass
    provider: ProviderName

    #: Default subprocess timeout in seconds; may be overridden per-call
    DEFAULT_TIMEOUT: int = 120

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._available_cache: Optional[bool] = None
        self._available_checked_at: float = 0.0
        self._availability_ttl: float = 60.0  # seconds before re-checking

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_command(self, envelope: IRPEnvelope) -> list[str]:
        """
        Build the subprocess argv list for the given IRP envelope.

        Must not launch any process — pure transformation only.
        """

    @abstractmethod
    def _parse_output(self, raw: str, returncode: int) -> dict:
        """
        Parse subprocess raw stdout into a structured dict.

        Returns a dict suitable for `ProviderResult.parsed`.
        Should not raise; return empty dict on parse failure.
        """

    @abstractmethod
    def env_vars(self) -> list[str]:
        """
        Return required environment variable names for this provider.

        Used by the availability checker and the settings UI.
        Example: ["ANTHROPIC_API_KEY"]
        """

    @abstractmethod
    def _extract_text(self, parsed: dict, raw: str) -> str:
        """Extract the primary text response from parsed output."""

    @abstractmethod
    def _extract_tokens(self, parsed: dict) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from parsed output."""

    # ------------------------------------------------------------------
    # Core call method
    # ------------------------------------------------------------------

    async def call(self, envelope: IRPEnvelope) -> ProviderResult:
        """
        Execute a relay call to this provider.

        Returns a ProviderResult regardless of success or failure —
        errors are captured in ProviderResult.error and .available=False
        on hard failures.

        Never raises.
        """
        t_start = time.monotonic()
        command = self._build_command(envelope)

        logger.debug(
            "adapter=%s command=%s timeout=%ds",
            self.provider.value,
            " ".join(command[:4]),  # first 4 tokens only (privacy)
            self.timeout,
        )

        raw_output = ""
        returncode = -1

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_get_subprocess_kwargs(),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                latency_ms = int((time.monotonic() - t_start) * 1000)
                logger.warning(
                    "adapter=%s timed out after %ds",
                    self.provider.value,
                    self.timeout,
                )
                return ProviderResult(
                    provider=self.provider,
                    raw_output="",
                    error=f"Provider timed out after {self.timeout}s",
                    latency_ms=latency_ms,
                    available=True,  # timed out ≠ unavailable
                )

            returncode = proc.returncode
            raw_output = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            if returncode != 0:
                latency_ms = int((time.monotonic() - t_start) * 1000)
                error_detail = stderr_text or f"Exit code {returncode}"
                logger.warning(
                    "adapter=%s exit=%d stderr=%s",
                    self.provider.value,
                    returncode,
                    error_detail[:200],
                )
                return ProviderResult(
                    provider=self.provider,
                    raw_output=raw_output,
                    error=error_detail,
                    latency_ms=latency_ms,
                    available=returncode != 127,  # 127 = command not found
                )

        except FileNotFoundError:
            latency_ms = int((time.monotonic() - t_start) * 1000)
            logger.warning("adapter=%s binary not found", self.provider.value)
            self._available_cache = False
            self._available_checked_at = time.monotonic()
            return ProviderResult(
                provider=self.provider,
                raw_output="",
                error=f"Provider CLI binary not found: {command[0]}",
                latency_ms=latency_ms,
                available=False,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - t_start) * 1000)
            logger.exception("adapter=%s unexpected error: %s", self.provider.value, exc)
            return ProviderResult(
                provider=self.provider,
                raw_output="",
                error=f"Unexpected error: {exc}",
                latency_ms=latency_ms,
                available=True,
            )

        latency_ms = int((time.monotonic() - t_start) * 1000)
        parsed = self._parse_output(raw_output, returncode)
        text = self._extract_text(parsed, raw_output)
        input_tokens, output_tokens = self._extract_tokens(parsed)

        return ProviderResult(
            provider=self.provider,
            raw_output=raw_output,
            parsed=parsed,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            available=True,
        )

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    async def check_available(self) -> bool:
        """
        Return True if this provider's CLI binary is accessible and the
        required environment variables are set.

        Results are cached for `_availability_ttl` seconds to avoid
        hammering the shell on every routing decision.
        """
        now = time.monotonic()
        if (
            self._available_cache is not None
            and (now - self._available_checked_at) < self._availability_ttl
        ):
            return self._available_cache

        available = await self._do_availability_check()
        self._available_cache = available
        self._available_checked_at = now
        return available

    async def _do_availability_check(self) -> bool:
        """
        Override in subclasses for provider-specific availability checks.
        Default: run `<binary> --version` and check exit code.
        """
        import shutil
        import os

        # Check required env vars first (fast)
        for var in self.env_vars():
            if not os.environ.get(var):
                logger.debug(
                    "adapter=%s missing env var %s", self.provider.value, var
                )
                return False

        # Check binary exists
        binary = self._build_command.__func__  # noqa: not ideal but prevents circular
        # Fall through to version check below — binary name extracted in subclass
        return await self._check_binary_version()

    async def _check_binary_version(self) -> bool:
        """
        Run `<binary> --version` as a quick availability probe.
        Subclasses must override `_binary_name` property.
        """
        binary = getattr(self, "_binary_name", None)
        if not binary:
            return True  # skip check if not configured

        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_get_subprocess_kwargs(),
            )
            try:
                await asyncio.wait_for(proc.communicate(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                return False
            return proc.returncode == 0
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def invalidate_availability_cache(self) -> None:
        """Force re-check on next availability query."""
        self._available_cache = None
        self._available_checked_at = 0.0
