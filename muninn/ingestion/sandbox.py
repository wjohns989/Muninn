"""
Parser security sandbox — subprocess isolation for binary file parsers (Phase 17).

Provides `sandboxed_parse_binary()`, a drop-in replacement for in-process PDF/DOCX
parsing that runs the parser in a separate OS process to contain exploits.

Security threat model addressed:
  1. Parser library exploits (CVEs in pypdf / python-docx)
     → Exploit code runs in child process, cannot affect parent's memory space
  2. Memory/decompression bombs (zip bombs inside DOCX, font bombs in PDF)
     → `timeout` enforces a hard wall-clock limit; subprocess is terminated
  3. Path traversal via embedded URIs or malformed filenames
     → Child receives only the resolved absolute path; no other env secrets
  4. Output flood attacks (pathologically large extracted text)
     → Child enforces MAX_OUTPUT_CHARS before serialization; parent additionally
        caps stdout read at MAX_STDOUT_BYTES

Design principles:
  - Cross-platform: works on Windows (no fork), Linux, macOS via subprocess.run
  - No network activity in child process
  - JSON protocol over stdout — structured, safe, easy to validate
  - Graceful fallback: if the subprocess mechanism fails AND the caller opts in,
    a direct in-process parse is available (for environments without subprocess support)

Usage:
    from muninn.ingestion.sandbox import sandboxed_parse_binary

    text = sandboxed_parse_binary(Path("document.pdf"), "pdf")
    text = sandboxed_parse_binary(Path("report.docx"), "docx", timeout=60.0)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Maximum bytes to accept from the subprocess stdout to prevent output flooding.
# This is a secondary defense — the subprocess itself also caps at MAX_OUTPUT_CHARS.
MAX_STDOUT_BYTES = 4 * 1024 * 1024  # 4 MB

# Module path for the subprocess worker
_WORKER_MODULE = "muninn.ingestion._parser_subprocess"

# Allowlist of environment variable names that the parser subprocess may receive.
# This prevents secrets inherited from the parent process (auth tokens, API keys,
# database connection strings, etc.) from being accessible to a compromised child.
_SANDBOX_ENV_ALLOWLIST: frozenset[str] = frozenset({
    # Execution path — needed to locate Python itself and shared libraries
    "PATH",
    # Windows system variables — required for Win32 subsystem, DLL loading
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "USERPROFILE",
    "TEMP", "TMP",  # temp dir (Python may write .pyc here)
    # POSIX/macOS execution environment
    "HOME", "USER", "LOGNAME",
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",  # shared library paths
    # Python runtime — include only if explicitly set by the caller to customise
    # the interpreter (e.g., virtual-env activation or custom PYTHONPATH).
    "PYTHONPATH", "PYTHONHOME", "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV",  # some packages inspect this to find stdlib paths
})


def _make_sandbox_env() -> Dict[str, str]:
    """Build a minimal environment dict for the parser subprocess.

    Only variables in _SANDBOX_ENV_ALLOWLIST are forwarded.  All other
    variables — including MUNINN_AUTH_TOKEN, OPENAI_API_KEY, database URLs,
    and any other application secrets present in the parent process — are
    stripped, so a compromised parser library cannot exfiltrate them.
    """
    return {k: v for k, v in os.environ.items() if k in _SANDBOX_ENV_ALLOWLIST}


def sandboxed_parse_binary(
    path: Path,
    source_type: str,
    *,
    timeout: float = 30.0,
    max_bytes: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    max_cpu_seconds: Optional[int] = None,
    fallback_in_process: bool = False,
) -> str:
    """
    Parse a binary file (PDF or DOCX) in an isolated subprocess.

    The subprocess runs `python -m muninn.ingestion._parser_subprocess <type> <path>`
    and communicates results via stdout JSON. The parent process trusts only the
    JSON output; any exception in the child is captured and re-raised here as
    RuntimeError, keeping the parent process safe.

    Args:
        path: Absolute (or resolvable) path to the file to parse.
        source_type: "pdf" or "docx".
        timeout: Maximum seconds to wait for the subprocess (default 30s).
                 On timeout the subprocess is killed and RuntimeError is raised.
        max_bytes: Optional maximum file size in bytes. If set and the file size
                   exceeds this value, parsing is rejected before subprocess launch.
        max_memory_mb: Optional maximum memory (MB) for the parser subprocess.
                       Enforced via POSIX rlimit if available.
        max_cpu_seconds: Optional CPU time limit (seconds) for the parser subprocess.
                         Enforced via POSIX rlimit if available.
        fallback_in_process: If True and subprocess fails due to infrastructure
                             reasons (not parser errors), attempt in-process parse.
                             Default False — safer to raise than silently bypass.

    Returns:
        Extracted text content, potentially truncated to MAX_OUTPUT_CHARS.

    Raises:
        RuntimeError: Parser error, timeout, unexpected subprocess failure, or
                      invalid JSON response.
        ValueError: source_type is not "pdf" or "docx".
    """
    if source_type not in {"pdf", "docx"}:
        raise ValueError(
            f"sandboxed_parse_binary only supports 'pdf' and 'docx', got: {source_type!r}"
        )

    def _coerce_positive_int(name: str, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer, got: {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got: {value}")
        return value

    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise RuntimeError(f"File not found: {resolved_path}")

    if max_bytes is not None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got: {max_bytes}")
        file_size = resolved_path.stat().st_size
        if file_size > max_bytes:
            raise RuntimeError(
                f"Binary parser rejected file larger than max_bytes "
                f"({file_size} > {max_bytes} bytes) for {resolved_path.name}"
            )

    if (max_memory_mb is not None or max_cpu_seconds is not None) and os.name != "posix":
        raise RuntimeError(
            "Parser resource limits (max_memory_mb/max_cpu_seconds) "
            "require POSIX rlimits and are unsupported on this platform."
        )

    max_memory_mb = _coerce_positive_int("max_memory_mb", max_memory_mb)
    max_cpu_seconds = _coerce_positive_int("max_cpu_seconds", max_cpu_seconds)

    cmd = [
        sys.executable,
        "-m",
        _WORKER_MODULE,
        source_type,
        str(resolved_path),
    ]

    env = _make_sandbox_env()
    if max_memory_mb is not None:
        env["MUNINN_PARSER_MAX_MEMORY_MB"] = str(max_memory_mb)
    if max_cpu_seconds is not None:
        env["MUNINN_PARSER_MAX_CPU_SECONDS"] = str(max_cpu_seconds)

    try:
        kwargs = {
            "args": cmd,
            "capture_output": True,
            "timeout": timeout,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            
        result = subprocess.run(**kwargs)
    except subprocess.TimeoutExpired as exc:
        # Kill is implicit after TimeoutExpired when using capture_output
        raise RuntimeError(
            f"Binary parser subprocess timed out after {timeout:.0f}s "
            f"(file: {resolved_path.name}, type: {source_type})"
        ) from exc
    except FileNotFoundError as exc:
        # Python executable not found — unusual but possible in constrained envs
        if fallback_in_process:
            return _in_process_fallback(resolved_path, source_type)
        raise RuntimeError(
            f"Python executable not found for subprocess launch: {sys.executable}"
        ) from exc

    # Decode stdout safely — errors='replace' prevents UnicodeDecodeError on
    # unexpected binary output from a misbehaving parser.
    raw_stdout = result.stdout[:MAX_STDOUT_BYTES].decode("utf-8", errors="replace").strip()

    if result.returncode not in (0, 1):
        # Exit code 2 = usage error in our worker; anything else is unexpected
        stderr_preview = result.stderr[:200].decode("utf-8", errors="replace").strip()
        if fallback_in_process and result.returncode != 1:
            return _in_process_fallback(resolved_path, source_type)
        raise RuntimeError(
            f"Parser subprocess exited with unexpected code {result.returncode} "
            f"for {source_type} file. stderr: {stderr_preview!r}"
        )

    # Parse JSON response
    if not raw_stdout:
        if result.returncode == 1:
            raise RuntimeError(
                f"Parser subprocess failed silently for {source_type} file "
                f"'{resolved_path.name}' (no JSON output, exit code 1)"
            )
        # Exit 0 with no output: treat as empty document
        return ""

    try:
        payload = json.loads(raw_stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Parser subprocess returned invalid JSON for {source_type} file "
            f"'{resolved_path.name}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Parser subprocess returned unexpected JSON type "
            f"({type(payload).__name__}) for {source_type} file"
        )

    # Check for error response
    if "error" in payload:
        raise RuntimeError(
            f"Parser subprocess error for {source_type} '{resolved_path.name}': "
            f"{payload['error']}"
        )

    # Extract text
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise RuntimeError(
            f"Parser subprocess returned non-string 'text' field "
            f"({type(text).__name__}) for {source_type} file"
        )

    return text


def _in_process_fallback(path: Path, source_type: str) -> str:
    """
    Direct in-process parse as a last-resort fallback.

    Only called when fallback_in_process=True and subprocess infrastructure failed
    (not a parser error). This bypasses sandbox isolation — use only in environments
    where subprocess is known to be unavailable (e.g., some serverless contexts).
    """
    if source_type == "pdf":
        try:
            from pypdf import PdfReader  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "PDF parsing requires optional dependency 'pypdf'."
            ) from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())

    if source_type == "docx":
        try:
            from docx import Document  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "DOCX parsing requires optional dependency 'python-docx'."
            ) from exc
        doc = Document(str(path))
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Unsupported source_type for fallback: {source_type!r}")
