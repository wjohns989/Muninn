from __future__ import annotations

"""
Phase 21: Zero-Trust Parser Isolation & Ingestion Safety (SOTA+)

Testing the subprocess sandbox for binary document parsing (Phase 17).
Verifies that:
1. Malformed binary files do not crash the parent process.
2. The timeout parameter reliably kills hanging parsers (zip bombs / font bombs).
3. Invalid types or missing files are handled gracefully.
"""

import sys
import tempfile
import types
from pathlib import Path

import pytest

from muninn.ingestion.sandbox import sandboxed_parse_binary, MAX_STDOUT_BYTES


def test_sandboxed_parse_missing_file():
    """Missing files raise explicit RuntimeError before subprocess launch."""
    with pytest.raises(RuntimeError, match="File not found"):
        sandboxed_parse_binary(Path("/does/not/exist.pdf"), "pdf")


def test_sandboxed_parse_invalid_type():
    """Only supported types bypass the guard."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tf:
        path = Path(tf.name)
        with pytest.raises(ValueError, match="only supports 'pdf' and 'docx'"):
            sandboxed_parse_binary(path, "txt")


def test_sandboxed_parse_rejects_large_file():
    """Files larger than max_bytes are rejected before subprocess launch."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(b"x" * 128)
        path = Path(tf.name)

    try:
        with pytest.raises(RuntimeError, match="rejected file larger than max_bytes"):
            sandboxed_parse_binary(path, "pdf", max_bytes=64)
    finally:
        path.unlink(missing_ok=True)


def test_sandboxed_parse_invalid_max_bytes():
    """Non-positive max_bytes is rejected."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
        path = Path(tf.name)
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            sandboxed_parse_binary(path, "pdf", max_bytes=0)


def test_sandboxed_parse_malformed_docx():
    """Malformed DOCX should return an error from the subprocess and not crash the parent."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
        tf.write(b"this is not a valid zip archive or docx")
        path = Path(tf.name)
    
    try:
        # The python-docx library expects a valid ZIP archive
        with pytest.raises(RuntimeError, match="(Parser subprocess error|failed silently) .*"):
            sandboxed_parse_binary(path, "docx")
    finally:
        path.unlink(missing_ok=True)


def test_sandboxed_parse_malformed_pdf():
    """Malformed PDF should return an error from the subprocess and not crash the parent."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(b"this is definitely not a valid pdf %PDF-1.4 ... wait no it isn't")
        path = Path(tf.name)
    
    try:
        # pypdf expectations
        with pytest.raises(RuntimeError, match="(Parser subprocess error|failed silently) .*"):
            sandboxed_parse_binary(path, "pdf")
    finally:
        path.unlink(missing_ok=True)


def test_sandboxed_timeout_kills_process(monkeypatch):
    """
    Mock the worker module internally to simulate a process that hangs forever (e.g., zip bomb).
    The sandbox timeout parameter must reliably kill it and raise RuntimeError.
    """
    import os
    import textwrap
    
    # We create a fake worker that just sleeps, and point _WORKER_MODULE at it.
    with tempfile.TemporaryDirectory() as td:
        worker_path = Path(td) / "fake_worker.py"
        worker_path.write_text(textwrap.dedent("""
            import time
            import sys
            import json
            
            def main():
                # Hang forever
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                return 0
                
            if __name__ == "__main__":
                sys.exit(main())
        """))
        
        # We need the sandbox module to run `python fake_worker.py` instead of its real module
        # So we patch the subprocess run command.
        import muninn.ingestion.sandbox as sandbox_mod
        
        original_run = sandbox_mod.subprocess.run
        
        def mock_run(cmd, *args, **kwargs):
            # Intercept the module execution and inject our script
            if "-m" in cmd and sandbox_mod._WORKER_MODULE in cmd:
                # Replace ["python", "-m", "muninn.ingestion._parser_subprocess", ...]
                # with    ["python", str(worker_path), ...]
                new_cmd = [sys.executable, str(worker_path)] + cmd[3:]
                return original_run(new_cmd, *args, **kwargs)
            return original_run(cmd, *args, **kwargs)
            
        monkeypatch.setattr(sandbox_mod.subprocess, "run", mock_run)
        
        # Create a dummy valid file to pass the file_exists check
        dummy_file = Path(td) / "dummy.pdf"
        dummy_file.write_bytes(b"dummy")

        # Call with a short timeout.
        with pytest.raises(RuntimeError, match="subprocess timed out after 1s"):
            sandboxed_parse_binary(dummy_file, "pdf", timeout=1.0)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX-only rlimit behaviour")
def test_fractional_limits_are_rejected():
    """Fractional CPU/memory limits must be rejected rather than truncated."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
        path = Path(tf.name)
        with pytest.raises(ValueError, match="max_memory_mb must be a positive integer"):
            sandboxed_parse_binary(path, "pdf", max_memory_mb=0.5)
        with pytest.raises(ValueError, match="max_cpu_seconds must be a positive integer"):
            sandboxed_parse_binary(path, "pdf", max_cpu_seconds=0.5)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX-only rlimit behaviour")
def test_apply_resource_limits_failure_raises(monkeypatch):
    """setrlimit failures surface as RuntimeError so the subprocess exits non-zero."""
    import muninn.ingestion._parser_subprocess as ps

    # Prepare fake resource module that always fails
    def failing_setrlimit(kind, limits):
        raise OSError("nope")

    fake_resource = types.SimpleNamespace(
        RLIMIT_AS=1,
        RLIMIT_CPU=2,
        setrlimit=failing_setrlimit,
    )

    monkeypatch.setenv("MUNINN_PARSER_MAX_MEMORY_MB", "64")
    monkeypatch.setenv("MUNINN_PARSER_MAX_CPU_SECONDS", "2")
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    with pytest.raises(RuntimeError, match="Failed to set parser"):
        ps._apply_resource_limits()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX-only rlimit behaviour")
def test_apply_resource_limits_sets_expected_limits(monkeypatch):
    """Happy path applies both limits using RLIMIT_AS and RLIMIT_CPU."""
    import muninn.ingestion._parser_subprocess as ps

    calls: list[tuple[int, tuple[int, int]]] = []

    def recording_setrlimit(kind, limits):
        calls.append((kind, limits))

    fake_resource = types.SimpleNamespace(
        RLIMIT_AS=11,
        RLIMIT_CPU=22,
        setrlimit=recording_setrlimit,
    )

    monkeypatch.setenv("MUNINN_PARSER_MAX_MEMORY_MB", "4")
    monkeypatch.setenv("MUNINN_PARSER_MAX_CPU_SECONDS", "3")
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    ps._apply_resource_limits()

    assert calls == [
        (11, (4 * 1024 * 1024, 4 * 1024 * 1024)),
        (22, (3, 3)),
    ]