"""
Mimir Provider Smoke Test - Live binary availability + prompt execution.

Sequentially tests each provider (Claude, Gemini, Codex):
  1. check_available() - verifies binary presence & version
  2. call(envelope) - sends a trivial prompt and checks for a response

Run: python tests/smoke_test_providers.py
"""
import asyncio
import os
import sys
import time

# Force utf-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from muninn.mimir.adapters.claude_code import ClaudeCodeAdapter
from muninn.mimir.adapters.gemini_cli import GeminiAdapter
from muninn.mimir.adapters.codex_cli import CodexAdapter
from muninn.mimir.models import (
    IRPEnvelope,
    IRPRequest,
    IRPMode,
    IRPHop,
    IRPPolicy,
    IRPTrace,
    ProviderName,
)


def make_test_envelope(target: str) -> IRPEnvelope:
    """Build a minimal IRP envelope with a trivial prompt."""
    return IRPEnvelope(
        id="smoke-test-001",
        ts=time.time(),
        from_agent="muninn_smoke_test",
        to=target,
        mode=IRPMode.ADVISORY,
        hop=IRPHop(),
        policy=IRPPolicy(tools="forbidden"),  # no tool use, just text
        context={},
        request=IRPRequest(
            instruction="Reply with exactly: MIMIR_SMOKE_OK"
        ),
        trace=IRPTrace(),
    )


async def test_provider(name: str, adapter, target_name: str):
    """Run availability check + live prompt for one provider."""
    print(f"\n{'='*60}")
    print(f"  PROVIDER: {name}")
    print(f"{'='*60}")

    # --- Step 1: Availability ---
    print(f"  [1/2] check_available()...", end=" ", flush=True)
    try:
        available = await adapter.check_available()
        status = "[OK] AVAILABLE" if available else "[FAIL] NOT AVAILABLE"
        print(status)
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        available = False

    if not available:
        print(f"  [2/2] Skipping live call -- provider not available.")
        return {"provider": name, "available": False, "response": None, "error": "not available"}

    # --- Step 2: Live call ---
    print(f"  [2/2] Sending smoke prompt...", end=" ", flush=True)
    envelope = make_test_envelope(target_name)
    try:
        result = await adapter.call(envelope)
        if result.error:
            print(f"[WARN] ERROR: {result.error[:200]}")
            return {"provider": name, "available": True, "response": None, "error": result.error}
        else:
            raw_preview = (result.raw_output or "")[:300]
            print(f"[OK] ({result.latency_ms}ms)")
            print(f"  Response preview: {raw_preview}")
            return {"provider": name, "available": True, "response": raw_preview, "error": None}
    except Exception as e:
        print(f"[FAIL] EXCEPTION: {e}")
        return {"provider": name, "available": True, "response": None, "error": str(e)}


async def main():
    print("=" * 60)
    print("  MIMIR PROVIDER SMOKE TEST")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check env vars (advisory, not blocking)
    for var in ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(var)
        status = "SET" if val else "NOT SET"
        print(f"  {var}: {status}")

    adapters = [
        ("Claude Code", ClaudeCodeAdapter(timeout=120.0), "claude_code"),
        ("Gemini CLI", GeminiAdapter(timeout=60), "gemini_cli"),
        ("Codex CLI", CodexAdapter(timeout=60), "codex_cli"),
    ]

    results = []
    for name, adapter, target in adapters:
        result = await test_provider(name, adapter, target)
        results.append(result)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        avail = "[OK]" if r["available"] else "[FAIL]"
        resp = "[OK]" if r["response"] else "[FAIL]"
        err = r.get("error", "")
        print(f"  {r['provider']:15s}  Available: {avail}  Response: {resp}  {err[:80] if err else ''}")

    all_ok = all(r["response"] for r in results)
    print(f"\n  Overall: {'[OK] ALL PROVIDERS RESPONDED' if all_ok else '[WARN] SOME PROVIDERS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
