"""Probe the installed Windows MCP bridge against the one live local server.

Runs the real stdio entry point outside the checkout, lists tools, and makes a
read-only project-context call, optionally followed by two review-queue pages.
Never prints tokens, returned context, memory text, or continuation capabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def _failure_code(exc: Exception) -> str:
    """Static diagnostic labels only; never echo an untrusted error payload."""
    codes = {'review_call_failed', 'review_response_invalid', 'review_continuation_invalid',
             'bridge_closed', 'mcp_error', 'protocol_mismatch', 'context_tool_missing',
             'context_call_failed', 'review_tool_missing', 'bridge_timeout'}
    return str(exc) if str(exc) in codes else type(exc).__name__


def _review_page(result: dict, *, limit: int = 1) -> dict:
    """Validate the actual wire response without reporting private payloads."""
    if result.get("isError") is True:
        raise RuntimeError("review_call_failed")
    page = result.get("structuredContent")
    if page is None:
        content = result.get("content")
        if (not isinstance(content, list) or len(content) != 1
                or content[0].get("type") != "text"):
            raise RuntimeError("review_response_invalid")
        try:
            page = json.loads(content[0]["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("review_response_invalid") from exc
    if (not isinstance(page, dict) or page.get("limit") != limit
            or page.get("credential_or_withheld_excluded") is not True
            or not isinstance(page.get("matches"), list)
            or len(page["matches"]) > limit
            or type(page.get("has_more")) is not bool
            or type(page.get("snapshot_events")) is not int
            or (page["has_more"] and (not isinstance(page.get("next_cursor"), str)
                                      or not page["next_cursor"] or not page["matches"]))
            or (not page["has_more"] and page.get("next_cursor") is not None)):
        raise RuntimeError("review_response_invalid")
    ids = set()
    for item in page["matches"]:
        if (not isinstance(item, dict)
                or not isinstance(item.get("id"), str) or not item["id"]
                or item["id"] in ids
                or not isinstance(item.get("source_ref"), str) or not item["source_ref"]
                or item.get("state") not in {"provisional", "needs_user"}
                or any(not isinstance(item.get(key), str) or not item[key]
                       or len(item[key]) > 2048 for key in ("text", "quote"))):
            raise RuntimeError("review_response_invalid")
        ids.add(item["id"])
    return page


def _review_progress(previous: dict, current: dict) -> None:
    if (current["snapshot_events"] != previous["snapshot_events"]
            or current["next_cursor"] == previous["next_cursor"]
            or {item["id"] for item in previous["matches"]}
               & {item["id"] for item in current["matches"]}):
        raise RuntimeError("review_continuation_invalid")


def _installed_profiles_match() -> bool:
    """Attest four configured clients without printing config data."""
    try:
        try:
            import tomllib as toml_reader
        except ModuleNotFoundError:
            import tomli as toml_reader
        codex = toml_reader.loads(
            (Path.home() / ".codex" / "config.toml").read_text(encoding="utf-8")
        )["mcp_servers"]["muninn"]
        gemini = json.loads(
            (Path.home() / ".gemini" / "settings.json").read_text(encoding="utf-8")
        )["mcpServers"]["muninn"]
        claude = json.loads(
            (Path.home() / ".claude.json").read_text(encoding="utf-8")
        )["mcpServers"]["muninn"]
        desktop = json.loads(
            (Path.home() / "AppData" / "Roaming" / "Claude" /
             "claude_desktop_config.json").read_text(encoding="utf-8")
        )["mcpServers"]["muninn"]
        executable = Path(sys.executable).resolve(strict=True)
        for profile, allowed in (
            (codex, {"command", "args", "env", "startup_timeout_sec", "tool_timeout_sec"}),
            (gemini, {"command", "args", "env"}),
            (claude, {"type", "command", "args", "env"}),
            (desktop, {"command", "args", "env"}),
        ):
            if (not isinstance(profile, dict) or set(profile) - allowed
                    or (profile is claude and profile.get("type") != "stdio")
                    or profile.get("args") != ["-E", "-P", "-m", "muninn_mcp_bridge"]
                    or profile.get("env") != ({"MUNINN_MCP_TOOLSET": "core",
                                               "MUNINN_AGENT_NAME": "claude-desktop"}
                                              if profile is desktop else
                                              {"MUNINN_MCP_TOOLSET": "core"})
                    or not isinstance(profile.get("command"), str)
                    or Path(profile["command"]).resolve(strict=True) != executable):
                return False
    except (ImportError, OSError, KeyError, TypeError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-installed-profiles", action="store_true",
                        help="Require Codex, Claude Code/Desktop and Gemini configs to use this bridge")
    parser.add_argument("--verify-review-queue", action="store_true",
                        help="Read up to two noncredential ambiguity pages through the installed bridge")
    args = parser.parse_args()
    if os.name != "nt":
        print(json.dumps({"state": "unsupported_host"}))
        return 2
    if args.verify_installed_profiles and not _installed_profiles_match():
        print(json.dumps({"state": "failed", "reason": "installed_profile_mismatch"}))
        return 1
    env = os.environ.copy()
    env.pop("MUNINN_AUTH_TOKEN", None)
    env.pop("MUNINN_SERVER_URL", None)
    if args.verify_installed_profiles:
        # All verified profiles explicitly request core; reproduce their
        # environment rather than probing the unconfigured CLI's default set.
        env["MUNINN_MCP_TOOLSET"] = "core"
    started = time.monotonic()
    responses: queue.Queue[dict | None] = queue.Queue()
    proc = subprocess.Popen(
        [sys.executable, "-E", "-P", "-m", "muninn_mcp_bridge"],
        cwd=Path.home(), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
    )

    def receive() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                item = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(item, dict):
                responses.put(item)
        responses.put(None)

    reader = threading.Thread(target=receive, daemon=True)
    reader.start()

    def send(message: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
        proc.stdin.flush()

    def result_for(request_id: int) -> dict:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            item = responses.get(timeout=max(0.1, deadline - time.monotonic()))
            if item is None:
                raise RuntimeError("bridge_closed")
            if item.get("id") == request_id:
                if "error" in item or not isinstance(item.get("result"), dict):
                    raise RuntimeError("mcp_error")
                return item["result"]
        raise TimeoutError("bridge_timeout")

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "muninn-local-smoke", "version": "1.0"},
        }})
        initialized = result_for(1)
        if initialized.get("protocolVersion") != "2025-11-25":
            raise RuntimeError("protocol_mismatch")
        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = result_for(2)
        names = {item.get("name") for item in listed.get("tools", []) if isinstance(item, dict)}
        if "get_project_context" not in names:
            raise RuntimeError("context_tool_missing")
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "get_project_context",
            "arguments": {"project": "Muninn", "recent_limit": 1},
        }})
        context = result_for(3)
        if context.get("isError") is True or not isinstance(context.get("content"), list):
            raise RuntimeError("context_call_failed")
        review_proof = {}
        if args.verify_review_queue:
            if "search_cited_memories" not in names:
                raise RuntimeError("review_tool_missing")
            pages = []
            cursor = None
            for request_id in (4, 5):
                arguments = {"review_only": True, "limit": 1}
                if cursor is not None:
                    arguments["cursor"] = cursor
                send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {
                    "name": "search_cited_memories", "arguments": arguments,
                }})
                page = _review_page(result_for(request_id))
                if pages:
                    _review_progress(pages[-1], page)
                pages.append(page)
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            review_proof = {"review_call_ok": True, "review_pages": len(pages),
                            "review_items": sum(len(page["matches"]) for page in pages),
                            "review_continuation_ok": len(pages) == 2,
                            "review_snapshot_events": pages[0]["snapshot_events"]}
        print(json.dumps({"state": "passed", "tool_count": len(names),
                          "context_call_ok": True,
                          "installed_profiles_match": (True if args.verify_installed_profiles else None),
                          "elapsed_ms": round((time.monotonic() - started) * 1000),
                          **review_proof}, sort_keys=True))
        return 0
    except (BrokenPipeError, OSError, RuntimeError, TimeoutError, queue.Empty) as exc:
        print(json.dumps({"state": "failed", "reason": _failure_code(exc)}, sort_keys=True))
        return 1
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
