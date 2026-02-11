"""
Muninn History Ingestion Script
-------------------------------
Ingests conversation history from all supported assistants into the Muninn global memory.
Supports: Claude Code, Codex, Antigravity/Gemini CLI

Usage:
    python ingest_history.py [--dry-run] [--agent codex|claude|antigravity|all]
"""
import json
import requests
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
# Throttle: sleep between adds to avoid overwhelming the server + xLAM sidecar
ADD_DELAY = 0.3  # seconds between /add calls


def add_memory(content: str, agent_id: str, metadata: dict, dry_run: bool = False) -> bool:
    """Add a single memory entry to Muninn. Returns True on success."""
    if dry_run:
        print(f"  [DRY-RUN] Would add {len(content)} chars from {agent_id}")
        return True
    try:
        resp = requests.post(f"{BASE_URL}/add", json={
            "content": content,
            "user_id": "global_user",
            "agent_id": agent_id,
            "metadata": metadata
        }, timeout=30)
        if resp.status_code == 200:
            time.sleep(ADD_DELAY)
            return True
        else:
            print(f"  [WARN] Server returned {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print(f"  [WARN] Timeout adding memory (content too large?)")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def ingest_codex(dry_run: bool = False):
    """Ingest Codex CLI history from ~/.codex/history.jsonl"""
    print("\n=== Ingesting Codex History ===")
    path = Path("C:/Users/wjohn/.codex/history.jsonl")
    if not path.exists():
        print("  Codex history not found. Skipping.")
        return 0

    count = 0
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if len(text) < 30:
                    skipped += 1
                    continue

                # Cap at 8000 chars to keep extraction feasible
                text = text[:8000]
                meta = {
                    "source": "codex_history",
                    "session_id": data.get("session_id", "unknown"),
                    "ts": data.get("ts")
                }
                if add_memory(text, "codex", meta, dry_run):
                    count += 1
                if count % 25 == 0 and count > 0:
                    print(f"  Ingested {count} codex entries...")
            except json.JSONDecodeError:
                skipped += 1
            except Exception as e:
                print(f"  Error: {e}")
                skipped += 1

    print(f"  Done. Ingested {count} items, skipped {skipped}.")
    return count


def ingest_claude(dry_run: bool = False):
    """Ingest Claude Code conversation history from ~/.claude/projects/*/"""
    print("\n=== Ingesting Claude Code History ===")
    projects_dir = Path("C:/Users/wjohn/.claude/projects")
    if not projects_dir.exists():
        print("  Claude projects directory not found. Skipping.")
        return 0

    count = 0
    skipped = 0
    jsonl_files = list(projects_dir.glob("**/*.jsonl"))
    print(f"  Found {len(jsonl_files)} conversation files.")

    for jsonl_path in jsonl_files:
        # Skip agent-prefixed files (task subagent transcripts, usually redundant)
        if jsonl_path.stem.startswith("agent-"):
            continue

        session_id = jsonl_path.stem
        project_name = jsonl_path.parent.name

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    msg_type = data.get("type", "")
                    message = data.get("message", {})
                    if not isinstance(message, dict):
                        continue

                    role = message.get("role", "")
                    content = message.get("content", "")

                    # Extract text from user messages (direct string content)
                    if msg_type == "user" and role == "user" and isinstance(content, str):
                        text = content.strip()
                        if len(text) < 30:
                            skipped += 1
                            continue
                        text = text[:8000]
                        meta = {
                            "source": "claude_history",
                            "session_id": session_id,
                            "project": project_name,
                            "role": "user"
                        }
                        if add_memory(text, "claude", meta, dry_run):
                            count += 1

                    # Extract text from user messages (content blocks)
                    elif msg_type == "user" and role == "user" and isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if len(text) < 30:
                                    continue
                                text = text[:8000]
                                meta = {
                                    "source": "claude_history",
                                    "session_id": session_id,
                                    "project": project_name,
                                    "role": "user"
                                }
                                if add_memory(text, "claude", meta, dry_run):
                                    count += 1

                    # Extract substantive assistant text responses (skip tool calls)
                    elif role == "assistant" and isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "").strip()
                                # Only ingest substantial assistant responses
                                if len(text) < 100:
                                    continue
                                text = text[:8000]
                                meta = {
                                    "source": "claude_history",
                                    "session_id": session_id,
                                    "project": project_name,
                                    "role": "assistant"
                                }
                                if add_memory(text, "claude", meta, dry_run):
                                    count += 1

                    if count % 25 == 0 and count > 0 and count % 50 == 0:
                        print(f"  Ingested {count} claude entries...")

        except Exception as e:
            print(f"  Error reading {jsonl_path}: {e}")

    print(f"  Done. Ingested {count} items, skipped {skipped}.")
    return count


def ingest_antigravity(dry_run: bool = False):
    """Ingest Antigravity/Gemini brain output files."""
    print("\n=== Ingesting Antigravity Brain ===")
    root = Path("C:/Users/wjohn/.gemini/antigravity/brain")
    if not root.exists():
        print("  Antigravity brain not found. Skipping.")
        return 0

    count = 0
    skipped = 0
    files = list(root.glob("**/output.txt"))
    print(f"  Found {len(files)} output files.")

    for f_path in files:
        try:
            size = f_path.stat().st_size
            if size < 100 or size > 50000:
                skipped += 1
                continue

            content = f_path.read_text(encoding="utf-8")

            # Extract session ID from path (UUID-style directory name)
            session_id = "unknown"
            for p in f_path.parts:
                if len(p) == 36 and p.count("-") == 4:
                    session_id = p
                    break

            meta = {
                "source": "antigravity_brain",
                "file": str(f_path),
                "session_id": session_id
            }
            text = content[:8000]
            if add_memory(text, "antigravity", meta, dry_run):
                count += 1
            if count % 10 == 0 and count > 0:
                print(f"  Ingested {count} antigravity files...")
        except Exception as e:
            print(f"  Error ingesting {f_path}: {e}")
            skipped += 1

    print(f"  Done. Ingested {count} files, skipped {skipped}.")
    return count


def main():
    dry_run = "--dry-run" in sys.argv
    agent_filter = "all"
    if "--agent" in sys.argv:
        idx = sys.argv.index("--agent")
        if idx + 1 < len(sys.argv):
            agent_filter = sys.argv[idx + 1].lower()

    if dry_run:
        print("[DRY RUN MODE - no data will be written]")

    # Server health check
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        health = resp.json()
        print(f"Muninn server: {health.get('status')} | graph_nodes: {health.get('graph_nodes')}")
    except Exception:
        print("ERROR: Muninn server must be running at http://localhost:8000")
        sys.exit(1)

    total = 0
    if agent_filter in ("all", "codex"):
        total += ingest_codex(dry_run)
    if agent_filter in ("all", "claude"):
        total += ingest_claude(dry_run)
    if agent_filter in ("all", "antigravity"):
        total += ingest_antigravity(dry_run)

    print(f"\n=== Ingestion Complete ===")
    print(f"Total memories added: {total}")
    print("GraphRAG will process these in the background via Mem0's native pipeline.")


if __name__ == "__main__":
    main()
