"""Forward an agent hook event to the Muninn server. Standard library only, for fast startup.

Codex (and any host with command hooks) runs:  python /path/to/muninn/hook_client.py codex
It reads the hook JSON from stdin, posts it to MUNINN_SERVER_URL/hooks/<agent>
(with MUNINN_AUTH_TOKEN when set), and prints the server's answer, e.g. the
session-start briefing. It always exits 0 so a stopped server never blocks the agent.
"""

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_SERVER = "http://127.0.0.1:42069"
# Codex gives SessionEnd hooks 1 second; everything else may wait for the briefing.
FAST_EVENTS = {"SessionEnd", "Interrupt", "Stop"}


def main(argv) -> int:
    agent = argv[1] if len(argv) > 1 else "codex"
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    try:
        event = json.loads(raw).get("hook_event_name", "") if raw.strip() else ""
    except ValueError:
        return 0
    url = f"{os.environ.get('MUNINN_SERVER_URL', DEFAULT_SERVER).rstrip('/')}/hooks/{agent}"
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("MUNINN_AUTH_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=raw.encode("utf-8") or b"{}", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=0.8 if event in FAST_EVENTS else 8) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"muninn hook: server not reachable ({exc})", file=sys.stderr)
        return 0
    if body and body != "{}":
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
