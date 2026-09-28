"""Forward an agent hook event to the Muninn server. Standard library only, for fast startup.

Codex (and any host with command hooks) runs:  python /path/to/muninn/hook_client.py codex
It reads the hook JSON from stdin, posts it to MUNINN_SERVER_URL/hooks/<agent>
(with MUNINN_AUTH_TOKEN when set), and prints the server's answer, e.g. the
session-start briefing. It always exits 0 so a stopped server never blocks the agent.
"""

import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_SERVER = "http://127.0.0.1:42069"
# Codex gives SessionEnd hooks 1 second; everything else may wait for the briefing.
FAST_EVENTS = {"SessionEnd", "Interrupt", "Stop", "AfterAgent"}


def _auth_token() -> str:
    token = os.environ.get("MUNINN_AUTH_TOKEN", "").strip()
    if token or sys.platform != "win32":
        return token
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")
            return value.strip() if isinstance(value, str) else ""
    except (OSError, ValueError):
        return ""


def main(argv) -> int:
    agent = argv[1] if len(argv) > 1 else "codex"
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    try:
        event = json.loads(raw).get("hook_event_name", "") if raw.strip() else ""
    except ValueError:
        return 0
    server = argv[2] if len(argv) > 2 else os.environ.get("MUNINN_SERVER_URL", DEFAULT_SERVER)
    url = f"{server.rstrip('/')}/hooks/{agent}"
    headers = {"Content-Type": "application/json"}
    token = _auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=raw.encode("utf-8") or b"{}", headers=headers, method="POST")
    try:
        # System proxy settings can intercept even 127.0.0.1 on Windows. A
        # loopback hook must reach the local server directly.
        host = urllib.parse.urlsplit(url).hostname
        timeout = 0.8 if event in FAST_EVENTS else 8
        if host in {"127.0.0.1", "localhost", "::1"} and urllib.parse.urlsplit(url).scheme == "http":
            parts = urllib.parse.urlsplit(url)
            conn = http.client.HTTPConnection(host, parts.port or 80, timeout=timeout)
            try:
                target = parts.path + (f"?{parts.query}" if parts.query else "")
                conn.request("POST", target, body=raw.encode("utf-8") or b"{}", headers=headers)
                response = conn.getresponse()
                if response.status >= 400:
                    raise OSError(f"HTTP {response.status}")
                body = response.read().decode("utf-8", errors="replace").strip()
            finally:
                conn.close()
        else:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
        print(f"muninn hook: server not reachable ({exc})", file=sys.stderr)
        return 0
    if body and body != "{}":
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
