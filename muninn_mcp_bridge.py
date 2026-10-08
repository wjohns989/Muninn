"""Windows stdio MCP entry point for clients with stale process environments.

The shared Muninn service keeps its token in the Windows User environment.
Desktop clients can outlive that environment change, so their MCP subprocess
reads the current user value at launch. No token is written to client config,
stdout, arguments, or disk by this bridge.
"""

import importlib
import os
import sys

_LOCAL_SERVER = "http://127.0.0.1:42069"
_NO_START_FLAGS = (
    "MUNINN_MCP_AUTO_START",
    "MUNINN_MCP_AUTOSTART_ON_LAUNCH",
    "MUNINN_MCP_AUTOSTART_SERVER",
    "MUNINN_MCP_AUTOSTART_OLLAMA",
)


class BridgeError(RuntimeError):
    """The requested MCP connection cannot be made safely."""


def _read_windows_user_token() -> str | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, kind = winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")
        return value if kind == winreg.REG_SZ and isinstance(value, str) else None
    except FileNotFoundError:
        return None
    except (ImportError, OSError) as exc:
        raise BridgeError("Windows user token is unavailable") from exc


def prepare_environment() -> None:
    """Pin an authenticated, proxy-free loopback connection before MCP imports."""
    configured = os.environ.get("MUNINN_SERVER_URL", "")
    if configured and configured != _LOCAL_SERVER:
        raise BridgeError("Muninn MCP bridge requires its shared local endpoint")
    token = _read_windows_user_token()
    if (not isinstance(token, str) or len(token) < 32
            or any(not 33 <= ord(char) <= 126 for char in token)):
        raise BridgeError("A valid Windows user Muninn token is required")

    os.environ["MUNINN_AUTH_TOKEN"] = token
    os.environ["MUNINN_SERVER_URL"] = _LOCAL_SERVER
    os.environ["MUNINN_NO_AUTH"] = "0"
    for name in _NO_START_FLAGS:
        os.environ[name] = "0"
    # requests must never route a local bearer through an inherited proxy.
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


def main() -> int:
    try:
        prepare_environment()
    except (BridgeError, OSError, ValueError):
        print("Muninn MCP bridge: authenticated local connection unavailable", file=sys.stderr)
        return 2
    wrapper = importlib.import_module("mcp_wrapper")
    wrapper.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
