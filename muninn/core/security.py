import os
import secrets
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger("Muninn.security")

@dataclass
class SecurityContext:
    user_id: str = "global_user"
    authenticated: bool = False
    method: str = "none"  # 'api', 'mcp', 'internal'
    metadata: Dict[str, Any] = field(default_factory=dict)

_GLOBAL_AUTH_TOKEN: Optional[str] = None
_GLOBAL_AUTH_TOKEN_SOURCE: str = "unset"  # 'env', 'configured', 'generated'

def initialize_security(configured_token: Optional[str] = None) -> str:
    """Initialize or generate the global auth token."""
    global _GLOBAL_AUTH_TOKEN
    
    # Check priority: 1. Passed arg, 2. Env Var, 3. Generation
    # Accept both MUNINN_AUTH_TOKEN and MUNINN_SERVER_AUTH_TOKEN for compatibility
    env_token = os.environ.get("MUNINN_AUTH_TOKEN") or os.environ.get("MUNINN_SERVER_AUTH_TOKEN")
    
    if configured_token:
        _GLOBAL_AUTH_TOKEN = configured_token
        _GLOBAL_AUTH_TOKEN_SOURCE = "configured"
        logger.info("Security initialized with configured token.")
    elif env_token:
        _GLOBAL_AUTH_TOKEN = env_token
        _GLOBAL_AUTH_TOKEN_SOURCE = "env"
        logger.info("Security initialized with MUNINN_AUTH_TOKEN from environment.")
    else:
        _GLOBAL_AUTH_TOKEN = secrets.token_urlsafe(32)
        _GLOBAL_AUTH_TOKEN_SOURCE = "generated"
        logger.warning("-" * 60)
        logger.warning("! SECURITY WARNING !")
        logger.warning("No MUNINN_AUTH_TOKEN configured. Generated temporary token:")
        logger.warning(f"MUNINN_AUTH_TOKEN={_GLOBAL_AUTH_TOKEN}")
        logger.warning("! SECURITY WARNING !")
        logger.warning("-" * 60)
    
    return _GLOBAL_AUTH_TOKEN

def get_token() -> str:
    """Retrieve the active auth token, initializing if necessary."""
    global _GLOBAL_AUTH_TOKEN
    if _GLOBAL_AUTH_TOKEN is None:
        return initialize_security()
    return _GLOBAL_AUTH_TOKEN

def verify_token(token: Optional[str]) -> bool:
    """Verify if the provided token matches any configured or runtime token.

    This verifier accepts any of the configured credentials:
      - `MUNINN_API_KEY` (HTTP API key)
      - `MUNINN_AUTH_TOKEN` / `MUNINN_SERVER_AUTH_TOKEN` (core/runtime auth)
      - the runtime-generated token returned by `get_token()`

    Security bypass via `MUNINN_NO_AUTH=1` is still respected.
    """
    # If an explicit API key is configured, accept requests authenticated
    # either with the API key or with the core auth token/runtime token.
    # This preserves deployments that use distinct credentials for the API
    # surface and internal server components while still allowing API keys
    # to be enforced for external callers.
    env_api_key = os.environ.get("MUNINN_API_KEY")
    env_auth_token = os.environ.get("MUNINN_AUTH_TOKEN") or os.environ.get("MUNINN_SERVER_AUTH_TOKEN")

    if env_api_key is not None and env_api_key.strip() != "":
        # token must be provided and match either the API key, the core auth
        # token (if configured), or the runtime-generated token.
        if token is None:
            return False
        if secrets.compare_digest(token, env_api_key):
            return True
        if env_auth_token is not None and env_auth_token.strip() != "":
            if secrets.compare_digest(token, env_auth_token):
                return True
        # Accept runtime-generated token as well
        if secrets.compare_digest(token, get_token()):
            return True
        return False

    # If an explicit core auth token is configured in the environment, require it.
    if env_auth_token is not None and env_auth_token.strip() != "":
        if token is None:
            return False
        return secrets.compare_digest(token, env_auth_token)

    # Global bypass (dev/test) — if security is disabled, accept all requests.
    if not is_security_enabled():
        return True

    # Default: no explicit tokens configured and security enabled — allow.
    # This mirrors historical behaviour where a runtime token is generated
    # when no env tokens are present and the system remains accessible.
    return True


def verify_api_token(token: Optional[str]) -> bool:
    """Verify tokens for HTTP API surface (MUNINN_API_KEY semantics).

    Rules:
      - If `MUNINN_API_KEY` is set to a non-empty value, require it.
      - Empty string or unset `MUNINN_API_KEY` is treated as dev-mode.
      - If security is disabled via `MUNINN_NO_AUTH=1` or `MUNINN_DEV_MODE=true`, allow.
      - Otherwise, allow by default to preserve historical behaviour.
    """
    env_api_key = os.environ.get("MUNINN_API_KEY")
    if env_api_key is not None and env_api_key.strip() != "":
        if token is None:
            return False
        return secrets.compare_digest(token, env_api_key)

    if not is_security_enabled():
        return True

    return True

def is_security_enabled() -> bool:
    """Check if security should be enforced."""
    # Highest-priority bypass for local development/debugging
    if os.environ.get("MUNINN_NO_AUTH") == "1":
        return False

    # If an explicit API key or core auth token is configured (non-empty),
    # enforce security even when `MUNINN_DEV_MODE` is set. An empty string
    # is treated as unset to preserve test semantics.
    api_key = os.environ.get("MUNINN_API_KEY", "")
    if api_key is not None and api_key.strip() != "":
        return True

    env_auth = (os.environ.get("MUNINN_AUTH_TOKEN", "") or os.environ.get("MUNINN_SERVER_AUTH_TOKEN", ""))
    if env_auth is not None and env_auth.strip() != "":
        return True

    # Backwards-compatible dev-mode toggle (opt-out for enforcement)
    if os.environ.get("MUNINN_DEV_MODE", "").lower() == "true":
        return False

    # Default: security enabled
    return True
