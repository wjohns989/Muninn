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

def initialize_security(configured_token: Optional[str] = None) -> str:
    """Initialize or generate the global auth token."""
    global _GLOBAL_AUTH_TOKEN
    
    # Check priority: 1. Passed arg, 2. Env Var, 3. Generation
    env_token = os.environ.get("MUNINN_AUTH_TOKEN")
    
    if configured_token:
        _GLOBAL_AUTH_TOKEN = configured_token
        logger.info("Security initialized with configured token.")
    elif env_token:
        _GLOBAL_AUTH_TOKEN = env_token
        logger.info("Security initialized with MUNINN_AUTH_TOKEN from environment.")
    else:
        _GLOBAL_AUTH_TOKEN = secrets.token_urlsafe(32)
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
    """Verify if the provided token matches the global token."""
    if token is None:
        return False
    expected = get_token()
    return secrets.compare_digest(token, expected)

def is_security_enabled() -> bool:
    """Check if security should be enforced (always true in v3.7.0+)."""
    return True
