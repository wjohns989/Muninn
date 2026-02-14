"""
Muninn SDK exceptions.
"""

from __future__ import annotations

from typing import Any, Optional


class MuninnError(RuntimeError):
    """Base class for SDK errors."""


class MuninnConnectionError(MuninnError):
    """Raised when the SDK cannot reach the Muninn server."""


class MuninnAPIError(MuninnError):
    """Raised when the server returns an HTTP or API-level error."""

    def __init__(
        self,
        detail: str,
        *,
        status_code: Optional[int] = None,
        path: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        self.status_code = status_code
        self.path = path
        self.payload = payload
        status_hint = f" (status={status_code})" if status_code is not None else ""
        path_hint = f" [{path}]" if path else ""
        super().__init__(f"{detail}{status_hint}{path_hint}")
