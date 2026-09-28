"""Conservative, bounded agent-facing transcript text sanitizer.

This is defense in depth, not a credential classifier. Original text remains
available only through explicit local archive access; this function never writes
the supplied plaintext to storage or logs.
"""

from __future__ import annotations

import re

from muninn.history.importer import redact as redact_import_text
from muninn.mimir.policy import IRPRedactionPolicy, PolicyEngine

_SENSITIVE_LINE = re.compile(
    r"(?i)(?:\b(?:api[_ -]?key|access[_ -]?token|secret|password|passwd|"
    r"credential|authorization|bearer|private[_ -]?key|cookie)\b)"
)
_OPAQUE_VALUE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_+/=-]{28,}(?![A-Za-z0-9])")
_WINDOWS_HOME = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s\"']+")
_POSIX_HOME = re.compile(r"(?<!\w)/home/[^/\s\"']+")


def sanitize_agent_span(value: str, *, max_chars: int = 4000) -> str:
    """Release a small redacted view; suppress lines that mention credentials."""
    if not isinstance(value, str) or not 1 <= max_chars <= 4000:
        raise ValueError("Invalid bounded history span")
    if len(value) > 12000:
        raise ValueError("History span exceeds sanitizer input bound")
    safe = redact_import_text(value)
    safe, _ = PolicyEngine.redact(safe, IRPRedactionPolicy.STRICT)
    safe = _OPAQUE_VALUE.sub("[REDACTED_OPAQUE_VALUE]", safe)
    safe = _WINDOWS_HOME.sub("[REDACTED_USER_HOME]", safe)
    safe = _POSIX_HOME.sub("[REDACTED_USER_HOME]", safe)
    safe = "\n".join(
        "[REDACTED_SENSITIVE_LINE]" if _SENSITIVE_LINE.search(line) else line
        for line in safe.splitlines()
    )
    return safe[:max_chars]
