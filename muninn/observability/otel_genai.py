"""
Optional OpenTelemetry GenAI instrumentation.

Design goals:
- Zero hard dependency: behaves as no-op when opentelemetry is not installed.
- Opt-in by feature flag (`otel_genai`) and env toggles.
- Privacy by default: does not record raw user content unless explicitly enabled.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from muninn.version import __version__


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


class OTelGenAITracer:
    """Thin wrapper over OpenTelemetry trace API with graceful fallback."""

    def __init__(self, *, enabled: bool = False):
        self.enabled = enabled
        self.capture_content = _env_bool("MUNINN_OTEL_CAPTURE_CONTENT", "0")
        self.capture_content_max_chars = max(
            0, min(_env_int("MUNINN_OTEL_CAPTURE_CONTENT_MAX_CHARS", 1000), 16000)
        )
        self._trace = None
        self._tracer = None

        if not enabled:
            return

        try:
            from opentelemetry import trace  # type: ignore

            self._trace = trace
            self._tracer = trace.get_tracer(
                instrumenting_module_name="muninn.observability.otel_genai",
                instrumenting_library_version=__version__,
            )
        except Exception:
            self._trace = None
            self._tracer = None

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._tracer is not None)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """Create span if instrumentation is available, else no-op."""
        if not self.active:
            yield
            return

        with self._tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if value is None:
                    continue
                span.set_attribute(key, value)
            yield

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to current span when active."""
        if not self.active:
            return
        current_span = self._trace.get_current_span()
        if not current_span:
            return
        current_span.add_event(name, attributes=attributes or {})

    def maybe_content(self, text: Optional[str]) -> Optional[str]:
        """Conditionally include content when capture toggle is enabled."""
        if not self.capture_content:
            return None
        if text is None:
            return None
        if self.capture_content_max_chars <= 0:
            return None
        return text[: self.capture_content_max_chars]
