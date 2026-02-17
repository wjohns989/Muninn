import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("Muninn.mcp.metrics")

class McpMetrics:
    """
    Tracks performance and payload metrics for a single MCP tool call.
    """
    def __init__(self, msg_id: Any, name: str):
        self.msg_id = msg_id
        self.name = name
        self.response_count = 0
        self.response_bytes_total = 0
        self.response_bytes_max = 0
        self.saw_error = False
        self.started_monotonic = time.monotonic()

    def record_response(self, message: Dict[str, Any], serialized: str) -> None:
        """Record a single JSON-RPC response message."""
        if self.msg_id != message.get("id"):
            return
        
        # +1 for the newline delimiter usually added by the transport
        payload_size_bytes = len(serialized.encode("utf-8")) + 1
        
        self.response_count += 1
        self.response_bytes_total += payload_size_bytes
        self.response_bytes_max = max(self.response_bytes_max, payload_size_bytes)
        
        if isinstance(message.get("error"), dict):
            self.saw_error = True

    def get_outcome(self) -> str:
        """Determine result outcome based on seen messages."""
        if self.saw_error:
            return "error"
        if self.response_count > 0:
            return "success"
        return "no_response"

    def log_telemetry(
        self,
        initial_budget_ms: Optional[float],
        remaining_budget_ms: Optional[float],
        warn_threshold_ms: float
    ) -> None:
        """Log normalized telemetry for the tool call."""
        elapsed_ms = max(0.0, (time.monotonic() - self.started_monotonic) * 1000.0)
        outcome = self.get_outcome()
        
        budget_str = "n/a" if initial_budget_ms is None else f"{initial_budget_ms:.1f}"
        remaining_str = "n/a" if remaining_budget_ms is None else f"{remaining_budget_ms:.1f}"
        
        log_method = logger.warning if elapsed_ms >= warn_threshold_ms else logger.info
        log_method(
            "Tool call telemetry: name=%s id=%r outcome=%s elapsed_ms=%.1f responses=%d "
            "response_bytes_total=%d response_bytes_max=%d budget_ms=%s remaining_budget_ms=%s",
            self.name,
            self.msg_id,
            outcome,
            elapsed_ms,
            self.response_count,
            self.response_bytes_total,
            self.response_bytes_max,
            budget_str,
            remaining_str,
        )
