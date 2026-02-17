import sys
import json
import logging
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, BinaryIO, Callable

from .state import _TRANSPORT_CLOSED, _RPC_WRITE_LOCK, _DISPATCH_EXECUTOR_LOCK

logger = logging.getLogger("Muninn.mcp.server")

class McpServer:
    """
    Handles JSON-RPC communication over stdio with thread-pooled dispatching.
    """
    def __init__(
        self,
        dispatch_fn: Callable[[Dict[str, Any]], None],
        max_workers: Optional[int] = None,
        queue_limit: Optional[int] = None
    ):
        self.dispatch_fn = dispatch_fn
        self.max_workers = max_workers or max(1, int(os.environ.get("MUNINN_MCP_DISPATCH_MAX_WORKERS", "8")))
        self.queue_limit = queue_limit or max(
            self.max_workers,
            int(os.environ.get("MUNINN_MCP_DISPATCH_QUEUE_LIMIT", str(self.max_workers * 8))),
        )
        
        self.transport_closed = _TRANSPORT_CLOSED
        self.write_lock = _RPC_WRITE_LOCK
        
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = _DISPATCH_EXECUTOR_LOCK
        self._queue_semaphore = threading.BoundedSemaphore(self.queue_limit)

    def get_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="muninn-mcp-dispatch",
                )
            return self._executor

    def stop(self):
        """Shut down the dispatcher and close transport."""
        self.transport_closed.set()
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)

    def send_rpc(self, message: Dict[str, Any]) -> None:
        """Serialize and send a JSON-RPC message to stdout."""
        if self.transport_closed.is_set():
            return
        
        try:
            serialized = json.dumps(message)
            with self.write_lock:
                if self.transport_closed.is_set():
                    return
                # Standard stdout write for MCP protocol
                sys.stdout.write(serialized + "\n")
                sys.stdout.flush()
        except (BrokenPipeError, OSError) as exc:
            self.transport_closed.set()
            logger.warning("MCP stdio transport closed while sending: %s", exc)

    def send_error(self, msg_id: Any, code: int, message: str) -> None:
        """Convenience method for sending JSON-RPC errors."""
        self.send_rpc({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
            },
        })

    def read_message(self, stream: BinaryIO) -> Optional[Dict[str, Any]]:
        """
        Read one inbound JSON-RPC message from a binary stream.
        Supports Content-Length framing and newline-delimited JSON.
        """
        while True:
            line = stream.readline()
            if not line:
                return None
            if not line.strip():
                continue

            lowered = line.lower()
            if lowered.startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                    if content_length <= 0:
                        raise ValueError("content length must be positive")
                except Exception:
                    logger.warning("Invalid Content-Length header: %r", line)
                    if not self._consume_framing_headers(stream):
                        return None
                    continue

                if not self._consume_framing_headers(stream):
                    return None

                payload = stream.read(content_length)
                if not payload or len(payload) != content_length:
                    return None
                
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                
                if isinstance(msg, dict):
                    return msg
                continue

            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            
            if isinstance(msg, dict):
                return msg
            continue

    def _consume_framing_headers(self, stream: BinaryIO) -> bool:
        while True:
            header_line = stream.readline()
            if not header_line:
                return False
            if header_line in (b"\r\n", b"\n"):
                return True

    def submit_dispatch(self, msg: Dict[str, Any]) -> bool:
        """Submit a message for background dispatch if a slot is available."""
        if not self._queue_semaphore.acquire(blocking=False):
            return False

        try:
            future = self.get_executor().submit(self._dispatch_guarded, msg)
        except Exception:
            self._queue_semaphore.release()
            raise

        future.add_done_callback(lambda f: self._queue_semaphore.release())
        return True

    def _dispatch_guarded(self, msg: Dict[str, Any]) -> None:
        try:
            self.dispatch_fn(msg)
        except Exception:
            logger.exception("Unexpected error during RPC dispatch")
            msg_id = msg.get("id")
            if msg_id is not None and not self.transport_closed.is_set():
                self.send_error(msg_id, -32603, "Internal error during request dispatch.")