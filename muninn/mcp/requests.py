import time
import logging
import requests
from typing import Optional, Dict, Any, Union
from .lifecycle import is_circuit_open, mark_success, mark_failure, BackendCircuitOpenError
from muninn.core.security import get_token

logger = logging.getLogger("Muninn.mcp.requests")

class _BackendCircuitOpenError(Exception): pass
class _RequestDeadlineExceededError(Exception): pass

def _remaining_seconds_from_deadline(deadline_epoch: Optional[float]) -> Optional[float]:
    """
    Compute remaining budget for either epoch-based or monotonic-based deadlines.

    Historical callers have used both clocks. We accept either and choose the
    smallest non-negative remaining budget as the effective constraint.
    """
    if deadline_epoch is None:
        return None

    now_epoch = time.time()
    now_mono = time.monotonic()
    remaining_epoch = deadline_epoch - now_epoch
    remaining_mono = deadline_epoch - now_mono

    non_negative = [r for r in (remaining_epoch, remaining_mono) if r >= 0.0]
    if non_negative:
        return min(non_negative)
    return max(remaining_epoch, remaining_mono)

def get_remaining_deadline_seconds(deadline_epoch: Optional[float]) -> Optional[float]:
    remaining = _remaining_seconds_from_deadline(deadline_epoch)
    if remaining is None:
        return None
    return max(0.0, remaining)

def make_request_with_retry(
    method: str,
    url: str,
    deadline_epoch: Optional[float] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
    **kwargs
) -> requests.Response:
    """Wrapper that checks for facade monkeypatches before executing."""
    try:
        import mcp_wrapper
        if hasattr(mcp_wrapper, "make_request_with_retry"):
            func = mcp_wrapper.make_request_with_retry
            # If it's been patched by something outside this module, use the patch.
            if func.__module__ != __name__:
                return func(method, url, deadline_epoch=deadline_epoch, timeout=timeout, max_retries=max_retries, **kwargs)
    except (ImportError, AttributeError):
        pass
    return _make_request_with_retry_internal(method, url, deadline_epoch, timeout, max_retries, **kwargs)

def _make_request_with_retry_internal(
    method: str,
    url: str,
    deadline_epoch: Optional[float] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
    **kwargs
) -> requests.Response:
    if is_circuit_open():
        raise BackendCircuitOpenError(f"Backend circuit is open, skipping request to {url}")

    remaining = _remaining_seconds_from_deadline(deadline_epoch)
    if remaining is not None and remaining <= 0:
        # Use local exception class
        raise _RequestDeadlineExceededError(f"Deadline exceeded before request to {url}")

    effective_timeout = min(timeout, max(0.001, remaining)) if remaining is not None else timeout
    
    # Inject security token for unified auth (Phase 10)
    headers = kwargs.get("headers", {})
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    kwargs["headers"] = headers

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=effective_timeout, **kwargs)
            resp.raise_for_status()
            mark_success()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                # Re-check deadline
                remaining = _remaining_seconds_from_deadline(deadline_epoch)
                if remaining is not None and remaining <= 0:
                    break
                effective_timeout = min(timeout, max(0.001, remaining)) if remaining is not None else timeout
            continue
        except requests.HTTPError as e:
            # Don't retry 4xx errors
            if 400 <= e.response.status_code < 500:
                mark_success() # It's a valid response, just an error
                return e.response
            last_err = e
            continue

    if last_err:
        mark_failure(last_err)
        raise last_err
    raise RuntimeError("Unexpected end of request loop")
