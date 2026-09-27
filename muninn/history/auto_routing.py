"""Conservative, read-only resource routing for automatic thread enrichment.

The CPU capture and search path never calls this module. Missing GPU telemetry
means no local model is loaded; a separately approved cloud route may be used.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import httpx

_MIB = 1024 * 1024
DEFAULT_MODEL_HINTS = ("qwen35", "qwen2.5:7b")


@dataclass(frozen=True)
class GpuState:
    free_mib: int
    total_mib: int
    utilization_percent: int
    sampled_at: float
    loaded_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class Route:
    provider: str  # ollama | openrouter | deferred
    model: Optional[str]
    reason: str
    free_mib: Optional[int] = None


def probe_gpu(*, now: Optional[float] = None) -> Optional[GpuState]:
    """Use the current NVIDIA reading; never infer headroom from static GPU size."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        # An unknown multi-GPU topology is not permission to select a device.
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        free, total, utilization = (int(value.strip()) for value in lines[0].split(","))
        if not (0 < free <= total and 0 <= utilization <= 100):
            return None
        return GpuState(free, total, utilization, time.time() if now is None else now)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def probe_ollama(base_url: str) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Inspect installed and resident models without invoking inference."""
    base = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as client:
            tags = client.get(f"{base}/api/tags")
            running = client.get(f"{base}/api/ps")
            tags.raise_for_status()
            running.raise_for_status()
        installed = tags.json().get("models") or []
        loaded = tuple(str(item.get("name") or item.get("model") or "")
                       for item in (running.json().get("models") or []))
        return [item for item in installed if isinstance(item, dict)], loaded
    except (httpx.HTTPError, ValueError, TypeError):
        return [], ()


def choose_route(
    gpu: Optional[GpuState], installed: Iterable[Mapping[str, Any]], *,
    cloud_allowed: bool = False, cloud_available: bool = False,
    model_hints: tuple[str, ...] = DEFAULT_MODEL_HINTS,
    max_gpu_utilization: int = 15,
    reserve_mib: int = 1536,
    inference_overhead_mib: int = 1024,
    now: Optional[float] = None,
) -> Route:
    """Prefer proven local candidates when idle, otherwise ZDR or defer.

    ``size`` is Ollama's model byte size, not a VRAM guarantee. The overhead and
    reserve deliberately prevent tight fits; a runtime OOM still defers work.
    """
    current = time.time() if now is None else now
    fallback_reason = "gpu_telemetry_unavailable"
    if gpu is not None and current - gpu.sampled_at <= 10 and gpu.sampled_at <= current + 1:
        if gpu.loaded_models:
            fallback_reason = "ollama_model_already_resident"
        elif gpu.utilization_percent > max_gpu_utilization:
            fallback_reason = "gpu_busy"
        else:
            candidates = list(installed)
            for hint in model_hints:
                for item in candidates:
                    name = str(item.get("name") or item.get("model") or "")
                    if hint.lower() not in name.lower():
                        continue
                    try:
                        size_mib = (int(item["size"]) + _MIB - 1) // _MIB
                    except (KeyError, TypeError, ValueError):
                        continue
                    if gpu.free_mib >= size_mib + inference_overhead_mib + reserve_mib:
                        return Route("ollama", name, "idle_gpu_headroom", gpu.free_mib)
            fallback_reason = "no_eligible_model_fits"
    if cloud_allowed and cloud_available:
        return Route("openrouter", None, fallback_reason, gpu.free_mib if gpu else None)
    return Route("deferred", None, fallback_reason, gpu.free_mib if gpu else None)


def configured_model_hints() -> tuple[str, ...]:
    raw = os.environ.get("MUNINN_AUTO_LOCAL_MODEL_HINTS", "")
    hints = tuple(item.strip() for item in raw.split(",") if item.strip())
    return hints or DEFAULT_MODEL_HINTS


def guarded_openrouter_available(daily_cap_usd: float = 1.0) -> bool:
    """Require a provider-enforced daily key limit before automatic egress.

    Application-side estimates cannot enforce a hard dollar cap if a response
    costs more than forecast. A dedicated OpenRouter key with a daily limit can.
    The key is never returned, logged, or placed in the route decision.
    """
    from muninn.history import llm_settings

    key = llm_settings.api_key()
    if not key or daily_cap_usd <= 0:
        return False
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{llm_settings.OPENROUTER_API}/key",
                                  headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
        data = response.json().get("data") or {}
        limit = float(data["limit"])
        remaining = float(data["limit_remaining"])
        return data.get("limit_reset") == "daily" and 0 < limit <= daily_cap_usd and remaining > 0
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return False
