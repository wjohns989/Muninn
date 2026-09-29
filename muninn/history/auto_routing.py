"""Conservative, read-only resource routing for automatic thread enrichment.

The CPU capture and search path never calls this module. Missing GPU telemetry
means no local model is loaded; a separately approved cloud route may be used.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

import httpx

_MIB = 1024 * 1024
DEFAULT_MODEL_HINTS = ("qwen2.5:7b", "qwen35")
COMPLEX_MODEL_HINTS = DEFAULT_MODEL_HINTS
COMPLEX_THREAD_TURNS = 60


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


def model_hints_for_thread(turns_imported: int, *,
                           configured: Optional[tuple[str, ...]] = None) -> tuple[str, ...]:
    """Favor the measured fast model until Q8 demonstrates better real-history accuracy.

    The turn count is the durable imported count in the history catalog. An
    explicit model order always wins over this conservative default. On the
    first real-history comparison, Q8 took longer and hallucinated more applied
    patches, so thread length alone must not select it.
    """
    if configured is not None:
        return configured
    if os.environ.get("MUNINN_AUTO_LOCAL_MODEL_HINTS", "").strip():
        return configured_model_hints()
    return COMPLEX_MODEL_HINTS if turns_imported >= COMPLEX_THREAD_TURNS else DEFAULT_MODEL_HINTS


def _local_setting(name: str) -> str:
    value = os.environ.get(name, "")
    if value or os.name != "nt":
        return value
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, name)[0])
    except (ImportError, FileNotFoundError, OSError):
        return ""


def _legacy_remote_policy() -> tuple[bool, float, float, bool]:
    daily, monthly = openrouter_budget_ceiling()
    return (_local_setting("MUNINN_STRICT_REMOTE_ANALYSIS").lower() in {"1", "true"},
            daily, monthly, _local_setting("MUNINN_OPENROUTER_BUDGET_OVERRIDE") == "1")


def remote_policy_snapshot(root: Path | None):
    """One fail-closed policy view for strict on-demand and background routes."""
    from muninn.history.remote_policy import PolicyError, RemotePolicy, read_policy

    if root is None:
        enabled, daily, monthly, override = _legacy_remote_policy()
        return RemotePolicy(enabled, daily, monthly, override, 0, "legacy_environment")
    try:
        return read_policy(root, _legacy_remote_policy)
    except PolicyError:
        return RemotePolicy(False, 0.0, 0.0, False, -1, "unavailable")


def openrouter_budget_ceiling(policy_root: Path | None = None) -> tuple[float, float]:
    """User-local policy, capped at $10/day and $100/month absent override."""
    if policy_root is not None:
        policy = remote_policy_snapshot(policy_root)
        return (policy.daily_usd, policy.monthly_usd) if policy.enabled else (0.0, 0.0)
    try:
        daily = float(_local_setting("MUNINN_OPENROUTER_MAX_DAILY_USD") or "10")
        monthly = float(_local_setting("MUNINN_OPENROUTER_MAX_MONTHLY_USD") or "100")
        if not (0 < daily < float("inf") and 0 < monthly < float("inf")):
            return 0.0, 0.0
        if _local_setting("MUNINN_OPENROUTER_BUDGET_OVERRIDE") != "1":
            daily, monthly = min(daily, 10.0), min(monthly, 100.0)
        return daily, monthly
    except ValueError:
        return 0.0, 0.0


def guarded_openrouter_available(daily_cap_usd: float | None = None,
                                 monthly_cap_usd: float | None = None,
                                 *, policy_root: Path | None = None) -> bool:
    """Require a finite provider-enforced key cap and both period ceilings.

    Application-side estimates cannot enforce a hard dollar cap if a response
    costs more than forecast. A dedicated OpenRouter key enforces its chosen
    reset period; the secondary period uses provider-reported usage as a guard.
    The key is never returned, logged, or placed in the route decision.
    """
    from muninn.history import llm_settings

    key = llm_settings.api_key()
    configured_daily, configured_monthly = openrouter_budget_ceiling(policy_root)
    daily_cap = min(configured_daily, daily_cap_usd) if daily_cap_usd is not None else configured_daily
    monthly_cap = min(configured_monthly, monthly_cap_usd) if monthly_cap_usd is not None else configured_monthly
    if not key or daily_cap <= 0 or monthly_cap <= 0:
        return False
    endpoint = urlsplit(llm_settings.OPENROUTER_API)
    if (endpoint.scheme != "https" or endpoint.netloc != "openrouter.ai"
            or endpoint.path != "/api/v1" or endpoint.query or endpoint.fragment):
        return False
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(f"{llm_settings.OPENROUTER_API}/key",
                                  headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
        data = response.json().get("data") or {}
        limit = float(data["limit"])
        remaining = float(data["limit_remaining"])
        if data.get("disabled") or not (0 < limit < float("inf") and remaining > 0):
            return False
        daily_used = float(data["usage_daily"])
        monthly_used = float(data["usage_monthly"])
        if not (0 <= daily_used < daily_cap and 0 <= monthly_used < monthly_cap):
            return False
        reset = data.get("limit_reset")
        if reset == "daily":
            return limit <= daily_cap
        if reset == "monthly":
            return limit <= monthly_cap
        return False
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return False
