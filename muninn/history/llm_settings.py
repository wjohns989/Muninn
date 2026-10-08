"""Environment-only OpenRouter credentials and nonsecret model settings."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

OPENROUTER_API = "https://openrouter.ai/api/v1"
KEYS_PAGE = "https://openrouter.ai/settings/keys"
PRIVACY_PAGE = "https://openrouter.ai/settings/privacy"

# Chosen from OpenRouter's live zero-data-retention endpoint list (2026-09-26): endpoints with
# strict structured outputs, 128k+ context and healthy uptime, ranked by independent
# extraction/summarization results and cost per thread. OpenRouter falls back down the list
# when a model has no ZDR endpoint available.
# All three accept about 1M tokens on ZDR endpoints. DeepSeek V4 Flash comes first among the
# fallbacks because it has many ZDR hosts, which keeps bulk runs moving if Luna's single ZDR
# host (Azure) is busy.
# GPT-6 Luna Pro is Luna served with reasoning mode "pro", at the same price on its Azure ZDR
# endpoints; verified live (strict JSON Schema, ZDR routing). Fallbacks run on other hosts, so an
# Azure outage does not stop a run. OpenRouter accepts at most 3 models per request.
DEFAULT_MODEL = "openai/gpt-6-luna-pro"
FALLBACK_MODELS = ("deepseek/deepseek-v4-flash", "google/gemini-3.5-flash-lite")
MAX_MODELS = 3
# ":batch" variants only work through OpenRouter's asynchronous Batch API, which also keeps inputs and
# results for up to 30 days; Muninn calls models directly, so the suffix is dropped.
_BATCH_SUFFIX = ":batch"


def settings_path() -> Path:
    from muninn.platform import get_config_dir

    return get_config_dir() / "openrouter.json"


def load() -> Dict[str, Any]:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        data.pop("api_key", None)
        return data
    except (OSError, ValueError):
        return {}


def _save(data: Dict[str, Any]) -> Path:
    data = dict(data)
    data.pop("api_key", None)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


def api_key() -> Optional[str]:
    for name in ("MUNINN_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    value = _windows_user_env("MUNINN_OPENROUTER_API_KEY")
    if value:
        return value
    return None


def _windows_user_env(name: str) -> Optional[str]:
    """Read one User-scoped environment value after this process already started.

    Windows stores User environment variables under HKCU. A long-lived service
    does not receive later environment-block updates, so process env alone can
    miss a key the operator just saved. Never enumerate or log this value.
    """
    if os.name != "nt" or name != "MUNINN_OPENROUTER_API_KEY":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
        if isinstance(value, str):
            return value.strip() or None
        return None
    except (ImportError, FileNotFoundError, PermissionError, OSError):
        return None


def key_source() -> Optional[str]:
    for name in ("MUNINN_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(name, "").strip():
            return f"environment ({name})"
    if _windows_user_env("MUNINN_OPENROUTER_API_KEY"):
        return "user environment (MUNINN_OPENROUTER_API_KEY)"
    return None


def models() -> List[str]:
    """Primary model first, then fallbacks: MUNINN_INSIGHTS_MODEL, saved choice, or the default."""
    chosen = os.environ.get("MUNINN_INSIGHTS_MODEL", "").strip() or str(load().get("model") or "").strip()
    primary = normalize_model(chosen) or DEFAULT_MODEL
    return ([primary] + [m for m in (DEFAULT_MODEL, *FALLBACK_MODELS) if m != primary])[:MAX_MODELS]


def normalize_model(model: Optional[str]) -> str:
    """Model id usable with chat completions (drops the Batch API ':batch' suffix)."""
    model = (model or "").strip()
    return model[: -len(_BATCH_SUFFIX)] if model.endswith(_BATCH_SUFFIX) else model


def save_key(key: str, model: Optional[str] = None) -> Path:
    """Make a key available only to this process; never persist credential material."""
    key = key.strip()
    if key:
        os.environ["MUNINN_OPENROUTER_API_KEY"] = key
    data = load()
    data.update({"saved_at": time.time(), "declined": False})
    if model:
        data["model"] = normalize_model(model)
    return _save(data)


def save_model(model: Optional[str]) -> Path:
    data = load()
    if model:
        data["model"] = normalize_model(model)
    else:
        data.pop("model", None)
    return _save(data)


def decline() -> Path:
    """Remember that the user chose local Ollama, so the CLI stops asking."""
    data = load()
    data.update({"declined": True, "declined_at": time.time()})
    return _save(data)


def forget() -> None:
    try:
        settings_path().unlink()
    except FileNotFoundError:
        pass


def should_prompt() -> bool:
    return api_key() is None and not load().get("declined")


def verify_key(key: str, timeout: float = 15.0) -> Tuple[bool, str]:
    """Ask OpenRouter about the key (GET /key): (valid, description)."""
    try:
        response = httpx.get(f"{OPENROUTER_API}/key", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except httpx.HTTPError as exc:
        return False, f"could not reach OpenRouter ({exc})"
    if response.status_code == 401:
        return False, "OpenRouter rejected the key"
    if response.status_code >= 400:
        return False, f"OpenRouter answered {response.status_code}"
    data = response.json().get("data") or {}
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    credit = "no spending limit" if limit is None else f"{remaining} of {limit} credits left"
    return True, f"key '{data.get('label') or 'unnamed'}' accepted ({credit})"
