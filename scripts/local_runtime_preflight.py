"""Read-only readiness without exposing environment values or process arguments."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import psutil


def inspect_runtime(repo: Path, *, authenticated: bool = False) -> dict:
    server = (repo / "server.py").resolve()
    owned = []
    for process in psutil.process_iter(["pid", "name", "exe", "cwd", "cmdline"]):
        try:
            args = process.info["cmdline"] or []
            if not args or "python" not in (process.info["name"] or "").lower():
                continue
            cwd = Path(process.info["cwd"] or ".")
            if any((cwd / arg).resolve() == server for arg in args[1:] if arg.endswith(".py")):
                owned.append({"pid": process.pid, "interpreter": process.info["exe"]})
        except (psutil.Error, OSError, ValueError):
            continue
    try:
        owners = sorted({item.pid for item in psutil.net_connections(kind="tcp")
                         if item.laddr and item.laddr.port == 42069
                         and item.status == psutil.CONN_LISTEN})
    except psutil.Error:
        owners = None
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get("http://127.0.0.1:42069/health")
        health = response.status_code
    except httpx.HTTPError:
        health = "unreachable"
    report = {"muninn_processes": owned, "listener_owners": owners, "health_http": health}
    if authenticated:
        from muninn.history.auto_routing import _local_setting

        token = _local_setting("MUNINN_AUTH_TOKEN")
        report["auth_configured"] = bool(token)
        if token:
            with httpx.Client(timeout=15, trust_env=False) as client:
                base = "http://127.0.0.1:42069"
                anonymous = client.get(base + "/profiles/model")
                protected = client.get(base + "/profiles/model", headers={"Authorization": "Bearer " + token})
                root = client.get(base + "/")
                history = client.get(base + "/history/status", headers={"Authorization": "Bearer " + token})
            report.update(anonymous_protected_http=anonymous.status_code,
                          authenticated_protected_http=protected.status_code,
                          anonymous_root_contains_token=token in root.text,
                          history_http=history.status_code)
            if history.status_code == 200:
                data = history.json().get("data", {})
                vault = data.get("vault", {})
                report.update(history_security=data.get("history_security"),
                              archive_ready=vault.get("ready"), archive=vault.get("archive"),
                              capture_queue=data.get("capture_queue"))
    return report


def inspect_models() -> dict:
    from dataclasses import asdict, replace
    from muninn.history.auto_routing import probe_gpu, probe_ollama, choose_route
    base = "http://127.0.0.1:11434"
    gpu = probe_gpu()
    installed, loaded = probe_ollama(base)
    if gpu is not None:
        gpu = replace(gpu, loaded_models=loaded)
    return {"gpu": asdict(gpu) if gpu is not None else None,
            "installed_models": [{"name": item.get("name") or item.get("model"),
                                  "size": item.get("size")} for item in installed],
            "routes": [{"requested": name, "route": asdict(choose_route(
                gpu, installed, cloud_allowed=False, model_hints=(name,)))}
                       for name in ("qwen2.5:7b", "muninn-qwen35-defiant-q8-test:latest")]}


if __name__ == "__main__":
    import sys

    report = inspect_runtime(Path(__file__).resolve().parents[1],
                             authenticated="--authenticated" in sys.argv)
    if "--models" in sys.argv:
        report["models"] = inspect_models()
    print(json.dumps(report, sort_keys=True))
