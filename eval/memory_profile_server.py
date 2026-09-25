"""Benchmark-only Muninn server wrapper with private heap telemetry.

This module refuses to start unless its parent harness supplies an isolated
temporary root, a non-production loopback port, and a synthetic control token.
It is not imported by the production server.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
import tracemalloc
from pathlib import Path

from fastapi import HTTPException, Request

from eval.memory_profile_benchmark import BENCHMARK_TOKEN_HEADER, IsolationGuard

_UVICORN_SERVER = None


def _validated_settings() -> tuple[Path, Path, str, int, str]:
    if os.environ.get("MUNINN_BENCHMARK_MODE") != "1":
        raise RuntimeError("Benchmark server requires MUNINN_BENCHMARK_MODE=1")
    root = Path(os.environ["MUNINN_BENCHMARK_ROOT"])
    data_dir = Path(os.environ["MUNINN_DATA_DIR"])
    host = os.environ.get("MUNINN_HOST", "127.0.0.1")
    port = int(os.environ["MUNINN_PORT"])
    token = os.environ.get("MUNINN_BENCHMARK_TOKEN", "")
    if not token:
        raise RuntimeError("Benchmark control token is missing")
    IsolationGuard(root=root, data_dir=data_dir, host=host, port=port).validate()
    return root, data_dir, host, port, token


def _resource_state(server_module) -> dict:
    memory = getattr(server_module, "memory", None)
    if memory is None:
        return {"memory_initialized": False}
    bm25 = getattr(memory, "_bm25", None)
    feedback_cache = getattr(memory, "_feedback_multiplier_cache", {})
    return {
        "memory_initialized": bool(getattr(memory, "_initialized", False)),
        "embedding_loaded": getattr(memory, "_embed_model", None) is not None,
        "embedding_backend": (
            "fastembed" if getattr(memory, "_embed_model", None) is not None else "ollama_fallback"
        ),
        "reranker_loaded": getattr(memory, "_reranker", None) is not None,
        "conflict_detector_loaded": getattr(memory, "_conflict_detector", None) is not None,
        "consolidation_conflict_detector_loaded": (
            getattr(getattr(memory, "_consolidation", None), "_conflict_detector", None) is not None
        ),
        "vector_store_loaded": getattr(memory, "_vectors", None) is not None,
        "graph_store_loaded": getattr(memory, "_graph", None) is not None,
        "metadata_store_loaded": getattr(memory, "_metadata", None) is not None,
        "feedback_cache_size": len(feedback_cache),
        "bm25_document_count": int(getattr(bm25, "size", 0)) if bm25 is not None else 0,
    }


def main() -> int:
    _, _, host, port, token = _validated_settings()
    tracemalloc.start(int(os.environ.get("MUNINN_BENCHMARK_TRACEMALLOC_FRAMES", "10")))

    import uvicorn

    import server as server_module
    app = server_module.app

    def _authorize(request: Request) -> None:
        if request.headers.get(BENCHMARK_TOKEN_HEADER) != token:
            raise HTTPException(status_code=404, detail="Not found")

    @app.get("/_benchmark/snapshot", include_in_schema=False)
    async def benchmark_snapshot(request: Request):
        _authorize(request)
        current, peak = tracemalloc.get_traced_memory()
        return {
            "python_heap_bytes": current,
            "python_peak_bytes": peak,
            "async_task_count": len(asyncio.all_tasks()),
            "thread_count": threading.active_count(),
            "gc_counts": list(gc.get_count()),
            "resources": _resource_state(server_module),
        }

    @app.post("/_benchmark/shutdown", include_in_schema=False)
    async def benchmark_shutdown(request: Request):
        _authorize(request)
        if _UVICORN_SERVER is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        _UVICORN_SERVER.should_exit = True
        return {"accepted": True}

    global _UVICORN_SERVER
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level=os.environ.get("MUNINN_LOG_LEVEL", "warning"),
        access_log=False,
        lifespan="on",
    )
    _UVICORN_SERVER = uvicorn.Server(config)
    _UVICORN_SERVER.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
