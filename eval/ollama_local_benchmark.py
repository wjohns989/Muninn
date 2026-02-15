from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = ROOT / "eval" / "ollama_model_matrix.json"
DEFAULT_PROMPTS_PATH = ROOT / "eval" / "ollama_benchmark_prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "reports" / "ollama"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Prompt line {line_number} is not a JSON object.")
            prompt_id = str(item.get("id", "")).strip()
            prompt_text = str(item.get("prompt", "")).strip()
            category = str(item.get("category", "")).strip()
            if not prompt_id or not prompt_text:
                raise ValueError(
                    f"Prompt line {line_number} must contain non-empty 'id' and 'prompt'."
                )
            prompts.append({"id": prompt_id, "prompt": prompt_text, "category": category})
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=True,
    )


def _installed_models() -> dict[str, dict[str, str]]:
    result = _run(["ollama", "list"])
    lines = [line.rstrip("\n") for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return {}

    models: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        # NAME ID SIZE MODIFIED...
        name = parts[0]
        model_id = parts[1]
        size = f"{parts[2]} {parts[3]}"
        modified = " ".join(parts[4:]).strip() if len(parts) > 4 else ""
        models[name] = {"id": model_id, "size": size, "modified": modified}
    return models


def _matrix_models(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    models = matrix.get("models")
    if not isinstance(models, list):
        raise ValueError("Matrix file must contain a 'models' list.")
    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict) or "tag" not in item:
            raise ValueError("Each model entry must be an object with a 'tag'.")
        normalized.append(item)
    return normalized


def _resolve_target_models(
    matrix: dict[str, Any], include_optional: bool, only: list[str] | None
) -> list[str]:
    matrix_entries = _matrix_models(matrix)
    if only:
        requested = {model.strip() for model in only if model.strip()}
        return [entry["tag"] for entry in matrix_entries if entry["tag"] in requested]
    if include_optional:
        return [entry["tag"] for entry in matrix_entries]
    return [entry["tag"] for entry in matrix_entries if bool(entry.get("default_enabled", False))]


def cmd_list(args: argparse.Namespace) -> int:
    matrix = _load_json(Path(args.matrix).resolve())
    installed = _installed_models()
    entries = _matrix_models(matrix)

    print("Model matrix status:")
    for entry in entries:
        tag = str(entry["tag"])
        status = "installed" if tag in installed else "missing"
        default_label = "default" if entry.get("default_enabled", False) else "optional"
        detail = installed.get(tag, {})
        suffix = f" | size={detail.get('size')} | modified={detail.get('modified')}" if detail else ""
        print(f"- {tag}: {status} ({default_label}){suffix}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    matrix = _load_json(Path(args.matrix).resolve())
    installed = _installed_models()
    selected = _resolve_target_models(
        matrix=matrix,
        include_optional=bool(args.include_optional),
        only=args.models.split(",") if args.models else None,
    )
    if not selected:
        print("No target models selected from matrix.")
        return 1

    print(f"Selected {len(selected)} model(s) for sync.")
    for model in selected:
        if model in installed:
            print(f"[skip] {model} already installed.")
            continue
        if args.dry_run:
            print(f"[dry-run] would pull {model}")
            continue
        print(f"[pull] {model}")
        proc = subprocess.run(["ollama", "pull", model], text=True)
        if proc.returncode != 0:
            print(f"[fail] pull failed for {model}", file=sys.stderr)
            return proc.returncode
    return 0


def _post_generate(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_version(url: str, timeout_seconds: int) -> str:
    req = request.Request(f"{url.rstrip('/')}/api/version", method="GET")
    with request.urlopen(req, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return str(payload.get("version", "unknown"))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    fraction = rank - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction)


def cmd_benchmark(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix).resolve()
    prompts_path = Path(args.prompts).resolve()
    output_dir = Path(args.output_dir).resolve()
    ollama_url = args.ollama_url.strip()
    timeout_seconds = int(args.timeout_seconds)
    num_predict = int(args.num_predict)
    repeats = int(args.repeats)
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if num_predict < 1:
        raise ValueError("--num-predict must be >= 1")

    matrix = _load_json(matrix_path)
    prompts = _load_prompts(prompts_path)
    installed = _installed_models()

    selected = _resolve_target_models(
        matrix=matrix,
        include_optional=bool(args.include_optional),
        only=args.models.split(",") if args.models else None,
    )
    selected = [model for model in selected if model in installed]
    if not selected:
        print("No selected models are installed. Run sync first.", file=sys.stderr)
        return 1

    try:
        ollama_version = _post_version(ollama_url, timeout_seconds)
    except error.URLError as exc:
        print(f"Ollama endpoint unavailable at {ollama_url}: {exc}", file=sys.stderr)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else output_dir / f"report_{run_id}.json"

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_url": ollama_url,
        "ollama_version": ollama_version,
        "matrix_path": str(matrix_path),
        "prompts_path": str(prompts_path),
        "repeats": repeats,
        "num_predict": num_predict,
        "models": {},
    }

    for model in selected:
        print(f"[benchmark] model={model}")
        model_runs: list[dict[str, Any]] = []
        for prompt in prompts:
            for rep in range(repeats):
                payload = {
                    "model": model,
                    "prompt": prompt["prompt"],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": num_predict},
                }
                wall_start = time.perf_counter()
                try:
                    result = _post_generate(ollama_url, payload, timeout_seconds)
                except Exception as exc:  # noqa: BLE001
                    model_runs.append(
                        {
                            "prompt_id": prompt["id"],
                            "category": prompt["category"],
                            "repeat_index": rep,
                            "error": str(exc),
                        }
                    )
                    continue
                wall_seconds = time.perf_counter() - wall_start
                eval_count = int(result.get("eval_count", 0) or 0)
                eval_duration_ns = int(result.get("eval_duration", 0) or 0)
                eval_tps = 0.0
                if eval_count > 0 and eval_duration_ns > 0:
                    eval_tps = eval_count / (eval_duration_ns / 1_000_000_000.0)
                model_runs.append(
                    {
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "repeat_index": rep,
                        "wall_seconds": round(wall_seconds, 6),
                        "total_seconds_api": round(
                            (int(result.get("total_duration", 0) or 0) / 1_000_000_000.0), 6
                        ),
                        "prompt_eval_count": int(result.get("prompt_eval_count", 0) or 0),
                        "eval_count": eval_count,
                        "eval_tokens_per_second": round(eval_tps, 4),
                        "done_reason": result.get("done_reason"),
                        "error": None,
                    }
                )

        successful = [run for run in model_runs if run.get("error") is None]
        latencies = [float(run["wall_seconds"]) for run in successful]
        tps_values = [float(run["eval_tokens_per_second"]) for run in successful if run["eval_tokens_per_second"] > 0]
        report["models"][model] = {
            "runs": model_runs,
            "summary": {
                "total_runs": len(model_runs),
                "successful_runs": len(successful),
                "failed_runs": len(model_runs) - len(successful),
                "avg_wall_seconds": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
                "p95_wall_seconds": round(_percentile(latencies, 0.95), 6) if latencies else 0.0,
                "avg_eval_tokens_per_second": round(sum(tps_values) / len(tps_values), 4)
                if tps_values
                else 0.0,
            },
        }

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"Benchmark report written to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Ollama model sync and benchmark utility for Muninn profile testing."
    )
    parser.add_argument(
        "--matrix",
        default=str(DEFAULT_MATRIX_PATH),
        help="Path to model matrix JSON.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List matrix model availability in local Ollama.")
    list_parser.set_defaults(func=cmd_list)

    sync_parser = subparsers.add_parser("sync", help="Pull missing models from matrix.")
    sync_parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also pull optional models from matrix.",
    )
    sync_parser.add_argument(
        "--models",
        help="Comma-separated explicit model tags to pull (overrides default selection).",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pull plan without pulling.",
    )
    sync_parser.set_defaults(func=cmd_sync)

    bench_parser = subparsers.add_parser("benchmark", help="Run local benchmark prompts against installed models.")
    bench_parser.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS_PATH),
        help="Path to benchmark prompts JSONL.",
    )
    bench_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated benchmark reports.",
    )
    bench_parser.add_argument(
        "--output",
        help="Explicit report output path; defaults to output-dir/report_<timestamp>.json.",
    )
    bench_parser.add_argument(
        "--models",
        help="Comma-separated explicit model tags to benchmark (must exist in matrix and be installed).",
    )
    bench_parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional matrix models when --models is not provided.",
    )
    bench_parser.add_argument(
        "--repeats",
        default=1,
        type=int,
        help="Number of repeated runs per prompt per model.",
    )
    bench_parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for local Ollama server.",
    )
    bench_parser.add_argument(
        "--timeout-seconds",
        default=240,
        type=int,
        help="HTTP timeout for a single generation request.",
    )
    bench_parser.add_argument(
        "--num-predict",
        default=192,
        type=int,
        help="Maximum generated tokens per request.",
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stdout)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return int(exc.returncode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
