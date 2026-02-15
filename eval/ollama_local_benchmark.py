from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = ROOT / "eval" / "ollama_model_matrix.json"
DEFAULT_PROMPTS_PATH = ROOT / "eval" / "ollama_benchmark_prompts.jsonl"
DEFAULT_PROMOTION_POLICY_PATH = ROOT / "eval" / "ollama_profile_promotion_policy.json"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "reports" / "ollama"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MUNINN_URL = "http://127.0.0.1:42069"
SUPPORTED_PROFILE_NAMES = ("low_latency", "balanced", "high_reasoning")
PROFILE_USAGE_GUIDANCE = {
    "low_latency": "Runtime helper during active coding and tool-heavy IDE workflows.",
    "balanced": "Default implementation assistant for day-to-day coding tasks.",
    "high_reasoning": "Planning/architecture/refactor analysis when deeper reasoning is required.",
}
LEGACY_TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".cs",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".sql",
    ".sh",
    ".ps1",
    ".rb",
    ".php",
    ".html",
    ".css",
}
STOPWORDS = {
    "that",
    "this",
    "with",
    "from",
    "return",
    "class",
    "function",
    "const",
    "let",
    "var",
    "true",
    "false",
    "null",
    "none",
    "void",
    "async",
    "await",
    "import",
    "export",
    "default",
    "global",
    "project",
    "memory",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
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
            normalized = dict(item)
            normalized["id"] = prompt_id
            normalized["prompt"] = prompt_text
            normalized["category"] = category
            prompts.append(normalized)
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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()

    if not (candidate.startswith("{") and candidate.endswith("}")):
        object_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if object_match:
            candidate = object_match.group(0).strip()
        else:
            return None

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _safe_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _score_live_response(prompt: dict[str, Any], response_text: str) -> tuple[float, dict[str, Any]]:
    checks: dict[str, float] = {}
    text = (response_text or "").strip()
    lower_text = text.lower()

    checks["non_empty"] = 1.0 if text else 0.0

    required_json_keys = prompt.get("required_json_keys") or []
    if isinstance(required_json_keys, list) and required_json_keys:
        payload = _extract_json_object(text)
        if payload is None:
            checks["required_json_keys"] = 0.0
        else:
            keys = [str(k).strip() for k in required_json_keys if str(k).strip()]
            if keys:
                present = sum(1 for key in keys if key in payload)
                checks["required_json_keys"] = present / len(keys)

    required_substrings = prompt.get("required_substrings") or []
    if isinstance(required_substrings, list) and required_substrings:
        needles = [str(s).strip().lower() for s in required_substrings if str(s).strip()]
        if needles:
            present = sum(1 for needle in needles if needle in lower_text)
            checks["required_substrings"] = present / len(needles)

    forbidden_substrings = prompt.get("forbidden_substrings") or []
    if isinstance(forbidden_substrings, list) and forbidden_substrings:
        needles = [str(s).strip().lower() for s in forbidden_substrings if str(s).strip()]
        if needles:
            present = sum(1 for needle in needles if needle in lower_text)
            checks["forbidden_substrings"] = max(0.0, 1.0 - (present / len(needles)))

    output_format = str(prompt.get("output_format", "")).strip().lower()
    if output_format == "code":
        likely_code = bool(
            re.search(r"\bdef\b|\bclass\b|=>|\{|\}|import\s+|from\s+\w+\s+import", text)
        )
        checks["code_format"] = 1.0 if likely_code else 0.0

    exact_line_count = prompt.get("exact_line_count")
    if isinstance(exact_line_count, int) and exact_line_count > 0:
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        checks["exact_line_count"] = 1.0 if len(non_empty_lines) == exact_line_count else 0.0

    min_words = prompt.get("min_words")
    if isinstance(min_words, int) and min_words > 0:
        word_count = len(_tokenize_words(text))
        checks["min_words"] = 1.0 if word_count >= min_words else 0.0

    score = _safe_average(list(checks.values()))
    return score, {"checks": checks}


def _read_text_file(path: Path, max_chars: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    content = content.replace("\r\n", "\n")
    if len(content) > max_chars:
        return content[:max_chars]
    return content


def _extract_expected_keywords(text: str, minimum: int = 3, maximum: int = 6) -> list[str]:
    tokens = [t for t in _tokenize_words(text) if len(t) >= 4 and t not in STOPWORDS]
    if not tokens:
        return []
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = [token for token, _count in ranked[:maximum]]
    if len(selected) < minimum:
        return []
    return selected


def _build_legacy_cases(
    roots: list[Path],
    *,
    max_cases_per_root: int,
    snippet_chars: int,
) -> list[dict[str, Any]]:
    all_cases: list[dict[str, Any]] = []
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.exists() or not resolved_root.is_dir():
            continue

        root_cases = 0
        for path in sorted(resolved_root.rglob("*")):
            if root_cases >= max_cases_per_root:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in LEGACY_TEXT_EXTENSIONS:
                continue
            snippet = _read_text_file(path, max_chars=snippet_chars)
            if len(snippet.strip()) < 120:
                continue
            expected_keywords = _extract_expected_keywords(snippet)
            if len(expected_keywords) < 3:
                continue
            relative_path = str(path.resolve().relative_to(resolved_root)).replace("\\", "/")
            prompt = (
                "Extract memory-oriented structure from this legacy project snippet. "
                "Return strict JSON with keys summary, entities, risks, action_items.\n"
                f"source_path={relative_path}\n"
                "snippet:\n"
                f"{snippet}"
            )
            case_id = f"{resolved_root.name}:{relative_path}"
            all_cases.append(
                {
                    "id": case_id,
                    "category": "legacy_ingestion",
                    "root": str(resolved_root),
                    "source_path": str(path.resolve()),
                    "prompt": prompt,
                    "expected_keywords": expected_keywords,
                    "required_json_keys": ["summary", "entities", "risks", "action_items"],
                    "min_words": 20,
                }
            )
            root_cases += 1
    return all_cases


def _score_legacy_response(case: dict[str, Any], response_text: str) -> tuple[float, dict[str, Any]]:
    checks: dict[str, float] = {}
    text = (response_text or "").strip()
    payload = _extract_json_object(text)
    if payload is None:
        checks["json_parse"] = 0.0
        checks["required_json_keys"] = 0.0
        checks["keyword_coverage"] = 0.0
        checks["min_words"] = 0.0
        return 0.0, {"checks": checks}

    checks["json_parse"] = 1.0
    required_json_keys = [str(k).strip() for k in case.get("required_json_keys", []) if str(k).strip()]
    if required_json_keys:
        present = sum(1 for key in required_json_keys if key in payload)
        checks["required_json_keys"] = present / len(required_json_keys)

    flattened = json.dumps(payload, ensure_ascii=False).lower()
    expected_keywords = [str(k).strip().lower() for k in case.get("expected_keywords", []) if str(k).strip()]
    if expected_keywords:
        present = sum(1 for keyword in expected_keywords if keyword in flattened)
        checks["keyword_coverage"] = present / len(expected_keywords)

    min_words = int(case.get("min_words") or 0)
    if min_words > 0:
        checks["min_words"] = 1.0 if len(_tokenize_words(flattened)) >= min_words else 0.0

    score = _safe_average(list(checks.values()))
    return score, {"checks": checks}


def _entry_for_model(matrix: dict[str, Any], tag: str) -> dict[str, Any]:
    for entry in _matrix_models(matrix):
        if str(entry.get("tag", "")).strip() == tag:
            return entry
    return {}


def _resource_efficiency(summary: dict[str, Any], matrix_entry: dict[str, Any]) -> dict[str, float]:
    avg_ability = float(summary.get("avg_ability_score", 0.0) or 0.0)
    avg_wall = float(summary.get("avg_wall_seconds", 0.0) or 0.0)
    vram_min_gb = float(matrix_entry.get("vram_min_gb", 0.0) or 0.0)
    return {
        "vram_min_gb": vram_min_gb,
        "ability_per_second": round(avg_ability / avg_wall, 6) if avg_wall > 0 else 0.0,
        "ability_per_vram_gb": round(avg_ability / vram_min_gb, 6) if vram_min_gb > 0 else 0.0,
    }


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _suite_model_summary(report: dict[str, Any], model: str) -> dict[str, Any] | None:
    models = report.get("models")
    if not isinstance(models, dict):
        return None
    details = models.get(model)
    if not isinstance(details, dict):
        return None
    summary = details.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def _weighted_average(
    first: float | None,
    second: float | None,
    *,
    first_weight: float,
    second_weight: float,
) -> float | None:
    weighted = 0.0
    weight_sum = 0.0
    if first is not None:
        weighted += first * first_weight
        weight_sum += first_weight
    if second is not None:
        weighted += second * second_weight
        weight_sum += second_weight
    if weight_sum <= 0:
        return None
    return weighted / weight_sum


def _load_promotion_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path)
    profiles = policy.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Promotion policy must define non-empty 'profiles' object.")
    for profile_name in ("low_latency", "balanced", "high_reasoning"):
        if profile_name not in profiles:
            raise ValueError(f"Promotion policy missing required profile '{profile_name}'.")
    return policy


def _evaluate_candidate(
    *,
    model: str,
    gate: dict[str, Any],
    live_summary: dict[str, Any] | None,
    legacy_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    violations: list[str] = []

    live_ability = _to_float((live_summary or {}).get("avg_ability_score"))
    legacy_ability = _to_float((legacy_summary or {}).get("avg_ability_score"))
    live_p95 = _to_float((live_summary or {}).get("p95_wall_seconds"))
    legacy_p95 = _to_float((legacy_summary or {}).get("p95_wall_seconds"))
    live_efficiency = _to_float(
        ((live_summary or {}).get("resource_efficiency") or {}).get("ability_per_vram_gb")
    )

    require_live = bool(gate.get("require_live_suite", True))
    require_legacy = bool(gate.get("require_legacy_suite", True))

    if require_live and live_summary is None:
        violations.append("missing_live_suite")
    if require_legacy and legacy_summary is None:
        violations.append("missing_legacy_suite")

    min_live_ability = _to_float(gate.get("min_live_ability"))
    if min_live_ability is not None:
        if live_ability is None:
            violations.append("missing_live_ability")
        elif live_ability < min_live_ability:
            violations.append("live_ability_below_threshold")

    min_legacy_ability = _to_float(gate.get("min_legacy_ability"))
    if min_legacy_ability is not None:
        if legacy_ability is None:
            violations.append("missing_legacy_ability")
        elif legacy_ability < min_legacy_ability:
            violations.append("legacy_ability_below_threshold")

    max_live_p95 = _to_float(gate.get("max_live_p95_seconds"))
    if max_live_p95 is not None:
        if live_p95 is None:
            violations.append("missing_live_p95")
        elif live_p95 > max_live_p95:
            violations.append("live_p95_above_threshold")

    max_legacy_p95 = _to_float(gate.get("max_legacy_p95_seconds"))
    if max_legacy_p95 is not None:
        if legacy_p95 is None:
            if require_legacy:
                violations.append("missing_legacy_p95")
        elif legacy_p95 > max_legacy_p95:
            violations.append("legacy_p95_above_threshold")

    min_ability_per_vram = _to_float(gate.get("min_live_ability_per_vram_gb"))
    if min_ability_per_vram is not None:
        if live_efficiency is None:
            violations.append("missing_live_ability_per_vram")
        elif live_efficiency < min_ability_per_vram:
            violations.append("live_ability_per_vram_below_threshold")

    live_weight = _to_float(gate.get("live_weight")) or 0.6
    legacy_weight = _to_float(gate.get("legacy_weight")) or 0.4
    combined_ability = _weighted_average(
        live_ability,
        legacy_ability,
        first_weight=live_weight,
        second_weight=legacy_weight,
    )

    objective = gate.get("objective") or {}
    ability_weight = _to_float(objective.get("ability_weight")) or 1.0
    resource_weight = _to_float(objective.get("resource_weight")) or 0.25
    latency_penalty = _to_float(objective.get("latency_penalty")) or 0.02
    live_avg_wall = _to_float((live_summary or {}).get("avg_wall_seconds")) or 0.0
    composite_score = (
        (combined_ability or 0.0) * ability_weight
        + (live_efficiency or 0.0) * resource_weight
        - live_avg_wall * latency_penalty
    )

    return {
        "model": model,
        "passed": len(violations) == 0,
        "violations": violations,
        "live_summary": live_summary,
        "legacy_summary": legacy_summary,
        "combined_ability_score": round(combined_ability, 6) if combined_ability is not None else None,
        "composite_score": round(composite_score, 6),
    }


def _pick_recommendation(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [result for result in results if result.get("passed") is True]
    if not passing:
        return None
    ranked = sorted(
        passing,
        key=lambda item: (
            float(item.get("composite_score", 0.0)),
            float(item.get("combined_ability_score") or 0.0),
        ),
        reverse=True,
    )
    return ranked[0]


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


def _http_json_request(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = request.Request(
        url,
        data=encoded,
        headers=headers,
        method=method.upper(),
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object response from {url}")
        return parsed


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _timestamp_context(now: datetime | None = None) -> dict[str, str]:
    current = now or datetime.now(timezone.utc)
    return {
        "created_at_utc": current.isoformat(),
        "run_id": current.strftime("%Y%m%dT%H%M%SZ"),
    }


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value else None


def _git_commit_reachable_from(commit_sha: str, ref: str) -> tuple[bool, str | None]:
    try:
        normalized_commit_sha = _normalize_commit_sha(commit_sha)
    except ValueError as exc:
        return False, str(exc)
    if not normalized_commit_sha:
        return False, "Empty commit SHA provided for ancestry verification."

    normalized_ref = str(ref or "").strip()
    if not normalized_ref:
        return False, "Empty ref provided for commit ancestry verification."
    try:
        ref_check = subprocess.run(
            ["git", "rev-parse", "--verify", "--", normalized_ref],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"git executable unavailable: {exc}"

    if ref_check.returncode != 0:
        stderr = (ref_check.stderr or "").strip()
        return False, f"Ref '{normalized_ref}' is not resolvable. {stderr}".strip()

    try:
        resolved_ref_sha = _normalize_commit_sha(ref_check.stdout)
    except ValueError:
        return (
            False,
            f"Ref '{normalized_ref}' did not resolve to a valid commit SHA.",
        )
    if not resolved_ref_sha:
        return (
            False,
            f"Ref '{normalized_ref}' did not resolve to a valid commit SHA.",
        )

    try:
        ancestor_check = subprocess.run(
            ["git", "merge-base", "--is-ancestor", normalized_commit_sha, resolved_ref_sha],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"git executable unavailable: {exc}"

    if ancestor_check.returncode == 0:
        return True, None
    if ancestor_check.returncode == 1:
        return False, None
    stderr = (ancestor_check.stderr or "").strip()
    return False, f"git merge-base failed. {stderr}".strip()


def _normalize_commit_sha(value: str | None) -> str | None:
    candidate = str(value or "").strip().lower()
    if not candidate:
        return None
    if not re.fullmatch(r"[0-9a-f]{7,40}", candidate):
        raise ValueError(
            "Invalid commit SHA format in approval manifest context. "
            "Expected 7-40 lowercase hex characters."
        )
    return candidate


def _build_change_context(
    *,
    pr_number: Any,
    pr_url: Any,
    commit_sha: Any,
    branch_name: Any,
) -> dict[str, Any]:
    normalized_pr_number: int | None = None
    if pr_number is not None:
        normalized_pr_number = int(pr_number)
        if normalized_pr_number <= 0:
            raise ValueError("--pr-number must be > 0 when provided.")

    normalized_pr_url = str(pr_url or "").strip()
    if normalized_pr_url:
        if not (
            normalized_pr_url.startswith("https://")
            or normalized_pr_url.startswith("http://")
        ):
            raise ValueError("--pr-url must start with https:// or http://.")

    normalized_commit_sha = _normalize_commit_sha(str(commit_sha or ""))
    if normalized_commit_sha is None:
        normalized_commit_sha = _normalize_commit_sha(_git_output(["rev-parse", "HEAD"]))

    normalized_branch = str(branch_name or "").strip()
    if not normalized_branch:
        normalized_branch = str(_git_output(["rev-parse", "--abbrev-ref", "HEAD"]) or "")

    return {
        "pr_number": normalized_pr_number,
        "pr_url": normalized_pr_url or None,
        "commit_sha": normalized_commit_sha,
        "branch_name": normalized_branch or None,
    }


def _validated_manifest_change_context(
    value: Any,
    *,
    require_change_context: bool,
    require_pr_number: bool,
    require_commit_sha: bool,
    require_branch_name: bool,
) -> dict[str, Any]:
    if value is None:
        if require_change_context or require_pr_number or require_commit_sha or require_branch_name:
            raise ValueError(
                "Approval manifest is missing change_context while provenance enforcement is enabled."
            )
        return {
            "pr_number": None,
            "pr_url": None,
            "commit_sha": None,
            "branch_name": None,
        }

    if not isinstance(value, dict):
        raise ValueError("Approval manifest change_context must be an object when present.")

    pr_number_raw = value.get("pr_number")
    pr_number: int | None = None
    if pr_number_raw is not None:
        pr_number = int(pr_number_raw)
        if pr_number <= 0:
            raise ValueError("Approval manifest change_context.pr_number must be > 0.")

    pr_url = str(value.get("pr_url") or "").strip() or None
    if pr_url and not (pr_url.startswith("https://") or pr_url.startswith("http://")):
        raise ValueError("Approval manifest change_context.pr_url must start with https:// or http://.")

    commit_sha = _normalize_commit_sha(str(value.get("commit_sha") or ""))
    branch_name = str(value.get("branch_name") or "").strip() or None

    if require_pr_number and pr_number is None:
        raise ValueError(
            "Approval manifest change_context.pr_number is required by --require-pr-number."
        )
    if require_commit_sha and commit_sha is None:
        raise ValueError(
            "Approval manifest change_context.commit_sha is required by --require-commit-sha."
        )
    if require_branch_name and branch_name is None:
        raise ValueError(
            "Approval manifest change_context.branch_name is required by --require-branch-name."
        )

    return {
        "pr_number": pr_number,
        "pr_url": pr_url,
        "commit_sha": commit_sha,
        "branch_name": branch_name,
    }


def _validated_profile_payload(
    active: dict[str, Any], *, source: str | None = None
) -> dict[str, str]:
    payload = {
        "model_profile": _validate_profile_name(
            "model_profile", str(active.get("model_profile", ""))
        ),
        "runtime_model_profile": _validate_profile_name(
            "runtime_model_profile", str(active.get("runtime_model_profile", ""))
        ),
        "ingestion_model_profile": _validate_profile_name(
            "ingestion_model_profile", str(active.get("ingestion_model_profile", ""))
        ),
        "legacy_ingestion_model_profile": _validate_profile_name(
            "legacy_ingestion_model_profile",
            str(active.get("legacy_ingestion_model_profile", "")),
        ),
    }
    if source is not None:
        payload["source"] = source
    return payload


def _muninn_api_request(
    muninn_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    url = f"{muninn_url.rstrip('/')}{path}"
    return _http_json_request(
        method=method,
        url=url,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )


def _unwrap_success_envelope(payload: dict[str, Any], *, endpoint: str) -> dict[str, Any]:
    if "success" in payload and "data" in payload:
        if payload.get("success") is not True:
            raise ValueError(f"{endpoint} returned success=false payload")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"{endpoint} expected object 'data' envelope")
        return data
    return payload


def _validate_profile_name(field_name: str, profile: str) -> str:
    candidate = str(profile or "").strip()
    if candidate not in SUPPORTED_PROFILE_NAMES:
        raise ValueError(
            f"Unsupported {field_name} '{profile}'. "
            f"Expected one of {SUPPORTED_PROFILE_NAMES}."
        )
    return candidate


def _checkpoint_target_policy(checkpoint: dict[str, Any]) -> dict[str, str]:
    target = checkpoint.get("target_policy")
    if not isinstance(target, dict):
        raise ValueError("Checkpoint is missing 'target_policy' object.")
    return _validated_profile_payload(target)


def _gate_recommendation_models(gate_report: dict[str, Any]) -> dict[str, str]:
    models: dict[str, str] = {}
    profiles = gate_report.get("profiles")
    if not isinstance(profiles, dict):
        return models
    for profile_name, payload in profiles.items():
        if not isinstance(payload, dict):
            continue
        recommendation = payload.get("recommendation")
        if not isinstance(recommendation, dict):
            continue
        model = str(recommendation.get("model", "")).strip()
        if model:
            models[profile_name] = model
    return models


def _apply_profile_policy_checkpoint(
    *,
    run_id: str,
    gate_report: dict[str, Any],
    gate_report_path: str,
    muninn_url: str,
    timeout_seconds: int,
    source: str,
    model_profile: str,
    runtime_model_profile: str,
    ingestion_model_profile: str,
    legacy_ingestion_model_profile: str,
    checkpoint_output: Path,
    dry_run: bool,
    allow_gate_failures: bool,
) -> dict[str, Any]:
    if not bool(gate_report.get("passed", False)) and not allow_gate_failures:
        raise ValueError(
            "Refusing to apply profile policy because profile gate did not pass. "
            "Use --allow-apply-when-gate-fails to override."
        )

    target_policy = {
        "model_profile": _validate_profile_name("model_profile", model_profile),
        "runtime_model_profile": _validate_profile_name(
            "runtime_model_profile", runtime_model_profile
        ),
        "ingestion_model_profile": _validate_profile_name(
            "ingestion_model_profile", ingestion_model_profile
        ),
        "legacy_ingestion_model_profile": _validate_profile_name(
            "legacy_ingestion_model_profile", legacy_ingestion_model_profile
        ),
    }

    recommendation_models = _gate_recommendation_models(gate_report)
    required_profiles = {
        target_policy["model_profile"],
        target_policy["runtime_model_profile"],
        target_policy["ingestion_model_profile"],
        target_policy["legacy_ingestion_model_profile"],
    }
    missing_recommendations = sorted(
        profile
        for profile in required_profiles
        if profile not in recommendation_models
    )
    if missing_recommendations:
        raise ValueError(
            "Refusing to apply profile policy because gate report has no passing recommendation for: "
            + ", ".join(missing_recommendations)
        )

    current_policy = _unwrap_success_envelope(
        _muninn_api_request(
            muninn_url,
            "/profiles/model",
            method="GET",
            timeout_seconds=timeout_seconds,
        ),
        endpoint="GET /profiles/model",
    )
    current_active = current_policy.get("active")
    if not isinstance(current_active, dict):
        raise ValueError("GET /profiles/model response missing 'active' policy object.")

    apply_payload = {**target_policy, "source": source}
    apply_result: dict[str, Any]
    if dry_run:
        apply_result = {
            "event": "MODEL_PROFILE_POLICY_APPLY_DRY_RUN",
            "applied": False,
            "payload": apply_payload,
        }
    else:
        response = _unwrap_success_envelope(
            _muninn_api_request(
                muninn_url,
                "/profiles/model",
                method="POST",
                payload=apply_payload,
                timeout_seconds=timeout_seconds,
            ),
            endpoint="POST /profiles/model",
        )
        apply_result = {
            "event": "MODEL_PROFILE_POLICY_APPLIED",
            "applied": True,
            "payload": apply_payload,
            "response": response,
        }

    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "run_id": run_id,
        "event": "MODEL_PROFILE_POLICY_CHECKPOINT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_report_passed": bool(gate_report.get("passed", False)),
        "gate_report_path": gate_report_path,
        "muninn_url": muninn_url,
        "source": source,
        "target_policy": target_policy,
        "recommendation_models": recommendation_models,
        "previous_policy": current_policy,
        "apply_result": apply_result,
    }
    with checkpoint_output.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, indent=2)
        f.write("\n")

    return {
        "event": "MODEL_PROFILE_POLICY_CHECKPOINT_WRITTEN",
        "checkpoint_path": str(checkpoint_output),
        "target_policy": target_policy,
        "recommendation_models": recommendation_models,
        "applied": bool(apply_result.get("applied", False)),
        "apply_result": apply_result,
    }


def cmd_rollback_policy(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    muninn_url = str(args.muninn_url).strip()
    timeout_seconds = int(args.muninn_timeout_seconds)
    source = str(args.source).strip() or "rollback_policy_cli"
    dry_run = bool(args.dry_run)

    checkpoint = _load_json(checkpoint_path)
    previous_policy = checkpoint.get("previous_policy")
    if not isinstance(previous_policy, dict):
        raise ValueError("Checkpoint is missing 'previous_policy' object.")
    active = previous_policy.get("active")
    if not isinstance(active, dict):
        raise ValueError("Checkpoint previous_policy is missing 'active' policy object.")

    restore_payload = _validated_profile_payload(active, source=source)

    rollback_result: dict[str, Any]
    if dry_run:
        rollback_result = {
            "event": "MODEL_PROFILE_POLICY_ROLLBACK_DRY_RUN",
            "applied": False,
            "payload": restore_payload,
        }
    else:
        response = _unwrap_success_envelope(
            _muninn_api_request(
                muninn_url,
                "/profiles/model",
                method="POST",
                payload=restore_payload,
                timeout_seconds=timeout_seconds,
            ),
            endpoint="POST /profiles/model",
        )
        rollback_result = {
            "event": "MODEL_PROFILE_POLICY_ROLLBACK_APPLIED",
            "applied": True,
            "payload": restore_payload,
            "response": response,
        }

    time_ctx = _timestamp_context()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"policy_rollback_{time_ctx['run_id']}.json"
    )
    payload = {
        "run_id": time_ctx["run_id"],
        "event": "MODEL_PROFILE_POLICY_ROLLBACK_COMPLETED",
        "created_at_utc": time_ctx["created_at_utc"],
        "checkpoint_path": str(checkpoint_path),
        "muninn_url": muninn_url,
        "result": rollback_result,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Policy rollback report written to: {output_path}")
    return 0


def cmd_approval_manifest(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = _load_json(checkpoint_path)
    _checkpoint_target_policy(checkpoint)

    decision = str(args.decision).strip().lower()
    if decision not in {"approved", "rejected"}:
        raise ValueError("--decision must be either 'approved' or 'rejected'.")

    approved_by = str(args.approved_by).strip()
    if not approved_by:
        raise ValueError("--approved-by must be non-empty.")

    change_context = _build_change_context(
        pr_number=getattr(args, "pr_number", None),
        pr_url=getattr(args, "pr_url", None),
        commit_sha=getattr(args, "commit_sha", None),
        branch_name=getattr(args, "branch_name", None),
    )

    time_ctx = _timestamp_context()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"policy_approval_{time_ctx['run_id']}.json"
    )

    payload = {
        "run_id": time_ctx["run_id"],
        "event": "MODEL_PROFILE_POLICY_APPROVAL_RECORDED",
        "created_at_utc": time_ctx["created_at_utc"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "decision": decision,
        "approved_by": approved_by,
        "notes": str(args.notes or "").strip(),
        "source": str(args.source).strip() or "policy_approval_cli",
        "change_context": change_context,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Policy approval manifest written to: {output_path}")
    return 0


def cmd_apply_checkpoint(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.approval_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    muninn_url = str(args.muninn_url).strip()
    timeout_seconds = int(args.muninn_timeout_seconds)
    source = str(args.source).strip() or "apply_checkpoint_cli"
    dry_run = bool(args.dry_run)
    require_change_context = bool(getattr(args, "require_change_context", False))
    require_pr_number = bool(getattr(args, "require_pr_number", False))
    require_commit_sha = bool(getattr(args, "require_commit_sha", False))
    require_branch_name = bool(getattr(args, "require_branch_name", False))
    require_commit_reachable_from = str(
        getattr(args, "require_commit_reachable_from", "") or ""
    ).strip()

    checkpoint = _load_json(checkpoint_path)
    target_policy = _checkpoint_target_policy(checkpoint)
    manifest = _load_json(manifest_path)

    decision = str(manifest.get("decision", "")).strip().lower()
    if decision != "approved":
        raise ValueError(
            f"Approval manifest decision is '{decision or 'missing'}'; expected 'approved'."
        )

    recorded_sha = str(manifest.get("checkpoint_sha256", "")).strip().lower()
    actual_sha = _sha256_file(checkpoint_path)
    if recorded_sha != actual_sha:
        raise ValueError("Approval manifest checkpoint_sha256 does not match checkpoint file.")

    recorded_path = str(manifest.get("checkpoint_path", "")).strip()
    if recorded_path and Path(recorded_path).resolve() != checkpoint_path:
        raise ValueError("Approval manifest checkpoint_path does not match provided checkpoint.")

    change_context = _validated_manifest_change_context(
        manifest.get("change_context"),
        require_change_context=require_change_context,
        require_pr_number=require_pr_number,
        require_commit_sha=require_commit_sha,
        require_branch_name=require_branch_name,
    )
    if require_commit_reachable_from:
        commit_sha = str(change_context.get("commit_sha") or "").strip()
        if not commit_sha:
            raise ValueError(
                "Approval manifest change_context.commit_sha is required when "
                "--require-commit-reachable-from is set."
            )
        reachable, verify_error = _git_commit_reachable_from(
            commit_sha, require_commit_reachable_from
        )
        if verify_error:
            raise ValueError(
                "Unable to verify commit ancestry for --require-commit-reachable-from. "
                f"{verify_error}"
            )
        if not reachable:
            raise ValueError(
                "Approval manifest change_context.commit_sha "
                f"'{commit_sha}' is not reachable from '{require_commit_reachable_from}'."
            )

    apply_payload = {**target_policy, "source": source}
    apply_result: dict[str, Any]
    if dry_run:
        apply_result = {
            "event": "MODEL_PROFILE_POLICY_APPLY_CHECKPOINT_DRY_RUN",
            "applied": False,
            "payload": apply_payload,
        }
    else:
        response = _unwrap_success_envelope(
            _muninn_api_request(
                muninn_url,
                "/profiles/model",
                method="POST",
                payload=apply_payload,
                timeout_seconds=timeout_seconds,
            ),
            endpoint="POST /profiles/model",
        )
        apply_result = {
            "event": "MODEL_PROFILE_POLICY_APPLY_CHECKPOINT_COMPLETED",
            "applied": True,
            "payload": apply_payload,
            "response": response,
        }

    time_ctx = _timestamp_context()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"policy_apply_checkpoint_{time_ctx['run_id']}.json"
    )
    payload = {
        "run_id": time_ctx["run_id"],
        "event": "MODEL_PROFILE_POLICY_CHECKPOINT_APPLY_RECORDED",
        "created_at_utc": time_ctx["created_at_utc"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_sha,
        "approval_manifest_path": str(manifest_path),
        "approved_by": str(manifest.get("approved_by", "")),
        "change_context": change_context,
        "muninn_url": muninn_url,
        "result": apply_result,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Checkpoint apply report written to: {output_path}")
    return 0


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
        "suite": "live",
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
                        "response_chars": len(str(result.get("response", "") or "")),
                        "response_preview": str(result.get("response", "") or "")[:320],
                        "ability_score": None,
                        "ability_checks": {},
                        "error": None,
                    }
                )
                ability_score, ability_details = _score_live_response(
                    prompt=prompt,
                    response_text=str(result.get("response", "") or ""),
                )
                model_runs[-1]["ability_score"] = round(float(ability_score), 6)
                model_runs[-1]["ability_checks"] = ability_details

        successful = [run for run in model_runs if run.get("error") is None]
        latencies = [float(run["wall_seconds"]) for run in successful]
        tps_values = [float(run["eval_tokens_per_second"]) for run in successful if run["eval_tokens_per_second"] > 0]
        ability_scores = [
            float(run["ability_score"])
            for run in successful
            if isinstance(run.get("ability_score"), (int, float))
        ]
        pass_threshold = float(args.ability_pass_threshold)
        passed_runs = sum(1 for score in ability_scores if score >= pass_threshold)
        matrix_entry = _entry_for_model(matrix, model)

        summary = {
            "total_runs": len(model_runs),
            "successful_runs": len(successful),
            "failed_runs": len(model_runs) - len(successful),
            "avg_wall_seconds": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
            "p95_wall_seconds": round(_percentile(latencies, 0.95), 6) if latencies else 0.0,
            "avg_eval_tokens_per_second": round(sum(tps_values) / len(tps_values), 4)
            if tps_values
            else 0.0,
            "avg_ability_score": round(sum(ability_scores) / len(ability_scores), 6)
            if ability_scores
            else 0.0,
            "ability_pass_threshold": pass_threshold,
            "ability_pass_rate": round((passed_runs / len(ability_scores)), 6) if ability_scores else 0.0,
        }
        summary["resource_efficiency"] = _resource_efficiency(summary, matrix_entry)

        report["models"][model] = {
            "runs": model_runs,
            "summary": summary,
            "matrix_entry": matrix_entry,
        }

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"Benchmark report written to: {output_path}")
    return 0


def _parse_legacy_roots(raw: str) -> list[Path]:
    roots = [Path(item.strip()).resolve() for item in raw.split(",") if item.strip()]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def cmd_legacy_benchmark(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix).resolve()
    output_dir = Path(args.output_dir).resolve()
    ollama_url = args.ollama_url.strip()
    timeout_seconds = int(args.timeout_seconds)
    num_predict = int(args.num_predict)
    repeats = int(args.repeats)
    max_cases_per_root = int(args.max_cases_per_root)
    snippet_chars = int(args.snippet_chars)
    pass_threshold = float(args.ability_pass_threshold)

    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if num_predict < 1:
        raise ValueError("--num-predict must be >= 1")
    if max_cases_per_root < 1:
        raise ValueError("--max-cases-per-root must be >= 1")
    if snippet_chars < 256:
        raise ValueError("--snippet-chars must be >= 256")

    roots = _parse_legacy_roots(args.legacy_roots)
    if not roots:
        raise ValueError("--legacy-roots must include at least one directory path")

    matrix = _load_json(matrix_path)
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

    cases = _build_legacy_cases(
        roots=roots,
        max_cases_per_root=max_cases_per_root,
        snippet_chars=snippet_chars,
    )
    if not cases:
        print("No legacy benchmark cases were generated from --legacy-roots.", file=sys.stderr)
        return 1

    try:
        ollama_version = _post_version(ollama_url, timeout_seconds)
    except error.URLError as exc:
        print(f"Ollama endpoint unavailable at {ollama_url}: {exc}", file=sys.stderr)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else output_dir / f"legacy_report_{run_id}.json"

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "legacy_ingestion",
        "ollama_url": ollama_url,
        "ollama_version": ollama_version,
        "matrix_path": str(matrix_path),
        "legacy_roots": [str(root) for root in roots],
        "legacy_case_count": len(cases),
        "repeats": repeats,
        "num_predict": num_predict,
        "models": {},
    }

    for model in selected:
        print(f"[legacy-benchmark] model={model}")
        model_runs: list[dict[str, Any]] = []
        for case in cases:
            for rep in range(repeats):
                payload = {
                    "model": model,
                    "prompt": case["prompt"],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": num_predict},
                }
                wall_start = time.perf_counter()
                try:
                    result = _post_generate(ollama_url, payload, timeout_seconds)
                except Exception as exc:  # noqa: BLE001
                    model_runs.append(
                        {
                            "case_id": case["id"],
                            "category": case["category"],
                            "source_path": case["source_path"],
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

                response_text = str(result.get("response", "") or "")
                ability_score, ability_details = _score_legacy_response(case, response_text)
                model_runs.append(
                    {
                        "case_id": case["id"],
                        "category": case["category"],
                        "source_path": case["source_path"],
                        "expected_keywords": case.get("expected_keywords", []),
                        "repeat_index": rep,
                        "wall_seconds": round(wall_seconds, 6),
                        "total_seconds_api": round(
                            (int(result.get("total_duration", 0) or 0) / 1_000_000_000.0), 6
                        ),
                        "prompt_eval_count": int(result.get("prompt_eval_count", 0) or 0),
                        "eval_count": eval_count,
                        "eval_tokens_per_second": round(eval_tps, 4),
                        "done_reason": result.get("done_reason"),
                        "response_chars": len(response_text),
                        "response_preview": response_text[:320],
                        "ability_score": round(float(ability_score), 6),
                        "ability_checks": ability_details,
                        "error": None,
                    }
                )

        successful = [run for run in model_runs if run.get("error") is None]
        latencies = [float(run["wall_seconds"]) for run in successful]
        tps_values = [float(run["eval_tokens_per_second"]) for run in successful if run["eval_tokens_per_second"] > 0]
        ability_scores = [
            float(run["ability_score"])
            for run in successful
            if isinstance(run.get("ability_score"), (int, float))
        ]
        passed_runs = sum(1 for score in ability_scores if score >= pass_threshold)
        matrix_entry = _entry_for_model(matrix, model)

        summary = {
            "total_runs": len(model_runs),
            "successful_runs": len(successful),
            "failed_runs": len(model_runs) - len(successful),
            "avg_wall_seconds": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
            "p95_wall_seconds": round(_percentile(latencies, 0.95), 6) if latencies else 0.0,
            "avg_eval_tokens_per_second": round(sum(tps_values) / len(tps_values), 4)
            if tps_values
            else 0.0,
            "avg_ability_score": round(sum(ability_scores) / len(ability_scores), 6)
            if ability_scores
            else 0.0,
            "ability_pass_threshold": pass_threshold,
            "ability_pass_rate": round((passed_runs / len(ability_scores)), 6) if ability_scores else 0.0,
        }
        summary["resource_efficiency"] = _resource_efficiency(summary, matrix_entry)

        report["models"][model] = {
            "runs": model_runs,
            "summary": summary,
            "matrix_entry": matrix_entry,
        }

    report["cases_preview"] = [
        {
            "id": case["id"],
            "source_path": case["source_path"],
            "expected_keywords": case.get("expected_keywords", []),
        }
        for case in cases[:20]
    ]
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    if args.dump_cases:
        cases_output = Path(args.dump_cases).resolve()
        with cases_output.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"Legacy case corpus written to: {cases_output}")

    print(f"Legacy benchmark report written to: {output_path}")
    return 0


def cmd_profile_gate(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix).resolve()
    policy_path = Path(args.policy).resolve()
    live_report_path = Path(args.live_report).resolve()
    legacy_report_path = Path(args.legacy_report).resolve() if args.legacy_report else None
    output_dir = Path(args.output_dir).resolve()

    matrix = _load_json(matrix_path)
    policy = _load_promotion_policy(policy_path)
    live_report = _load_json(live_report_path)
    legacy_report = _load_json(legacy_report_path) if legacy_report_path else {}

    start_time = datetime.now(timezone.utc)
    run_id = start_time.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).resolve() if args.output else output_dir / f"profile_gate_{run_id}.json"
    )

    profiles = policy.get("profiles", {})
    result_profiles: dict[str, Any] = {}
    any_failures = False

    for profile_name, gate in profiles.items():
        if not isinstance(gate, dict):
            raise ValueError(f"Policy profile '{profile_name}' must be an object.")
        candidates = gate.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Policy profile '{profile_name}' requires non-empty candidates list.")

        candidate_results: list[dict[str, Any]] = []
        for model in [str(item).strip() for item in candidates if str(item).strip()]:
            live_summary = _suite_model_summary(live_report, model)
            legacy_summary = _suite_model_summary(legacy_report, model) if legacy_report else None
            evaluated = _evaluate_candidate(
                model=model,
                gate=gate,
                live_summary=live_summary,
                legacy_summary=legacy_summary,
            )
            matrix_entry = _entry_for_model(matrix, model)
            if matrix_entry:
                evaluated["matrix_entry"] = matrix_entry
            candidate_results.append(evaluated)

        recommendation = _pick_recommendation(candidate_results)
        profile_passed = recommendation is not None
        if not profile_passed:
            any_failures = True

        result_profiles[profile_name] = {
            "gate": gate,
            "passed": profile_passed,
            "recommendation": recommendation,
            "candidates": candidate_results,
        }

    output_payload = {
        "run_id": run_id,
        "event": "PROFILE_PROMOTION_GATE_EVALUATED",
        "started_at_utc": start_time.isoformat(),
        "matrix_path": str(matrix_path),
        "policy_path": str(policy_path),
        "live_report_path": str(live_report_path),
        "legacy_report_path": str(legacy_report_path) if legacy_report_path else None,
        "passed": not any_failures,
        "profiles": result_profiles,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
        f.write("\n")

    for profile_name, profile_result in result_profiles.items():
        recommendation = profile_result.get("recommendation")
        if recommendation:
            print(
                f"[gate] {profile_name}: PASS "
                f"-> {recommendation['model']} (score={recommendation['composite_score']})"
            )
        else:
            print(f"[gate] {profile_name}: FAIL (no passing candidate)")

    print(f"Profile gate report written to: {output_path}")
    if any_failures and not bool(args.allow_failures):
        return 1
    return 0


def _recommendation_summary(
    *,
    profile_name: str,
    recommendation: dict[str, Any] | None,
    live_report: dict[str, Any],
    legacy_report: dict[str, Any] | None,
) -> dict[str, Any]:
    usage = PROFILE_USAGE_GUIDANCE.get(profile_name, "Profile-specific model guidance.")
    if not recommendation:
        return {
            "profile": profile_name,
            "usage": usage,
            "model": None,
            "status": "no_passing_candidate",
            "composite_score": None,
            "evidence": None,
        }

    model = str(recommendation.get("model", "")).strip()
    live_summary = _suite_model_summary(live_report, model) if model else None
    legacy_summary = _suite_model_summary(legacy_report or {}, model) if model else None

    evidence = {
        "live_avg_ability_score": _to_float((live_summary or {}).get("avg_ability_score")),
        "live_p95_wall_seconds": _to_float((live_summary or {}).get("p95_wall_seconds")),
        "live_ability_per_vram_gb": _to_float(
            ((live_summary or {}).get("resource_efficiency") or {}).get("ability_per_vram_gb")
        ),
        "legacy_avg_ability_score": _to_float((legacy_summary or {}).get("avg_ability_score")),
        "legacy_p95_wall_seconds": _to_float((legacy_summary or {}).get("p95_wall_seconds")),
    }

    return {
        "profile": profile_name,
        "usage": usage,
        "model": model,
        "status": "recommended",
        "composite_score": _to_float(recommendation.get("composite_score")),
        "evidence": evidence,
    }


def cmd_dev_cycle(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%SZ")

    live_output = Path(args.live_output).resolve() if args.live_output else output_dir / f"cycle_live_{run_id}.json"
    legacy_output = (
        Path(args.legacy_output).resolve()
        if args.legacy_output
        else output_dir / f"cycle_legacy_{run_id}.json"
    )
    gate_output = Path(args.gate_output).resolve() if args.gate_output else output_dir / f"cycle_gate_{run_id}.json"
    summary_output = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"dev_cycle_summary_{run_id}.json"
    )

    benchmark_args = argparse.Namespace(
        matrix=args.matrix,
        prompts=args.prompts,
        output_dir=str(output_dir),
        output=str(live_output),
        models=args.models,
        include_optional=bool(args.include_optional),
        repeats=int(args.repeats),
        ollama_url=args.ollama_url,
        timeout_seconds=int(args.timeout_seconds),
        num_predict=int(args.num_predict),
        ability_pass_threshold=float(args.ability_pass_threshold),
    )
    bench_rc = cmd_benchmark(benchmark_args)
    if bench_rc != 0:
        return bench_rc

    legacy_args = argparse.Namespace(
        matrix=args.matrix,
        legacy_roots=args.legacy_roots,
        output_dir=str(output_dir),
        output=str(legacy_output),
        models=args.models,
        include_optional=bool(args.include_optional),
        repeats=int(args.repeats),
        max_cases_per_root=int(args.max_cases_per_root),
        snippet_chars=int(args.snippet_chars),
        ollama_url=args.ollama_url,
        timeout_seconds=int(args.timeout_seconds),
        num_predict=int(args.num_predict),
        ability_pass_threshold=float(args.ability_pass_threshold),
        dump_cases=args.dump_cases,
    )
    legacy_rc = cmd_legacy_benchmark(legacy_args)
    if legacy_rc != 0:
        return legacy_rc

    gate_args = argparse.Namespace(
        matrix=args.matrix,
        policy=args.policy,
        live_report=str(live_output),
        legacy_report=str(legacy_output),
        output_dir=str(output_dir),
        output=str(gate_output),
        allow_failures=bool(args.allow_gate_failures),
    )
    gate_rc = cmd_profile_gate(gate_args)

    live_report = _load_json(live_output)
    legacy_report = _load_json(legacy_output)
    gate_report = _load_json(gate_output)
    profiles = gate_report.get("profiles", {})
    recommendations: dict[str, Any] = {}
    for profile_name, payload in profiles.items():
        recommendation = payload.get("recommendation") if isinstance(payload, dict) else None
        recommendations[profile_name] = _recommendation_summary(
            profile_name=profile_name,
            recommendation=recommendation if isinstance(recommendation, dict) else None,
            live_report=live_report,
            legacy_report=legacy_report,
        )

    cycle_summary = {
        "run_id": run_id,
        "event": "DEV_CYCLE_MODEL_BENCHMARK_COMPLETED",
        "started_at_utc": run_started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "live_report": str(live_output),
            "legacy_report": str(legacy_output),
            "gate_report": str(gate_output),
        },
        "passed": bool(gate_report.get("passed", False)),
        "recommendations": recommendations,
    }

    if bool(args.apply_policy):
        checkpoint_output = (
            Path(args.checkpoint_output).resolve()
            if args.checkpoint_output
            else output_dir / f"profile_policy_checkpoint_{run_id}.json"
        )
        policy_apply = _apply_profile_policy_checkpoint(
            run_id=run_id,
            gate_report=gate_report,
            gate_report_path=str(gate_output),
            muninn_url=str(args.muninn_url).strip(),
            timeout_seconds=int(args.muninn_timeout_seconds),
            source=str(args.apply_source).strip() or f"dev_cycle_{run_id}",
            model_profile=str(args.target_model_profile),
            runtime_model_profile=str(args.target_runtime_model_profile),
            ingestion_model_profile=str(args.target_ingestion_model_profile),
            legacy_ingestion_model_profile=str(args.target_legacy_ingestion_model_profile),
            checkpoint_output=checkpoint_output,
            dry_run=bool(args.apply_dry_run),
            allow_gate_failures=bool(args.allow_apply_when_gate_fails),
        )
        cycle_summary["policy_apply"] = policy_apply

    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(cycle_summary, f, indent=2)
        f.write("\n")

    print(f"Dev-cycle benchmark summary written to: {summary_output}")
    for profile_name, item in recommendations.items():
        model = item.get("model")
        usage = item.get("usage")
        if model:
            print(f"[dev-cycle] {profile_name}: {model} -> {usage}")
        else:
            print(f"[dev-cycle] {profile_name}: no recommendation -> {usage}")

    if gate_rc != 0 and not bool(args.allow_gate_failures):
        return gate_rc
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
    bench_parser.add_argument(
        "--ability-pass-threshold",
        default=0.75,
        type=float,
        help="Ability score threshold used for pass-rate summaries.",
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    legacy_parser = subparsers.add_parser(
        "legacy-benchmark",
        help="Benchmark extraction quality on legacy project snippets (old projects).",
    )
    legacy_parser.add_argument(
        "--legacy-roots",
        required=True,
        help="Comma-separated list of project root directories to build legacy benchmark cases from.",
    )
    legacy_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated benchmark reports.",
    )
    legacy_parser.add_argument(
        "--output",
        help="Explicit report output path; defaults to output-dir/legacy_report_<timestamp>.json.",
    )
    legacy_parser.add_argument(
        "--models",
        help="Comma-separated explicit model tags to benchmark (must exist in matrix and be installed).",
    )
    legacy_parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional matrix models when --models is not provided.",
    )
    legacy_parser.add_argument(
        "--repeats",
        default=1,
        type=int,
        help="Number of repeated runs per legacy case per model.",
    )
    legacy_parser.add_argument(
        "--max-cases-per-root",
        default=12,
        type=int,
        help="Maximum generated benchmark cases per root directory.",
    )
    legacy_parser.add_argument(
        "--snippet-chars",
        default=1800,
        type=int,
        help="Maximum chars read from each source file when generating legacy cases.",
    )
    legacy_parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for local Ollama server.",
    )
    legacy_parser.add_argument(
        "--timeout-seconds",
        default=300,
        type=int,
        help="HTTP timeout for a single generation request.",
    )
    legacy_parser.add_argument(
        "--num-predict",
        default=192,
        type=int,
        help="Maximum generated tokens per request.",
    )
    legacy_parser.add_argument(
        "--ability-pass-threshold",
        default=0.75,
        type=float,
        help="Ability score threshold used for pass-rate summaries.",
    )
    legacy_parser.add_argument(
        "--dump-cases",
        help="Optional JSONL output path for generated legacy benchmark cases.",
    )
    legacy_parser.set_defaults(func=cmd_legacy_benchmark)

    gate_parser = subparsers.add_parser(
        "profile-gate",
        help="Evaluate profile-promotion gates from live and legacy benchmark reports.",
    )
    gate_parser.add_argument(
        "--policy",
        default=str(DEFAULT_PROMOTION_POLICY_PATH),
        help="Path to profile-promotion gate policy JSON.",
    )
    gate_parser.add_argument(
        "--live-report",
        required=True,
        help="Path to benchmark report produced by the 'benchmark' command.",
    )
    gate_parser.add_argument(
        "--legacy-report",
        help="Path to benchmark report produced by the 'legacy-benchmark' command.",
    )
    gate_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated gate reports.",
    )
    gate_parser.add_argument(
        "--output",
        help="Explicit gate report output path; defaults to output-dir/profile_gate_<timestamp>.json.",
    )
    gate_parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success exit code even when one or more profile gates fail.",
    )
    gate_parser.set_defaults(func=cmd_profile_gate)

    cycle_parser = subparsers.add_parser(
        "dev-cycle",
        help="Run live+legacy benchmarking and profile-gate in one operator-triggered development cycle.",
    )
    cycle_parser.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS_PATH),
        help="Path to benchmark prompts JSONL.",
    )
    cycle_parser.add_argument(
        "--policy",
        default=str(DEFAULT_PROMOTION_POLICY_PATH),
        help="Path to profile-promotion gate policy JSON.",
    )
    cycle_parser.add_argument(
        "--legacy-roots",
        required=True,
        help="Comma-separated legacy project roots for ingestion-like benchmark cases.",
    )
    cycle_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated reports.",
    )
    cycle_parser.add_argument("--output", help="Optional summary output path.")
    cycle_parser.add_argument("--live-output", help="Optional live benchmark report output path.")
    cycle_parser.add_argument("--legacy-output", help="Optional legacy benchmark report output path.")
    cycle_parser.add_argument("--gate-output", help="Optional profile-gate report output path.")
    cycle_parser.add_argument("--models", help="Comma-separated explicit model tags to benchmark.")
    cycle_parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional matrix models when --models is not provided.",
    )
    cycle_parser.add_argument("--repeats", default=1, type=int, help="Repeated runs per case.")
    cycle_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Local Ollama base URL.")
    cycle_parser.add_argument(
        "--timeout-seconds",
        default=300,
        type=int,
        help="HTTP timeout for each generation request.",
    )
    cycle_parser.add_argument(
        "--num-predict",
        default=192,
        type=int,
        help="Maximum generated tokens per request.",
    )
    cycle_parser.add_argument(
        "--ability-pass-threshold",
        default=0.75,
        type=float,
        help="Ability score threshold used for pass-rate summaries.",
    )
    cycle_parser.add_argument(
        "--max-cases-per-root",
        default=12,
        type=int,
        help="Maximum generated legacy benchmark cases per root directory.",
    )
    cycle_parser.add_argument(
        "--snippet-chars",
        default=1800,
        type=int,
        help="Maximum chars read from each source file when generating legacy cases.",
    )
    cycle_parser.add_argument(
        "--dump-cases",
        help="Optional JSONL output path for generated legacy benchmark cases.",
    )
    cycle_parser.add_argument(
        "--allow-gate-failures",
        action="store_true",
        help="Return success even if profile gate fails one or more profiles.",
    )
    cycle_parser.add_argument(
        "--apply-policy",
        action="store_true",
        help="Apply profile defaults to running Muninn server when gate evidence is acceptable.",
    )
    cycle_parser.add_argument(
        "--apply-dry-run",
        action="store_true",
        help="Write checkpoint payload without posting profile update to server.",
    )
    cycle_parser.add_argument(
        "--allow-apply-when-gate-fails",
        action="store_true",
        help="Allow policy apply even when profile gate report failed.",
    )
    cycle_parser.add_argument(
        "--muninn-url",
        default=DEFAULT_MUNINN_URL,
        help="Muninn server base URL for profile policy apply actions.",
    )
    cycle_parser.add_argument(
        "--muninn-timeout-seconds",
        default=20,
        type=int,
        help="HTTP timeout for profile policy apply requests.",
    )
    cycle_parser.add_argument(
        "--apply-source",
        default="dev_cycle_cli",
        help="Audit source label used when applying profile policy.",
    )
    cycle_parser.add_argument(
        "--checkpoint-output",
        help="Optional explicit output path for policy-apply checkpoint artifact.",
    )
    cycle_parser.add_argument(
        "--target-model-profile",
        default="balanced",
        help="Target default extraction profile to apply when --apply-policy is used.",
    )
    cycle_parser.add_argument(
        "--target-runtime-model-profile",
        default="low_latency",
        help="Target runtime profile to apply when --apply-policy is used.",
    )
    cycle_parser.add_argument(
        "--target-ingestion-model-profile",
        default="balanced",
        help="Target ingestion profile to apply when --apply-policy is used.",
    )
    cycle_parser.add_argument(
        "--target-legacy-ingestion-model-profile",
        default="balanced",
        help="Target legacy-ingestion profile to apply when --apply-policy is used.",
    )
    cycle_parser.set_defaults(func=cmd_dev_cycle)

    rollback_parser = subparsers.add_parser(
        "rollback-policy",
        help="Rollback profile policy from a previously written apply checkpoint artifact.",
    )
    rollback_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint written by dev-cycle --apply-policy flow.",
    )
    rollback_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated rollback reports.",
    )
    rollback_parser.add_argument(
        "--output",
        help="Optional rollback report output path.",
    )
    rollback_parser.add_argument(
        "--muninn-url",
        default=DEFAULT_MUNINN_URL,
        help="Muninn server base URL for rollback request.",
    )
    rollback_parser.add_argument(
        "--muninn-timeout-seconds",
        default=20,
        type=int,
        help="HTTP timeout for rollback request.",
    )
    rollback_parser.add_argument(
        "--source",
        default="rollback_policy_cli",
        help="Audit source label used when rolling back profile policy.",
    )
    rollback_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write rollback report without posting rollback request to server.",
    )
    rollback_parser.set_defaults(func=cmd_rollback_policy)

    approval_parser = subparsers.add_parser(
        "approval-manifest",
        help="Create approval/rejection manifest for a profile-policy checkpoint.",
    )
    approval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to profile-policy checkpoint artifact.",
    )
    approval_parser.add_argument(
        "--decision",
        required=True,
        choices=["approved", "rejected"],
        help="Approval decision for the checkpoint.",
    )
    approval_parser.add_argument(
        "--approved-by",
        required=True,
        help="Reviewer or operator identifier approving/rejecting the checkpoint.",
    )
    approval_parser.add_argument(
        "--notes",
        help="Optional rationale notes for approval/rejection.",
    )
    approval_parser.add_argument(
        "--pr-number",
        type=int,
        help="Optional PR number associated with this approval decision.",
    )
    approval_parser.add_argument(
        "--pr-url",
        help="Optional PR URL associated with this approval decision.",
    )
    approval_parser.add_argument(
        "--commit-sha",
        help="Optional commit SHA associated with this approval decision.",
    )
    approval_parser.add_argument(
        "--branch-name",
        help="Optional branch name associated with this approval decision.",
    )
    approval_parser.add_argument(
        "--source",
        default="policy_approval_cli",
        help="Audit source label for the approval decision.",
    )
    approval_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated approval manifests.",
    )
    approval_parser.add_argument(
        "--output",
        help="Optional approval manifest output path.",
    )
    approval_parser.set_defaults(func=cmd_approval_manifest)

    apply_checkpoint_parser = subparsers.add_parser(
        "apply-checkpoint",
        help="Apply profile policy from checkpoint using an approved manifest.",
    )
    apply_checkpoint_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint written by dev-cycle --apply-policy flow.",
    )
    apply_checkpoint_parser.add_argument(
        "--approval-manifest",
        required=True,
        help="Path to approval manifest created by approval-manifest command.",
    )
    apply_checkpoint_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated apply reports.",
    )
    apply_checkpoint_parser.add_argument(
        "--output",
        help="Optional checkpoint apply report output path.",
    )
    apply_checkpoint_parser.add_argument(
        "--muninn-url",
        default=DEFAULT_MUNINN_URL,
        help="Muninn server base URL for checkpoint apply request.",
    )
    apply_checkpoint_parser.add_argument(
        "--muninn-timeout-seconds",
        default=20,
        type=int,
        help="HTTP timeout for checkpoint apply request.",
    )
    apply_checkpoint_parser.add_argument(
        "--source",
        default="apply_checkpoint_cli",
        help="Audit source label used when applying checkpoint.",
    )
    apply_checkpoint_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write apply report without posting apply request to server.",
    )
    apply_checkpoint_parser.add_argument(
        "--require-change-context",
        action="store_true",
        help="Require approval manifest to include change_context object.",
    )
    apply_checkpoint_parser.add_argument(
        "--require-pr-number",
        action="store_true",
        help="Require approval manifest change_context.pr_number.",
    )
    apply_checkpoint_parser.add_argument(
        "--require-commit-sha",
        action="store_true",
        help="Require approval manifest change_context.commit_sha.",
    )
    apply_checkpoint_parser.add_argument(
        "--require-branch-name",
        action="store_true",
        help="Require approval manifest change_context.branch_name.",
    )
    apply_checkpoint_parser.add_argument(
        "--require-commit-reachable-from",
        help="Require manifest commit SHA to be reachable from this git ref/branch.",
    )
    apply_checkpoint_parser.set_defaults(func=cmd_apply_checkpoint)

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
