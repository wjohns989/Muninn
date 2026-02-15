from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from eval import ollama_local_benchmark as bench


def test_extract_json_object_from_fenced_block() -> None:
    text = """```json
{"summary":"ok","entities":["RouterPolicy"]}
```"""
    payload = bench._extract_json_object(text)
    assert payload == {"summary": "ok", "entities": ["RouterPolicy"]}


def test_score_live_response_with_rubric_checks() -> None:
    prompt = {
        "required_json_keys": ["goal", "constraints", "risks", "assumptions"],
        "required_substrings": ["goal", "constraints"],
    }
    response = '{"goal":"g","constraints":["c"],"risks":["r"],"assumptions":["a"]}'
    score, details = bench._score_live_response(prompt, response)
    assert score >= 0.95
    assert details["checks"]["required_json_keys"] == 1.0
    assert details["checks"]["required_substrings"] == 1.0


def test_build_legacy_cases_from_project_roots(tmp_path: Path) -> None:
    project = tmp_path / "legacy_project"
    project.mkdir(parents=True, exist_ok=True)
    source = project / "router_policy.py"
    source.write_text(
        (
            "def choose_runtime_profile(vram_budget_gb, workload_mode):\n"
            "    profile_policy = 'low_latency' if workload_mode == 'coding' else 'balanced'\n"
            "    return profile_policy\n"
            "class ProfileAuditEvent:\n"
            "    pass\n"
        ),
        encoding="utf-8",
    )

    cases = bench._build_legacy_cases(
        roots=[project],
        max_cases_per_root=5,
        snippet_chars=2000,
    )
    assert cases
    case = cases[0]
    assert case["category"] == "legacy_ingestion"
    assert "source_path" in case
    assert len(case["expected_keywords"]) >= 3


def test_score_legacy_response_keyword_coverage() -> None:
    case = {
        "required_json_keys": ["summary", "entities", "risks", "action_items"],
        "expected_keywords": ["profile", "latency", "policy"],
        "min_words": 6,
    }
    response = (
        '{"summary":"Profile policy selects low latency runtime mode.",'
        '"entities":["RuntimeProfilePolicy"],'
        '"risks":["Policy drift"],'
        '"action_items":["Track profile latency metrics"]}'
    )
    score, details = bench._score_legacy_response(case, response)
    assert score >= 0.8
    assert details["checks"]["required_json_keys"] == 1.0
    assert details["checks"]["keyword_coverage"] >= 2 / 3


def test_resource_efficiency_uses_vram_and_latency() -> None:
    summary = {"avg_ability_score": 0.9, "avg_wall_seconds": 3.0}
    matrix_entry = {"vram_min_gb": 6}
    efficiency = bench._resource_efficiency(summary, matrix_entry)
    assert efficiency["ability_per_second"] == 0.3
    assert efficiency["ability_per_vram_gb"] == 0.15


def test_evaluate_candidate_passes_thresholds() -> None:
    gate = {
        "require_live_suite": True,
        "require_legacy_suite": True,
        "min_live_ability": 0.7,
        "min_legacy_ability": 0.65,
        "max_live_p95_seconds": 20.0,
        "max_legacy_p95_seconds": 30.0,
        "min_live_ability_per_vram_gb": 0.05,
    }
    live_summary = {
        "avg_ability_score": 0.78,
        "p95_wall_seconds": 12.0,
        "avg_wall_seconds": 8.0,
        "resource_efficiency": {"ability_per_vram_gb": 0.11},
    }
    legacy_summary = {
        "avg_ability_score": 0.72,
        "p95_wall_seconds": 18.0,
    }
    result = bench._evaluate_candidate(
        model="qwen3:8b",
        gate=gate,
        live_summary=live_summary,
        legacy_summary=legacy_summary,
    )
    assert result["passed"] is True
    assert result["violations"] == []
    assert result["combined_ability_score"] is not None


def test_cmd_profile_gate_recommends_model(tmp_path: Path) -> None:
    matrix = {
        "models": [
            {"tag": "xlam:latest", "vram_min_gb": 1},
            {"tag": "qwen3:8b", "vram_min_gb": 6},
        ]
    }
    policy = {
        "profiles": {
            "low_latency": {
                "candidates": ["xlam:latest", "qwen3:8b"],
                "require_live_suite": True,
                "require_legacy_suite": False,
                "min_live_ability": 0.6,
                "max_live_p95_seconds": 25.0,
                "min_live_ability_per_vram_gb": 0.01,
            },
            "balanced": {
                "candidates": ["qwen3:8b"],
                "require_live_suite": True,
                "require_legacy_suite": False,
                "min_live_ability": 0.7,
                "max_live_p95_seconds": 25.0,
                "min_live_ability_per_vram_gb": 0.01,
            },
            "high_reasoning": {
                "candidates": ["qwen3:8b"],
                "require_live_suite": True,
                "require_legacy_suite": False,
                "min_live_ability": 0.7,
                "max_live_p95_seconds": 30.0,
                "min_live_ability_per_vram_gb": 0.01,
            },
        }
    }
    live_report = {
        "models": {
            "xlam:latest": {
                "summary": {
                    "avg_ability_score": 0.66,
                    "avg_wall_seconds": 4.0,
                    "p95_wall_seconds": 6.5,
                    "resource_efficiency": {"ability_per_vram_gb": 0.66},
                }
            },
            "qwen3:8b": {
                "summary": {
                    "avg_ability_score": 0.8,
                    "avg_wall_seconds": 12.0,
                    "p95_wall_seconds": 17.0,
                    "resource_efficiency": {"ability_per_vram_gb": 0.12},
                }
            },
        }
    }

    matrix_path = tmp_path / "matrix.json"
    policy_path = tmp_path / "policy.json"
    live_path = tmp_path / "live.json"
    out_path = tmp_path / "gate.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    live_path.write_text(json.dumps(live_report), encoding="utf-8")

    args = SimpleNamespace(
        matrix=str(matrix_path),
        policy=str(policy_path),
        live_report=str(live_path),
        legacy_report=None,
        output_dir=str(tmp_path),
        output=str(out_path),
        allow_failures=False,
    )
    rc = bench.cmd_profile_gate(args)
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["profiles"]["low_latency"]["recommendation"]["model"] == "xlam:latest"
