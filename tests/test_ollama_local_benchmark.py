from __future__ import annotations

from pathlib import Path

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
