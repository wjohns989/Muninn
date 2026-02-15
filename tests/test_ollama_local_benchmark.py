from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

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


def test_evaluate_candidate_flags_missing_legacy_p95_when_required() -> None:
    gate = {
        "require_live_suite": True,
        "require_legacy_suite": True,
        "max_legacy_p95_seconds": 20.0,
    }
    live_summary = {
        "avg_ability_score": 0.8,
        "p95_wall_seconds": 10.0,
        "avg_wall_seconds": 7.0,
        "resource_efficiency": {"ability_per_vram_gb": 0.1},
    }
    legacy_summary = {"avg_ability_score": 0.8}
    result = bench._evaluate_candidate(
        model="qwen3:8b",
        gate=gate,
        live_summary=live_summary,
        legacy_summary=legacy_summary,
    )
    assert result["passed"] is False
    assert "missing_legacy_p95" in result["violations"]


def test_cmd_dev_cycle_writes_role_recommendations(tmp_path: Path, monkeypatch) -> None:
    live_output = tmp_path / "live.json"
    legacy_output = tmp_path / "legacy.json"
    gate_output = tmp_path / "gate.json"
    summary_output = tmp_path / "summary.json"

    def fake_benchmark(args: SimpleNamespace) -> int:
        payload = {
            "models": {
                "xlam:latest": {
                    "summary": {
                        "avg_ability_score": 0.7,
                        "p95_wall_seconds": 6.2,
                        "resource_efficiency": {"ability_per_vram_gb": 0.6},
                    }
                },
                "qwen3:8b": {
                    "summary": {
                        "avg_ability_score": 0.82,
                        "p95_wall_seconds": 12.0,
                        "resource_efficiency": {"ability_per_vram_gb": 0.12},
                    }
                },
            }
        }
        Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    def fake_legacy(args: SimpleNamespace) -> int:
        payload = {
            "models": {
                "xlam:latest": {"summary": {"avg_ability_score": 0.68, "p95_wall_seconds": 8.5}},
                "qwen3:8b": {"summary": {"avg_ability_score": 0.8, "p95_wall_seconds": 16.0}},
            }
        }
        Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    def fake_gate(args: SimpleNamespace) -> int:
        payload = {
            "passed": True,
            "profiles": {
                "low_latency": {"recommendation": {"model": "xlam:latest", "composite_score": 0.9}},
                "balanced": {"recommendation": {"model": "qwen3:8b", "composite_score": 0.86}},
                "high_reasoning": {"recommendation": {"model": "qwen3:8b", "composite_score": 0.88}},
            },
        }
        Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    monkeypatch.setattr(bench, "cmd_benchmark", fake_benchmark)
    monkeypatch.setattr(bench, "cmd_legacy_benchmark", fake_legacy)
    monkeypatch.setattr(bench, "cmd_profile_gate", fake_gate)

    args = SimpleNamespace(
        matrix=str(tmp_path / "matrix.json"),
        prompts=str(tmp_path / "prompts.jsonl"),
        policy=str(tmp_path / "policy.json"),
        legacy_roots=str(tmp_path),
        output_dir=str(tmp_path),
        output=str(summary_output),
        live_output=str(live_output),
        legacy_output=str(legacy_output),
        gate_output=str(gate_output),
        models="xlam:latest,qwen3:8b",
        include_optional=False,
        repeats=1,
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=30,
        num_predict=64,
        ability_pass_threshold=0.75,
        max_cases_per_root=5,
        snippet_chars=800,
        dump_cases=None,
        allow_gate_failures=False,
        apply_policy=False,
        apply_dry_run=False,
        allow_apply_when_gate_fails=False,
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        apply_source="dev_cycle_cli",
        checkpoint_output=None,
        target_model_profile="balanced",
        target_runtime_model_profile="low_latency",
        target_ingestion_model_profile="balanced",
        target_legacy_ingestion_model_profile="balanced",
    )

    rc = bench.cmd_dev_cycle(args)
    assert rc == 0
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["recommendations"]["low_latency"]["model"] == "xlam:latest"
    assert "Runtime helper" in summary["recommendations"]["low_latency"]["usage"]
    assert summary["recommendations"]["balanced"]["model"] == "qwen3:8b"


def test_cmd_dev_cycle_apply_policy_writes_checkpoint(tmp_path: Path, monkeypatch) -> None:
    live_output = tmp_path / "live_apply.json"
    legacy_output = tmp_path / "legacy_apply.json"
    gate_output = tmp_path / "gate_apply.json"
    summary_output = tmp_path / "summary_apply.json"
    checkpoint_output = tmp_path / "checkpoint_apply.json"

    def fake_benchmark(args: SimpleNamespace) -> int:
        payload = {
            "models": {
                "xlam:latest": {
                    "summary": {
                        "avg_ability_score": 0.72,
                        "p95_wall_seconds": 6.0,
                        "resource_efficiency": {"ability_per_vram_gb": 0.6},
                    }
                },
                "qwen3:8b": {
                    "summary": {
                        "avg_ability_score": 0.84,
                        "p95_wall_seconds": 11.0,
                        "resource_efficiency": {"ability_per_vram_gb": 0.12},
                    }
                },
            }
        }
        Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    def fake_legacy(args: SimpleNamespace) -> int:
        payload = {
            "models": {
                "xlam:latest": {"summary": {"avg_ability_score": 0.68, "p95_wall_seconds": 8.0}},
                "qwen3:8b": {"summary": {"avg_ability_score": 0.8, "p95_wall_seconds": 15.0}},
            }
        }
        Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    def fake_gate(args: SimpleNamespace) -> int:
        payload = {
            "passed": True,
            "profiles": {
                "low_latency": {"recommendation": {"model": "xlam:latest", "composite_score": 0.91}},
                "balanced": {"recommendation": {"model": "qwen3:8b", "composite_score": 0.86}},
                "high_reasoning": {"recommendation": {"model": "qwen3:8b", "composite_score": 0.88}},
            },
        }
        Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_muninn_request(
        muninn_url: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
        timeout_seconds: int = 20,
    ) -> dict[str, object]:
        calls.append((method, path, payload))
        if method == "GET" and path == "/profiles/model":
            return {
                "success": True,
                "data": {
                    "active": {
                        "model_profile": "balanced",
                        "runtime_model_profile": "low_latency",
                        "ingestion_model_profile": "balanced",
                        "legacy_ingestion_model_profile": "balanced",
                    }
                },
            }
        if method == "POST" and path == "/profiles/model":
            assert payload is not None
            assert payload["runtime_model_profile"] == "low_latency"
            assert payload["ingestion_model_profile"] == "balanced"
            return {
                "success": True,
                "data": {"event": "MODEL_PROFILE_POLICY_UPDATED", "audit_event_id": 7},
            }
        raise AssertionError(f"Unexpected request {method} {path}")

    monkeypatch.setattr(bench, "cmd_benchmark", fake_benchmark)
    monkeypatch.setattr(bench, "cmd_legacy_benchmark", fake_legacy)
    monkeypatch.setattr(bench, "cmd_profile_gate", fake_gate)
    monkeypatch.setattr(bench, "_muninn_api_request", fake_muninn_request)

    args = SimpleNamespace(
        matrix=str(tmp_path / "matrix.json"),
        prompts=str(tmp_path / "prompts.jsonl"),
        policy=str(tmp_path / "policy.json"),
        legacy_roots=str(tmp_path),
        output_dir=str(tmp_path),
        output=str(summary_output),
        live_output=str(live_output),
        legacy_output=str(legacy_output),
        gate_output=str(gate_output),
        models="xlam:latest,qwen3:8b",
        include_optional=False,
        repeats=1,
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=30,
        num_predict=64,
        ability_pass_threshold=0.75,
        max_cases_per_root=5,
        snippet_chars=800,
        dump_cases=None,
        allow_gate_failures=False,
        apply_policy=True,
        apply_dry_run=False,
        allow_apply_when_gate_fails=False,
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        apply_source="dev_cycle_test",
        checkpoint_output=str(checkpoint_output),
        target_model_profile="balanced",
        target_runtime_model_profile="low_latency",
        target_ingestion_model_profile="balanced",
        target_legacy_ingestion_model_profile="balanced",
    )

    rc = bench.cmd_dev_cycle(args)
    assert rc == 0
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["policy_apply"]["applied"] is True
    checkpoint = json.loads(checkpoint_output.read_text(encoding="utf-8"))
    assert checkpoint["target_policy"]["runtime_model_profile"] == "low_latency"
    assert checkpoint["apply_result"]["applied"] is True
    assert any(method == "POST" and path == "/profiles/model" for method, path, _ in calls)


def test_cmd_rollback_policy_applies_previous_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "policy_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "previous_policy": {
                    "active": {
                        "model_profile": "balanced",
                        "runtime_model_profile": "low_latency",
                        "ingestion_model_profile": "balanced",
                        "legacy_ingestion_model_profile": "balanced",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_muninn_request(
        muninn_url: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
        timeout_seconds: int = 20,
    ) -> dict[str, object]:
        calls.append({"method": method, "path": path, "payload": payload})
        assert method == "POST"
        assert path == "/profiles/model"
        assert payload is not None
        assert payload["runtime_model_profile"] == "low_latency"
        return {"success": True, "data": {"event": "MODEL_PROFILE_POLICY_UPDATED"}}

    monkeypatch.setattr(bench, "_muninn_api_request", fake_muninn_request)

    rollback_output = tmp_path / "rollback_report.json"
    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        output_dir=str(tmp_path),
        output=str(rollback_output),
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        source="rollback_test",
        dry_run=False,
    )

    rc = bench.cmd_rollback_policy(args)
    assert rc == 0
    report = json.loads(rollback_output.read_text(encoding="utf-8"))
    assert report["result"]["applied"] is True
    assert calls[0]["path"] == "/profiles/model"


def test_cmd_approval_manifest_records_checkpoint_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        bench,
        "_git_output",
        lambda args: "44f2192" if args == ["rev-parse", "HEAD"] else "main",
    )

    manifest_output = tmp_path / "approval_manifest.json"
    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        decision="approved",
        approved_by="ops@example",
        notes="Reviewed benchmark evidence.",
        pr_number=27,
        pr_url="https://github.com/wjohns989/Muninn/pull/27",
        commit_sha=None,
        branch_name=None,
        source="approval_manifest_test",
        output_dir=str(tmp_path),
        output=str(manifest_output),
    )

    rc = bench.cmd_approval_manifest(args)
    assert rc == 0
    payload = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert payload["decision"] == "approved"
    assert payload["approved_by"] == "ops@example"
    assert payload["checkpoint_path"] == str(checkpoint_path.resolve())
    assert payload["checkpoint_sha256"] == bench._sha256_file(checkpoint_path.resolve())
    assert payload["source"] == "approval_manifest_test"
    assert payload["change_context"]["pr_number"] == 27
    assert payload["change_context"]["pr_url"] == "https://github.com/wjohns989/Muninn/pull/27"
    assert payload["change_context"]["commit_sha"] == "44f2192"
    assert payload["change_context"]["branch_name"] == "main"


def test_cmd_apply_checkpoint_applies_approved_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha = bench._sha256_file(checkpoint_path.resolve())
    manifest_path = tmp_path / "approval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "decision": "approved",
                "approved_by": "ops@example",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "change_context": {
                    "pr_number": 27,
                    "pr_url": "https://github.com/wjohns989/Muninn/pull/27",
                    "commit_sha": "44f2192",
                    "branch_name": "feat/phase4n-policy-approval-manifest",
                },
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_muninn_request(
        muninn_url: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
        timeout_seconds: int = 20,
    ) -> dict[str, object]:
        calls.append(
            {
                "muninn_url": muninn_url,
                "path": path,
                "method": method,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        assert path == "/profiles/model"
        assert method == "POST"
        assert payload is not None
        assert payload["model_profile"] == "balanced"
        assert payload["runtime_model_profile"] == "low_latency"
        assert payload["source"] == "apply_checkpoint_test"
        return {"success": True, "data": {"event": "MODEL_PROFILE_POLICY_UPDATED"}}

    monkeypatch.setattr(bench, "_muninn_api_request", fake_muninn_request)

    apply_output = tmp_path / "apply_report.json"
    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        approval_manifest=str(manifest_path),
        output_dir=str(tmp_path),
        output=str(apply_output),
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        source="apply_checkpoint_test",
        dry_run=False,
    )

    rc = bench.cmd_apply_checkpoint(args)
    assert rc == 0
    report = json.loads(apply_output.read_text(encoding="utf-8"))
    assert report["result"]["applied"] is True
    assert report["checkpoint_sha256"] == checkpoint_sha
    assert report["approved_by"] == "ops@example"
    assert report["change_context"]["pr_number"] == 27
    assert report["change_context"]["commit_sha"] == "44f2192"
    assert len(calls) == 1


def test_cmd_approval_manifest_rejects_invalid_commit_sha(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        decision="approved",
        approved_by="ops@example",
        notes="",
        pr_number=None,
        pr_url=None,
        commit_sha="not-a-sha",
        branch_name=None,
        source="approval_manifest_test",
        output_dir=str(tmp_path),
        output=str(tmp_path / "approval_manifest.json"),
    )
    with pytest.raises(ValueError, match="Invalid commit SHA format"):
        bench.cmd_approval_manifest(args)


def test_cmd_apply_checkpoint_rejects_non_approved_manifest(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "approval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "decision": "rejected",
                "approved_by": "ops@example",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": bench._sha256_file(checkpoint_path.resolve()),
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        approval_manifest=str(manifest_path),
        output_dir=str(tmp_path),
        output=str(tmp_path / "apply_report.json"),
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        source="apply_checkpoint_test",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="expected 'approved'"):
        bench.cmd_apply_checkpoint(args)


def test_cmd_apply_checkpoint_rejects_sha_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "approval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "decision": "approved",
                "approved_by": "ops@example",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        approval_manifest=str(manifest_path),
        output_dir=str(tmp_path),
        output=str(tmp_path / "apply_report.json"),
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        source="apply_checkpoint_test",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="checkpoint_sha256 does not match"):
        bench.cmd_apply_checkpoint(args)


def test_cmd_apply_checkpoint_rejects_path_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )
    wrong_path = tmp_path / "other_checkpoint.json"
    manifest_path = tmp_path / "approval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "decision": "approved",
                "approved_by": "ops@example",
                "checkpoint_path": str(wrong_path.resolve()),
                "checkpoint_sha256": bench._sha256_file(checkpoint_path.resolve()),
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        approval_manifest=str(manifest_path),
        output_dir=str(tmp_path),
        output=str(tmp_path / "apply_report.json"),
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        source="apply_checkpoint_test",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="checkpoint_path does not match"):
        bench.cmd_apply_checkpoint(args)


def test_cmd_apply_checkpoint_rejects_non_object_change_context(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "target_policy": {
                    "model_profile": "balanced",
                    "runtime_model_profile": "low_latency",
                    "ingestion_model_profile": "balanced",
                    "legacy_ingestion_model_profile": "balanced",
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "approval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "decision": "approved",
                "approved_by": "ops@example",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": bench._sha256_file(checkpoint_path.resolve()),
                "change_context": "invalid",
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        approval_manifest=str(manifest_path),
        output_dir=str(tmp_path),
        output=str(tmp_path / "apply_report.json"),
        muninn_url="http://127.0.0.1:42069",
        muninn_timeout_seconds=20,
        source="apply_checkpoint_test",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="change_context must be an object"):
        bench.cmd_apply_checkpoint(args)
