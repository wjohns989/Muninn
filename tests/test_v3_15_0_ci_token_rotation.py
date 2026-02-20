"""
Phase 18 (v3.15.0) — CI Benchmark Workflow & Token Rotation tests.

Covers:
  1. TestCIBenchmarkWorkflow    — .github/workflows/benchmark.yml structure & correctness
  2. TestTokenRotationCLI       — muninn.cli rotate-token command
  3. TestTokenFilePersistence   — token file write / resolution logic
  4. TestMCPConfigPatcher       — _patch_mcp_config JSON patching
  5. TestVersionBump315         — version == 3.15.0
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest


# ===========================================================================
# 1. TestCIBenchmarkWorkflow
# ===========================================================================

class TestCIBenchmarkWorkflow:
    """Validates the .github/workflows/benchmark.yml file."""

    _WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "benchmark.yml"

    def test_workflow_file_exists(self):
        assert self._WORKFLOW_PATH.exists(), (
            f"CI workflow not found at {self._WORKFLOW_PATH}. "
            "Expected .github/workflows/benchmark.yml to be created in Phase 18."
        )

    def test_workflow_is_valid_yaml(self):
        """Workflow must parse as valid YAML."""
        import yaml
        content = self._WORKFLOW_PATH.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), "Workflow YAML must parse to a dict at top level"

    def test_workflow_name_set(self):
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        assert "name" in parsed, "Workflow must have a 'name' field"
        assert parsed["name"], "Workflow name must not be empty"

    @staticmethod
    def _get_on_section(parsed: dict) -> dict:
        """Return the workflow 'on:' section.

        PyYAML safe_load uses YAML 1.1 rules where bare 'on' is a boolean True.
        GitHub Actions YAML uses 'on:' as a string key, but PyYAML will serialize
        it as True. We check both to be safe.
        """
        # PyYAML parses 'on:' as True (YAML 1.1 boolean); also try string key
        return parsed.get(True, parsed.get("on", {})) or {}

    def test_workflow_triggers_on_pull_request_to_main(self):
        """Workflow must trigger on pull_request to main branch."""
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        on_section = self._get_on_section(parsed)
        assert "pull_request" in on_section, "Workflow must trigger on pull_request"
        pr = on_section["pull_request"]
        if pr is not None:
            branches = pr.get("branches", [])
            assert "main" in branches, "pull_request trigger must target 'main' branch"

    def test_workflow_triggers_on_push_to_main(self):
        """Workflow must also trigger on push to main."""
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        on_section = self._get_on_section(parsed)
        assert "push" in on_section, "Workflow must trigger on push"
        push = on_section["push"]
        if push is not None:
            branches = push.get("branches", [])
            assert "main" in branches, "push trigger must target 'main' branch"

    def test_workflow_has_workflow_dispatch(self):
        """Workflow must support manual dispatch via workflow_dispatch."""
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        on_section = self._get_on_section(parsed)
        assert "workflow_dispatch" in on_section, "Workflow must support workflow_dispatch"

    def test_workflow_uses_ubuntu_latest(self):
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        jobs = parsed.get("jobs", {})
        assert jobs, "Workflow must define at least one job"
        for job_id, job in jobs.items():
            runs_on = job.get("runs-on", "")
            assert "ubuntu" in str(runs_on).lower(), (
                f"Job '{job_id}' must run on ubuntu-latest (got: {runs_on!r})"
            )

    def test_workflow_uses_checkout_v4(self):
        """Workflow must use actions/checkout@v4."""
        content = self._WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "actions/checkout@v4" in content, (
            "Workflow must use actions/checkout@v4 (not v3 or earlier)"
        )

    def test_workflow_uses_setup_python_v5(self):
        """Workflow must use actions/setup-python@v5."""
        content = self._WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "actions/setup-python@v5" in content, (
            "Workflow must use actions/setup-python@v5"
        )

    def test_workflow_runs_benchmark_dry_run(self):
        """Workflow step must invoke eval.run_benchmark with --dry-run."""
        content = self._WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "run_benchmark" in content, "Workflow must invoke eval.run_benchmark"
        assert "--dry-run" in content, "Workflow must pass --dry-run flag"

    def test_workflow_uploads_artifact(self):
        """Workflow must upload the benchmark report as an artifact."""
        content = self._WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "upload-artifact" in content, (
            "Workflow must upload benchmark report as an artifact"
        )

    def test_workflow_has_timeout(self):
        """Every job must specify a timeout to prevent runaway CI costs."""
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        jobs = parsed.get("jobs", {})
        for job_id, job in jobs.items():
            assert "timeout-minutes" in job, (
                f"Job '{job_id}' must set timeout-minutes to control CI costs"
            )
            assert job["timeout-minutes"] > 0

    def test_workflow_permissions_contents_read(self):
        """Workflow must declare minimal permissions (contents: read)."""
        import yaml
        parsed = yaml.safe_load(self._WORKFLOW_PATH.read_text(encoding="utf-8"))
        # permissions can be at workflow level or job level
        wp = parsed.get("permissions", {})
        jobs = parsed.get("jobs", {})
        all_perms = [wp] + [j.get("permissions", {}) or {} for j in jobs.values()]
        # At least one scope must have contents: read (not write)
        has_read = any(p.get("contents") == "read" for p in all_perms if p)
        assert has_read, (
            "Workflow must declare permissions.contents: read at workflow or job level"
        )

    def test_workflow_python_version_311_or_later(self):
        """Must target Python 3.11+ consistent with existing CI."""
        import yaml
        content = self._WORKFLOW_PATH.read_text(encoding="utf-8")
        # Check for python-version string somewhere in the workflow
        assert "3.11" in content or "3.12" in content or "3.13" in content, (
            "Workflow must target Python 3.11 or later"
        )


# ===========================================================================
# 2. TestTokenRotationCLI
# ===========================================================================

class TestTokenRotationCLI:
    """Tests for muninn.cli rotate-token command."""

    def test_cli_module_importable(self):
        from muninn import cli  # noqa: F401

    def test_cli_has_main_function(self):
        from muninn.cli import main
        assert callable(main)

    def test_cli_has_build_parser(self):
        from muninn.cli import build_parser
        parser = build_parser()
        assert parser is not None

    def test_rotate_token_subcommand_exists(self):
        from muninn.cli import build_parser
        parser = build_parser()
        # Parsing with rotate-token should not raise
        args = parser.parse_args(["rotate-token", "--dry-run"])
        assert args.command == "rotate-token"
        assert args.dry_run is True

    def test_rotate_token_dry_run_does_not_write_file(self, tmp_path):
        """--dry-run must not write the token file."""
        from muninn.cli import cmd_rotate_token, build_parser
        token_file = tmp_path / ".muninn_token_test"
        args = build_parser().parse_args(["rotate-token", "--dry-run", "--token-file", str(token_file)])
        rc = cmd_rotate_token(args)
        assert rc == 0
        assert not token_file.exists(), "--dry-run must not write the token file"

    def test_rotate_token_writes_token_file(self, tmp_path):
        """Without --dry-run, token file must be written."""
        from muninn.cli import cmd_rotate_token, build_parser
        token_file = tmp_path / ".muninn_token"
        args = build_parser().parse_args(["rotate-token", "--token-file", str(token_file)])
        rc = cmd_rotate_token(args)
        assert rc == 0
        assert token_file.exists(), "rotate-token must create the token file"

    def test_rotate_token_file_contains_valid_token(self, tmp_path):
        """Written token must be 32-byte urlsafe base64 (~43 chars)."""
        from muninn.cli import cmd_rotate_token, build_parser
        token_file = tmp_path / ".muninn_token"
        args = build_parser().parse_args(["rotate-token", "--token-file", str(token_file)])
        cmd_rotate_token(args)
        token = token_file.read_text(encoding="utf-8").strip()
        assert len(token) >= 40, f"Token too short: {len(token)} chars"
        # Must be URL-safe base64 charset (no +, /)
        import re
        assert re.match(r"^[A-Za-z0-9_\-]+$", token), (
            f"Token contains invalid chars for URL-safe base64: {token[:10]}..."
        )

    def test_rotate_token_each_call_produces_unique_token(self, tmp_path):
        """Two consecutive rotate-token calls must produce different tokens."""
        from muninn.cli import cmd_rotate_token, build_parser
        tokens = set()
        for i in range(3):
            token_file = tmp_path / f".muninn_token_{i}"
            args = build_parser().parse_args(["rotate-token", "--token-file", str(token_file)])
            cmd_rotate_token(args)
            tokens.add(token_file.read_text(encoding="utf-8").strip())
        assert len(tokens) == 3, "rotate-token must generate a unique token each time"

    def test_rotate_token_only_prints_just_token(self, tmp_path, capsys):
        """--token-only must print exactly one line: the token."""
        from muninn.cli import cmd_rotate_token, build_parser
        token_file = tmp_path / ".tok"
        args = build_parser().parse_args(["rotate-token", "--token-file", str(token_file), "--token-only"])
        rc = cmd_rotate_token(args)
        assert rc == 0
        captured = capsys.readouterr()
        output = captured.out.strip()
        assert "\n" not in output, "--token-only must not produce multi-line output"
        assert len(output) >= 40, "--token-only output must be the token (≥40 chars)"

    def test_rotate_token_dry_run_token_only(self, tmp_path, capsys):
        """--dry-run --token-only must print the token without writing files."""
        from muninn.cli import cmd_rotate_token, build_parser
        token_file = tmp_path / ".tok"
        args = build_parser().parse_args([
            "rotate-token", "--token-file", str(token_file),
            "--dry-run", "--token-only"
        ])
        rc = cmd_rotate_token(args)
        assert rc == 0
        assert not token_file.exists()
        captured = capsys.readouterr()
        assert len(captured.out.strip()) >= 40

    def test_main_entrypoint_rotate_token(self, tmp_path, monkeypatch):
        """main() must support rotate-token and return 0 on success."""
        from muninn import cli
        token_file = tmp_path / ".token"
        monkeypatch.setattr(
            sys, "argv",
            ["muninn.cli", "rotate-token", "--token-file", str(token_file), "--dry-run"]
        )
        rc = cli.main()
        assert rc == 0


# ===========================================================================
# 3. TestTokenFilePersistence
# ===========================================================================

class TestTokenFilePersistence:
    """Tests _resolve_token_file resolution logic."""

    def test_explicit_token_file_takes_precedence(self, tmp_path):
        from muninn.cli import _resolve_token_file
        explicit = tmp_path / "my_token"
        resolved = _resolve_token_file(explicit)
        assert resolved == explicit

    def test_env_var_token_file_resolution(self, tmp_path):
        from muninn.cli import _resolve_token_file
        env_path = str(tmp_path / "env_token")
        with patch.dict(os.environ, {"MUNINN_TOKEN_FILE": env_path}, clear=False):
            resolved = _resolve_token_file(None)
        assert str(resolved) == env_path

    def test_default_token_file_is_dot_muninn_token(self):
        from muninn.cli import _resolve_token_file, _DEFAULT_TOKEN_FILE
        # No explicit path, no env var
        with patch.dict(os.environ, {}, clear=False):
            env_backup = os.environ.pop("MUNINN_TOKEN_FILE", None)
            resolved = _resolve_token_file(None)
            if env_backup is not None:
                os.environ["MUNINN_TOKEN_FILE"] = env_backup
        assert resolved == _DEFAULT_TOKEN_FILE

    def test_explicit_overrides_env_var(self, tmp_path):
        from muninn.cli import _resolve_token_file
        explicit = tmp_path / "explicit"
        env_path = str(tmp_path / "env_path")
        with patch.dict(os.environ, {"MUNINN_TOKEN_FILE": env_path}, clear=False):
            resolved = _resolve_token_file(explicit)
        assert resolved == explicit


# ===========================================================================
# 4. TestMCPConfigPatcher
# ===========================================================================

class TestMCPConfigPatcher:
    """Tests _patch_mcp_config MCP JSON config patching."""

    def _write_config(self, path: Path, cfg: Dict[str, Any]) -> None:
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def test_patches_muninn_auth_token_in_env(self, tmp_path):
        """Must update MUNINN_AUTH_TOKEN if present in muninn server env."""
        from muninn.cli import _patch_mcp_config
        cfg_path = tmp_path / "claude_desktop_config.json"
        self._write_config(cfg_path, {
            "mcpServers": {
                "muninn": {
                    "command": "python",
                    "args": ["-m", "mcp_wrapper"],
                    "env": {"MUNINN_AUTH_TOKEN": "old_token"},
                }
            }
        })
        result = _patch_mcp_config(cfg_path, "new_token_abc")
        assert result is True
        updated = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert updated["mcpServers"]["muninn"]["env"]["MUNINN_AUTH_TOKEN"] == "new_token_abc"

    def test_does_not_patch_nonexistent_file(self, tmp_path):
        from muninn.cli import _patch_mcp_config
        missing = tmp_path / "no_such_file.json"
        result = _patch_mcp_config(missing, "tok")
        assert result is False

    def test_does_not_patch_if_no_muninn_server(self, tmp_path):
        """Config with no muninn server must not be modified."""
        from muninn.cli import _patch_mcp_config
        cfg_path = tmp_path / "config.json"
        self._write_config(cfg_path, {
            "mcpServers": {
                "other_server": {"env": {"OTHER_TOKEN": "old"}},
            }
        })
        original = cfg_path.read_text(encoding="utf-8")
        result = _patch_mcp_config(cfg_path, "tok")
        assert result is False
        assert cfg_path.read_text(encoding="utf-8") == original, "File must not be modified"

    def test_does_not_patch_if_no_auth_token_env_key(self, tmp_path):
        """Muninn server present but env has no MUNINN_AUTH_TOKEN — skip."""
        from muninn.cli import _patch_mcp_config
        cfg_path = tmp_path / "config.json"
        self._write_config(cfg_path, {
            "mcpServers": {
                "muninn": {"command": "python", "env": {}},
            }
        })
        result = _patch_mcp_config(cfg_path, "tok")
        assert result is False

    def test_dry_run_does_not_write_file(self, tmp_path):
        """dry_run=True must return True (would patch) but not write."""
        from muninn.cli import _patch_mcp_config
        cfg_path = tmp_path / "config.json"
        self._write_config(cfg_path, {
            "mcpServers": {
                "muninn": {"env": {"MUNINN_AUTH_TOKEN": "old"}},
            }
        })
        original = cfg_path.read_text(encoding="utf-8")
        result = _patch_mcp_config(cfg_path, "new_token", dry_run=True)
        assert result is True
        assert cfg_path.read_text(encoding="utf-8") == original, (
            "dry_run=True must not modify the file"
        )

    def test_patches_case_insensitive_muninn_server_name(self, tmp_path):
        """Server name matching is case-insensitive ('Muninn', 'MUNINN', 'muninn-mcp')."""
        from muninn.cli import _patch_mcp_config
        for server_name in ("muninn", "Muninn", "MUNINN", "muninn-mcp", "my-muninn-server"):
            cfg_path = tmp_path / f"config_{server_name}.json"
            self._write_config(cfg_path, {
                "mcpServers": {
                    server_name: {"env": {"MUNINN_AUTH_TOKEN": "old"}},
                }
            })
            result = _patch_mcp_config(cfg_path, "new")
            assert result is True, f"Should have patched server named '{server_name}'"
            updated = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert updated["mcpServers"][server_name]["env"]["MUNINN_AUTH_TOKEN"] == "new"

    def test_invalid_json_returns_false(self, tmp_path):
        """Malformed JSON config must not raise — return False."""
        from muninn.cli import _patch_mcp_config
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text("{ this is not valid json ]", encoding="utf-8")
        result = _patch_mcp_config(cfg_path, "tok")
        assert result is False

    def test_multiple_muninn_servers_all_patched(self, tmp_path):
        """All muninn servers in a config must be patched."""
        from muninn.cli import _patch_mcp_config
        cfg_path = tmp_path / "config.json"
        self._write_config(cfg_path, {
            "mcpServers": {
                "muninn": {"env": {"MUNINN_AUTH_TOKEN": "old"}},
                "muninn-dev": {"env": {"MUNINN_AUTH_TOKEN": "old-dev"}},
                "context7": {"env": {"OTHER_KEY": "irrelevant"}},
            }
        })
        result = _patch_mcp_config(cfg_path, "new_multi")
        assert result is True
        updated = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert updated["mcpServers"]["muninn"]["env"]["MUNINN_AUTH_TOKEN"] == "new_multi"
        assert updated["mcpServers"]["muninn-dev"]["env"]["MUNINN_AUTH_TOKEN"] == "new_multi"
        # context7 must not be touched
        assert "MUNINN_AUTH_TOKEN" not in updated["mcpServers"]["context7"]["env"]


# ===========================================================================
# 5. TestVersionBump315
# ===========================================================================

class TestVersionBump315:
    """Validates version bump to 3.15.0."""

    def test_version_module_is_315(self):
        from muninn.version import __version__
        parts = tuple(int(x) for x in __version__.split("."))
        assert parts >= (3, 15, 0), (
            f"Expected version >= 3.15.0, got {__version__}. "
            "Bump muninn/version.py and pyproject.toml for Phase 18."
        )

    def test_pyproject_toml_version_matches(self):
        import re
        pyproject = (
            Path(__file__).resolve().parent.parent / "pyproject.toml"
        ).read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([\d.]+)"', pyproject, re.MULTILINE)
        assert m is not None, "pyproject.toml must contain a version = \"...\" line"
        from muninn.version import __version__
        assert m.group(1) == __version__, (
            f"pyproject.toml version ({m.group(1)}) must match muninn/version.py ({__version__})"
        )
