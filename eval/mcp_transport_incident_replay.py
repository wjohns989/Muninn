from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOG_PREFIX_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ")
DEFAULT_SIGNATURE_PATTERNS = [
    r"MCP stdio transport closed while sending JSON-RPC message",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_prefixed_timestamp(line: str) -> datetime | None:
    match = LOG_PREFIX_TS_RE.match(line)
    if not match:
        return None
    ts_text = match.group("ts")
    try:
        parsed = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _split_command(command: str) -> list[str]:
    return shlex.split(command, posix=False)


def _decode_subprocess_output(completed: subprocess.CompletedProcess[bytes]) -> str:
    output_bytes = (completed.stdout or b"") + (completed.stderr or b"")
    if not output_bytes:
        return ""
    for encoding in ("utf-8", "cp1252"):
        try:
            return output_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return output_bytes.decode("utf-8", errors="replace")


def _run_text_command(command: str) -> tuple[int, str, list[str]]:
    run_tokens = _split_command(command)
    completed = subprocess.run(
        run_tokens,
        check=False,
        shell=False,
        capture_output=True,
    )
    output = _decode_subprocess_output(completed)
    return completed.returncode, output, run_tokens


def _try_parse_json_output(output: str) -> Any:
    text = (output or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"mcp_transport_incident_replay_{run_id}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan wrapper logs for transport-closure signatures and trigger diagnostics "
            "capture when incidents are detected."
        )
    )
    parser.add_argument("--log-path", type=Path, default=Path("mcp_wrapper.log"))
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument(
        "--signature-pattern",
        action="append",
        dest="signature_patterns",
        default=None,
        help=(
            "Regex pattern to classify a line as a transport incident signature. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument("--min-signature-count", type=int, default=1)
    parser.add_argument("--max-match-samples", type=int, default=20)
    parser.add_argument(
        "--include-log-sha256",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include SHA-256 digest of the scanned log file in replay report provenance.",
    )
    parser.add_argument(
        "--require-log-path-exists",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when log-path does not exist in the replay scan environment.",
    )
    parser.add_argument(
        "--diagnostics-command",
        default="python -m eval.mcp_transport_diagnostics --lookback-hours 24",
        help="Command to run when signatures are detected (or when always-run is enabled).",
    )
    parser.add_argument(
        "--always-run-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--fail-on-diagnostics-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--report-dir", type=Path, default=Path("eval/reports/mcp_transport"))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.lookback_hours <= 0:
        raise ValueError("--lookback-hours must be positive")
    if args.min_signature_count < 0:
        raise ValueError("--min-signature-count must be non-negative")
    if args.max_match_samples < 0:
        raise ValueError("--max-match-samples must be non-negative")

    signature_patterns = list(args.signature_patterns or DEFAULT_SIGNATURE_PATTERNS)
    if not signature_patterns:
        raise ValueError("At least one signature pattern is required")

    compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
    for pattern in signature_patterns:
        try:
            compiled_patterns.append((pattern, re.compile(pattern, flags=re.IGNORECASE)))
        except re.error as exc:
            raise ValueError(f"Invalid signature regex pattern: {pattern}") from exc

    now = _utc_now()
    window_start = now - timedelta(hours=float(args.lookback_hours))
    run_id = now.strftime("%Y%m%d_%H%M%S")
    output_path = args.output.resolve() if args.output is not None else _report_path(args.report_dir, run_id)

    line_count = 0
    window_line_count = 0
    total_signature_count = 0
    counts_by_pattern: dict[str, int] = {pattern: 0 for pattern, _ in compiled_patterns}
    sample_matches: list[dict[str, Any]] = []

    log_path_exists = args.log_path.exists()
    log_file_metadata: dict[str, Any] = {
        "path": str(args.log_path),
        "exists": log_path_exists,
        "size_bytes": None,
        "modified_at": None,
        "sha256": None,
    }
    if log_path_exists:
        stat_result = args.log_path.stat()
        log_file_metadata["size_bytes"] = int(stat_result.st_size)
        log_file_metadata["modified_at"] = datetime.fromtimestamp(
            stat_result.st_mtime, tz=timezone.utc
        ).isoformat()
        if args.include_log_sha256:
            log_file_metadata["sha256"] = _compute_sha256(args.log_path)

    if log_path_exists:
        with args.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line_count += 1
                ts = _parse_prefixed_timestamp(raw_line)
                if ts is None or ts < window_start:
                    continue
                window_line_count += 1
                line_text = raw_line.rstrip("\r\n")
                for pattern, compiled in compiled_patterns:
                    if not compiled.search(line_text):
                        continue
                    counts_by_pattern[pattern] = counts_by_pattern.get(pattern, 0) + 1
                    total_signature_count += 1
                    if len(sample_matches) < args.max_match_samples:
                        sample_matches.append(
                            {
                                "timestamp": ts.isoformat(),
                                "pattern": pattern,
                                "line": line_text,
                            }
                        )
                    break
    triggered = bool(args.always_run_diagnostics or total_signature_count >= args.min_signature_count)

    diagnostics_return_code: int | None = None
    diagnostics_output = ""
    diagnostics_parsed: dict[str, Any] | None = None
    diagnostics_resolved_command: list[str] | None = None
    diagnostics_artifact_path: str | None = None

    if triggered and args.diagnostics_command.strip():
        diagnostics_return_code, diagnostics_output, diagnostics_resolved_command = _run_text_command(
            args.diagnostics_command
        )
        parsed = _try_parse_json_output(diagnostics_output)
        if isinstance(parsed, dict):
            diagnostics_parsed = parsed
            inputs = parsed.get("inputs")
            if isinstance(inputs, dict):
                output_candidate = inputs.get("output")
                if isinstance(output_candidate, str) and output_candidate.strip():
                    diagnostics_artifact_path = output_candidate

    report = {
        "run_id": run_id,
        "completed_at": now.isoformat(),
        "inputs": {
            "log_path": str(args.log_path),
            "lookback_hours": float(args.lookback_hours),
            "signature_patterns": signature_patterns,
            "min_signature_count": args.min_signature_count,
            "max_match_samples": args.max_match_samples,
            "diagnostics_command": args.diagnostics_command,
            "always_run_diagnostics": bool(args.always_run_diagnostics),
            "fail_on_diagnostics_error": bool(args.fail_on_diagnostics_error),
            "output": str(output_path),
        },
        "results": {
            "scan": {
                "line_count": line_count,
                "window_line_count": window_line_count,
                "window_start": window_start.isoformat(),
                "window_end": now.isoformat(),
                "log_path_exists": log_path_exists,
                "log_file": log_file_metadata,
                "total_signature_count": total_signature_count,
                "counts_by_pattern": counts_by_pattern,
                "sample_matches": sample_matches,
            },
            "triggered": triggered,
            "diagnostics": {
                "executed": bool(triggered and args.diagnostics_command.strip()),
                "resolved_command": diagnostics_resolved_command,
                "return_code": diagnostics_return_code,
                "artifact_path": diagnostics_artifact_path,
                "parsed": diagnostics_parsed,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.require_log_path_exists and not log_path_exists:
        return 4
    if triggered and args.fail_on_diagnostics_error:
        if not args.diagnostics_command.strip():
            return 3
        if diagnostics_return_code not in (None, 0):
            return int(diagnostics_return_code)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
