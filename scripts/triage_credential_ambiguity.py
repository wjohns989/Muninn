"""One bounded locally unlocked pass over encrypted credential ambiguities.

Run in a local interactive terminal. The passphrase is never an argument or
environment variable. No candidate text, model reply, or secret is printed.
Without --apply this only previews decision counts (reveals are still audited).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import stat
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx

from muninn.history.ambiguity_triage import (
    CandidateForReview,
    classify_local,
    deterministic_decision,
)
from muninn.history.auto_routing import choose_route, probe_gpu, probe_ollama
from muninn.history.credential_review_source import CredentialReviewSource
from muninn.history.credential_store import CredentialStore
from muninn.history.triage_status import progress_identity, runtime_binding

_progress_runtime_binding = None

_PROGRESS_STAGES = {'zdr_readiness', 'waiting_for_remote_admission', 'awaiting_passphrase',
                    'passphrase_received', 'opening_pre_backup_vault', 'validating_pre_backup',
                    'opening_post_backup_vault', 'validating_post_backup', 'starting_review',
                    'reading_triage_status',
                    'validated_pre_triage_backup', 'validated_post_triage_backup',
                    'review_page', 'source_context_prepare', 'local_context_review',
                    'zdr_context_review', 'remote_context_review', 'triage_status'}
_PROGRESS_STATES = {'ready', 'failed', 'remote_admission_busy', 'remote_consent_revoked',
                    'remote_readiness_timeout', 'readiness_wait_cancelled', 'readiness_check_failed',
                    'progress_log_unavailable', 'interactive_terminal_required',
                    'unknown', 'disabled', 'key_missing', 'provider_unavailable',
                    'invalid_provider_data', 'key_disabled', 'key_cap_exceeds_local_threshold',
                    'key_exhausted', 'local_threshold_reached',
                    'remote_accounting_unconfigured', 'remote_accounting_unavailable',
                    'remote_accounting_invalid_cost', 'remote_accounting_invalid_reference',
                    'remote_accounting_invalid_policy', 'remote_cost_unresolved',
                    'credential_context_not_remote_safe', 'source_context_required',
                    'model_limit_reached'}
_PROGRESS_ERRORS = {'EOFError', 'VaultIntegrityError', 'VaultPermissionError',
                    'HTTPStatusError', 'AdmissionError', 'OSError', 'PermissionError',
                    'FileNotFoundError', 'RuntimeError', 'ValueError', 'TypeError',
                    'OperationalError', 'IntegrityError', 'DatabaseError',
                    'KeyboardInterrupt', 'UnicodeDecodeError', 'UnicodeEncodeError'}


def validate_progress_path(path):
    """No linked/reparse ancestor may redirect a progress destination."""
    if '..' in path.parts:
        raise ValueError('Progress path must not contain parent traversal')
    path = path.absolute()
    for component in (*reversed(path.parents), path):
        try:
            details = component.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(details.st_mode)
                or getattr(details, 'st_file_attributes', 0) & 0x400
                or (hasattr(component, 'is_junction') and component.is_junction())):
            raise ValueError('Progress path must not contain linked components')
    return path


def emit_progress(report, path=None):
    """Keep the interactive console intact; persist only operational counters."""
    if path is not None:
        from muninn.history.private_acl import verify_private
        validate_progress_path(path)
        verify_private(path)
        safe = {}
        counters = {'rows', 'page', 'model_calls', 'contexts_reused', 'groups_seen',
                    'left_pending', 'deferred_for_user', 'model_rejected', 'rule_rejected',
                    'source_context_pending', 'contexts_quota_deferred',
                    'contexts_route_deferred', 'credential_records', 'remaining_seconds',
                    'unresolved_admissions', 'http_status_code'}
        flags = {'passphrase_needed', 'review_resolved', 'applied', 'post_backup_unavailable'}
        for name, value in report.items():
            if name in counters and type(value) is int and 0 <= value < 2**63:
                safe[name] = value
            elif name in flags and type(value) is bool:
                safe[name] = value
            elif name == 'backup_state' and value == 'not_started':
                safe[name] = value
            elif (name in {'stage', 'backup_state', 'failure_stage'} and isinstance(value, str)
                  and value in _PROGRESS_STAGES):
                safe[name] = value
            elif name == 'error_category' and isinstance(value, str) and value in _PROGRESS_ERRORS:
                safe[name] = value
            elif (name in {'state', 'model_route'} and isinstance(value, str)
                  and value in _PROGRESS_STATES):
                safe[name] = value
            elif name in {'queue_counts', 'review_queue'} and isinstance(value, dict):
                safe[name] = {state: count for state, count in value.items()
                              if state in {'pending', 'accepted', 'rejected', 'deferred'}
                              and type(count) is int and 0 <= count < 2**63}
        # Existing file only: a removed destination is never silently recreated.
        safe.update(progress_identity(_progress_runtime_binding))
        with path.open('r+', encoding='utf-8') as stream:
            stream.seek(0, 2)
            stream.write(json.dumps(safe, sort_keys=True) + '\n')
            stream.flush()
    print(json.dumps(report, sort_keys=True), flush=True)


def wait_remote_readiness(policy_root, wait_seconds=0, *, on_progress=None):
    """Read-only admission wait BEFORE unlocking; never reserve or retry a call.

    Only an already-owned admission is transient here. Revocation, provider
    failures and budgets return to the operator, not a hidden retry loop.
    """
    from muninn.history.auto_routing import openrouter_key_status, remote_policy_snapshot
    from muninn.history.remote_accounting import AdmissionError, status
    deadline = time.monotonic() + wait_seconds
    report = {"stage": "zdr_readiness", "passphrase_needed": False}
    try:
        while True:
            policy = remote_policy_snapshot(policy_root)
            accounting = status(policy_root)
            if (type(policy.enabled) is not bool or not isinstance(accounting, dict)
                    or type(accounting.get('unresolved')) is not int
                    or not 0 <= accounting['unresolved'] <= 1):
                return {**report, 'state': 'readiness_check_failed'}
            state = 'remote_consent_revoked' if not policy.enabled else (
                'remote_admission_busy' if accounting['unresolved'] else 'ready')
            if state == 'ready':
                provider_status = openrouter_key_status(policy_root=policy_root)
                if (not isinstance(provider_status, dict)
                        or type(provider_status.get('admission_ready')) is not bool
                        or (not provider_status['admission_ready']
                            and provider_status.get('state') == 'ready')):
                    return {**report, 'state': 'readiness_check_failed'}
                state = 'ready' if provider_status['admission_ready'] else provider_status['state']
            report = {**report, 'state': state,
                      'unresolved_admissions': accounting['unresolved']}
            remaining = max(0, deadline - time.monotonic())
            if wait_seconds and not remaining and state in {'ready', 'remote_admission_busy'}:
                return {**report, 'state': 'remote_readiness_timeout'}
            if state != 'remote_admission_busy' or not wait_seconds:
                return report
            if on_progress is not None:
                on_progress({**report, 'stage': 'waiting_for_remote_admission',
                             'remaining_seconds': int(remaining)})
            time.sleep(min(60, remaining))
    except AdmissionError as exc:
        return {**report, 'state': exc.code}
    except KeyboardInterrupt:
        return {**report, 'state': 'readiness_wait_cancelled'}
    except Exception as exc:
        # Private configuration or HTTP exceptions must never enter the log.
        return {**report, 'state': 'readiness_check_failed',
                'error_category': type(exc).__name__}


def run_zdr(*, root, passphrase, limit, model_limit, model, apply, policy_root,
            archive_root, review_source, after, on_progress):
    """Same occurrence/evidence gate, without a local-model probe or dispatch."""
    from muninn.history.auto_routing import remote_policy_snapshot
    from muninn.history.credential_zdr import review_context
    from muninn.history.remote_accounting import AdmissionError
    if model_limit and (policy_root is None or archive_root is None and review_source is None):
        raise ValueError('ZDR review requires explicit policy root and authenticated source archive')
    store = CredentialStore(root)
    rows = store.list_ambiguities(status='pending', limit=limit,
                                  **({'after': after} if after is not None else {}))
    if on_progress:
        on_progress({'stage': 'review_page', 'rows': len(rows)})
    source = review_source or (CredentialReviewSource(archive_root) if model_limit else None)
    generation = remote_policy_snapshot(policy_root).generation if model_limit else None
    calls = reused = incomplete = rules = rejected = deferred = 0
    examined = set()
    reason = 'no_model_needed'
    for row in rows:
        candidate = store.reveal_ambiguity(row['id'], passphrase=passphrase)
        item = CandidateForReview(row['id'], row['name'], row['reason'], candidate)
        decision = deterministic_decision(item)
        if decision is not None:
            rules += decision.decision == 'rejected'
        elif not model_limit:
            examined.add(row['id'])
            continue
        else:
            prepared = source.prepare(row, candidate=candidate)
            if prepared is None:
                incomplete += 1
                examined.add(row['id'])
                reason = 'source_context_required'
                continue
            outcomes, matched, complete, unsafe = set(), 0, True, False
            for page, occurrence in source.inputs(prepared, row, candidate):
                matched += 1
                if calls >= model_limit:
                    reason, complete = 'model_limit_reached', False
                    break
                try:
                    result, dispatched, cached = review_context(
                        occurrence, source, prepared, page, policy_root=policy_root,
                        generation=generation, model=model)
                except AdmissionError as exc:
                    reason, complete = exc.code, False
                    unsafe = exc.code == 'credential_context_not_remote_safe'
                    break
                calls += dispatched
                reused += cached
                outcomes.add(result.decision)
                reason = 'zdr_context_review'
                if on_progress:
                    on_progress({'stage': reason, 'model_calls': calls, 'contexts_reused': reused})
            if unsafe:
                incomplete += 1
                deferred += 1
                examined.add(row['id'])
                continue  # local-only/user-review context must not starve later safe rows
            if not complete or not matched:
                incomplete += 1
                # Do not repeatedly attempt reservations for later rows when
                # a known batch, unknown transport, revocation or cap blocks.
                break
            from muninn.history.ambiguity_triage import ReviewDecision
            decision = ReviewDecision(item.id, 'rejected' if outcomes == {'rejected'} else 'deferred', 'zdr-model')
            rejected += decision.decision == 'rejected'
        examined.add(row['id'])
        deferred += decision.decision == 'deferred'
        if apply and decision.decision == 'rejected':
            store.decide_ambiguity(row['id'], passphrase=passphrase, decision='rejected',
                                   actor='zdr-agent' if decision.basis == 'zdr-model' else 'local-agent',
                                   reason='not-a-secret')
    cursor = after
    for row in rows:
        if row['id'] not in examined:
            break
        if 'created_at' in row:
            cursor = {'created_at': row['created_at'], 'id': row['id']}
    counts = store.ambiguity_status()
    return {'groups_seen': len(examined), 'rows_seen': len(rows), 'model_calls': calls,
            'contexts_reused': reused, 'source_context_pending': incomplete,
            'rule_rejected': rules, 'model_rejected': rejected, 'deferred_for_user': deferred,
            'left_pending': counts.get('pending', 0), 'model_route': reason,
            'applied': apply, 'queue_counts': counts, 'next_cursor': cursor}


def run(*, root: Path, passphrase: str, limit: int, model_limit: int,
        model: str, apply: bool, base_url: str,
        keep_alive: int | str = 0, archive_root: Path | None = None,
        review_source=None, after: dict | None = None, on_progress=None,
        provider: str = 'ollama', policy_root: Path | None = None) -> dict:
    if not 1 <= limit <= 100 or not 0 <= model_limit <= min(limit, 100):
        raise ValueError("Invalid local triage bounds")
    if provider == 'openrouter':
        return run_zdr(root=root, passphrase=passphrase, limit=limit, model_limit=model_limit,
                       model=model, apply=apply, policy_root=policy_root, archive_root=archive_root,
                       review_source=review_source, after=after, on_progress=on_progress)
    if provider != 'ollama':
        raise ValueError('Unsupported credential review provider')
    if base_url.rstrip("/") != "http://127.0.0.1:11434":
        raise ValueError("Credential triage requires loopback Ollama")
    store = CredentialStore(root)
    rows = store.list_ambiguities(status="pending", limit=limit,
                                  **({"after": after} if after is not None else {}))
    if on_progress:
        on_progress({"stage": "review_page", "rows": len(rows)})
    rule_decisions = []
    examined = set()
    model_inputs = []
    for row in rows:
        candidate = store.reveal_ambiguity(row["id"], passphrase=passphrase)
        item = CandidateForReview(row["id"], row["name"], row["reason"], candidate)
        rule = deterministic_decision(item)
        if rule is not None:
            rule_decisions.append(rule)
            examined.add(row["id"])
        else:
            model_inputs.append((row, item))
    model_decisions = []
    calls, reused, incomplete = 0, 0, 0
    quota_deferred, route_deferred = 0, 0
    route_reason = "no_model_needed" if not model_inputs else "deferred"
    if model_inputs and model_limit and (archive_root is not None or review_source is not None):
        review_source = review_source or CredentialReviewSource(archive_root)
        gpu = probe_gpu()
        installed, loaded = probe_ollama(base_url)
        if gpu is not None:
            gpu = replace(gpu, loaded_models=loaded)
        route = choose_route(gpu, installed, cloud_allowed=False,
                             model_hints=(model,))
        route_reason = route.reason
        if route.provider == "ollama" and route.model == model:
            installed_model = next((m for m in installed if (m.get("name") or m.get("model")) == model), {})
            digest = installed_model.get("digest")
            if not digest:
                route_reason = "model_identity_unavailable"
            else:
                identity = hashlib.sha256(f"credential-review-v2\0{model}\0{digest}".encode()).hexdigest()
                for row, original in model_inputs:
                    if calls >= model_limit:
                        route_reason = "model_limit_reached"
                        break
                    if on_progress:
                        on_progress({"stage": "source_context_prepare", "model_calls": calls})
                    prepared = review_source.prepare(row, candidate=original.candidate)
                    if prepared is None:
                        incomplete += 1
                        examined.add(row["id"])
                        continue
                    decisions, matched, complete, resumable = set(), 0, True, False
                    for page, item in review_source.inputs(prepared, row, original.candidate):
                        matched += 1
                        if not item.source_context:
                            complete = False
                            continue
                        cached = review_source.cached(prepared, page, identity)
                        if cached is not None:
                            decisions.add(cached)
                            reused += 1
                            continue
                        if calls >= model_limit:
                            complete = False
                            resumable = True
                            quota_deferred += 1
                            continue
                        # Acquire the shared local inference slot, then recheck
                        # GPU contention/headroom immediately before each call.
                        from muninn.extraction.ollama_slot import ollama_slot
                        with ollama_slot():
                            gpu = probe_gpu()
                            installed, loaded = probe_ollama(base_url)
                            if gpu is not None:
                                gpu = replace(gpu, loaded_models=loaded)
                            fresh = choose_route(gpu, installed, cloud_allowed=False, model_hints=(model,))
                            if fresh.provider != "ollama" or fresh.model != model:
                                complete = False
                                resumable = True
                                route_reason = fresh.reason
                                route_deferred += 1
                                continue
                            fresh_model = next((m for m in installed if (m.get("name") or m.get("model")) == model), {})
                            if fresh_model.get("digest") != digest:
                                complete = False
                                resumable = True
                                route_reason = "model_identity_changed"
                                route_deferred += 1
                                continue
                            route_reason = fresh.reason
                            result = classify_local([item], model=model,
                                                    base_url=base_url, keep_alive=keep_alive)[0]
                            calls += 1
                            if on_progress:
                                on_progress({"stage": "local_context_review", "model_calls": calls})
                        if apply:
                            review_source.record(prepared, page, identity, result.decision)
                        decisions.add(result.decision)
                    if complete and matched:
                        from muninn.history.ambiguity_triage import ReviewDecision
                        model_decisions.append(ReviewDecision(original.id,
                            "rejected" if decisions == {"rejected"} else "deferred", "local-model"))
                    else:
                        incomplete += 1
                    if not resumable:
                        examined.add(row["id"])
        else:
            route_reason = route.reason
    elif model_inputs and model_limit:
        route_reason = "source_context_required"
        examined.update(row["id"] for row, _item in model_inputs)
    elif not model_limit:
        examined.update(row["id"] for row, _item in model_inputs)
    if apply:
        for decision in [*rule_decisions, *model_decisions]:
            if decision.decision != "rejected":
                # An uncertain model output is not a resolution. Preserve the
                # pending item until evidence or explicit user review settles it.
                continue
            store.decide_ambiguity(
                decision.id, passphrase=passphrase, decision=decision.decision,
                actor="local-agent", reason="not-a-secret" if decision.decision == "rejected" else "",
            )
    queue_counts = store.ambiguity_status()
    next_cursor = after
    for row in rows:
        if row["id"] not in examined:
            break
        if "created_at" in row:
            next_cursor = {"created_at": row["created_at"], "id": row["id"]}
    return {
        "groups_seen": len(rows), "rows_seen": len(rows), "model_calls": calls,
        "contexts_reused": reused, "source_context_pending": incomplete,
        "rule_rejected": sum(d.decision == "rejected" for d in rule_decisions),
        "model_rejected": sum(d.decision == "rejected" for d in model_decisions),
        "deferred_for_user": sum(d.decision == "deferred" for d in [*rule_decisions, *model_decisions]),
        "left_pending": queue_counts.get("pending", 0),
        "model_route": "model_limit_reached" if quota_deferred else route_reason,
        "contexts_quota_deferred": quota_deferred,
        "contexts_route_deferred": route_deferred,
        "applied": apply, "queue_counts": queue_counts,
        "next_cursor": next_cursor,
    }


def main() -> int:
    global _progress_runtime_binding
    _progress_runtime_binding = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path,
                        help="Authenticated archive required for source-aware local model review")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--model-limit", type=int, default=12)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument('--provider', choices=('ollama', 'openrouter'), default='ollama')
    parser.add_argument('--policy-root', type=Path,
                        help='Existing managed ZDR consent and budget root; required for OpenRouter')
    parser.add_argument('--check-readiness', action='store_true',
                        help='Check managed remote admission before any local passphrase prompt')
    parser.add_argument('--wait-for-readiness', type=int, default=0, metavar='SECONDS',
                        help='Interactive read-only wait for a busy admission (1-86400); unlock only when ready')
    parser.add_argument('--progress-log', type=Path,
                        help='New owner-only live progress JSONL; counters/codes only, no credentials or cursors')
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-pages", type=int, default=1,
                        help="Process successive pages with one local unlock (1-10000)")
    parser.add_argument("--backup-after", type=Path,
                        help="Create and validate a new portable vault backup after review")
    parser.add_argument("--backup-before", type=Path,
                        help="Create and validate a new portable vault backup before review")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.provider == 'openrouter' and args.model == 'qwen2.5:7b':
        args.model = None  # configured remote primary, never a local name by default
    if args.provider == 'openrouter' and args.model_limit and (args.policy_root is None or args.archive_root is None):
        parser.error('OpenRouter review requires --policy-root and --archive-root')
    if args.check_readiness and (args.provider != 'openrouter' or args.policy_root is None):
        parser.error('--check-readiness requires OpenRouter and its existing --policy-root')
    if not 0 <= args.wait_for_readiness <= 86400:
        parser.error('--wait-for-readiness must be between 0 and 86400 seconds')
    if args.wait_for_readiness and (args.provider != 'openrouter' or args.policy_root is None
                                   or not args.model_limit or args.check_readiness):
        parser.error('Readiness waiting requires interactive OpenRouter model review, not --check-readiness')
    if args.progress_log is not None and args.check_readiness:
        parser.error('Progress logging requires the interactive workflow, not --check-readiness')
    if not 1 <= args.max_pages <= 10000:
        parser.error("--max-pages must be between 1 and 10000")
    if args.backup_after is not None and not args.apply:
        parser.error("--backup-after requires --apply")
    if args.backup_before is not None and not args.apply:
        parser.error("--backup-before requires --apply")
    try:
        vault_path = args.root.resolve(strict=True)
        destinations = []
        for name, destination in (("--backup-before", args.backup_before),
                                  ("--backup-after", args.backup_after)):
            if destination is None:
                continue
            if (destination.exists() or destination.is_symlink()
                    or (hasattr(destination, "is_junction") and destination.is_junction())):
                parser.error(f"{name} destination already exists")
            resolved = destination.resolve(strict=False)
            if resolved == vault_path or vault_path in resolved.parents:
                parser.error("Backup destination cannot be inside the live vault")
            destinations.append(resolved)
        if len(destinations) == 2 and destinations[0] == destinations[1]:
            parser.error("Pre- and post-triage backup destinations must differ")
        if args.progress_log is not None:
            args.progress_log = validate_progress_path(args.progress_log)
            progress_path = args.progress_log.resolve(strict=False)
            private_roots = [vault_path]
            if args.archive_root is not None:
                private_roots.append(args.archive_root.resolve(strict=True))
            if any(progress_path == root or root in progress_path.parents for root in private_roots):
                parser.error('Progress log cannot be inside the vault or archive')
            if any(progress_path == root or root in progress_path.parents for root in destinations):
                parser.error('Progress log cannot use a backup destination')
    except (OSError, RuntimeError, ValueError):
        parser.error("Invalid vault or backup destination")
    if args.archive_root is not None and args.policy_root is not None:
        _progress_runtime_binding = runtime_binding(args.root, args.archive_root,
            args.policy_root, Path(__file__).resolve().parents[1], Path(sys.executable))
    if (args.wait_for_readiness or args.progress_log is not None) and (
            not sys.stdin.isatty() or not sys.stdout.isatty()):
        print(json.dumps({"state": "interactive_terminal_required"}), flush=True)
        return 2
    if args.progress_log is not None:
        try:
            from muninn.history.private_acl import create_private_directory, create_private_file
            if not args.progress_log.parent.exists():
                create_private_directory(args.progress_log.parent)
            validate_progress_path(args.progress_log)
            create_private_file(args.progress_log)
        except Exception as exc:
            print(json.dumps({'state': 'progress_log_unavailable',
                              'error_category': type(exc).__name__}), flush=True)
            return 2
    progress_failed = False
    last_stage = None
    def emit(report):
        nonlocal progress_failed, last_stage
        # Last operation entered, not an error diagnosis. Never retain free text.
        stage = report.get('stage')
        if isinstance(stage, str) and stage in _PROGRESS_STAGES:
            last_stage = stage
        try:
            emit_progress(report, args.progress_log)
        except BaseException:
            progress_failed = True
            raise
    if args.provider == 'openrouter' and (args.model_limit or args.check_readiness):
        report = wait_remote_readiness(args.policy_root, args.wait_for_readiness,
            on_progress=emit)
        if args.check_readiness or report['state'] != 'ready':
            emit(report)
            return 130 if report['state'] == 'readiness_wait_cancelled' else (
                0 if report['state'] == 'ready' else 2)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(json.dumps({"state": "interactive_terminal_required"}))
        return 2
    backup_state = "not_started"
    try:
        emit({'stage': 'awaiting_passphrase', 'passphrase_needed': True})
        passphrase = getpass.getpass("Credential vault passphrase (hidden): ")
        emit({'stage': 'passphrase_received', 'passphrase_needed': False})
        if args.backup_before is not None:
            emit({'stage': 'opening_pre_backup_vault'})
            backup_store = CredentialStore(args.root)
            emit({'stage': 'validating_pre_backup'})
            count = backup_store.backup(args.backup_before, passphrase=passphrase)
            backup_state = "validated_pre_triage_backup"
            emit({"stage": "validated_pre_triage_backup",
                              "credential_records": count,
                              "review_queue": CredentialStore(args.root).ambiguity_status()})
        emit({'stage': 'starting_review'})
        review_source = (CredentialReviewSource(args.archive_root)
                         if args.archive_root is not None and args.model_limit else None)
        cursor = None
        for page in range(1, args.max_pages + 1):
            report = run(root=args.root, passphrase=passphrase, limit=args.limit,
                         model_limit=args.model_limit, model=args.model,
                         apply=args.apply, base_url=args.ollama_url,
                         archive_root=args.archive_root,
                         review_source=review_source,
                         provider=args.provider, policy_root=args.policy_root,
                         after=cursor,
                         on_progress=emit,
                         keep_alive="30s" if args.max_pages > 1 else 0)
            emit({"page": page, **report})
            if (not args.apply or report["groups_seen"] == 0
                    or (report["next_cursor"] == cursor and report["model_calls"] == 0)
                    or report["queue_counts"].get("pending", 0) == 0):
                break
            cursor = report["next_cursor"]
        if args.backup_after is not None:
            emit({'stage': 'opening_post_backup_vault'})
            backup_store = CredentialStore(args.root)
            emit({'stage': 'validating_post_backup'})
            count = backup_store.backup(args.backup_after, passphrase=passphrase)
            backup_state = "validated_post_triage_backup"
            emit({"stage": "validated_post_triage_backup",
                              "credential_records": count,
                              "review_queue": CredentialStore(args.root).ambiguity_status()})
        if args.apply:
            emit({'stage': 'reading_triage_status'})
            queue_status = CredentialStore(args.root).ambiguity_status()
            review_resolved = not any(queue_status.get(name, 0)
                                      for name in ("pending", "deferred"))
            emit({"stage": "triage_status",
                              "review_resolved": review_resolved,
                              "review_queue": queue_status})
            return 0 if review_resolved else 2
    except BaseException as exc:
        # Exception text can contain private source context; return only type.
        failure = {"state": "failed", "error_category": type(exc).__name__,
                          "passphrase_needed": False,
                          "backup_state": backup_state,
                          "post_backup_unavailable": args.backup_after is not None
                          and backup_state != "validated_post_triage_backup"}
        if last_stage is not None:
            failure['failure_stage'] = last_stage
        if isinstance(exc, httpx.HTTPStatusError):
            # The body, URL, headers and exception message may contain secrets.
            status = exc.response.status_code
            if type(status) is int and 100 <= status <= 599:
                failure["http_status_code"] = status
        from muninn.history.remote_accounting import AdmissionError
        if isinstance(exc, AdmissionError):
            failure['reason'] = exc.code
        if isinstance(exc, sqlite3.Error) and getattr(exc, "sqlite_errorname", "") in {
                "SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_READONLY", "SQLITE_FULL",
                "SQLITE_IOERR", "SQLITE_CANTOPEN", "SQLITE_CORRUPT"}:
            failure["sqlite_error_code"] = exc.sqlite_errorname
        # Healthy monitoring must not retain a stale hidden-input prompt.
        # A failed destination is never retried, recreated or repermissioned.
        try:
            emit_progress(failure, None if progress_failed else args.progress_log)
        except BaseException:
            emit_progress(failure)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
