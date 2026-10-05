"""One bounded local-only pass over encrypted credential ambiguity groups.

Run in a local interactive terminal. The passphrase is never an argument or
environment variable. No candidate text, model reply, or secret is printed.
Without --apply this only previews decision counts (reveals are still audited).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import sqlite3
import sys
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


def run(*, root: Path, passphrase: str, limit: int, model_limit: int,
        model: str, apply: bool, base_url: str,
        keep_alive: int | str = 0, archive_root: Path | None = None,
        review_source=None, after: dict | None = None, on_progress=None) -> dict:
    if not 1 <= limit <= 100 or not 0 <= model_limit <= min(limit, 100):
        raise ValueError("Invalid local triage bounds")
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
                    prepared = review_source.prepare(row)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path,
                        help="Authenticated archive required for source-aware local model review")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--model-limit", type=int, default=12)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-pages", type=int, default=1,
                        help="Process successive pages with one local unlock (1-10000)")
    parser.add_argument("--backup-after", type=Path,
                        help="Create and validate a new portable vault backup after review")
    parser.add_argument("--backup-before", type=Path,
                        help="Create and validate a new portable vault backup before review")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
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
    except (OSError, RuntimeError):
        parser.error("Invalid vault or backup destination")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(json.dumps({"state": "interactive_terminal_required"}))
        return 2
    backup_state = "not_started"
    try:
        passphrase = getpass.getpass("Credential vault passphrase (hidden): ")
        if args.backup_before is not None:
            count = CredentialStore(args.root).backup(args.backup_before,
                                                      passphrase=passphrase)
            backup_state = "validated_pre_triage_backup"
            print(json.dumps({"stage": "validated_pre_triage_backup",
                              "credential_records": count,
                              "review_queue": CredentialStore(args.root).ambiguity_status()},
                             sort_keys=True), flush=True)
        review_source = (CredentialReviewSource(args.archive_root)
                         if args.archive_root is not None and args.model_limit else None)
        cursor = None
        for page in range(1, args.max_pages + 1):
            report = run(root=args.root, passphrase=passphrase, limit=args.limit,
                         model_limit=args.model_limit, model=args.model,
                         apply=args.apply, base_url=args.ollama_url,
                         archive_root=args.archive_root,
                         review_source=review_source,
                         after=cursor,
                         on_progress=lambda report: print(json.dumps(report, sort_keys=True), flush=True),
                         keep_alive="30s" if args.max_pages > 1 else 0)
            print(json.dumps({"page": page, **report}, sort_keys=True), flush=True)
            if (not args.apply or report["groups_seen"] == 0
                    or (report["next_cursor"] == cursor and report["model_calls"] == 0)
                    or report["queue_counts"].get("pending", 0) == 0):
                break
            cursor = report["next_cursor"]
        if args.backup_after is not None:
            count = CredentialStore(args.root).backup(args.backup_after,
                                                      passphrase=passphrase)
            backup_state = "validated_post_triage_backup"
            print(json.dumps({"stage": "validated_post_triage_backup",
                              "credential_records": count,
                              "review_queue": CredentialStore(args.root).ambiguity_status()},
                             sort_keys=True), flush=True)
        if args.apply:
            queue_status = CredentialStore(args.root).ambiguity_status()
            review_resolved = not any(queue_status.get(name, 0)
                                      for name in ("pending", "deferred"))
            print(json.dumps({"stage": "triage_status",
                              "review_resolved": review_resolved,
                              "review_queue": queue_status},
                             sort_keys=True), flush=True)
            return 0 if review_resolved else 2
    except BaseException as exc:
        # Exception text can contain private source context; return only type.
        failure = {"state": "failed", "error_category": type(exc).__name__,
                          "backup_state": backup_state,
                          "post_backup_unavailable": args.backup_after is not None
                          and backup_state != "validated_post_triage_backup"}
        if isinstance(exc, httpx.HTTPStatusError):
            # The body, URL, headers and exception message may contain secrets.
            status = exc.response.status_code
            if type(status) is int and 100 <= status <= 599:
                failure["http_status_code"] = status
        if isinstance(exc, sqlite3.Error) and getattr(exc, "sqlite_errorname", "") in {
                "SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_READONLY", "SQLITE_FULL",
                "SQLITE_IOERR", "SQLITE_CANTOPEN", "SQLITE_CORRUPT"}:
            failure["sqlite_error_code"] = exc.sqlite_errorname
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
