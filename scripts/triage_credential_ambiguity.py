"""One bounded local-only pass over encrypted credential ambiguity groups.

Run in a local interactive terminal. The passphrase is never an argument or
environment variable. No candidate text, model reply, or secret is printed.
Without --apply this only previews decision counts (reveals are still audited).
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from dataclasses import replace
from pathlib import Path

from muninn.history.ambiguity_triage import (
    CandidateForReview, classify_local, deterministic_decision,
)
from muninn.history.auto_routing import choose_route, probe_gpu, probe_ollama
from muninn.history.credential_store import CredentialStore


def run(*, root: Path, passphrase: str, limit: int, model_limit: int,
        model: str, apply: bool, base_url: str,
        keep_alive: int | str = 0) -> dict:
    if not 1 <= limit <= 100 or not 0 <= model_limit <= min(limit, 100):
        raise ValueError("Invalid local triage bounds")
    if base_url.rstrip("/") != "http://127.0.0.1:11434":
        raise ValueError("Credential triage requires loopback Ollama")
    store = CredentialStore(root)
    groups = store.list_ambiguity_groups(status="pending", limit=limit)
    rule_decisions = []
    model_inputs = []
    for group in groups:
        candidate = store.reveal_ambiguity(group["representative_id"], passphrase=passphrase)
        item = CandidateForReview(group["representative_id"], group["name"],
                                  group["reason"], candidate)
        rule = deterministic_decision(item)
        if rule is not None:
            rule_decisions.append(rule)
        elif len(model_inputs) < model_limit:
            model_inputs.append(item)
    model_decisions = []
    route_reason = "no_model_needed" if not model_inputs else "deferred"
    if model_inputs:
        gpu = probe_gpu()
        installed, loaded = probe_ollama(base_url)
        if gpu is not None:
            gpu = replace(gpu, loaded_models=loaded)
        route = choose_route(gpu, installed, cloud_allowed=False,
                             model_hints=(model,))
        route_reason = route.reason
        if route.provider == "ollama" and route.model == model:
            for start in range(0, len(model_inputs), 12):
                model_decisions.extend(classify_local(
                    model_inputs[start:start + 12], model=model,
                    base_url=base_url, keep_alive=keep_alive,
                ))
    if apply:
        for decision in [*rule_decisions, *model_decisions]:
            store.decide_ambiguity_group(
                decision.id, passphrase=passphrase, decision=decision.decision,
                actor="local-agent", reason="not-a-secret" if decision.decision == "rejected" else "",
            )
    queue_counts = store.ambiguity_status()
    return {
        "groups_seen": len(groups), "rule_rejected": sum(d.decision == "rejected" for d in rule_decisions),
        "model_rejected": sum(d.decision == "rejected" for d in model_decisions),
        "deferred_for_user": sum(d.decision == "deferred" for d in [*rule_decisions, *model_decisions]),
        "left_pending": queue_counts.get("pending", 0), "model_route": route_reason,
        "applied": apply, "queue_counts": queue_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
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
        for page in range(1, args.max_pages + 1):
            report = run(root=args.root, passphrase=passphrase, limit=args.limit,
                         model_limit=args.model_limit, model=args.model,
                         apply=args.apply, base_url=args.ollama_url,
                         keep_alive="30s" if args.max_pages > 1 else 0)
            print(json.dumps({"page": page, **report}, sort_keys=True), flush=True)
            decided = (report["rule_rejected"] + report["model_rejected"]
                       + report["deferred_for_user"])
            if (not args.apply or report["groups_seen"] == 0 or decided == 0
                    or report["queue_counts"].get("pending", 0) == 0):
                break
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
        print(json.dumps({"state": "failed", "error_category": type(exc).__name__,
                          "backup_state": backup_state,
                          "post_backup_unavailable": args.backup_after is not None
                          and backup_state != "validated_post_triage_backup"},
                         sort_keys=True), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
