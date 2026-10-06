"""Bounded human consultation; no model calls or shared-bearer write authority."""
import json

from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError

_CHOICES = {"f": ("filed", "user_confirmed"), "r": ("rejected", "user_rejected"),
            "u": ("needs_user", "insufficient_context"),
            "c": ("needs_user", "possible_contradiction")}


def run_local_triage(archive, *, backup_before, limit=20, cursor=None):
    """Consult on one anchored page; confirm each exact candidate separately.

    The preimage is a ledger-only baseline, not a portable whole-archive backup.
    It is validated before the first decision; no writer opens on browse/skip.
    """
    reader = MemoryLedger(archive, read_only=True)
    page = reader.grouped_review_page(limit=limit, cursor=cursor)
    print(json.dumps({"stage": "grouped_review_page", **page}, sort_keys=True), flush=True)
    writer = None
    reviewed = 0
    for group in page["groups"]:
        for candidate in group["items"]:
            view = reader.source(candidate["id"], max_chars=2000, include_transcript_capability=False)
            if (view is None or view["memory"]["state"] != candidate["state"]
                    or view["context_state"] != "available" or "text" not in view["memory"]):
                print(json.dumps({"stage": "candidate_changed_or_withheld"}), flush=True)
                continue
            shown = {key: value for key, value in view.items()
                     if key not in {"transcript_capability", "transcript_tool"}}
            print(json.dumps({"stage": "consult_candidate", "source_group": group["source_group"],
                              "grouping_scope": "one_anchored_page", **shown}, sort_keys=True), flush=True)
            try:
                choice = input("Does this cited interpretation belong in its shown project/type? "
                               "[f] file, [r] reject, [u] needs clarification, "
                               "[c] possible contradiction, [s] skip, [q] quit: ").strip().lower()
                if choice == "q":
                    return 2
                if choice not in _CHOICES:
                    continue
                state, reason = _CHOICES[choice]
                confirmation = f"{state} {candidate['id']}"
                if input(f"Type '{confirmation}' to confirm this exact candidate: ") != confirmation:
                    continue
            except (EOFError, KeyboardInterrupt):
                return 2
            if writer is None:
                # Authenticate a read-only baseline BEFORE opening any writer.
                backup = reader.backup_review_preimage(backup_before)
                print(json.dumps({"stage": "validated_ledger_only_preimage", **backup},
                                 sort_keys=True), flush=True)
                writer = MemoryLedger(archive)
            writer.resolve_review(candidate["id"], state=state,
                                  expected_state=candidate["state"], reason=reason)
            current = reader.get(candidate["id"])
            if (current is None or current["state"] != state or any(current[key] != candidate[key]
                    for key in ("source_ref", "truth_status", "epistemic_kind", "type",
                                "project_ref", "project_basis", "event_at", "time_basis"))):
                raise MemoryLedgerIntegrityError("Review readback failed")
            reviewed += 1
            print(json.dumps({"stage": "review_recorded", "id": candidate["id"], "state": state,
                              "truth_status": current["truth_status"]}, sort_keys=True), flush=True)
    print(json.dumps({"stage": "review_page_complete", "reviewed": reviewed,
                      "next_cursor": page["next_cursor"], "has_more": page["has_more"],
                      "grouping_scope": "one_anchored_page", "whole_queue_resolved": False},
                     sort_keys=True), flush=True)
    return 0
