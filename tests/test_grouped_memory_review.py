"""Bounded local consultation on synthetic encrypted sources; no inference."""
import json

import pytest

from muninn.history.memory_ledger import MemoryLedger
from muninn.history.private_acl import create_private_directory
from muninn.history.source_evidence import SourceEvidenceStore
from tests.test_memory_ledger import fixture, record
from tests.test_memory_review_queue import files_digest


def proposals(tmp_path):
    text = "Keep citations. Preserve provenance."
    archive, entry, attempt, page = fixture(tmp_path, text=text, role="assistant")
    ledger = MemoryLedger(archive)
    ids = [record(ledger, entry, attempt, page, text=quote, quote=quote,
                  start=text.index(quote), type="fact")
           for quote in ("Keep citations.", "Preserve provenance.")]
    return ledger, ids


def add_source(ledger, path, *, cwd, timestamp):
    rows = [{"type": "session_meta", "payload": {"cwd": cwd}}] if cwd else []
    event = {"type": "response_item", "payload": {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Keep citations."}]}}
    if timestamp:
        event["timestamp"] = timestamp
    path.write_text("\n".join(json.dumps(row) for row in [*rows, event]) + "\n", encoding="utf-8")
    ledger.archive.archive_file(path, "codex")
    entry = ledger.archive._load_manifest()["files"][str(path.resolve())][0]
    units = SourceEvidenceStore(ledger.archive)
    attempt = units.build_snapshot(entry, 0)
    page = next(i for i in range(units.count_pages(entry, 0, attempt))
                if json.loads(units.get_page(entry, 0, attempt, i))["text"] == "Keep citations.")
    fresh = MemoryLedger(ledger.archive)
    return record(fresh, entry, attempt, page, text="Keep citations.", type="fact")


def test_grouping_uses_original_source_not_excerpt_citation_and_never_writes(tmp_path, monkeypatch):
    ledger, ids = proposals(tmp_path)
    readonly = MemoryLedger(ledger.archive, read_only=True)
    before = files_digest(ledger.archive.root)
    monkeypatch.setattr(readonly.units, "_store_screen_info", lambda *a, **kw: pytest.fail("browse wrote"))
    result = readonly.grouped_review_page()
    assert result["grouping_scope"] == "one_anchored_page"
    assert len(result["groups"]) == 1
    items = result["groups"][0]["items"]
    assert {item["id"] for item in items} == set(ids)
    assert len({item["source_ref"] for item in items}) == 2
    assert all(item["truth_status"] == "model_inferred" for item in items)
    assert files_digest(ledger.archive.root) == before
    assert "C:/synthetic-project" not in json.dumps(result)
    assert "source_group" not in json.dumps(readonly.review_page())


def test_other_sources_projects_and_unknown_attribution_are_not_merged(tmp_path):
    ledger, ids = proposals(tmp_path)
    other = add_source(ledger, tmp_path / "other.jsonl", cwd="C:/another-project",
                       timestamp="2026-09-29T12:00:00Z")
    unknown = add_source(ledger, tmp_path / "unknown.jsonl", cwd=None, timestamp=None)
    result = MemoryLedger(ledger.archive, read_only=True).grouped_review_page()
    assert len(result["groups"]) == 3
    assert sorted(len(group["items"]) for group in result["groups"]) == [1, 1, 2]
    by_id = {item["id"]: (group, item) for group in result["groups"] for item in group["items"]}
    assert by_id[other][0]["project_ref"] != by_id[ids[0]][0]["project_ref"]
    assert by_id[unknown][0]["project_ref"] is None
    assert by_id[unknown][1]["event_at"] is None
    assert by_id[unknown][1]["time_basis"] == "unknown"
    assert by_id[other][1]["event_at"] < by_id[ids[0]][1]["event_at"]


def test_grouped_pages_keep_original_anchor_and_exclude_credentials(tmp_path):
    ledger, ids = proposals(tmp_path)
    first = ledger.grouped_review_page(limit=1)
    assert first["has_more"] and first["next_cursor"]
    ledger.resolve_review(ids[1], state="needs_user", expected_state="provisional",
                          reason="possible_contradiction")
    second = ledger.grouped_review_page(limit=1, cursor=first["next_cursor"])
    assert second["groups"][0]["items"][0]["state"] == "needs_user"
    assert not second["has_more"]
    assert second["snapshot_events"] < second["current_events"]
    with pytest.raises(ValueError):
        ledger.grouped_review_page(limit=2, cursor=first["next_cursor"])
    # Existing page screening is still authoritative, not the group label.
    (tmp_path / "private").mkdir()
    archive, entry, attempt, page = fixture(tmp_path / "private")
    private = MemoryLedger(archive)
    record(private, entry, attempt, page, type="possible_credential")
    assert private.grouped_review_page()["groups"] == []


def test_grouping_does_not_grant_passphrase_authority(tmp_path):
    ledger, _ids = proposals(tmp_path)
    ledger.archive._unlocked_with_passphrase = False
    with pytest.raises(PermissionError):
        ledger.grouped_review_page()


def test_event_sorting_is_within_page_and_unknown_scope_stays_per_candidate(tmp_path):
    ledger, _ids = proposals(tmp_path)
    path = tmp_path / "dated.jsonl"
    rows = [{"type": "session_meta", "payload": {"cwd": "C:/dated-project"}}]
    for timestamp in ("2026-09-30T12:00:00Z", "2026-09-29T12:00:00Z"):
        rows.append({"timestamp": timestamp, "type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Keep citations."}]}})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    ledger.archive.archive_file(path, "codex")
    entry = ledger.archive._load_manifest()["files"][str(path.resolve())][0]
    units = SourceEvidenceStore(ledger.archive)
    attempt = units.build_snapshot(entry, 0)
    fresh = MemoryLedger(ledger.archive)
    ids = [record(fresh, entry, attempt, i, text="Keep citations.", type="fact")
           for i in range(units.count_pages(entry, 0, attempt))
           if json.loads(units.get_page(entry, 0, attempt, i))["text"] == "Keep citations."]
    group = next(g for g in fresh.grouped_review_page()["groups"]
                 if any(item["id"] == ids[0] for item in g["items"]))
    assert [item["id"] for item in group["items"]] == list(reversed(ids))
    seen, cursor = [], None
    while True:
        page = fresh.grouped_review_page(limit=1, cursor=cursor)
        seen.extend(item["id"] for g in page["groups"] for item in g["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == 4
    # A single source with two unknown-project excerpts is NOT treated as a
    # shared project just because the source's opaque identity matches.
    other = tmp_path / "unattributed"
    other.mkdir()
    archive, entry, attempt, page = fixture(other, text="Keep citations. Preserve provenance.",
                                           role="assistant", cwd=False)
    unknown = MemoryLedger(archive)
    for quote, start in (("Keep citations.", 0), ("Preserve provenance.", 16)):
        record(unknown, entry, attempt, page, text=quote, quote=quote, start=start, type="fact")
    groups = unknown.grouped_review_page()["groups"]
    assert len(groups) == 2 and all(len(g["items"]) == 1 and not g["scope_known"] for g in groups)


@pytest.mark.parametrize("failure", ["stale", "backup"])
def test_local_consumer_stale_or_failed_backup_prevents_decision(tmp_path, monkeypatch, failure):
    from muninn.history.memory_review import run_local_triage
    ledger, _ids = proposals(tmp_path)
    first = ledger.grouped_review_page()["groups"][0]["items"][0]
    parent = tmp_path / "backups"
    create_private_directory(parent)
    before = ledger.verify_all()
    responses = iter(["f", "filed " + first["id"]])

    def answer(prompt):
        response = next(responses)
        if failure == "stale" and response.startswith("filed "):
            ledger.resolve_review(first["id"], state="needs_user", expected_state="provisional",
                                  reason="insufficient_context")
        return response

    monkeypatch.setattr("builtins.input", answer)
    if failure == "backup":
        def broken_backup(*args, **kwargs):
            raise OSError("synthetic backup failure")
        monkeypatch.setattr(MemoryLedger, "backup_review_preimage", broken_backup)
    with pytest.raises(ValueError if failure == "stale" else OSError):
        run_local_triage(ledger.archive, backup_before=parent / "preimage")
    after = ledger.verify_all()
    assert after["decisions"] == before["decisions"] + (failure == "stale")
    assert ledger.get(first["id"])["state"] != "filed"


def test_multiple_explicit_decisions_share_one_verified_preimage(tmp_path, monkeypatch):
    from muninn.history.memory_review import run_local_triage
    ledger, _ids = proposals(tmp_path)
    items = ledger.grouped_review_page()["groups"][0]["items"]
    parent = tmp_path / "backups"
    create_private_directory(parent)
    calls = []
    original = MemoryLedger.backup_review_preimage

    def backup(self, path):
        calls.append(True)
        return original(self, path)

    monkeypatch.setattr(MemoryLedger, "backup_review_preimage", backup)
    responses = iter(["c", "needs_user " + items[0]["id"],
                      "r", "rejected " + items[1]["id"]])
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))
    assert run_local_triage(ledger.archive, backup_before=parent / "preimage") == 0
    assert calls == [True]
    assert ledger.get(items[0]["id"])["state"] == "needs_user"
    assert ledger.get(items[1]["id"])["state"] == "rejected"
    assert all(ledger.get(item["id"])["truth_status"] == item["truth_status"] for item in items)


def test_local_consumer_quit_and_bad_confirmation_never_backup_or_write(tmp_path, monkeypatch):
    from muninn.history.memory_review import run_local_triage
    ledger, _ids = proposals(tmp_path)
    before = files_digest(ledger.archive.root)
    parent = tmp_path / "backups"
    create_private_directory(parent)
    destination = parent / "preimage"
    answers = iter(["f", "wrong candidate", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert run_local_triage(ledger.archive, backup_before=destination) == 2
    assert not destination.exists()
    assert files_digest(ledger.archive.root) == before


def test_local_consumer_confirmed_decision_has_preimage_and_preserves_truth(tmp_path, monkeypatch, capsys):
    from muninn.history.memory_review import run_local_triage
    ledger, ids = proposals(tmp_path)
    first = MemoryLedger(ledger.archive, read_only=True).grouped_review_page()["groups"][0]["items"][0]
    parent = tmp_path / "backups"
    create_private_directory(parent)
    destination = parent / "preimage"
    answers = iter(["f", "filed " + first["id"], "s"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert run_local_triage(ledger.archive, backup_before=destination) == 0
    updated = MemoryLedger(ledger.archive).get(first["id"])
    assert updated["state"] == "filed"
    assert updated["truth_status"] == first["truth_status"]
    assert updated["source_ref"] == first["source_ref"]
    assert (destination / "memory-ledger.sqlite3").is_file()
    out = capsys.readouterr().out
    assert "transcript_capability" not in out
    assert "review_recorded" in out
    assert set(ids) == {item["id"] for item in ledger.review_page()["matches"]} | {first["id"]}
