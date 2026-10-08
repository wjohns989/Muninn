"""Local operator review against isolated encrypted archives; no model calls."""
import argparse
import io
import json
import sys

import pytest

from muninn import cli
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.private_acl import create_private_directory

PHRASE = "synthetic portable memory review phrase"
TEXT = "Keep the original source citations."


class TTY(io.StringIO):
    def isatty(self):
        return True


def candidate(tmp_path):
    source = tmp_path / "chat.jsonl"
    source.write_text(json.dumps({"timestamp": "2026-09-30T12:00:00Z",
        "type": "response_item", "payload": {"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": TEXT}]}}) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PHRASE)
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    units = SourceEvidenceStore(archive)
    attempt = units.build_snapshot(entry, 0)
    page = next(n for n in range(units.count_pages(entry, 0, attempt))
                if json.loads(units.get_page(entry, 0, attempt, n))["text"] == TEXT)
    ledger = MemoryLedger(archive)
    ident = ledger.record(entry, 0, attempt, page,
        {"type": "observation", "text": TEXT, "quote": TEXT, "start": 0},
        model_identity="a" * 64)
    return archive, ledger, ident


def args(root, action="status", **extra):
    values = dict(archive_root=root, action=action, record_id=None, state=None,
                  expected_state=None, reason=None, backup_before=None)
    values.update(extra)
    return argparse.Namespace(**values)


def terminal(monkeypatch, confirmation=""):
    out = TTY()
    monkeypatch.setattr(sys, "stdin", TTY())
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr("getpass.getpass", lambda _: PHRASE)
    monkeypatch.setattr("builtins.input", lambda _: confirmation)
    return out


def test_noninteractive_review_refuses_before_unlock_or_store_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr("getpass.getpass", lambda _: pytest.fail("must not prompt"))
    with pytest.raises(SystemExit, match="interactive local terminal"):
        cli.cmd_memories(args(tmp_path / "missing"))
    assert not (tmp_path / "missing").exists()


def test_parser_requires_explicit_review_binding_and_no_passphrase_argv():
    parser = cli.build_parser()
    parsed = parser.parse_args(["memories", "review", "--archive-root", "archive",
        "--record-id", "a" * 64, "--state", "filed", "--expected-state", "needs_user",
        "--reason", "source_supported", "--backup-before", "backup"])
    assert parsed.command == "memories" and parsed.expected_state == "needs_user"
    with pytest.raises(SystemExit):
        parser.parse_args(["memories", "status", "--archive-root", "archive", "--passphrase", PHRASE])


def test_status_and_get_keep_truth_provenance_and_hide_capabilities(tmp_path, monkeypatch):
    archive, ledger, ident = candidate(tmp_path)
    out = terminal(monkeypatch)
    assert cli.cmd_memories(args(archive.root)) == 0
    assert json.loads(out.getvalue())["provisional"] == 1
    out.seek(0); out.truncate()
    assert cli.cmd_memories(args(archive.root, "get", record_id=ident)) == 0
    shown = json.loads(out.getvalue())
    assert shown["memory"]["truth_status"] == ledger.get(ident)["truth_status"]
    assert shown["context"] == TEXT
    assert "transcript_capability" not in shown
    assert PHRASE not in out.getvalue()


def test_review_list_is_authenticated_read_only_and_keeps_the_compact_cursor(tmp_path, monkeypatch):
    archive, ledger, ident = candidate(tmp_path)
    before = ledger.verify_all()
    out = terminal(monkeypatch)
    assert cli.cmd_memories(args(archive.root, "review-list", limit=1, cursor=None)) == 0
    result = json.loads(out.getvalue())
    assert result["matches"][0]["id"] == ident
    assert result["matches"][0]["state"] == "provisional"
    assert not result["has_more"] and result["limit"] == 1
    assert PHRASE not in out.getvalue() and "transcript_capability" not in out.getvalue()
    assert ledger.verify_all() == before
    parsed = cli.build_parser().parse_args(["memories", "review-list", "--archive-root", "archive",
                                           "--limit", "1", "--cursor", "opaque"])
    assert parsed.cursor == "opaque" and parsed.limit == 1


def test_grouped_triage_cli_requires_backup_before_prompt_and_quit_is_read_only(tmp_path, monkeypatch):
    archive, ledger, _ident = candidate(tmp_path)
    out = terminal(monkeypatch, "q")
    parsed = cli.build_parser().parse_args(["memories", "triage", "--archive-root", str(archive.root),
                                          "--limit", "6"])
    monkeypatch.setattr("getpass.getpass", lambda _: pytest.fail("must validate before prompt"))
    with pytest.raises(SystemExit, match="--backup-before"):
        cli.cmd_memories(parsed)
    parsed.backup_before = tmp_path / "unused"
    monkeypatch.setattr("getpass.getpass", lambda _: PHRASE)
    before = ledger.verify_all()
    assert cli.cmd_memories(parsed) == 2
    assert ledger.verify_all() == before and not parsed.backup_before.exists()
    assert "grouped_review_page" in out.getvalue()
    assert PHRASE not in out.getvalue() and "transcript_capability" not in out.getvalue()


@pytest.mark.parametrize("state,reason", [("filed", "source_supported"), ("rejected", "user_rejected")])
def test_review_typed_confirmation_backup_then_cas(tmp_path, monkeypatch, state, reason):
    archive, ledger, ident = candidate(tmp_path)
    before = ledger.get(ident)
    backups = tmp_path / "backups"
    create_private_directory(backups)
    destination = backups / "before-review"
    out = terminal(monkeypatch, f"{state} {ident}")
    assert cli.cmd_memories(args(archive.root, "review", record_id=ident, state=state,
        expected_state="provisional", reason=reason, backup_before=destination)) == 0
    after = ledger.get(ident)
    assert after == {**before, "state": state}
    assert (destination / "memory-ledger.sqlite3").exists()
    assert json.loads(out.getvalue().splitlines()[-1])["state"] == state
    assert (ledger.search("citations")["total_matches"] == 0) is (state == "rejected")


def test_cancel_or_stale_state_does_not_write(tmp_path, monkeypatch):
    archive, ledger, ident = candidate(tmp_path)
    before = ledger.verify_all()
    terminal(monkeypatch, "no")
    destination = tmp_path / "cancelled-backup"
    assert cli.cmd_memories(args(archive.root, "review", record_id=ident, state="filed",
        expected_state="provisional", reason="user_confirmed", backup_before=destination)) == 2
    assert not destination.exists() and ledger.verify_all() == before
    terminal(monkeypatch, f"filed {ident}")
    with pytest.raises(SystemExit, match="state changed"):
        cli.cmd_memories(args(archive.root, "review", record_id=ident, state="filed",
            expected_state="needs_user", reason="user_confirmed", backup_before=destination))
    assert not destination.exists() and ledger.verify_all() == before


def test_wrong_passphrase_is_not_displayed_or_downgraded(tmp_path, monkeypatch):
    archive, ledger, _ = candidate(tmp_path)
    before = ledger.verify_all()
    out = terminal(monkeypatch)
    monkeypatch.setattr("getpass.getpass", lambda _: "wrong synthetic passphrase")
    with pytest.raises(SystemExit, match="VaultIntegrityError"):
        cli.cmd_memories(args(archive.root))
    assert "wrong synthetic" not in out.getvalue()
    assert ledger.verify_all() == before
