"""Small isolated checks for the explicit read-only comparison boundary."""
import json
import sqlite3
import sys

import pytest

from scripts import prepare_native_cited_comparison as comparison


def test_readonly_connection_cannot_modify_an_existing_database(tmp_path, monkeypatch):
    path = tmp_path / "isolated.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE proof(value INTEGER)")
        db.execute("INSERT INTO proof VALUES(1)")
    monkeypatch.setattr(comparison, "verify_private", lambda path: None)
    with comparison.readonly(path) as db:
        assert db.execute("SELECT value FROM proof").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            db.execute("UPDATE proof SET value=2")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM proof").fetchone()[0] == 1


def test_screened_emission_withholds_original_answers_by_default(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["comparison", "--archive-root", str(tmp_path),
                                     "--emit-screened-samples"])
    monkeypatch.setattr(comparison, "samples", lambda *args, **kwargs: [
        {"id": "sample-1", "input": {"text": "Isolated test input."},
         "openrouter_original": {"summary": "Do not send this answer to the child."}}])
    comparison.main()
    result = json.loads(capsys.readouterr().out)
    assert result == [{"id": "sample-1", "input": {"text": "Isolated test input."}}]


def test_invalid_limits_fail_before_archive_or_key_access(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "SecureHistoryArchive",
                        lambda *args: pytest.fail("Invalid limits accessed the archive"))
    for limit in (0, 11, True, "10"):
        with pytest.raises(ValueError):
            comparison.samples(tmp_path, limit=limit)


def test_original_answers_require_explicit_emission(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["comparison", "--archive-root", str(tmp_path),
                                     "--include-originals-for-parent"])
    monkeypatch.setattr(comparison, "samples", lambda *args, **kwargs: pytest.fail("Archive opened"))
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2
