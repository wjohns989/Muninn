"""Isolated preimage proof; never reads the installed policy or a live key."""

import sqlite3

import pytest

from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.remote_accounting import _MARKER as ADMISSION_MARKER, reserve
from muninn.history.remote_policy import write_policy
from scripts.backup_remote_policy_preimage import backup


def test_preimage_copies_and_verifies_managed_policy(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    write_policy(root, enabled=True, daily_usd=5.0, monthly_usd=50.0,
                 override_ceiling=False, fallback=lambda: (False, 1.0, 30.0, False))
    private = tmp_path / "private"
    create_private_directory(private)
    destination = private / "policy-before"
    report = backup(root, destination)
    assert report == {"state": "validated_preimage", "policy_generation": 1,
                      "policy_enabled": True, "accounting_state": "uninitialized",
                      "unresolved_admissions": 0, "restorable_as_active_policy": False}
    verify_private(destination)
    verify_private(destination / "managed")
    verify_private(destination / "policy.sqlite3")
    with sqlite3.connect(destination / "policy.sqlite3") as copy:
        assert copy.execute("SELECT daily_usd,monthly_usd FROM policy WHERE id=1").fetchone() == (5.0, 50.0)
    with pytest.raises(Exception):
        backup(root, destination)


def test_preimage_rejects_marker_written_before_accounting_commit(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    write_policy(root, enabled=True, daily_usd=5.0, monthly_usd=50.0,
                 override_ceiling=False, fallback=lambda: (False, 1.0, 30.0, False))
    admission_marker = root / "remote_policy" / "admission-managed"
    create_private_file(admission_marker)
    admission_marker.write_bytes(ADMISSION_MARKER)
    private = tmp_path / "private"
    create_private_directory(private)
    with pytest.raises(ValueError, match="marker and snapshot disagree"):
        backup(root, private / "policy-before")


def test_preimage_preserves_initialized_unresolved_admission(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    write_policy(root, enabled=True, daily_usd=5.0, monthly_usd=50.0,
                 override_ceiling=False, fallback=lambda: (False, 1.0, 30.0, False))
    reserve(root, 1, {"admission_ready": True, "usage_daily_usd": 0,
                      "usage_monthly_usd": 0})
    private = tmp_path / "private"
    create_private_directory(private)
    destination = private / "policy-before"
    report = backup(root, destination)
    assert report["accounting_state"] == "blocked"
    assert report["unresolved_admissions"] == 1
    assert (destination / "admission-managed").read_bytes() == ADMISSION_MARKER
    with sqlite3.connect(destination / "policy.sqlite3") as copy:
        assert copy.execute("SELECT count(*) FROM remote_admissions WHERE state='reserved'").fetchone() == (1,)
