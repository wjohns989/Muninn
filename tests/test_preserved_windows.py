"""Same-occurrence window matching; never processing/publication coverage."""
import json

import pytest

from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_cited_analysis_source import fixture


def setup_growth(tmp_path, *, build_prior=True, rewritten=False):
    archive, source, _cap = fixture(tmp_path, text="Keep exact source coordinates. " * 300)
    path = tmp_path / "chat.jsonl"
    first = archive._load_manifest()["files"][str(path.resolve())][0]
    old_plans = CitedWindowPlanStore(archive)
    old_attempt = old_plans.build_snapshot(first, 0) if build_prior else None
    addition = json.dumps({"type": "event_msg", "timestamp": "2026-09-30T12:02:00Z",
        "payload": {"type": "user_message", "message": "New occurrence with new context."}}) + "\n"
    if rewritten:
        path.write_text(path.read_text(encoding="utf-8").replace("exact", "other") + addition, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(addition)
    archive.archive_file(path, "codex")
    latest = archive._load_manifest()["files"][str(path.resolve())][1]
    plans = CitedWindowPlanStore(archive)
    attempt = plans.build_snapshot(latest, 1)
    return archive, plans, first, old_attempt, latest, attempt


def test_appended_occurrence_stays_new_while_exact_earlier_windows_match(tmp_path, monkeypatch):
    archive, plans, old, old_attempt, entry, attempt = setup_growth(tmp_path)
    old_count = plans.count_pages(old, 0, old_attempt)
    count = plans.count_pages(entry, 1, attempt)
    assert count > old_count > 2
    original = plans.build_snapshot
    def forbidden(*args, **kwargs):
        pytest.fail("Matching must not invent a prior window plan")
    monkeypatch.setattr(plans, "build_snapshot", forbidden)
    for ordinal in range(old_count):
        parent = plans.preserved_parent_window(entry, 1, attempt, ordinal)
        assert parent == plans.window_at(old, 0, old_attempt, ordinal)
        assert plans.source.reopen(parent) == plans.source.reopen(plans.window_at(entry, 1, attempt, ordinal))
    for ordinal in range(old_count, count):
        assert plans.preserved_parent_window(entry, 1, attempt, ordinal) is None
    monkeypatch.setattr(plans, "build_snapshot", original)
    assert not (archive.root / "capture-jobs.db").exists()  # Matching has no ACK/job effects.


@pytest.mark.parametrize("reason", ["no_prior_plan", "rewritten", "changed_geometry"])
def test_missing_proof_or_changed_partition_cannot_match_prior_window(tmp_path, reason):
    archive, plans, old, old_attempt, entry, attempt = setup_growth(
        tmp_path, build_prior=reason != "no_prior_plan", rewritten=reason == "rewritten")
    if reason == "changed_geometry":
        plans.window_chars = 1500
        attempt = plans.build_snapshot(entry, 1)
    assert plans.preserved_parent_window(entry, 1, attempt, 0) is None


@pytest.mark.parametrize("field", ["native_id", "physical_line", "ordinal"])
def test_input_text_digest_alone_cannot_substitute_for_source_occurrence(tmp_path, monkeypatch, field):
    archive, plans, old, old_attempt, entry, attempt = setup_growth(tmp_path)
    original = plans.source._window
    def mismatched_occurrence(descriptor):
        snapshot, page, window = original(descriptor)
        if descriptor["version"] == 1:
            page = {**page, "unit": {**page["unit"], field: "different-occurrence"}}
        return snapshot, page, window
    monkeypatch.setattr(plans.source, "_window", mismatched_occurrence)
    assert plans.preserved_parent_window(entry, 1, attempt, 0) is None


def test_matching_identical_text_in_different_file_is_not_a_parent(tmp_path):
    archive, plans, old, old_attempt, entry, attempt = setup_growth(tmp_path)
    copied = tmp_path / "another-project-chat.jsonl"
    copied.write_bytes((tmp_path / "chat.jsonl").read_bytes())
    archive.archive_file(copied, "codex")
    other = archive._load_manifest()["files"][str(copied.resolve())][0]
    other_plans = CitedWindowPlanStore(archive)
    other_attempt = other_plans.build_snapshot(other, 0)
    assert other_plans.preserved_parent_window(other, 0, other_attempt, 0) is None


def test_corrupt_prior_plan_is_error_not_a_match_or_silent_completion(tmp_path):
    archive, plans, old, old_attempt, entry, attempt = setup_growth(tmp_path)
    with plans._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=? AND ordinal=0",
                   (old_attempt,))
    with pytest.raises(ProjectionIntegrityError):
        plans.preserved_parent_window(entry, 1, attempt, 0)


def test_changed_current_entry_is_not_admitted_from_untrusted_metadata(tmp_path):
    archive, plans, old, old_attempt, entry, attempt = setup_growth(tmp_path)
    with pytest.raises(ProjectionIntegrityError):
        plans.preserved_parent_window({**entry, "sha256": "0" * 64}, 1, attempt, 0)
