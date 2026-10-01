"""Window derivation inherits sealed source EOF; no inference/coverage claim."""
import json

import pytest

from muninn.history.cited_windows import CitedWindowPlanStore
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_cited_analysis_source import fixture


def prepared(tmp_path):
    archive, source, _cap = fixture(tmp_path, text="Bounded source context. " * 500)
    entry = next(e for (blob, version), e in source.ledger._entries.items() if version == 0)
    plans = CitedWindowPlanStore(archive)
    source_attempt = plans.source.ledger.units.build_snapshot(entry, 0)
    return archive, plans, entry, source_attempt


def test_new_plan_from_sealed_units_needs_no_another_raw_pass(tmp_path, monkeypatch):
    archive, plans, entry, source_attempt = prepared(tmp_path)
    expected = list(plans._descriptors(entry, 0, source_attempt))
    def forbidden(*args, **kwargs):
        pytest.fail("A sealed unit attempt already authenticated complete raw EOF")
    monkeypatch.setattr(archive, "_iter_verified_entry", forbidden)
    attempt = plans.build_snapshot(entry, 0)
    actual = [json.loads(plans.get_page(entry, 0, attempt, ordinal)) for ordinal in range(len(expected))]
    assert actual == expected
    assert plans.verify_all()["windows"] == len(expected)


def test_initial_plan_uses_only_the_existing_two_source_unit_passes(tmp_path, monkeypatch):
    archive, source, _cap = fixture(tmp_path)
    entry = next(e for (blob, version), e in source.ledger._entries.items() if version == 0)
    plans = CitedWindowPlanStore(archive)
    original = archive._iter_verified_entry
    reads = []
    def counted(snapshot):
        reads.append(snapshot["blob"])
        yield from original(snapshot)
    monkeypatch.setattr(archive, "_iter_verified_entry", counted)
    attempt = plans.build_snapshot(entry, 0)
    assert plans.count_pages(entry, 0, attempt) > 0
    assert reads == [entry["blob"], entry["blob"]]


@pytest.mark.parametrize("damage", ["missing", "reordered", "late_tamper"])
def test_invalid_parent_fragment_stream_cannot_publish_derived_plan(tmp_path, damage):
    archive, plans, entry, source_attempt = prepared(tmp_path)
    with plans.source.ledger.units._connect() as db:
        if damage == "missing":
            db.execute("DELETE FROM pages WHERE attempt=? AND ordinal=1", (source_attempt,))
        elif damage == "reordered":
            db.execute("UPDATE pages SET ordinal=-1 WHERE attempt=? AND ordinal=0", (source_attempt,))
            db.execute("UPDATE pages SET ordinal=0 WHERE attempt=? AND ordinal=1", (source_attempt,))
            db.execute("UPDATE pages SET ordinal=1 WHERE attempt=? AND ordinal=-1", (source_attempt,))
        else:
            db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                       "WHERE attempt=? AND ordinal=(SELECT MAX(ordinal) FROM pages WHERE attempt=?)",
                       (source_attempt, source_attempt))
    with pytest.raises(ProjectionIntegrityError):
        plans.build_snapshot(entry, 0)
    with plans._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE state='complete'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
