"""Related cohorts retain source ACK ownership; isolated encrypted stores only."""
from collections import Counter

import pytest

from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from tests.test_analysis_publication_journal import queued, bind_stage, publish


def cohort_ack(tmp_path, count=2):
    journal, archive, job, stage, source = queued(tmp_path)
    stage["proposals"] = [{"type": "preference", "text": f"Keep source citations item {i}.",
        "quote": "Keep needle citations.", "start": 0} for i in range(count)]
    bind_stage(journal, job, stage)
    assert journal.begin_publication(job.job_id, job.lease_token)
    refs = publish(source, stage)
    assert journal.acknowledge_publication(job.job_id, job.lease_token, refs)
    return journal, archive, refs


def test_same_ack_related_candidates_share_one_owned_job(tmp_path):
    journal, archive, refs = cohort_ack(tmp_path)
    assert journal.discover_classifications() == {"acks": 1, "jobs": 1}
    job = journal.claim_classification()
    assert job["refs"] == refs
    assert journal.verify_classifications() == 1
    reopened = CaptureJournal(archive, recover=False)
    assert reopened.discover_classifications() == {"acks": 0, "jobs": 0}
    assert reopened.verify_classifications() == 1
    assert b"Keep source citations" not in journal.path.read_bytes()


def test_cohort_bounds_and_unknown_projects_stay_separate():
    from muninn.history.classification_enrollment import related_cohorts
    rows = [(str(i), "project") for i in range(25)]
    groups = related_cohorts(rows)
    assert list(map(len, groups)) == [12, 12, 1]
    assert Counter(ref for group in groups for ref in group) == Counter(ref for ref, _ in rows)
    assert related_cohorts([("a", None), ("b", None), ("c", "p"), ("d", "q")]) == [["a"], ["b"], ["c"], ["d"]]


def test_missing_or_foreign_member_blocks_recovery(tmp_path):
    journal, archive, refs = cohort_ack(tmp_path)
    journal.discover_classifications()
    with journal._connect() as db:
        db.execute("DELETE FROM memory_classification_members WHERE member_id=?", (journal._classification_member_id(refs[0]),))
    with pytest.raises(VaultIntegrityError, match="membership"):
        journal.verify_classifications()
    with pytest.raises(VaultIntegrityError, match="membership"):
        CaptureJournal(archive, recover=False)


def test_legacy_upgrade_preserves_existing_singleton_byte_for_byte(tmp_path):
    journal, archive, refs = cohort_ack(tmp_path, count=1)
    journal.discover_classifications()
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM memory_classification_jobs").fetchone())
        db.execute("DROP TABLE memory_classification_members")  # Isolated legacy-schema fixture.
    reopened = CaptureJournal(archive, recover=False)
    with reopened._connect() as db:
        assert tuple(db.execute("SELECT * FROM memory_classification_jobs").fetchone()) == before
    assert reopened.verify_classifications() == 1
    assert reopened.discover_classifications() == {"acks": 0, "jobs": 0}


def test_failed_legacy_seed_rolls_back_new_table_and_preserves_job(tmp_path):
    journal, archive, refs = cohort_ack(tmp_path, count=1)
    journal.discover_classifications()
    with journal._connect() as db:
        before = tuple(db.execute("SELECT * FROM memory_classification_jobs").fetchone())
        db.execute("DROP TABLE memory_classification_members")
        db.execute("DELETE FROM memory_classification_acks")  # Isolated damaged preimage.
    with pytest.raises(VaultIntegrityError, match="legacy ACK"):
        CaptureJournal(archive, recover=False)
    with journal._connect() as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_classification_members'").fetchone() is None
        assert tuple(db.execute("SELECT * FROM memory_classification_jobs").fetchone()) == before


@pytest.mark.parametrize("wrong", ["owner", "ack"])
def test_correctly_sealed_foreign_membership_is_rejected(tmp_path, wrong):
    from muninn.history.classification_enrollment import MEMBER_PURPOSE
    journal, archive, refs = cohort_ack(tmp_path, count=1)
    journal.discover_classifications()
    with journal._connect() as db:
        row = db.execute("SELECT * FROM memory_classification_members").fetchone()
        value = journal._open_search(row["sealed"], row["member_id"], MEMBER_PURPOSE)
        if wrong == "owner":
            db.execute("UPDATE memory_classification_members SET owner=?", ("d" * 32,))
        else:
            value["ack_id"] = "d" * 32
            db.execute("UPDATE memory_classification_members SET sealed=?",
                (journal._seal_search(value, row["member_id"], MEMBER_PURPOSE),))
    with pytest.raises(VaultIntegrityError, match="membership"):
        journal.verify_classifications()
