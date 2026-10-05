"""Operational window counts are separate from source coverage; no model calls."""
from tests.test_capture_window_jobs import window_fixture


def test_window_status_counts_distinguish_failed_parked_and_unplanned_sources(tmp_path):
    journal, _archive, receipt = window_fixture(tmp_path)
    queued = journal.queue_capture_windows(receipt, limit=3)
    with journal._connect() as db:
        ids = [row[0] for row in db.execute(
            "SELECT job_id FROM history_analysis_jobs WHERE lane=1 ORDER BY job_id")]
        assert len(ids) >= 2
        db.execute("UPDATE history_analysis_jobs SET state='failed',error_code='model_unavailable' "
                   "WHERE job_id=?", (ids[0],))
        db.execute("UPDATE history_analysis_jobs SET state='retry',error_code='source_not_remote_safe' "
                   "WHERE job_id=?", (ids[1],))
    before = journal.path.read_bytes()
    status = journal.enrichment_status()
    assert status['pending_sources'] == 1
    assert status['parked_private_windows'] == 1
    assert status['window_jobs'] == {
        'basis': 'all_capture_lane_jobs', 'total': queued['queued'],
        'states': {'failed': 1, 'retry': 1, **({'pending': len(ids) - 2} if len(ids) > 2 else {})},
    }
    assert journal.path.read_bytes() == before
