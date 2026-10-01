"""Status reads must not report corruption during a legitimate outbox commit."""
import threading

from tests.test_capture_enrichment import capture
from tests.test_capture_window_jobs import window_fixture


def test_schedule_status_pins_seal_and_mirrors_to_one_read_snapshot(tmp_path, monkeypatch):
    journal, archive, receipt = window_fixture(tmp_path)
    other = capture(archive, tmp_path / "other.jsonl", "Second source arrives.")["snapshot_receipt"]
    written = threading.Event()
    original_counts = journal._capture_schedule_counts
    original_write = journal._write_capture_schedule
    errors = []
    main_thread = threading.current_thread()

    def signal_write(db, value):
        result = original_write(db, value)
        if threading.current_thread() is not main_thread:
            written.set()
        return result

    def add_source():
        try:
            journal.enqueue_enrichment_receipt(other)
        except Exception as exc:
            errors.append(exc)

    writer = threading.Thread(target=add_source)
    started = False
    def interleave_counts(db):
        nonlocal started
        if threading.current_thread() is main_thread and not started:
            started = True
            writer.start()
            assert written.wait(2)
        return original_counts(db)

    monkeypatch.setattr(journal, "_capture_schedule_counts", interleave_counts)
    monkeypatch.setattr(journal, "_write_capture_schedule", signal_write)
    try:
        # The writer changes both sealed totals and rows between root read and
        # mirror reads. A consistent old OR new snapshot is valid, never a mix.
        report = journal.enrichment_status()
        assert report["pending_sources"] in (1, 2)
    finally:
        writer.join(2)
    assert not writer.is_alive() and not errors
    assert journal.enrichment_status()["pending_sources"] == 2
