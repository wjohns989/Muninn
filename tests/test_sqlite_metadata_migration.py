from pathlib import Path

from muninn.store.sqlite_metadata import SQLiteMetadataStore


def test_schema_meta_roundtrip(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    store.set_meta("user_scope_migration_complete", "1")

    assert store.get_meta("user_scope_migration_complete") == "1"
    assert store.get_meta("missing", "0") == "0"


def test_user_scope_backfill_failure_ledger(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "ledger.db")

    store.record_user_scope_backfill_failure("mem-1", "initial error")
    store.record_user_scope_backfill_failure("mem-1", "retry failed")
    store.record_user_scope_backfill_failure("mem-2", "other failure")

    assert store.count_user_scope_backfill_failures() == 2
    assert set(store.get_user_scope_backfill_failures()) == {"mem-1", "mem-2"}

    store.clear_user_scope_backfill_failure("mem-1")

    assert store.count_user_scope_backfill_failures() == 1
    assert store.get_user_scope_backfill_failures() == ["mem-2"]
