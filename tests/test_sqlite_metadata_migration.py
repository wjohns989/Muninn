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

from muninn.core.types import MemoryRecord, MemoryType, Provenance


def _record(memory_id: str, metadata: dict):
    return MemoryRecord(
        id=memory_id,
        content=f"content-{memory_id}",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        namespace="project-a",
        metadata=metadata,
    )


def test_count_and_fetch_missing_user_id(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "missing.db")
    store.add(_record("m1", {}))
    store.add(_record("m2", {"user_id": "user-1"}))
    store.add(_record("m3", {"other": True}))

    missing = store.get_missing_user_id_records(limit=10)

    assert store.count_missing_user_id() == 2
    assert {r.id for r in missing} == {"m1", "m3"}


def test_get_all_filters_by_user_id(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "scope.db")
    store.add(_record("u1", {"user_id": "user-1"}))
    store.add(_record("u2", {"user_id": "user-2"}))

    rows = store.get_all(limit=10, user_id="user-1")

    assert [r.id for r in rows] == ["u1"]
