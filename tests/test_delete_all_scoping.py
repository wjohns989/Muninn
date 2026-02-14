"""Tests for user-scoped delete_all — ensures multi-tenant isolation.

Validates that delete_all properly filters by user_id and namespace
so that one tenant cannot wipe another's data (Gemini security review).
"""

from pathlib import Path

import pytest

from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.core.types import MemoryRecord, MemoryType, Provenance


def _make_record(
    memory_id: str,
    user_id: str = "user-1",
    namespace: str = "global",
    content: str = "",
) -> MemoryRecord:
    """Create a minimal MemoryRecord for testing."""
    return MemoryRecord(
        id=memory_id,
        content=content or f"content-{memory_id}",
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.AUTO_EXTRACTED,
        namespace=namespace,
        metadata={"user_id": user_id},
    )


# ---------------------------------------------------------------------------
# SQLiteMetadataStore.delete_all scoping
# ---------------------------------------------------------------------------


class TestDeleteAllUserScoping:
    """Verify that delete_all respects user_id boundaries."""

    def test_delete_all_scoped_by_user_id(self, tmp_path: Path):
        """Only memories belonging to the specified user are removed."""
        store = SQLiteMetadataStore(tmp_path / "scope.db")
        store.add(_make_record("m1", user_id="alice"))
        store.add(_make_record("m2", user_id="alice"))
        store.add(_make_record("m3", user_id="bob"))

        count = store.delete_all(user_id="alice")

        assert count == 2
        remaining = store.get_all(limit=100)
        assert len(remaining) == 1
        assert remaining[0].id == "m3"

    def test_delete_all_scoped_by_namespace(self, tmp_path: Path):
        """Only memories in the specified namespace are removed."""
        store = SQLiteMetadataStore(tmp_path / "ns.db")
        store.add(_make_record("m1", namespace="proj-a"))
        store.add(_make_record("m2", namespace="proj-b"))
        store.add(_make_record("m3", namespace="proj-a"))

        count = store.delete_all(namespace="proj-a")

        assert count == 2
        remaining = store.get_all(limit=100)
        assert len(remaining) == 1
        assert remaining[0].id == "m2"

    def test_delete_all_scoped_by_user_and_namespace(self, tmp_path: Path):
        """Combined user_id + namespace narrows the scope correctly."""
        store = SQLiteMetadataStore(tmp_path / "both.db")
        store.add(_make_record("m1", user_id="alice", namespace="proj-a"))
        store.add(_make_record("m2", user_id="alice", namespace="proj-b"))
        store.add(_make_record("m3", user_id="bob", namespace="proj-a"))
        store.add(_make_record("m4", user_id="bob", namespace="proj-b"))

        count = store.delete_all(user_id="alice", namespace="proj-a")

        assert count == 1
        remaining = store.get_all(limit=100)
        remaining_ids = {r.id for r in remaining}
        assert remaining_ids == {"m2", "m3", "m4"}

    def test_delete_all_no_filters_wipes_everything(self, tmp_path: Path):
        """Calling without filters still deletes all (backward compat)."""
        store = SQLiteMetadataStore(tmp_path / "all.db")
        store.add(_make_record("m1", user_id="alice"))
        store.add(_make_record("m2", user_id="bob"))

        count = store.delete_all()

        assert count == 2
        assert store.count() == 0

    def test_delete_all_nonexistent_user_returns_zero(self, tmp_path: Path):
        """Deleting for a user with no memories is a no-op."""
        store = SQLiteMetadataStore(tmp_path / "empty.db")
        store.add(_make_record("m1", user_id="alice"))

        count = store.delete_all(user_id="charlie")

        assert count == 0
        assert store.count() == 1

    def test_delete_all_preserves_other_users(self, tmp_path: Path):
        """After scoped delete, other users' data is fully intact."""
        store = SQLiteMetadataStore(tmp_path / "preserve.db")
        store.add(_make_record("a1", user_id="alice", content="alice secret"))
        store.add(_make_record("a2", user_id="alice", content="alice note"))
        store.add(_make_record("b1", user_id="bob", content="bob secret"))
        store.add(_make_record("b2", user_id="bob", content="bob note"))

        store.delete_all(user_id="alice")

        bob_records = store.get_all(limit=100, user_id="bob")
        assert len(bob_records) == 2
        bob_ids = {r.id for r in bob_records}
        assert bob_ids == {"b1", "b2"}

        alice_records = store.get_all(limit=100, user_id="alice")
        assert len(alice_records) == 0

    def test_delete_all_user_id_no_sql_injection(self, tmp_path: Path):
        """User_id with SQL metacharacters does not cause injection.

        The LIKE pattern uses parameterised queries so injection is
        impossible. With JSON1 user_id filtering, the exact malicious
        identifier should match safely (no injection) and only that row
        is removed.
        """
        store = SQLiteMetadataStore(tmp_path / "inject.db")
        malicious_id = 'user"; DROP TABLE memories; --'
        store.add(_make_record("m1", user_id=malicious_id))
        store.add(_make_record("m2", user_id="normal"))

        # Should not raise and should not drop the table
        count = store.delete_all(user_id=malicious_id)

        # Exact match delete of only malicious-id row; no injection side effects
        assert count == 1
        # Table intact and other data preserved
        assert store.count() == 1
        remaining = store.get_all(limit=100)
        assert {r.id for r in remaining} == {"m2"}
