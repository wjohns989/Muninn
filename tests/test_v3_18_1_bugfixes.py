"""
Phase 19 bugfix tests (v3.18.1) — P0 + P2 regression coverage

Bugs fixed:
  P0 — hybrid.py: _record_matches_constraints didn't skip synthetic filter keys
       (memory_ids, scope), causing Scout's final re-rank to reject every record.
  P2 — handlers.py: _do_hunt_memory never forwarded synthesize=True and never
       surfaced the synthesis field for the LLM.
  P2 — dashboard.html: handleIngest() called /ingest (file ingestion) instead of /add.
  P2 — dashboard.html: forceConsolidation() called /maintenance/consolidate (non-existent)
       instead of /consolidation/run.

Tests:
  TestHybridSyntheticKeySkip   (6) — _record_matches_constraints skips memory_ids/scope
  TestMCPHuntSynthesisExposure (5) — _do_hunt_memory passes synthesize=True, injects
                                     synthesis_narrative into data list
  TestVersionBump3181          (2) — version >= 3.18.1, pyproject.toml sync

Total: 13 tests
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(record_id="rec1", namespace="global", memory_type="episodic", metadata=None):
    """Build a minimal MemoryRecord-like MagicMock for constraint tests."""
    record = MagicMock()
    record.id = record_id
    record.namespace = namespace
    record.memory_type = memory_type
    record.metadata = metadata or {"user_id": "global_user"}
    return record


# ---------------------------------------------------------------------------
# TestHybridSyntheticKeySkip
# ---------------------------------------------------------------------------

class TestHybridSyntheticKeySkip:
    """P0 regression: _record_matches_constraints must not re-evaluate synthetic keys."""

    def _get_retriever_class(self):
        from muninn.retrieval.hybrid import HybridRetriever
        return HybridRetriever

    def test_synthetic_keys_attribute_exists(self):
        """_SYNTHETIC_FILTER_KEYS frozenset must be defined on HybridRetriever."""
        cls = self._get_retriever_class()
        assert hasattr(cls, "_SYNTHETIC_FILTER_KEYS"), \
            "HybridRetriever must have _SYNTHETIC_FILTER_KEYS class attribute"

    def test_memory_ids_in_synthetic_keys(self):
        """'memory_ids' must be in _SYNTHETIC_FILTER_KEYS."""
        cls = self._get_retriever_class()
        assert "memory_ids" in cls._SYNTHETIC_FILTER_KEYS

    def test_scope_not_in_synthetic_keys(self):
        """'scope' must NOT be in _SYNTHETIC_FILTER_KEYS to prevent data leakage."""
        cls = self._get_retriever_class()
        assert "scope" not in cls._SYNTHETIC_FILTER_KEYS

    def test_record_not_rejected_by_memory_ids_filter(self):
        """A valid record must pass _record_matches_constraints even when
        filters contains memory_ids — the key must be skipped, not evaluated."""
        cls = self._get_retriever_class()
        retriever = object.__new__(cls)  # bypass __init__

        record = _make_record(record_id="abc123", metadata={"user_id": "u1"})
        # filters contains memory_ids — the exact pattern Scout uses
        result = retriever._record_matches_constraints(
            record,
            user_id="u1",
            namespaces=None,
            filters={"memory_ids": ["abc123", "def456", "ghi789"]},
        )
        assert result is True, \
            "_record_matches_constraints must return True when memory_ids is the only filter"

    def test_record_rejected_by_scope_filter(self):
        """A valid record must be rejected by _record_matches_constraints when filters
        contains a scope that does not match the record's scope."""
        cls = self._get_retriever_class()
        retriever = object.__new__(cls)

        record = _make_record(namespace="ns1", metadata={"user_id": "u1"})
        record.scope = "project"  # Default scope
        result = retriever._record_matches_constraints(
            record,
            user_id="u1",
            namespaces=None,
            filters={"scope": "global"},
        )
        assert result is False, \
            "_record_matches_constraints must return False when record scope does not match filter scope"

    def test_non_synthetic_filter_still_enforced(self):
        """Real (non-synthetic) filter keys must still be enforced."""
        cls = self._get_retriever_class()
        retriever = object.__new__(cls)

        # Record has memory_type="episodic"; filter requires "semantic"
        record = _make_record(metadata={"user_id": "u1"})
        record.memory_type = "episodic"

        result = retriever._record_matches_constraints(
            record,
            user_id="u1",
            namespaces=None,
            filters={"memory_type": "semantic"},
        )
        assert result is False, \
            "A non-matching real filter must still reject the record"


# ---------------------------------------------------------------------------
# TestMCPHuntSynthesisExposure
# ---------------------------------------------------------------------------

class TestMCPHuntSynthesisExposure:
    """P2 regression: _do_hunt_memory must forward synthesize=True and expose synthesis."""

    def _read_handler_source(self):
        """Return source lines of _do_hunt_memory for static checks."""
        import inspect
        from muninn.mcp.handlers import _do_hunt_memory
        return inspect.getsource(_do_hunt_memory)

    def test_synthesize_true_in_payload(self):
        """_do_hunt_memory must include 'synthesize': True in the POST payload."""
        source = self._read_handler_source()
        assert '"synthesize": True' in source or "'synthesize': True" in source, \
            "_do_hunt_memory must send synthesize=True in the payload"

    def test_synthesis_injected_into_data_list(self):
        """When server returns a non-empty synthesis, it must be prepended to data list."""
        from muninn.mcp import handlers

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": [{"id": "m1", "memory": "test memory", "score": 0.9}],
            "synthesis": "Scout found memories about Python patterns.",
        }

        with patch.object(handlers, "make_request_with_retry", return_value=mock_response):
            result = handlers._do_hunt_memory(
                {"query": "Python patterns", "limit": 5, "depth": 2},
                deadline=None,
            )

        assert isinstance(result["data"], list)
        assert result["data"][0].get("synthesis_narrative") == \
            "Scout found memories about Python patterns.", \
            "synthesis_narrative must be the first element of data list"
        assert result["data"][1]["id"] == "m1", \
            "Original memory records must follow synthesis_narrative"

    def test_empty_synthesis_does_not_alter_data(self):
        """When synthesis is empty string, data list must remain unchanged."""
        from muninn.mcp import handlers

        original_data = [{"id": "m1", "memory": "test", "score": 0.9}]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": list(original_data),
            "synthesis": "",
        }

        with patch.object(handlers, "make_request_with_retry", return_value=mock_response):
            result = handlers._do_hunt_memory(
                {"query": "test", "limit": 5, "depth": 2},
                deadline=None,
            )

        assert result["data"] == original_data, \
            "data list must be unmodified when synthesis is empty"

    def test_missing_synthesis_does_not_alter_data(self):
        """When server returns no synthesis key, data list must remain unchanged."""
        from muninn.mcp import handlers

        original_data = [{"id": "m2", "memory": "other", "score": 0.8}]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": list(original_data),
            # no 'synthesis' key — backward-compat with older server versions
        }

        with patch.object(handlers, "make_request_with_retry", return_value=mock_response):
            result = handlers._do_hunt_memory(
                {"query": "other", "limit": 5, "depth": 2},
                deadline=None,
            )

        assert result["data"] == original_data, \
            "data list must be unmodified when synthesis key is absent"

    def test_namespaces_forwarded(self):
        """namespaces arg must still be forwarded in the payload."""
        from muninn.mcp import handlers

        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "data": [], "synthesis": ""}
        captured = {}

        def capture_call(*args, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return mock_response

        with patch.object(handlers, "make_request_with_retry", side_effect=capture_call):
            handlers._do_hunt_memory(
                {"query": "q", "limit": 3, "depth": 1, "namespaces": ["project_x"]},
                deadline=None,
            )

        assert captured["json"]["namespaces"] == ["project_x"]
        assert captured["json"]["synthesize"] is True


# ---------------------------------------------------------------------------
# TestVersionBump3181
# ---------------------------------------------------------------------------

class TestVersionBump3181:
    """Version consistency checks for v3.18.1."""

    def test_version_at_least_3_18_1(self):
        """Package version is >= 3.18.1."""
        from muninn.version import __version__
        parts = tuple(int(x) for x in __version__.split("."))
        assert parts >= (3, 18, 1), f"Expected >= 3.18.1, got {__version__}"

    def test_pyproject_version_matches_package(self):
        """pyproject.toml version matches muninn.version.__version__."""
        import tomllib
        from pathlib import Path
        from muninn.version import __version__

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject = tomllib.load(f)
        assert pyproject["project"]["version"] == __version__
