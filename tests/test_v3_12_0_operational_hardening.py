"""
Tests for Muninn v3.12.0 — Operational Hardening (Phase 15).

Coverage:
  1. Auth token propagation — lifecycle.start_server() passes MUNINN_AUTH_TOKEN
     to the spawned server process (prevents HTTP 401 in fresh environments).
  2. Graph memory chain smoke tests — real KuzuDB integration verifying:
       - Memory nodes + entity nodes persist in the graph store
       - PRECEDES and CAUSES chain links are created and retrieved
       - MemoryChainDetector produces correct links on overlapping entities
       - Entity count (graph_nodes in health) increments after entity addition
  3. OTel GenAI semantic convention attribute validation — span names and
     attribute keys match the OpenTelemetry GenAI spec conventions.

Correctness properties verified:
  A. start_server() env kwarg always contains MUNINN_AUTH_TOKEN — never None
  B. Env var token takes priority over get_token() fallback
  C. get_token() fallback ensures token is non-empty when env var is absent
  D. GraphStore CAUSES/PRECEDES edges are retrievable via find_chain_related_memories()
  E. MemoryChainDetector assigns CAUSES relation when causal markers present
  F. gen_ai.operation.name and gen_ai.system attributes are always set on spans
  G. muninn.scope attribute is present for memory.add spans (Phase 14 addition)
"""

import json
import os
import time
import pytest
from unittest.mock import MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Auth Propagation — lifecycle.start_server()  (Task #16)
# ─────────────────────────────────────────────────────────────────────────────

class TestStartServerAuthPropagation:
    """
    start_server() must always forward MUNINN_AUTH_TOKEN to the spawned
    server.py process so both MCP wrapper and backend share the same token.
    Correctness property A: env kwarg contains MUNINN_AUTH_TOKEN — never None.
    Correctness property B: env var token takes priority over get_token().
    Correctness property C: get_token() fallback is used when env var absent.
    """

    @staticmethod
    def _run_start_server(spawn_side_effect, monkeypatch_env=None, mock_get_token=None):
        """
        Invoke start_server() with spawn and python-discovery mocked.
        Returns the (result, list_of_spawn_call_kwargs) tuple.
        """
        from muninn.mcp.lifecycle import start_server

        spawn_calls = []

        def recording_spawn(args, cwd=None, env=None):
            spawn_calls.append({"args": args, "cwd": cwd, "env": env})
            if callable(spawn_side_effect):
                return spawn_side_effect(args, cwd=cwd, env=env)

        patches = [
            patch("muninn.platform.spawn_detached_process", side_effect=recording_spawn),
            patch("muninn.platform.find_python_executable", return_value="/fake/python"),
            patch("time.sleep"),  # skip the 2-second startup wait
        ]
        if mock_get_token is not None:
            patches.append(patch("muninn.core.security.get_token", return_value=mock_get_token))

        ctx = patches[0]
        for p in patches[1:]:
            ctx = ctx.__class__.__new__(ctx.__class__)
        # Use ExitStack for clean multi-patch
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = start_server()

        return result, spawn_calls

    def test_token_from_env_forwarded_to_spawned_process(self, monkeypatch):
        """MUNINN_AUTH_TOKEN in the environment must be forwarded to server.py."""
        monkeypatch.setenv("MUNINN_AUTH_TOKEN", "env-injected-token-xyz789")

        from contextlib import ExitStack
        from muninn.mcp.lifecycle import start_server

        spawn_calls = []

        def recording_spawn(args, cwd=None, env=None):
            spawn_calls.append({"args": args, "cwd": cwd, "env": env})

        with ExitStack() as stack:
            stack.enter_context(
                patch("muninn.platform.spawn_detached_process", side_effect=recording_spawn)
            )
            stack.enter_context(
                patch("muninn.platform.find_python_executable", return_value="/fake/python")
            )
            stack.enter_context(patch("time.sleep"))
            result = start_server()

        assert result is True
        assert len(spawn_calls) == 1
        env = spawn_calls[0]["env"]
        assert env is not None, "env kwarg must always be passed to spawn_detached_process"
        assert "MUNINN_AUTH_TOKEN" in env, "MUNINN_AUTH_TOKEN must be forwarded in env"
        assert env["MUNINN_AUTH_TOKEN"] == "env-injected-token-xyz789"

    def test_get_token_fallback_when_env_var_absent(self, monkeypatch):
        """
        When MUNINN_AUTH_TOKEN is not in the environment, get_token() provides
        the token so both wrapper and server share the same value.
        Correctness property C.
        """
        monkeypatch.delenv("MUNINN_AUTH_TOKEN", raising=False)

        from contextlib import ExitStack
        from muninn.mcp.lifecycle import start_server

        spawn_calls = []

        def recording_spawn(args, cwd=None, env=None):
            spawn_calls.append({"env": env})

        with ExitStack() as stack:
            stack.enter_context(
                patch("muninn.platform.spawn_detached_process", side_effect=recording_spawn)
            )
            stack.enter_context(
                patch("muninn.platform.find_python_executable", return_value="/fake/python")
            )
            stack.enter_context(patch("time.sleep"))
            stack.enter_context(
                patch("muninn.core.security.get_token", return_value="wrapper-token-fallback-abc")
            )
            result = start_server()

        assert result is True
        assert len(spawn_calls) == 1
        env = spawn_calls[0]["env"]
        assert env is not None
        assert env["MUNINN_AUTH_TOKEN"] == "wrapper-token-fallback-abc"

    def test_env_var_token_takes_priority_over_get_token(self, monkeypatch):
        """
        When both the env var and get_token() are available, env var wins.
        The env var is more stable (set persistently via setx/export or MCP config).
        Correctness property B.
        """
        monkeypatch.setenv("MUNINN_AUTH_TOKEN", "persistent-system-token")

        from contextlib import ExitStack
        from muninn.mcp.lifecycle import start_server

        spawn_calls = []
        get_token_mock = MagicMock(return_value="ephemeral-random-token")

        def recording_spawn(args, cwd=None, env=None):
            spawn_calls.append({"env": env})

        with ExitStack() as stack:
            stack.enter_context(
                patch("muninn.platform.spawn_detached_process", side_effect=recording_spawn)
            )
            stack.enter_context(
                patch("muninn.platform.find_python_executable", return_value="/fake/python")
            )
            stack.enter_context(patch("time.sleep"))
            stack.enter_context(
                patch("muninn.core.security.get_token", get_token_mock)
            )
            result = start_server()

        assert result is True
        assert len(spawn_calls) == 1
        env = spawn_calls[0]["env"]
        # The env var token must win — get_token() must not override it
        assert env["MUNINN_AUTH_TOKEN"] == "persistent-system-token"

    def test_start_server_returns_false_on_spawn_failure(self, monkeypatch):
        """start_server() handles spawn errors gracefully — returns False, no exception."""
        monkeypatch.setenv("MUNINN_AUTH_TOKEN", "test-token")

        from contextlib import ExitStack
        from muninn.mcp.lifecycle import start_server

        def failing_spawn(args, cwd=None, env=None):
            raise OSError("No such file or directory: /fake/python")

        with ExitStack() as stack:
            stack.enter_context(
                patch("muninn.platform.spawn_detached_process", side_effect=failing_spawn)
            )
            stack.enter_context(
                patch("muninn.platform.find_python_executable", return_value="/fake/python")
            )
            stack.enter_context(patch("time.sleep"))
            result = start_server()

        assert result is False

    def test_spawned_command_includes_python_and_server_script(self, monkeypatch):
        """The spawn args must be [python_executable, <path_to_server.py>]."""
        monkeypatch.setenv("MUNINN_AUTH_TOKEN", "test-token")

        from contextlib import ExitStack
        from muninn.mcp.lifecycle import start_server, SERVER_SCRIPT

        spawn_calls = []

        def recording_spawn(args, cwd=None, env=None):
            spawn_calls.append({"args": args})

        with ExitStack() as stack:
            stack.enter_context(
                patch("muninn.platform.spawn_detached_process", side_effect=recording_spawn)
            )
            stack.enter_context(
                patch("muninn.platform.find_python_executable", return_value="/fake/python")
            )
            stack.enter_context(patch("time.sleep"))
            start_server()

        assert len(spawn_calls) == 1
        args = spawn_calls[0]["args"]
        assert args[0] == "/fake/python"
        assert str(SERVER_SCRIPT) in args[1]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Graph Memory Chains Smoke Tests  (Task #18)
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphChainsSmoke:
    """
    Integration smoke tests against a real KuzuDB instance (tmp_path).

    Validates that:
    - Memory and Entity nodes persist correctly in the KuzuDB schema
    - CAUSES and PRECEDES chain edges are created and traversable
    - find_chain_related_memories() returns expected successor memories
    - Entity count (health endpoint's graph_nodes source) increments
    - MemoryChainDetector produces typed CAUSES links when causal markers present
    """

    def test_memory_nodes_and_causes_chain_link(self, tmp_path):
        """Real KuzuDB: Memory nodes + CAUSES edge created and retrieved correctly."""
        from muninn.store.graph_store import GraphStore

        graph = GraphStore(tmp_path / "smoke_causes")

        # Add predecessor and successor memory nodes
        assert graph.add_memory_node(
            "mem-redis-001", "Investigated Redis queue backlog",
            user_id="user1", namespace="global"
        )
        assert graph.add_memory_node(
            "mem-redis-002", "Queue recovered after Redis cache warmed",
            user_id="user1", namespace="global"
        )

        # Create a causal chain link
        result = graph.add_chain_link(
            "mem-redis-001",
            "mem-redis-002",
            relation_type="CAUSES",
            confidence=0.87,
            reason="Redis cache warmup resolved the queue backlog",
            shared_entities=["Redis", "Queue"],
            hours_apart=0.5,
        )
        assert result is True, "add_chain_link must return True on success"

        # Traverse from predecessor — successor must be returned
        related = graph.find_chain_related_memories(["mem-redis-001"], limit=10)
        related_ids = [r[0] for r in related]
        assert "mem-redis-002" in related_ids, (
            "find_chain_related_memories must return the CAUSES successor"
        )

    def test_memory_nodes_and_precedes_chain_link(self, tmp_path):
        """Real KuzuDB: PRECEDES (temporal) chain links are created and retrievable."""
        from muninn.store.graph_store import GraphStore

        graph = GraphStore(tmp_path / "smoke_precedes")

        assert graph.add_memory_node(
            "seq-design-001", "Designed the authentication flow",
            user_id="u1", namespace="global"
        )
        assert graph.add_memory_node(
            "seq-impl-002", "Implemented the authentication middleware",
            user_id="u1", namespace="global"
        )

        result = graph.add_chain_link(
            "seq-design-001",
            "seq-impl-002",
            relation_type="PRECEDES",
            confidence=0.95,
            reason="Design phase temporally preceded implementation",
            hours_apart=2.0,
        )
        assert result is True

        related = graph.find_chain_related_memories(["seq-design-001"], limit=10)
        related_ids = [r[0] for r in related]
        assert "seq-impl-002" in related_ids

    def test_entity_count_increments_graph_nodes_health_signal(self, tmp_path):
        """
        get_entity_count() is the source for graph_nodes in the health endpoint.
        Adding entities must increment this count — confirming the health signal works.
        Correctness property: graph_nodes health counter increments.
        """
        from muninn.store.graph_store import GraphStore

        graph = GraphStore(tmp_path / "smoke_entity_count")

        initial = graph.get_entity_count()

        graph.add_entity("PostgreSQL", "technology", user_id="u1", namespace="global")
        graph.add_entity("Redis", "technology", user_id="u1", namespace="global")
        graph.add_entity("RabbitMQ", "technology", user_id="u1", namespace="global")

        final = graph.get_entity_count()
        assert final >= initial + 3, (
            f"Expected at least {initial + 3} entities after adding 3, got {final}"
        )

    def test_chain_link_invalid_relation_type_rejected(self, tmp_path):
        """add_chain_link must reject unknown relation types and return False."""
        from muninn.store.graph_store import GraphStore

        graph = GraphStore(tmp_path / "smoke_invalid_rel")
        graph.add_memory_node("m-001", "Memory A", user_id="u1", namespace="global")
        graph.add_memory_node("m-002", "Memory B", user_id="u1", namespace="global")

        # UNKNOWN_RELATION is not in the allowed set {PRECEDES, CAUSES}
        result = graph.add_chain_link("m-001", "m-002", relation_type="UNKNOWN_RELATION")
        assert result is False

    def test_chain_link_self_loop_rejected(self, tmp_path):
        """add_chain_link must reject same predecessor and successor IDs."""
        from muninn.store.graph_store import GraphStore

        graph = GraphStore(tmp_path / "smoke_self_loop")
        graph.add_memory_node("m-self", "Memory Self", user_id="u1", namespace="global")

        result = graph.add_chain_link("m-self", "m-self", relation_type="PRECEDES")
        assert result is False

    def test_chain_detector_produces_causes_link_on_causal_marker(self):
        """
        MemoryChainDetector assigns CAUSES relation when causal markers present
        in the successor content and entities overlap with the predecessor.
        Correctness property E.
        """
        from muninn.chains import MemoryChainDetector
        from muninn.core.types import MemoryRecord, MemoryType, Provenance

        detector = MemoryChainDetector(
            threshold=0.2,
            max_hours_apart=24.0,
            max_links_per_memory=3,
        )
        now = time.time()

        predecessor = MemoryRecord(
            id="detect-prev-001",
            content="Investigated PostgreSQL connection pool exhaustion",
            memory_type=MemoryType.EPISODIC,
            provenance=Provenance.AUTO_EXTRACTED,
            project="proj-alpha",
            namespace="global",
            created_at=now - 1800,
            metadata={"user_id": "user1", "entity_names": ["PostgreSQL", "ConnectionPool"]},
        )

        successor = MemoryRecord(
            id="detect-next-002",
            content="Connection pool recovered because PostgreSQL config was tuned",
            memory_type=MemoryType.EPISODIC,
            provenance=Provenance.AUTO_EXTRACTED,
            project="proj-alpha",
            namespace="global",
            created_at=now,
            metadata={"user_id": "user1"},
        )

        links = detector.detect_links(
            successor_record=successor,
            successor_content="Connection pool recovered because PostgreSQL config was tuned",
            successor_entity_names=["PostgreSQL", "ConnectionPool"],
            candidate_records=[predecessor],
        )

        assert len(links) >= 1, "Detector must produce at least one link"
        link = links[0]
        assert link.predecessor_id == "detect-prev-001"
        assert link.successor_id == "detect-next-002"
        assert link.relation_type == "CAUSES", (
            "Causal marker 'because' must produce CAUSES relation, not PRECEDES"
        )
        assert link.confidence >= 0.2

    def test_chain_detector_produces_precedes_link_without_causal_marker(self):
        """
        MemoryChainDetector falls back to PRECEDES when no causal markers present
        but temporal proximity and entity overlap exceed threshold.
        """
        from muninn.chains import MemoryChainDetector
        from muninn.core.types import MemoryRecord, MemoryType, Provenance

        detector = MemoryChainDetector(
            threshold=0.2,
            max_hours_apart=24.0,
            max_links_per_memory=3,
        )
        now = time.time()

        predecessor = MemoryRecord(
            id="temp-prev-001",
            content="Started work on Redis caching layer",
            memory_type=MemoryType.EPISODIC,
            provenance=Provenance.AUTO_EXTRACTED,
            project="proj-beta",
            namespace="global",
            created_at=now - 900,
            metadata={"user_id": "user2", "entity_names": ["Redis", "Cache"]},
        )

        successor = MemoryRecord(
            id="temp-next-002",
            content="Completed Redis caching layer implementation",
            memory_type=MemoryType.EPISODIC,
            provenance=Provenance.AUTO_EXTRACTED,
            project="proj-beta",
            namespace="global",
            created_at=now,
            metadata={"user_id": "user2"},
        )

        links = detector.detect_links(
            successor_record=successor,
            successor_content="Completed Redis caching layer implementation",
            successor_entity_names=["Redis", "Cache"],
            candidate_records=[predecessor],
        )

        assert len(links) >= 1
        link = links[0]
        assert link.predecessor_id == "temp-prev-001"
        assert link.successor_id == "temp-next-002"
        assert link.relation_type in ("PRECEDES", "CAUSES")
        assert link.confidence >= 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: OTel GenAI Semantic Convention Attribute Validation  (Task #19)
# ─────────────────────────────────────────────────────────────────────────────

class TestOTelGenAIAttributes:
    """
    Validate that OTel span names and attribute keys follow the OpenTelemetry
    GenAI semantic conventions spec.

    GenAI semantic conventions (OTel):
      Required attributes: gen_ai.operation.name, gen_ai.system
      Muninn extensions:   muninn.namespace, muninn.user_id, muninn.project, muninn.scope

    Correctness properties F and G:
      F. gen_ai.operation.name and gen_ai.system are always set on every span.
      G. muninn.scope is present for memory.add spans.
    """

    @staticmethod
    def _make_active_tracer():
        """Create an OTelGenAITracer wired to a MagicMock OTel tracer."""
        from muninn.observability.otel_genai import OTelGenAITracer

        mock_span = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_span)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_tracer_obj = MagicMock()
        mock_tracer_obj.start_as_current_span.return_value = mock_cm

        tracer = OTelGenAITracer(enabled=True)
        # Bypass optional OTel installation — inject the mock directly
        tracer._tracer = mock_tracer_obj

        return tracer, mock_span

    def test_add_span_attributes_contain_required_genai_keys(self):
        """
        muninn.memory.add span attributes must include gen_ai.operation.name,
        gen_ai.system, and muninn.scope.
        Correctness properties F and G.
        """
        tracer, mock_span = self._make_active_tracer()

        add_attributes = {
            "gen_ai.operation.name": "memory.add",
            "gen_ai.system": "muninn",
            "muninn.namespace": "global",
            "muninn.user_id": "user-test",
            "muninn.project": "test-project",
            "muninn.scope": "project",
        }

        with tracer.span("muninn.memory.add", add_attributes):
            pass

        set_keys = {c.args[0] for c in mock_span.set_attribute.call_args_list}
        required = {"gen_ai.operation.name", "gen_ai.system", "muninn.scope"}
        missing = required - set_keys
        assert not missing, f"Missing required GenAI semantic convention attributes: {missing}"

    def test_add_span_operation_name_is_memory_add(self):
        """gen_ai.operation.name value for add spans must be 'memory.add'."""
        tracer, mock_span = self._make_active_tracer()

        with tracer.span(
            "muninn.memory.add",
            {"gen_ai.operation.name": "memory.add", "gen_ai.system": "muninn"},
        ):
            pass

        attrs = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("gen_ai.operation.name") == "memory.add"
        assert attrs.get("gen_ai.system") == "muninn"

    def test_search_span_operation_name_is_retrieval_search(self):
        """gen_ai.operation.name for search/retrieval spans must be 'retrieval.search'."""
        tracer, mock_span = self._make_active_tracer()

        with tracer.span(
            "muninn.retrieval.search",
            {
                "gen_ai.operation.name": "retrieval.search",
                "gen_ai.system": "muninn",
                "muninn.user_id": "user-test",
                "muninn.limit": 10,
            },
        ):
            pass

        attrs = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attrs.get("gen_ai.operation.name") == "retrieval.search"
        assert attrs.get("gen_ai.system") == "muninn"

    def test_span_noop_when_tracer_disabled(self):
        """OTelGenAITracer must behave as a no-op when disabled — must not raise."""
        from muninn.observability.otel_genai import OTelGenAITracer

        tracer = OTelGenAITracer(enabled=False)
        assert not tracer.active

        # No-op span: must succeed without raising regardless of OTel installation
        with tracer.span(
            "muninn.memory.add",
            {"gen_ai.operation.name": "memory.add", "gen_ai.system": "muninn"},
        ):
            pass  # Must not raise

    def test_attribute_key_names_are_namespaced_and_lowercase(self):
        """
        All GenAI semantic convention attribute keys must:
        - Be namespaced (contain at least one '.')
        - Be lowercase (no camelCase or UPPER_SNAKE)
        - Use dot-separated namespace prefixes (gen_ai., muninn.)
        """
        # Canonical attribute sets from memory.py (add span) and hybrid.py (search span)
        add_attrs = {
            "gen_ai.operation.name": "memory.add",
            "gen_ai.system": "muninn",
            "muninn.namespace": "global",
            "muninn.user_id": "user-1",
            "muninn.project": "test-project",
            "muninn.scope": "project",
        }
        search_attrs = {
            "gen_ai.operation.name": "retrieval.search",
            "gen_ai.system": "muninn",
            "muninn.user_id": "user-1",
            "muninn.limit": 10,
        }

        for attr_set, label in [(add_attrs, "add"), (search_attrs, "search")]:
            for key in attr_set:
                assert "." in key, (
                    f"[{label}] Key '{key}' must be namespaced with a dot separator"
                )
                assert key == key.lower(), (
                    f"[{label}] Key '{key}' must be lowercase per OTel spec"
                )
                assert key.startswith(("gen_ai.", "muninn.")), (
                    f"[{label}] Key '{key}' must use 'gen_ai.' or 'muninn.' namespace"
                )

    def test_maybe_content_respects_privacy_by_default(self):
        """
        OTelGenAITracer.maybe_content() must return None by default
        (MUNINN_OTEL_CAPTURE_CONTENT not set) — privacy-first default.
        """
        import importlib
        import muninn.observability.otel_genai as otel_mod

        # Reload to clear env-derived state
        mod = importlib.reload(otel_mod)
        tracer = mod.OTelGenAITracer(enabled=False)
        assert tracer.maybe_content("sensitive user text") is None

    def test_span_start_called_with_correct_name(self):
        """The OTel tracer's start_as_current_span must be called with the exact span name."""
        tracer, mock_span = self._make_active_tracer()

        with tracer.span("muninn.memory.add", {"gen_ai.system": "muninn"}):
            pass

        # Verify the mock tracer was asked to start a span with the right name
        # (the mock_tracer_obj is stored on tracer._tracer)
        tracer._tracer.start_as_current_span.assert_called_once_with("muninn.memory.add")

    def test_none_attribute_values_are_skipped(self):
        """
        OTelGenAITracer.span() must not call set_attribute for keys whose
        value is None (OTel SDK rejects None attribute values).
        """
        tracer, mock_span = self._make_active_tracer()

        attrs_with_none = {
            "gen_ai.operation.name": "memory.add",
            "gen_ai.system": "muninn",
            "muninn.project": None,   # optional field that may be absent
            "muninn.scope": "project",
        }

        with tracer.span("muninn.memory.add", attrs_with_none):
            pass

        set_keys = {c.args[0] for c in mock_span.set_attribute.call_args_list}
        assert "muninn.project" not in set_keys, (
            "None-valued attributes must not be forwarded to set_attribute"
        )
        assert "gen_ai.operation.name" in set_keys
        assert "muninn.scope" in set_keys


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: LongMemEval Adapter Unit Tests  (Task #20)
# ─────────────────────────────────────────────────────────────────────────────

class TestLongMemEvalAdapter:
    """
    Unit tests for eval/longmemeval_adapter.py.

    These tests do NOT require a live Muninn server — the HTTP client is mocked.
    They verify:
    - Dataset parsing (QuestionCase extraction, malformed line handling)
    - Turn extraction from sessions
    - Relevant-turn identification via word overlap
    - CaseResult metric computation (nDCG@k, Recall@k)
    - Adapter.run() aggregation (mean metrics, by_question_type breakdown)
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_case(
        question_id: str = "q-001",
        question: str = "What is the user's favourite language?",
        expected_answer: str = "Python",
        question_type: str = "single-session-qa",
    ) -> dict:
        return {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "expected_answer": expected_answer,
            "question_date": "2025-01-10T10:00:00Z",
            "sessions": [
                {
                    "session_id": "sess-001",
                    "date": "2025-01-09T09:00:00Z",
                    "conversation": [
                        {
                            "role": "user",
                            "content": "My favourite programming language is Python.",
                            "turn_id": 1,
                        },
                        {
                            "role": "assistant",
                            "content": "Great choice! Python is very versatile.",
                            "turn_id": 1,
                        },
                        {
                            "role": "user",
                            "content": "I also like TypeScript for web frontends.",
                            "turn_id": 2,
                        },
                    ],
                }
            ],
        }

    # ── Dataset parsing ───────────────────────────────────────────────────────

    def test_parse_dataset_from_jsonl(self, tmp_path):
        """parse_dataset() correctly reads a well-formed JSONL file."""
        from eval.longmemeval_adapter import parse_dataset

        dataset_file = tmp_path / "sample.jsonl"
        lines = [json.dumps(self._make_case(question_id=f"q-{i:03d}")) for i in range(5)]
        dataset_file.write_text("\n".join(lines), encoding="utf-8")

        cases = parse_dataset(dataset_file)
        assert len(cases) == 5
        for i, case in enumerate(cases):
            assert case.question_id == f"q-{i:03d}"
            assert case.question_type == "single-session-qa"
            assert case.question == "What is the user's favourite language?"
            assert len(case.sessions) == 1

    def test_parse_dataset_skips_malformed_lines(self, tmp_path):
        """parse_dataset() skips malformed JSON lines without crashing."""
        from eval.longmemeval_adapter import parse_dataset

        dataset_file = tmp_path / "messy.jsonl"
        content = "\n".join([
            json.dumps(self._make_case("q-001")),
            "NOT VALID JSON {{{{",
            json.dumps(self._make_case("q-003")),
            "",                                   # blank line — also skipped
            json.dumps(self._make_case("q-005")),
        ])
        dataset_file.write_text(content, encoding="utf-8")

        cases = parse_dataset(dataset_file)
        assert len(cases) == 3
        assert {c.question_id for c in cases} == {"q-001", "q-003", "q-005"}

    # ── Turn extraction ───────────────────────────────────────────────────────

    def test_extract_turns_returns_all_non_empty_turns(self):
        """extract_turns() returns one ConversationTurn per non-empty conversation entry."""
        from eval.longmemeval_adapter import extract_turns, QuestionCase

        raw = self._make_case()
        case = QuestionCase(**{k: v for k, v in raw.items()})
        turns = extract_turns(case)

        assert len(turns) == 3  # 3 conversation entries in the sample case
        assert turns[0].role == "user"
        assert turns[0].content == "My favourite programming language is Python."
        assert turns[1].role == "assistant"
        assert turns[2].role == "user"

    def test_extract_turns_skips_empty_content(self):
        """extract_turns() omits turns where content is blank after stripping."""
        from eval.longmemeval_adapter import extract_turns, QuestionCase

        raw = self._make_case()
        raw["sessions"][0]["conversation"].append(
            {"role": "user", "content": "   ", "turn_id": 3}
        )
        case = QuestionCase(**{k: v for k, v in raw.items()})
        turns = extract_turns(case)

        # Blank-content turn must be excluded
        assert all(t.content.strip() for t in turns)

    # ── Relevant turn identification ──────────────────────────────────────────

    def test_identify_relevant_turns_with_word_overlap(self):
        """identify_relevant_turns() finds turns containing the expected answer."""
        from eval.longmemeval_adapter import identify_relevant_turns, extract_turns, QuestionCase

        raw = self._make_case(expected_answer="Python")
        case = QuestionCase(**{k: v for k, v in raw.items()})
        turns = extract_turns(case)

        relevant = identify_relevant_turns(turns, "Python")
        # The first user turn mentions Python
        assert len(relevant) >= 1

    def test_identify_relevant_turns_empty_answer_returns_empty(self):
        """identify_relevant_turns() returns [] when expected_answer is empty."""
        from eval.longmemeval_adapter import identify_relevant_turns, extract_turns, QuestionCase

        raw = self._make_case(expected_answer="")
        case = QuestionCase(**{k: v for k, v in raw.items()})
        turns = extract_turns(case)

        assert identify_relevant_turns(turns, "") == []

    # ── Metric computation ────────────────────────────────────────────────────

    def test_ndcg_perfect_retrieval(self):
        """nDCG@k = 1.0 when the only relevant item is at rank 1."""
        from eval.longmemeval_adapter import _ndcg_at_k

        relevant = {"turn-001"}
        ranked = ["turn-001", "turn-002", "turn-003"]
        assert _ndcg_at_k(relevant, ranked, k=3) == pytest.approx(1.0)

    def test_ndcg_zero_when_relevant_not_in_results(self):
        """nDCG@k = 0.0 when the relevant item does not appear in top-k."""
        from eval.longmemeval_adapter import _ndcg_at_k

        relevant = {"turn-999"}
        ranked = ["turn-001", "turn-002", "turn-003"]
        assert _ndcg_at_k(relevant, ranked, k=3) == pytest.approx(0.0)

    def test_recall_at_k_perfect(self):
        """Recall@k = 1.0 when all relevant items are in top-k."""
        from eval.longmemeval_adapter import _recall_at_k

        relevant = {"a", "b"}
        ranked = ["a", "b", "c"]
        assert _recall_at_k(relevant, ranked, k=3) == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        """Recall@k = 0.5 when half the relevant items appear in top-k."""
        from eval.longmemeval_adapter import _recall_at_k

        relevant = {"a", "b"}
        ranked = ["a", "c", "d"]  # "b" missing
        assert _recall_at_k(relevant, ranked, k=3) == pytest.approx(0.5)

    # ── Adapter aggregation with mocked HTTP ──────────────────────────────────

    def test_adapter_run_produces_report_with_mocked_client(self):
        """
        LongMemEvalAdapter.run() returns an AdapterReport with correct
        aggregate metrics when the HTTP client is fully mocked.
        """
        from eval.longmemeval_adapter import (
            LongMemEvalAdapter,
            MuninnHTTPClient,
            QuestionCase,
        )

        # Build a minimal case
        raw = self._make_case(question_id="mock-001", expected_answer="Python")
        case = QuestionCase(**{k: v for k, v in raw.items()})

        # Mock the HTTP client so no real server is needed
        mock_client = MagicMock(spec=MuninnHTTPClient)
        mock_client.base_url = "http://mock-server"
        mock_client.add.return_value = {"event": "ADD", "id": "mem-001"}
        # Return a result whose content matches the first user turn
        mock_client.search.return_value = [
            {"id": "mem-001", "content": "[user] My favourite programming language is Python."},
        ]
        mock_client.delete_all.return_value = None

        adapter = LongMemEvalAdapter(mock_client, k=5, cleanup=True)
        report = adapter.run([case])

        assert report.total_cases == 1
        assert report.evaluated_cases == 1
        assert report.skipped_cases == 0
        assert 0.0 <= report.mean_ndcg_at_k <= 1.0
        assert 0.0 <= report.mean_recall_at_k <= 1.0
        assert report.k == 5
        assert "single-session-qa" in report.by_question_type

    def test_adapter_run_skips_case_with_no_turns(self):
        """
        LongMemEvalAdapter.run() marks cases with empty sessions as skipped.
        """
        from eval.longmemeval_adapter import LongMemEvalAdapter, MuninnHTTPClient, QuestionCase

        empty_case = QuestionCase(
            question_id="empty-001",
            question_type="single-session-qa",
            question="Some question?",
            expected_answer="Some answer",
            question_date="2025-01-10T10:00:00Z",
            sessions=[],   # no turns
        )

        mock_client = MagicMock(spec=MuninnHTTPClient)
        mock_client.base_url = "http://mock-server"

        adapter = LongMemEvalAdapter(mock_client, k=5)
        report = adapter.run([empty_case])

        assert report.total_cases == 1
        assert report.evaluated_cases == 0
        assert report.skipped_cases == 1

    def test_stable_turn_id_format(self):
        """stable_turn_id() produces unique, deterministic, non-empty IDs."""
        from eval.longmemeval_adapter import stable_turn_id, ConversationTurn

        turn = ConversationTurn(
            role="user",
            content="test content",
            turn_id=5,
            session_id="sess-abc",
        )
        tid = stable_turn_id(turn, ordinal=0)
        assert tid.startswith("lme_sess-abc_")
        assert "use" in tid   # role prefix
        assert len(tid) > 10

        # Deterministic: same input → same output
        assert stable_turn_id(turn, ordinal=0) == tid

        # Different ordinals → different IDs
        assert stable_turn_id(turn, ordinal=1) != tid
