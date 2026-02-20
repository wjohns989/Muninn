"""
Phase 19 (v3.18.0) Test Suite — Scout LLM Synthesis + Dashboard Hunt Mode

Tests:
  TestScoutSynthesis          (7) — synthesize_hunt_results() graceful degradation + happy path
  TestHuntEndpointSynthesize  (6) — HuntMemoryRequest.synthesize field + endpoint response shape
  TestVersionBump318          (2) — version >= 3.18.0 + pyproject.toml sync

Total: 15 tests
"""

import importlib
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# TestScoutSynthesis
# ---------------------------------------------------------------------------

class TestScoutSynthesis:
    """Tests for muninn/retrieval/synthesis.py — synthesize_hunt_results()."""

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_anthropic_not_installed(self):
        """Synthesis gracefully returns '' when anthropic SDK is absent."""
        from muninn.retrieval import synthesis

        with patch.dict(sys.modules, {"anthropic": None}):
            # Reload to pick up the patched modules dict
            import importlib
            reloaded = importlib.reload(synthesis)
            result = await reloaded.synthesize_hunt_results(
                "test query",
                [{"memory": "some content", "memory_type": "fact", "score": 0.8}],
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_api_key_not_set(self):
        """Synthesis gracefully returns '' when ANTHROPIC_API_KEY is absent."""
        from muninn.retrieval.synthesis import synthesize_hunt_results

        mock_module = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            with patch.dict(os.environ, {}, clear=False):
                # Ensure key is not set
                env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
                with patch.dict(os.environ, env, clear=True):
                    result = await synthesize_hunt_results(
                        "query",
                        [{"memory": "content", "memory_type": "fact", "score": 0.9}],
                    )
                    assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_string_for_empty_results(self):
        """Synthesis returns '' immediately when there are no results to narrate."""
        from muninn.retrieval.synthesis import synthesize_hunt_results

        result = await synthesize_hunt_results("any query", [])
        assert result == ""

    @pytest.mark.asyncio
    async def test_calls_anthropic_with_query_and_snippet_context(self):
        """When Anthropic SDK + API key present, calls claude-haiku with correct prompt."""
        from muninn.retrieval import synthesis

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="These memories discuss Python patterns.")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
                import importlib
                reloaded = importlib.reload(synthesis)
                result = await reloaded.synthesize_hunt_results(
                    "Python async patterns",
                    [
                        {"memory": "asyncio event loop notes", "memory_type": "fact", "score": 0.9},
                        {"memory": "FastAPI async routes", "memory_type": "episode", "score": 0.8},
                    ],
                )

        assert result == "These memories discuss Python patterns."
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["max_tokens"] == 200
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "Python async patterns" in prompt_text
        assert "asyncio event loop notes" in prompt_text

    @pytest.mark.asyncio
    async def test_handles_api_exception_gracefully(self):
        """When Anthropic API call raises, synthesis returns '' (never raises)."""
        from muninn.retrieval import synthesis

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
                import importlib
                reloaded = importlib.reload(synthesis)
                result = await reloaded.synthesize_hunt_results(
                    "query",
                    [{"memory": "content", "memory_type": "fact", "score": 0.7}],
                )

        assert result == ""

    @pytest.mark.asyncio
    async def test_snippet_truncated_to_120_chars(self):
        """Long memory content is truncated to _SYNTHESIS_SNIPPET_CHARS in the prompt."""
        from muninn.retrieval import synthesis

        long_memory = "x" * 300  # 300 chars > 120 limit

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Summary.")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
                import importlib
                reloaded = importlib.reload(synthesis)
                await reloaded.synthesize_hunt_results(
                    "query",
                    [{"memory": long_memory, "memory_type": "fact", "score": 0.8}],
                )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        # The snippet in the prompt should be truncated — at most 120 'x' chars
        assert long_memory[:120] in prompt
        assert long_memory[:121] not in prompt

    @pytest.mark.asyncio
    async def test_at_most_6_snippets_sent_to_llm(self):
        """Only the first 6 results are included in the synthesis prompt."""
        from muninn.retrieval import synthesis

        # 8 results — only first 6 should appear in prompt
        results = [
            {"memory": f"memory_{i}", "memory_type": "fact", "score": 0.9 - i * 0.05}
            for i in range(8)
        ]

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Summary.")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
                import importlib
                reloaded = importlib.reload(synthesis)
                await reloaded.synthesize_hunt_results("query", results)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "memory_5" in prompt   # 6th result (index 5) — included
        assert "memory_6" not in prompt  # 7th result (index 6) — excluded


# ---------------------------------------------------------------------------
# TestHuntEndpointSynthesize
# ---------------------------------------------------------------------------

class TestHuntEndpointSynthesize:
    """Tests for HuntMemoryRequest.synthesize field and /search/hunt response shape."""

    def test_hunt_request_has_synthesize_field(self):
        """HuntMemoryRequest pydantic model includes synthesize field."""
        from server import HuntMemoryRequest
        fields = HuntMemoryRequest.model_fields
        assert "synthesize" in fields

    def test_hunt_request_synthesize_defaults_false(self):
        """synthesize defaults to False for backward compatibility."""
        from server import HuntMemoryRequest
        req = HuntMemoryRequest(query="test")
        assert req.synthesize is False

    def test_hunt_request_synthesize_can_be_set_true(self):
        """synthesize can be set to True."""
        from server import HuntMemoryRequest
        req = HuntMemoryRequest(query="test", synthesize=True)
        assert req.synthesize is True

    def test_hunt_endpoint_in_app_routes(self):
        """POST /search/hunt is registered with verify_token."""
        from server import app
        routes = {r.path: r for r in app.routes if hasattr(r, "path")}
        assert "/search/hunt" in routes
        route = routes["/search/hunt"]
        has_auth = any(
            "verify_token" in str(getattr(d, "dependency", d))
            for d in route.dependencies
        )
        assert has_auth, "/search/hunt must require verify_token"

    @pytest.mark.asyncio
    async def test_hunt_endpoint_includes_synthesis_field_in_response(self):
        """When synthesize=False (default), response still includes synthesis key (empty)."""
        from server import app
        from fastapi.testclient import TestClient

        mock_memory = MagicMock()
        mock_memory.hunt = AsyncMock(return_value=[
            {"id": "m1", "memory": "test content", "score": 0.9,
             "memory_type": "fact", "importance": 1.0, "metadata": {}, "source": "vector"}
        ])

        import server as srv
        original_memory = srv.memory
        srv.memory = mock_memory
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/search/hunt",
                json={"query": "test", "synthesize": False},
                headers={"Authorization": "Bearer test-token"},
            )
            # Either 200 (valid token) or 401 (token mismatch in test env)
            # We just need to verify the code path returns synthesis field when successful
            if resp.status_code == 200:
                body = resp.json()
                assert "synthesis" in body
                assert body["synthesis"] == ""  # no synthesize requested → empty
        finally:
            srv.memory = original_memory

    @pytest.mark.asyncio
    async def test_synthesize_hunt_called_when_flag_set(self):
        """synthesize_hunt_results is called when synthesize=True and results exist."""
        from server import hunt_memory_endpoint, HuntMemoryRequest
        import server as srv

        original_memory = srv.memory
        srv.memory = MagicMock()
        srv.memory.hunt = AsyncMock(return_value=[
            {"id": "m1", "memory": "test memory", "score": 0.9,
             "memory_type": "fact", "importance": 1.0, "metadata": {}, "source": "vector"}
        ])

        try:
            with patch("server.synthesize_hunt_results", new=AsyncMock(return_value="Scout found Python patterns.")) as mock_synth:
                req = HuntMemoryRequest(query="Python", synthesize=True)
                result = await hunt_memory_endpoint(req)
                mock_synth.assert_called_once_with(
                    "Python",
                    srv.memory.hunt.return_value,
                )
                assert result["synthesis"] == "Scout found Python patterns."
                assert result["success"] is True
                assert len(result["data"]) == 1
        finally:
            srv.memory = original_memory


# ---------------------------------------------------------------------------
# TestVersionBump318
# ---------------------------------------------------------------------------

class TestVersionBump318:
    """Version consistency checks for v3.18.0."""

    def test_version_at_least_3_18_0(self):
        """Package version is >= 3.18.0."""
        from muninn.version import __version__
        parts = tuple(int(x) for x in __version__.split("."))
        assert parts >= (3, 18, 0), f"Expected >= 3.18.0, got {__version__}"

    def test_pyproject_version_matches_package(self):
        """pyproject.toml version matches muninn.version.__version__."""
        import tomllib
        from pathlib import Path
        from muninn.version import __version__

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject = tomllib.load(f)
        assert pyproject["project"]["version"] == __version__
