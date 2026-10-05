"""The installed compact profile must describe and expose complete recall paths."""
import pytest

from muninn.mcp.definitions import toolset_schemas
from muninn.mcp.prompts import protocol_for


def test_core_profile_can_finish_pending_transcript_build():
    names = {schema["name"] for schema in toolset_schemas("core")}
    assert {"start_secure_history_transcript", "poll_secure_history_transcript",
            "read_secure_history_transcript_page"} <= names
    assert len(names) <= 20


def test_core_guidance_does_not_require_full_only_tools():
    instructions = protocol_for("core")
    for full_only in ("get_thread", "get_project_goal", "update_memory", "correct_fact"):
        assert full_only not in instructions
    assert "get_project_goal" in {schema["name"] for schema in toolset_schemas("full")}


@pytest.mark.parametrize("profile", ["core", "full", "readonly"])
def test_agent_protocol_distinguishes_recall_sources_and_authority(profile):
    instructions = protocol_for(profile)
    for fragment in ("search_cited_memories", "get_cited_memory_source",
                     "start_secure_history_search", "search_credential_metadata",
                     "not verified facts", "does not grant authority"):
        assert fragment in instructions
    assert "not federated" in instructions


def test_chatgpt_protocol_does_not_advertise_unavailable_tools():
    assert "search_cited_memories" not in protocol_for("chatgpt")
