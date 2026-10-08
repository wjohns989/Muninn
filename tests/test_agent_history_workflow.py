"""The installed compact profile must describe and expose complete recall paths."""
import pytest

from muninn.mcp.definitions import toolset_schemas
from muninn.mcp.prompts import protocol_for
from scripts.smoke_user_bridge_live import _failure_code, _recall_tools_present


def test_core_profile_can_finish_pending_transcript_build():
    names = {schema["name"] for schema in toolset_schemas("core")}
    assert {"start_secure_history_transcript", "poll_secure_history_transcript",
            "read_secure_history_transcript_page"} <= names
    assert len(names) <= 20


def test_live_probe_rejects_an_old_catalog_missing_transcript_poll():
    names = {schema["name"] for schema in toolset_schemas("core")}
    assert _recall_tools_present(names)
    for required in ("search_cited_memories", "get_cited_memory_source",
                     "start_secure_history_transcript", "poll_secure_history_transcript",
                     "read_secure_history_transcript_page"):
        assert not _recall_tools_present(names - {required})
    assert _failure_code(RuntimeError("transcript_workflow_missing")) == "transcript_workflow_missing"


def test_transcript_guidance_names_the_exact_pending_and_ready_continuations():
    schemas = {schema["name"]: schema for schema in toolset_schemas("core")}
    start = schemas["start_secure_history_transcript"]["description"]
    assert "poll_secure_history_transcript" in start
    assert "original capability" in start
    assert "Start once" in start
    assert "cited source" in start
    assert "read_secure_history_transcript_page" in start
    instructions = protocol_for("core")
    assert "original capability" in instructions
    assert "next_cursor" in instructions
    assert "not a job ID" in instructions
    assert "transcript_capability" in instructions


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
