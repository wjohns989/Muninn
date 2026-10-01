"""Analysis contract invalidation, not model quality or reuse-coverage proof."""
import copy
import hashlib
import json

import pytest

from muninn.history import secure_analysis as analysis


@pytest.fixture
def window():
    return {"text": "Keep exact citations.", "provider": "codex", "role": "user",
            "event_at": 1790830800.0, "time_basis": "source_event", "project_ref": "a" * 64,
            "project_basis": "source_cwd", "boundary_hit": False,
            "citation_ranges": [{"start": 0, "length": 21}]}


def identity(window, model="fixture-model", digest="a" * 64):
    return analysis._cited_model_identity(window, "ollama", model, digest)


def test_current_identity_preserves_existing_staged_hash_contract(window):
    expected = hashlib.sha256(json.dumps({"version": analysis._CITED_VERSION,
        "schema": analysis._CITED_SCHEMA, "messages": analysis._cited_prompt(window),
        "provider": "ollama", "model": "fixture-model", "weights_digest": "a" * 64},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert identity(window) == expected


@pytest.mark.parametrize("change", ["classifier", "schema", "prompt", "weights", "model",
                                    "text", "role", "timestamp", "time_basis", "project_basis", "ranges"])
def test_material_contract_or_input_changes_invalidate_prior_identity(window, monkeypatch, change):
    prior = identity(window)
    model, digest = "fixture-model", "a" * 64
    if change == "classifier":
        monkeypatch.setattr(analysis, "_CITED_VERSION", analysis._CITED_VERSION + "-semantic-change")
    elif change == "schema":
        schema = copy.deepcopy(analysis._CITED_SCHEMA)
        schema["properties"]["proposals"]["maxItems"] = 11
        monkeypatch.setattr(analysis, "_CITED_SCHEMA", schema)
    elif change == "prompt":
        original = analysis._cited_prompt
        def updated(content):
            messages = original(content)
            messages[0]["content"] += " Changed interpretation rule."
            return messages
        monkeypatch.setattr(analysis, "_cited_prompt", updated)
    elif change == "weights":
        digest = "b" * 64
    elif change == "model":
        model = "different-fixture-model"
    else:
        key, value = {"text": ("text", "Keep different citations."), "role": ("role", "assistant"),
            "timestamp": ("event_at", window["event_at"] + 1), "time_basis": ("time_basis", "unknown"),
            "project_basis": ("project_basis", "unknown"),
            "ranges": ("citation_ranges", [{"start": 1, "length": 20}])}[change]
        window[key] = value
    assert identity(window, model, digest) != prior


def test_new_extraction_uses_the_same_recomputable_identity(window):
    class Source:
        def reopen(self, descriptor):
            return window
        def validated_proposals(self, descriptor, proposals):
            assert proposals == []
    content = json.dumps({"summary": "Preserve citations.", "decisions": [], "open_items": [],
                          "uncertainty": "No broader context.", "proposals": []})
    outcome = analysis._cited_outcome(content, Source(), {"fixture": True},
                                      "ollama", "fixture-model", "a" * 64)
    assert outcome["extraction"]["model_identity"] == identity(window)
