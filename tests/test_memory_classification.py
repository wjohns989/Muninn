"""Classification proposals never establish truth, filing or dispatch authority."""
import json
from dataclasses import replace

import pytest

from muninn.history.memory_classification import (
    ClassificationError, prepare_classification, revalidate_classification, validate_classification,
)
from muninn.history.memory_ledger import MemoryLedger
from tests.test_memory_ledger import fixture, MODEL


def prepared(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path)
    writer = MemoryLedger(archive)
    refs = writer.record_batch(entry, 0, attempt, [{"page": page, "proposal": {
        "type": "preference", "text": "Keep source citations.",
        "quote": "I want source citations kept.", "start": 0}}], model_identity=MODEL)
    reader = MemoryLedger(archive, read_only=True)
    return archive, writer, reader, refs, prepare_classification(reader, refs)


def reply(plan, **changes):
    ref = plan.payload()["candidates"][0]["id"]
    row = {"id": ref, "bucket": "preference", "disposition": "accepted",
           "evidence_refs": [ref], "reason": "source_supported", "confidence": 0.98}
    return json.dumps({"items": [{**row, **changes}]})


def test_real_encrypted_context_is_safe_bound_readonly_and_not_filing(tmp_path):
    archive, writer, reader, refs, plan = prepared(tmp_path)
    before = writer.verify_all()
    payload = plan.payload()
    assert payload["comparison_scope"] == "selected_peers_only"
    assert payload["global_conflict_coverage"] is False
    assert payload["candidates"][0]["role"] == "user"
    assert "transcript_capability" not in plan.payload_json
    assert "synthetic-project" not in plan.payload_json
    binding = plan.bindings()[0]
    assert binding["decision_seq"] == 1 and binding["expected_state"] == "provisional"
    assert not binding["human_reviewed"]
    assert refs[0] not in plan.payload_json
    assert revalidate_classification(reader, plan)
    result = validate_classification(plan, reply(plan))
    assert result[0]["disposition"] == "accepted"
    assert result[0]["truth_status"] == "model_inferred"
    assert result[0]["global_conflict_coverage"] is False
    assert writer.verify_all() == before
    assert writer.get(refs[0])["state"] == "provisional"
    assert not (archive.root / "blind-index").exists()


@pytest.mark.parametrize("changes", [
    {"id": "f" * 64}, {"bucket": "possible_credential"}, {"bucket": "conflict"},
    {"confidence": True}, {"confidence": -1}, {"confidence": 2},
    {"confidence": 10 ** 400},
    {"evidence_refs": []}, {"evidence_refs": ["f" * 64]},
    {"reason": "user_confirmed"}, {"project_ref": "invented"},
    {"truth_status": "verified"}, {"global_conflict_coverage": True},
    {"disposition": "conflict", "reason": "conflicting_evidence"},
])
def test_rejects_unsupported_authority_evidence_and_shape(tmp_path, changes):
    _archive, _writer, _reader, _refs, plan = prepared(tmp_path)
    with pytest.raises(ClassificationError, match="^classification_reply_invalid$"):
        validate_classification(plan, reply(plan, **changes))


def test_low_confidence_is_consultation_not_accepted_placement(tmp_path):
    *_unused, plan = prepared(tmp_path)
    value = validate_classification(plan, reply(plan, confidence=0.94))[0]
    assert value["disposition"] == "needs_user" and value["reason"] == "ambiguous_type"


@pytest.mark.parametrize("raw", ['{"items":[],"items":[]}', '{"items":[NaN]}', '{}', 'private-invalid'])
def test_invalid_reply_diagnostics_never_echo_contents(tmp_path, raw):
    *_unused, plan = prepared(tmp_path)
    with pytest.raises(ClassificationError, match="^classification_reply_invalid$"):
        validate_classification(plan, raw)


def test_changed_preparation_identity_is_not_a_reusable_result(tmp_path):
    *_unused, plan = prepared(tmp_path)
    altered = replace(plan, bindings_json=plan.bindings_json.replace('"decision_seq":1', '"decision_seq":2'))
    with pytest.raises(ClassificationError, match="classification_input_changed"):
        validate_classification(altered, reply(plan))


def test_writer_and_human_decisions_do_not_grant_classifier_authority(tmp_path):
    _archive, writer, reader, refs, _plan = prepared(tmp_path)
    with pytest.raises(ClassificationError, match="classification_readonly_required"):
        prepare_classification(writer, refs)
    writer.resolve_review(refs[0], state="filed", expected_state="provisional", reason="user_confirmed")
    with pytest.raises(ClassificationError, match="classification_human_or_terminal"):
        prepare_classification(reader, refs)
    with pytest.raises(ClassificationError, match="classification_human_or_terminal"):
        revalidate_classification(reader, _plan)


def test_unknown_scope_is_not_invented_from_context(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path, cwd=None)
    writer = MemoryLedger(archive)
    refs = writer.record_batch(entry, 0, attempt, [{"page": page, "proposal": {
        "type": "preference", "text": "Keep source citations.",
        "quote": "I want source citations kept.", "start": 0}}], model_identity=MODEL)
    with pytest.raises(ClassificationError, match="classification_unknown_scope"):
        prepare_classification(MemoryLedger(archive, read_only=True), refs)


def test_credentials_and_metadata_only_sources_are_not_model_inputs(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path, text="API_KEY=synthetic-test-secret")
    writer = MemoryLedger(archive)
    refs = writer.record_batch(entry, 0, attempt, [{"page": page, "proposal": {
        "type": "possible_credential", "text": "A key assignment.",
        "quote": "API_KEY=synthetic-test-secret", "start": 0}}], model_identity=MODEL)
    with pytest.raises(ClassificationError, match="classification_human_or_terminal|classification_withheld"):
        prepare_classification(MemoryLedger(archive, read_only=True), refs)


@pytest.mark.parametrize("change", [
    {"payload_json": "private-invalid"}, {"payload_json": "[]"},
    {"bindings_json": "null"}, {"bindings_json": "[{}]"},
    {"input_sha256": None}, {"payload_json": None},
])
def test_malformed_retained_input_has_static_errors(tmp_path, change):
    *_unused, plan = prepared(tmp_path)
    broken = replace(plan, **change)
    for check in (lambda: broken.payload(), lambda: broken.bindings(),
                  lambda: validate_classification(broken, reply(plan)),
                  lambda: revalidate_classification(_unused[2], broken)):
        with pytest.raises(ClassificationError, match="^classification_input_invalid$"):
            check()


def peers(tmp_path):
    text = "Keep citations. Preserve provenance."
    archive, entry, attempt, page = fixture(tmp_path, text=text, role="assistant")
    writer = MemoryLedger(archive)
    refs = writer.record_batch(entry, 0, attempt, [{"page": page, "proposal": {
        "type": "preference", "text": quote, "quote": quote, "start": text.index(quote)}}
        for quote in ("Keep citations.", "Preserve provenance.")], model_identity=MODEL)
    return writer, MemoryLedger(archive, read_only=True), refs


def test_rejected_peer_is_not_evidence_for_accepted_placement(tmp_path):
    writer, reader, refs = peers(tmp_path)
    writer.resolve_review(refs[1], state="rejected", expected_state="provisional", reason="user_rejected")
    plan = prepare_classification(reader, refs[:1], peer_refs=refs[1:])
    with pytest.raises(ClassificationError, match="classification_reply_invalid"):
        validate_classification(plan, reply(plan, evidence_refs=["m0", "m1"]))
    assert validate_classification(plan, reply(plan, disposition="needs_user",
        reason="missing_evidence", evidence_refs=["m0", "m1"]))[0]["disposition"] == "needs_user"


def test_same_state_human_peer_revision_invalidates_preparation(tmp_path):
    writer, reader, refs = peers(tmp_path)
    writer.resolve_review(refs[1], state="filed", expected_state="provisional", reason="user_confirmed")
    plan = prepare_classification(reader, refs[:1], peer_refs=refs[1:])
    writer.resolve_review(refs[1], state="filed", expected_state="filed", reason="user_confirmed")
    with pytest.raises(ClassificationError, match="classification_input_changed"):
        revalidate_classification(reader, plan)
    assert writer.get(refs[0])["state"] == "provisional"


@pytest.mark.parametrize("timestamp,cwd,category", [
    (False, True, "classification_unknown_time"),
    (True, None, "classification_unknown_scope"),
])
def test_missing_original_provenance_is_not_invented(tmp_path, timestamp, cwd, category):
    archive, entry, attempt, page = fixture(tmp_path, timestamp=timestamp, cwd=cwd)
    writer = MemoryLedger(archive)
    refs = writer.record_batch(entry, 0, attempt, [{"page": page, "proposal": {
        "type": "preference", "text": "Keep source citations.",
        "quote": "I want source citations kept.", "start": 0}}], model_identity=MODEL)
    with pytest.raises(ClassificationError, match=category):
        prepare_classification(MemoryLedger(archive, read_only=True), refs)


def test_cross_project_peer_is_not_classification_evidence(tmp_path):
    from tests.test_grouped_memory_review import add_source
    writer, _reader, refs = peers(tmp_path)
    other = add_source(writer, tmp_path / "other.jsonl", cwd="C:/another-project",
                       timestamp="2026-09-29T12:00:00Z")
    reader = MemoryLedger(writer.archive, read_only=True)
    with pytest.raises(ClassificationError, match="classification_cross_project"):
        prepare_classification(reader, refs[:1], peer_refs=[other])


def test_context_coordinate_and_partial_coverage_are_not_lost(tmp_path, monkeypatch):
    _archive, _writer, reader, refs, plan = prepared(tmp_path)
    row = plan.payload()["candidates"][0]
    assert row["context_coordinate"] == "source_fragment"
    assert row["partial_visible_ranges"] is False
    original = reader.source
    def source(*args, **kwargs):
        view = original(*args, **kwargs)
        return {**view, "context_coordinate": "cited_window", "context_start": 19,
                "partial_visible_ranges": True, "context_truncated": True}
    monkeypatch.setattr(reader, "source", source)
    projected = prepare_classification(reader, refs).payload()["candidates"][0]
    assert projected["context_coordinate"] == "cited_window" and projected["context_start"] == 19
    assert projected["citation"]["quote_start"] == 0
    assert projected["partial_visible_ranges"] and projected["context_truncated"]


def test_preparation_does_not_write_private_stores(tmp_path):
    from tests.test_memory_review_queue import files_digest
    archive, _writer, reader, refs, _plan = prepared(tmp_path)
    before = files_digest(archive.root)
    prepare_classification(reader, refs)
    assert files_digest(archive.root) == before


def test_real_projected_source_keeps_range_coordinates_without_secret(tmp_path):
    from tests.test_projected_memory_publication import mixed
    archive, source, descriptor, proposal, view = mixed(tmp_path)
    refs = source.record_proposals(descriptor, [proposal], model_identity=MODEL, source_view=view)
    reader = MemoryLedger(archive, read_only=True)
    plan = prepare_classification(reader, refs)
    row = plan.payload()["candidates"][0]
    assert row["partial_visible_ranges"] is True
    assert row["context_coordinate"] == "cited_window"
    assert row["citation"]["quote_start"] == proposal["start"]
    assert "API_KEY" not in plan.payload_json and "short-value" not in plan.payload_json
    assert revalidate_classification(reader, plan)
