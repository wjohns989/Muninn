"""Isolated portable archives; no provider dispatch or live vault access."""
import json

import pytest

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.cited_analysis_source import CitedAnalysisSource, CitedSourceError
from muninn.history.secure_archive import SecureHistoryArchive

PHRASE = "synthetic portable recovery phrase"
MODEL = "a" * 64


def fixture(tmp_path, text="Keep SQLite for orbital-widget caching."):
    path = tmp_path / "chat.jsonl"
    rows = [{"type": "session_meta", "payload": {"cwd": "C:/synthetic-project"}},
            {"type": "event_msg", "timestamp": "2026-09-30T12:00:00Z",
             "payload": {"type": "user_message", "message": text}},
            {"type": "event_msg", "timestamp": "2026-09-30T12:01:00Z",
             "payload": {"type": "user_message", "message": "Later harmless conversation."}}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PHRASE)
    entry = archive.archive_file(path, "codex")
    # Capability binds the real authenticated manifest entry, not fixture metadata.
    entry = archive._load_manifest()["files"][str(path.resolve())][0]
    source = CitedAnalysisSource(archive)
    cap = SecureHistoryBlindIndex(archive)._capability(entry, 0, "orbital")
    return archive, source, cap


def test_cited_window_reopens_exact_source_and_omits_private_location(tmp_path):
    archive, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    window = CitedAnalysisSource(archive).reopen(descriptor)
    assert window["text"] == "Keep SQLite for orbital-widget caching."
    assert window["role"] == "user" and window["event_at"] is not None
    assert window["time_basis"] == "provider_record" and window["project_ref"]
    assert "cwd" not in window and "native_id" not in window
    assert "synthetic-project" not in json.dumps(window)
    assert "text" not in descriptor and "path" not in descriptor
    assert source.remote_input(descriptor) == window


@pytest.mark.parametrize("field,value", [("input_sha256", "b" * 64), ("page", 999999),
                                       ("length", 0), ("offset", True), ("parser_version", -1)])
def test_descriptor_cannot_be_repointed_or_changed_silently(tmp_path, field, value):
    archive, source, cap = fixture(tmp_path)
    descriptor = {**source.prepare(cap), field: value}
    with pytest.raises(CitedSourceError):
        CitedAnalysisSource(archive).reopen(descriptor)


def test_late_corrupt_source_page_blocks_prepared_window(tmp_path):
    archive, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    with source.ledger.units._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=? AND ordinal=?",
                   (descriptor["attempt"], descriptor["page"]))
    with pytest.raises(CitedSourceError):
        CitedAnalysisSource(archive).reopen(descriptor)


def test_prepare_drains_source_after_an_early_hit(tmp_path, monkeypatch):
    archive, source, cap = fixture(tmp_path)
    original = source.ledger.units.fragments
    def corrupt_late(*args, **kwargs):
        for part in original(*args, **kwargs):
            yield part
        raise CitedSourceError("isolated late source failure")
    monkeypatch.setattr(source.ledger.units, "fragments", corrupt_late)
    with pytest.raises(CitedSourceError):
        source.prepare(cap)
    assert source.ledger.verify_all()["events"] == 0


def test_large_source_selects_pertinent_bounded_window_without_claiming_coverage(tmp_path):
    text = "ordinary words " * 20000 + " Keep SQLite for orbital-widget caching."
    archive, source, cap = fixture(tmp_path, text=text)
    descriptor = source.prepare(cap)
    window = source.reopen(descriptor)
    assert "orbital-widget" in window["text"] and len(window["text"]) <= 3000
    assert descriptor["page"] > 1
    assert "complete" not in descriptor
    quote = "Keep SQLite for orbital-widget caching."
    refs = source.record_proposals(descriptor, [{"type": "decision", "text": "Keep SQLite.",
                "quote": quote, "start": window["text"].index(quote)}], model_identity=MODEL)
    assert source.ledger.get(refs[0])["state"] == "provisional"
    assert source.ledger.get(refs[0])["proposal_origin"] == "model"


def test_secret_elsewhere_in_whole_unit_denies_remote_but_not_local(tmp_path):
    text = "Keep SQLite for orbital-widget caching. " + "ordinary words " * 1000
    text += " SERVICE_API_KEY=synthetic$secret"
    archive, source, cap = fixture(tmp_path, text=text)
    descriptor = source.prepare(cap)
    assert "orbital-widget" in source.reopen(descriptor)["text"]
    assert source.remote_input(descriptor) is None
    refs = source.record_proposals(descriptor, [{"type": "decision", "text": "Keep SQLite.",
                "quote": "Keep SQLite", "start": 0}], model_identity=MODEL)
    assert source.ledger.get(refs[0])["state"] == "pending"
    assert "text" not in source.ledger.get(refs[0])


def test_proposal_quotes_validate_before_any_batch_publication(tmp_path):
    archive, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    good = {"type": "decision", "text": "Keep SQLite.", "quote": "Keep SQLite", "start": 0}
    with pytest.raises(CitedSourceError):
        source.record_proposals(descriptor, [good, {**good, "quote": "invented quote"}], model_identity=MODEL)
    assert source.ledger.verify_all()["events"] == 0
    refs = source.record_proposals(descriptor, [good], model_identity=MODEL)
    assert source.record_proposals(descriptor, [good], model_identity=MODEL) == refs
    assert source.record_proposals(descriptor, [], model_identity=MODEL) == []
    assert source.ledger.verify_all()["candidates"] == 1


@pytest.mark.parametrize("items", [None, [None], [{}], [None] * 13])
def test_malformed_model_proposals_are_bounded_and_fail_closed(tmp_path, items):
    archive, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    with pytest.raises(CitedSourceError):
        source.record_proposals(descriptor, items, model_identity=MODEL)
    assert source.ledger.verify_all()["events"] == 0


def test_fragment_boundary_query_does_not_authorize_a_cross_fragment_quote(tmp_path):
    archive, source, cap = fixture(tmp_path, text="x" * 4094 + "orbital-widget caching.")
    descriptor = source.prepare(cap)
    window = source.reopen(descriptor)
    assert descriptor["boundary_hit"] is True
    assert "orbital-widget" in window["text"] and len(window["text"]) <= 3000
    assert len(window["citation_ranges"]) == 2
    with pytest.raises(CitedSourceError):
        source.record_proposals(descriptor, [{"type": "fact", "text": "orbital-widget",
                "quote": "orbital-widget", "start": window["text"].index("orbital-widget")}], model_identity=MODEL)
    assert source.ledger.verify_all()["candidates"] == 0


@pytest.mark.parametrize("prefix", ["ordinary " * 400, "ß " * 1800])
def test_near_end_and_casefold_coordinates_keep_actual_hit_and_source_quote(tmp_path, prefix):
    archive, source, cap = fixture(tmp_path, text=prefix + "orbital-widget caching.")
    descriptor = source.prepare(cap)
    window = source.reopen(descriptor)
    assert "orbital-widget caching." in window["text"] and descriptor["offset"] > 0
    quote = "orbital-widget caching."
    refs = source.record_proposals(descriptor, [{"type": "fact", "text": quote,
                "quote": quote, "start": window["text"].index(quote)}], model_identity=MODEL)
    assert source.ledger.get(refs[0])["quote"] == quote


def test_unsealed_attempt_and_no_query_match_do_not_grant_a_window(tmp_path):
    archive, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    entry = source.ledger._entries[(descriptor["blob"], descriptor["version"])]
    missing = SecureHistoryBlindIndex(archive)._capability(entry, 0, "missingterm")
    assert source.prepare(missing) is None
    with source.ledger.units._connect() as db:
        db.execute("UPDATE attempts SET state='building' WHERE attempt=?", (descriptor["attempt"],))
    with pytest.raises(CitedSourceError):
        source.reopen(descriptor)


def test_invalid_type_and_past_window_quote_do_not_publish(tmp_path):
    archive, source, cap = fixture(tmp_path)
    descriptor = source.prepare(cap)
    good = {"type": "decision", "text": "Keep SQLite.", "quote": "Keep SQLite", "start": 0}
    for changes in [{"type": []}, {"start": True}, {"start": 999999}, {"text": "x" * 2049}]:
        with pytest.raises(CitedSourceError):
            source.record_proposals(descriptor, [{**good, **changes}], model_identity=MODEL)
    assert source.ledger.verify_all()["candidates"] == 0
