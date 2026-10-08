"""Batch recovery contract on isolated encrypted archives; no provider calls."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.historical_batch import (
    MODEL,
    BatchError,
    BatchOutbox,
    billed_cost,
    payload,
    prepare_items,
    terminal_results,
    validate_item,
)
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.transcript_units import PARSER_VERSION


def items():
    return [{"custom_id": f"{n:032x}", "job_id": f"{n + 10:032x}",
             "window": {"format": 1, "blob": "a" * 32, "sha256": "b" * 64,
                        "version": 0, "attempt": "c" * 32, "page": n, "offset": 0,
                        "length": 12, "parser_version": PARSER_VERSION,
                        "input_sha256": "d" * 64, "boundary_hit": False, "prefix": None},
             "body": {"messages": [{"role": "user", "content": "private-source-marker"}]}}
            for n in (1, 2)]


def submitted():
    return {"id": "batch_fixture", "model": MODEL, "endpoint": "/v1/chat/completions",
            "completion_window": "24h", "status": "validating",
            "request_counts": {"total": 2, "completed": 0, "failed": 0}}


def completed():
    return {**submitted(), "status": "completed",
            "request_counts": {"total": 2, "completed": 2, "failed": 0},
            "usage": {"cost": 0.0012, "is_byok": False},
            "results": [{"custom_id": row["custom_id"], "error": None,
                         "response": {"status_code": 200, "body": {"model": MODEL,
                                       "choices": [{"message": {"content": "private-output-marker"}}]}}}
                        for row in items()]}


@pytest.fixture
def outbox(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "isolated batch recovery passphrase")
    return BatchOutbox(archive)


def test_payload_order_and_provider_pin_never_weaken_normal_zdr():
    result = payload(items())
    assert list(result) == ["endpoint", "model", "provider", "completion_window", "requests"]
    assert result["provider"] == {"only": ["openai"]}
    assert result["model"] == MODEL and not result["model"].endswith(":batch")
    assert "job_id" not in result["requests"][0] and "window" not in result["requests"][0]
    assert "private-source-marker" in json.dumps(result)
    assert "zdr" not in result["provider"]  # This is a separately authorized retention exception.


@pytest.mark.parametrize("change", ["duplicate_custom", "duplicate_job", "bad_descriptor", "too_many"])
def test_bad_local_binding_is_rejected(change):
    rows = items()
    if change == "duplicate_custom":
        rows[1]["custom_id"] = rows[0]["custom_id"]
    elif change == "duplicate_job":
        rows[1]["job_id"] = rows[0]["job_id"]
    elif change == "bad_descriptor":
        rows[0]["window"]["length"] = 3001
    else:
        rows *= 13
    with pytest.raises(ValueError):
        payload(rows)


def test_preparation_requires_whole_source_privacy_and_body_screening():
    class Source:
        def remote_input(self, descriptor):
            return None
    with pytest.raises(BatchError, match="source_not_remote_safe"):
        prepare_items(Source(), [(items()[0]["job_id"], items()[0]["window"])])


def test_encrypted_cas_survives_restart_and_never_resubmits_unknown(outbox):
    ident = outbox.prepare(items(), consent_generation=3)
    assert outbox.begin_submission(ident, 0) == 1
    reopened = BatchOutbox(outbox.archive)
    assert reopened.read(ident)["state"] == "submission_unknown"
    with pytest.raises(BatchError, match="state_conflict"):
        reopened.begin_submission(ident, 0)
    assert reopened.save_submission(ident, 1, submitted()) == 2
    assert reopened.save_terminal(ident, 2, completed()) == 3
    stored = BatchOutbox(outbox.archive).read(ident)
    assert stored["terminal"] == completed()
    raw = outbox.path.read_bytes()
    assert b"private-source-marker" not in raw and b"private-output-marker" not in raw
    assert b"batch_fixture" not in raw and MODEL.encode() not in raw


def test_exactly_one_competing_submit_marker_wins(outbox):
    ident = outbox.prepare(items(), consent_generation=1)
    def mark(_):
        try:
            return outbox.begin_submission(ident, 0)
        except BatchError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(mark, range(2)), key=str) == [1, "conflict"]


def test_prepared_records_are_detached_from_mutable_caller(outbox):
    rows = items()
    ident = outbox.prepare(rows, consent_generation=1)
    rows[0]["body"]["messages"][0]["content"] = "modified"
    assert outbox.read(ident)["items"] == items()


def test_passphrase_unlocked_archive_recovers_outbox_without_dpapi(outbox):
    ident = outbox.prepare(items(), consent_generation=1)
    outbox.begin_submission(ident, 0)
    archive = SecureHistoryArchive(outbox.archive.root, "isolated batch recovery passphrase")
    assert BatchOutbox(archive).read(ident)["state"] == "submission_unknown"


@pytest.mark.parametrize("damage", ["ciphertext", "state", "revision", "schema", "missing_pair"])
def test_tampering_or_missing_store_fails_closed(outbox, damage):
    ident = outbox.prepare(items(), consent_generation=1)
    if damage == "missing_pair":
        outbox.marker.unlink()
        with pytest.raises(VaultIntegrityError):
            BatchOutbox(outbox.archive)
        return
    with sqlite3.connect(outbox.path) as db:
        if damage == "ciphertext":
            db.execute("UPDATE batches SET sealed=?", (b"not authentic",))
        elif damage == "state":
            db.execute("UPDATE batches SET state='cleaned'")
        elif damage == "revision":
            db.execute("UPDATE batches SET revision=7")
        else:
            db.execute("DELETE FROM sentinel")
    with pytest.raises(VaultIntegrityError):
        outbox.read(ident)


def test_results_reconcile_by_id_not_order_and_allow_item_failure():
    data = completed()
    data["results"].reverse()
    assert set(terminal_results(items(), data)) == {r["custom_id"] for r in items()}
    data["results"][0].update(response=None, error={"code": "fixture_failure"})
    data["request_counts"].update(completed=1, failed=1)
    assert len(terminal_results(items(), data)) == 2


@pytest.mark.parametrize("damage", ["duplicate", "missing", "extra", "wrong_model",
                                    "both", "neither", "counts", "bool_counts"])
def test_results_never_invent_success_from_misaligned_response(damage):
    data = completed()
    if damage == "duplicate":
        data["results"][1]["custom_id"] = data["results"][0]["custom_id"]
    elif damage == "missing":
        data["results"].pop()
    elif damage == "extra":
        data["results"].append(deepcopy(data["results"][0]))
    elif damage == "wrong_model":
        data["results"][0]["response"]["body"]["model"] = "unapproved-model"
    elif damage == "both":
        data["results"][0]["error"] = {"code": "failed"}
    elif damage == "neither":
        data["results"][0]["response"] = None
    elif damage == "bool_counts":
        data["request_counts"]["failed"] = False
    else:
        data["request_counts"]["completed"] = 1
    with pytest.raises(BatchError):
        terminal_results(items(), data)


@pytest.mark.parametrize("terminal", ["failed", "expired", "cancelled"])
def test_terminal_failure_is_saved_but_not_fabricated_as_completed(outbox, terminal):
    ident = outbox.prepare(items(), consent_generation=1)
    outbox.begin_submission(ident, 0)
    outbox.save_submission(ident, 1, submitted())
    data = {**completed(), "status": terminal, "results": None}
    outbox.save_terminal(ident, 2, data)
    assert outbox.read(ident)["terminal"] == data
    with pytest.raises(BatchError, match="not_completed"):
        terminal_results(items(), data)


@pytest.mark.parametrize("usage", [None, {}, {"cost": 0, "is_byok": True},
                                  {"cost": None, "is_byok": False},
                                  {"cost": True, "is_byok": False},
                                  {"cost": -1, "is_byok": False},
                                  {"cost": float("nan"), "is_byok": False}])
def test_unresolved_cost_is_never_zero(usage):
    with pytest.raises(BatchError, match="cost_unresolved"):
        billed_cost({"usage": usage})
    assert billed_cost({"usage": {"cost": 0, "is_byok": False}}) == 0


def test_cleanup_cannot_precede_durable_terminal_save_or_fake_provider_receipt(outbox):
    ident = outbox.prepare(items(), consent_generation=1)
    outbox.begin_submission(ident, 0)
    outbox.save_submission(ident, 1, submitted())
    receipt = {"id": "batch_fixture", "deletion": {"openrouter": "deleted",
               "upstream": {"provider": "openai", "status": "unsupported"}}}
    with pytest.raises(BatchError, match="state_conflict"):
        outbox.save_cleanup(ident, 2, receipt)
    outbox.save_terminal(ident, 2, completed())
    with pytest.raises(BatchError, match="deletion_unverified"):
        outbox.save_cleanup(ident, 3, {**receipt, "id": "another_batch"})
    outbox.save_cleanup(ident, 3, receipt)
    reopened = BatchOutbox(outbox.archive).read(ident)
    assert reopened["state"] == "cleaned" and reopened["terminal"] == completed()
    assert reopened["deletion"]["deletion"]["upstream"]["status"] == "unsupported"


@pytest.mark.parametrize("defect", [None, "schema", "unsupported_quote", "structure", "truncated"])
def test_transport_success_requires_actual_cited_output_validation(tmp_path, defect):
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    from muninn.history.cited_windows import CitedWindowPlanStore
    from muninn.history.secure_analysis import ModelOutputInvalid
    from tests.test_capture_window_jobs import window_fixture

    journal, archive, receipt = window_fixture(tmp_path, text="A local capture observation.")
    journal.queue_capture_windows(receipt, limit=1)
    job = journal.claim_analysis(include_capture=True)
    plans = CitedWindowPlanStore(archive)
    entry = plans.source.ledger._entries[(job.target["blob"], job.target["version"])]
    descriptor = plans.window_at(entry, job.target["version"], job.target["plan_attempt"],
                                 job.target["ordinal"])
    source = CitedAnalysisSource(archive)
    prepared = prepare_items(source, [(job.job_id, descriptor)])
    text = source.reopen(descriptor)["text"]
    output = {"summary": "A local observation.", "decisions": [], "open_items": [],
              "uncertainty": "", "proposals": [{"type": "fact", "text": "A local observation.",
                "quote": text, "start": 0}]}
    if defect == "schema":
        output.pop("proposals")
    elif defect == "unsupported_quote":
        output["proposals"][0]["quote"] = "This quote is absent from the authenticated source."
    row = {"custom_id": prepared[0]["custom_id"], "error": None,
           "response": {"status_code": 200, "body": {"model": MODEL, "choices": [{
             "finish_reason": "stop", "message": {"content": json.dumps(output)}}]}}}
    if defect == "structure":
        row["response"]["body"]["choices"] = {}
    elif defect == "truncated":
        row["response"]["body"]["choices"][0]["finish_reason"] = "length"
    response = {**completed(), "results": [row],
                "request_counts": {"total": 1, "completed": 1, "failed": 0}}
    box = BatchOutbox(archive)
    ident = box.prepare(prepared, consent_generation=1)
    box.begin_submission(ident, 0)
    box.save_submission(ident, 1, {**submitted(), "request_counts": {"total": 1}})
    box.save_terminal(ident, 2, response)
    matched = terminal_results(prepared, box.read(ident)["terminal"])
    assert len(matched) == 1  # Provider success, not yet accepted memory.
    if defect is None:
        checked = validate_item(source, prepared[0], row)
        assert checked["extraction"]["proposals"][0]["quote"] == text
        assert journal.capture_window_status(receipt)["acknowledged"] == 0
    else:
        with pytest.raises((BatchError, ModelOutputInvalid)):
            validate_item(source, prepared[0], row)
    assert box.read(ident)["terminal"] == response  # Invalid replies retained, not erased/retried.
