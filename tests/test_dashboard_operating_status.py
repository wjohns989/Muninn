"""Focused sanitized fixtures for dashboard spend and interpretation status."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_history_interpretation_status_uses_backend_schema_and_keeps_unknown_distinct():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    page = Path(__file__).resolve().parents[1].joinpath("dashboard.html").read_text(encoding="utf-8")
    assert 'id="history-interpretation-status"' in page
    source = "async function loadHistoryStatus()" + page.split(
        "async function loadHistoryStatus()", 1,
    )[1].split("async function loadRemotePolicy()", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
const document = {getElementById(id) {
    if (!elements.has(id)) {
        const node = {textContent: '', classList: {contains: () => true}};
        Object.defineProperty(node, 'innerHTML', {set() { throw Error('HTML sink'); }});
        elements.set(id, node);
    }
    return elements.get(id);
}};
const context = vm.createContext({document, Date});
vm.runInContext("let historyStatusSequence = 0; let AUTH_TOKEN = 'fixture'; " + __SOURCE__, context);
const state = {vault: {ready: true, archive: {generation: 1, snapshots: 0, sources: 0}},
    last_secure_index: {archive_generation: 1, ready: 0, total: 0, missing: 0, complete: true},
    capture_queue: {}, hook_receipts: [], hook_receipts_error: null,
    capture_enrichment: {configured: true, pending_sources: 0, parked_private_windows: 0,
        window_jobs: {basis: 'all_capture_lane_jobs', total: 4,
            states: {succeeded: 2, reused: 1, pending: 1}},
        historical_batch: {state: 'awaiting_provider', items: 1, repair_only: true,
            parent_items: 9, repair_round: 1},
        backlog_drain: {active: false, halted_reason: null},
        historical_enrollment: {queued: 2, existing: 3, excluded: 1, complete: false}}};
let failStatus = false;
context.api = async path => { assert.equal(path, '/history/status');
    return failStatus ? {error: 'fixture unavailable'} : {data: state};
};
(async () => {
    await vm.runInContext('loadHistoryStatus()', context);
    let text = document.getElementById('history-interpretation-status').textContent;
    assert.match(text, /0 pending source versions/);
    assert.match(text, /0 parked private windows/);
    assert.match(text, /catch-up inactive/);
    assert.match(text, /queued 2, existing 3, excluded 1/);
    assert.match(text, /Enrollment is not interpretation completion/);
    let windows = document.getElementById('history-window-status').textContent;
    assert.match(windows, /all capture jobs/);
    assert.match(windows, /2 succeeded.*1 reused.*1 pending/);
    assert.match(windows, /remaining window total unknown/);
    let batch = document.getElementById('history-batch-status').textContent;
    assert.match(batch, /1 window.*9.window checkpoint.*round 1/);
    assert.match(batch, /waiting for provider/);
    state.capture_enrichment.historical_batch.health = {state: 'degraded_unknown', age_seconds: 14400,
        completed_requests: 0, failed_requests: 0, total_requests: 60};
    await vm.runInContext('loadHistoryStatus()', context);
    batch = document.getElementById('history-batch-status').textContent;
    assert.match(batch, /Delayed: no reported progress; internal activity unknown/);
    assert.match(batch, /4.0 hours; 0 completed \/ 60 requests, 0 failed/);
    assert.match(batch, /No streaming results/);
    state.capture_enrichment.window_jobs.states.pending = '<img>';
    state.capture_enrichment.historical_batch = {state: '<img>', items: '<script>'};
    await vm.runInContext('loadHistoryStatus()', context);
    assert.match(document.getElementById('history-window-status').textContent, /unknown/);
    assert.doesNotMatch(document.getElementById('history-batch-status').textContent, /<img|<script/);
    state.capture_enrichment = {configured: true, pending_sources: 4, parked_private_windows: 7,
        backlog_drain: {active: true, halted_reason: '<img>'},
        historical_enrollment: {queued: 2, existing: 3, excluded: 1, complete: true}};
    await vm.runInContext('loadHistoryStatus()', context);
    text = document.getElementById('history-interpretation-status').textContent;
    assert.match(text, /4 pending source versions.*7 parked private windows/);
    assert.match(text, /catch-up active/);
    assert.doesNotMatch(text, /<img/);
    state.capture_enrichment = {pending_sources: 0};
    await vm.runInContext('loadHistoryStatus()', context);
    text = document.getElementById('history-interpretation-status').textContent;
    assert.match(text, /0 pending source versions/);
    assert.match(text, /parked private windows; catch-up status unknown/);
    assert.match(text, /enrollment not reported/);
    state.capture_enrichment = null;
    await vm.runInContext('loadHistoryStatus()', context);
    assert.match(document.getElementById('history-interpretation-status').textContent, /unavailable/);
    state.capture_enrichment = {pending_sources: 9, parked_private_windows: 8,
        backlog_drain: {active: true}, historical_enrollment: {queued: 7, existing: 6,
            excluded: 5, complete: false}};
    await vm.runInContext('loadHistoryStatus()', context);
    assert.match(document.getElementById('history-interpretation-status').textContent, /9 pending/);
    failStatus = true;
    await vm.runInContext('loadHistoryStatus()', context);
    text = document.getElementById('history-interpretation-status').textContent;
    assert.match(text, /unavailable.*pending and parked counts are unknown/);
    assert.doesNotMatch(text, /9 pending|8 parked|catch-up active|queued 7/);
    assert.match(document.getElementById('history-window-status').textContent, /unavailable/);
    assert.match(document.getElementById('history-batch-status').textContent, /unavailable/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__SOURCE__", json.dumps(source))
    checked = subprocess.run([node, "-"], input=harness.encode("utf-8"),
                             capture_output=True, timeout=15, check=False)
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")


def test_dedicated_key_spend_is_provider_reported_and_unknown_is_not_zero():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    page = Path(__file__).resolve().parents[1].joinpath("dashboard.html").read_text(encoding="utf-8")
    source = "async function loadRemoteKeyStatus(enabled)" + page.split(
        "async function loadRemoteKeyStatus(enabled)", 1,
    )[1].split("async function saveRemotePolicy()", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const status = {textContent: ''};
Object.defineProperty(status, 'innerHTML', {set() { throw Error('HTML sink'); }});
const document = {getElementById(id) { assert.equal(id, 'remote-key-status'); return status; }};
const context = vm.createContext({document});
vm.runInContext(__SOURCE__, context);
(async () => {
    let calls = 0;
    context.api = async path => { calls++; assert.equal(path, '/history/secure/remote-policy/key-status');
        return {data: {state: 'ready', usage_daily_usd: 0.1234567, usage_monthly_usd: 2.5,
            key_limit_usd: 5, key_remaining_usd: 4.8, key_reset: 'daily'}};
    };
    await vm.runInContext('loadRemoteKeyStatus(true)', context);
    assert.equal(calls, 1);
    assert.match(status.textContent, /Dedicated-key usage: \$0\.123457 today, \$2\.500000 this month/);
    assert.match(status.textContent, /not this run or app-ledger spend/);
    context.api = async () => ({data: {state: 'ready', usage_daily_usd: 0,
        usage_monthly_usd: null, key_limit_usd: 5, key_remaining_usd: 5, key_reset: 'daily'}});
    await vm.runInContext('loadRemoteKeyStatus(true)', context);
    assert.match(status.textContent, /\$0\.000000 today, \$unknown this month/);
    context.api = async () => ({error: 'provider unavailable'});
    await vm.runInContext('loadRemoteKeyStatus(true)', context);
    assert.match(status.textContent, /status unavailable/);
    assert.doesNotMatch(status.textContent, /Dedicated-key usage: \$0/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__SOURCE__", json.dumps(source))
    checked = subprocess.run([node, "-"], input=harness.encode("utf-8"),
                             capture_output=True, timeout=15, check=False)
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")
