"""Actual frontend functions, sanitized data and no running service/model."""
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "dashboard.html"


def function(start, end):
    page = PAGE.read_text(encoding="utf-8")
    assert start in page, f"Missing real UI function: {start}"
    return start + page.split(start, 1)[1].split(end, 1)[0]


def run_js(source, checks):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    harness = "const assert = require('node:assert/strict');\n" + source + "\n" + checks
    result = subprocess.run([node, "-"], input=harness.encode(), capture_output=True,
                            timeout=15, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_operating_view_has_distinct_units_enrollment_and_unknown_states():
    source = function("function operatingView(data)", "function renderOperatingStatus(data)")
    run_js(source, r"""
const empty = operatingView(null);
assert.equal(empty.succeeded, 'Unknown');
assert.equal(empty.pendingVersions, 'Unknown');
assert.match(empty.batch, /unknown/i);
const data = {capture_enrichment: {pending_sources: 6067, parked_private_windows: 3,
    automatic_remote_only: true,
    historical_enrollment: {complete: true, queued: 2457, existing: 924, excluded: 848},
    historical_versions_enrollment: {complete: true, queued: 2124, existing: 3631, excluded: 848},
    backlog_drain: {active: false, remaining_seconds: 0, halted_reason: null},
    historical_batch: {items: 1, provider_requests: 1, state: 'awaiting_provider',
        repair_only: true, parent_items: 9, repair_round: 1},
    window_jobs: {basis: 'all_capture_lane_jobs', total: 12,
        states: {succeeded: 2, reused: 1, pending: 4, retry: 3, failed: 1, outcome_unknown: 1}}}};
const view = operatingView(data);
assert.equal(view.succeeded, '2');
assert.equal(view.reused, '1');
assert.equal(view.runnable, '4');
assert.equal(view.parked, '3');
assert.equal(view.pendingVersions, '6,067');
assert.match(view.latest, /2,457.*924.*848/);
assert.match(view.versions, /2,124.*3,631.*848/);
assert.match(view.batch, /1 window.*1 provider request.*waiting for provider/);
assert.match(view.batch, /9.window checkpoint.*round 1/);
assert.match(view.drain, /not completion/);
assert.match(view.route, /remote.only/i);
assert.match(view.coverage, /remaining window total.*unknown/i);
data.capture_enrichment.window_jobs.total = 13;
assert.equal(operatingView(data).succeeded, 'Unknown');
data.capture_enrichment.historical_batch = {items: 2, provider_requests: 3, state: '<img>'};
assert.doesNotMatch(operatingView(data).batch, /<img|3 provider/);
data.capture_enrichment.pending_sources = -1;
assert.equal(operatingView(data).pendingVersions, 'Unknown');
data.capture_enrichment.historical_versions_enrollment.queued = '<img>';
assert.doesNotMatch(operatingView(data).versions, /<img/);
""")


def test_operating_load_invalidates_errors_and_old_session_completions():
    source = function("async function loadOperatingStatus()", "function historyMessage(message)")
    run_js(source, r"""
let AUTH_TOKEN = 'test'; let SECURITY_ENABLED = true;
let sessionEpoch = 1; let operatingStatusSequence = 0;
const events = [];
const elements = new Map();
const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: ''});
    return elements.get(id);
}};
function renderOperatingStatus(data) {events.push(data);}
let resolve;
let api = async path => {assert.equal(path, '/history/status');
    return new Promise(r => resolve = r);};
(async () => {
    const old = loadOperatingStatus();
    sessionEpoch++; AUTH_TOKEN = 'new-test';
    resolve({data: {privateFixture: 'must never repaint'}}); await old;
    assert.equal(events.length, 0);
    api = async () => ({error: 'failed'});
    await loadOperatingStatus();
    assert.equal(events.at(-1), null);
    assert.match(document.getElementById('operations-refreshed').textContent, /unavailable/i);
    AUTH_TOKEN = '';
    await loadOperatingStatus();
    assert.equal(events.at(-1), null);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_api_discards_old_token_response_and_only_current_401_locks():
    source = function("async function api(endpoint", "document.addEventListener('DOMContentLoaded'")
    # Include the complete declaration tail; the extractor starts at its prefix.
    run_js(source, r"""
let API_BASE = 'http://fixture'; let AUTH_TOKEN = 'test'; let sessionEpoch = 1;
let locks = 0; function lockSession() {locks++; sessionEpoch++; AUTH_TOKEN = '';}
let resolve;
let fetch = async () => new Promise(r => resolve = r);
(async () => {
    let p = api('/fixture');
    sessionEpoch++; AUTH_TOKEN = 'new';
    resolve({status: 200, ok: true, json: async () => ({privateFixture: 'old'})});
    let reply = await p; assert.equal(reply.privateFixture, undefined);
    assert.equal(locks, 0);
    p = api('/fixture'); sessionEpoch++; AUTH_TOKEN = 'newer';
    resolve({status: 401, ok: false}); await p; assert.equal(locks, 0);
    fetch = async () => ({status: 401, ok: false});
    reply = await api('/fixture'); assert.equal(locks, 1);
    assert.equal(reply.error, 'Unauthorized');
    AUTH_TOKEN = 'last';
    let resolveJson;
    fetch = async () => ({status: 200, ok: true, json: () => new Promise(r => resolveJson = r)});
    p = api('/fixture'); await Promise.resolve();
    sessionEpoch++; // Same token string after reauthentication must still invalidate the reply.
    resolveJson({privateFixture: 'must not return'});
    reply = await p; assert.equal(reply.privateFixture, undefined);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_public_shell_has_no_mock_counts_and_retains_all_existing_tools():
    page = PAGE.read_text(encoding="utf-8")
    assert 'id="tab-operations"' in page
    assert 'id="lock-session"' in page
    assert 'id="brain-pulse"' not in page
    assert 'id="stat-total">Unknown<' in page
    for tab in ("overview", "ingest", "search", "history", "credentials", "system"):
        assert f'id="tab-{tab}"' in page
    assert 'id="run-accounting-ledger"' in page
    assert 'Local settled charges: unknown.' in page
    assert "Recovery readiness: not reported" in page
    assert "Review queue: not reported" in page


def test_cost_view_is_on_demand_negative_or_missing_usage_is_unknown():
    source = function("async function loadOperatingCosts()", "function historyMessage(message)")
    run_js(source, r"""
let AUTH_TOKEN = 'fixture'; let SECURITY_ENABLED = true;
let operatingCostSequence = 0; let sessionEpoch = 0;
const status = {textContent: ''};
Object.defineProperty(status, 'innerHTML', {set() {throw Error('HTML sink');}});
const document = {getElementById(id) {assert.equal(id, 'operations-cost-status'); return status;}};
let calls = 0;
let api = async (path, method) => {calls++; assert.equal(path, '/history/secure/remote-policy/key-status');
    assert.equal(method, 'GET');
    return {data: {state: 'ready', usage_daily_usd: 0, usage_monthly_usd: null,
        key_limit_usd: 5, key_remaining_usd: 4, key_reset: 'daily'}};};
(async () => {
    assert.equal(calls, 0); await loadOperatingCosts();
    assert.match(status.textContent, /\$0\.000000 today, unknown this month/);
    assert.match(status.textContent, /not this run or app-ledger spend/);
    api = async () => ({data: {state: '<img>', usage_daily_usd: -1, usage_monthly_usd: '<script>'}});
    await loadOperatingCosts(); assert.match(status.textContent, /unknown today, unknown this month/);
    assert.doesNotMatch(status.textContent, /<img|<script|-1/);
    let resolve;
    api = () => new Promise(r => resolve = r);
    const old = loadOperatingCosts(); sessionEpoch++;
    resolve({data: {usage_daily_usd: 999}}); await old;
    assert.doesNotMatch(status.textContent, /999/);
    api = async () => ({error: 'fixture failure'}); await loadOperatingCosts();
    assert.match(status.textContent, /unavailable.*prior values cleared/);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_lock_clears_private_dom_capabilities_and_pollers():
    source = function("function lockSession()", "function startStatusPolling()")
    run_js(source, r"""
let AUTH_TOKEN = 'fixture'; let sessionEpoch = 0; let authAttemptSequence = 0;
let historySearchSequence = 0, historyTranscriptSequence = 0, historyStatusSequence = 0;
let overviewStatusSequence = 0, operatingStatusSequence = 0, operatingCostSequence = 0;
let remotePolicyLoadSequence = 0, resourceStatusSequence = 0, credentialSearchSequence = 0;
let healthStatusSequence = 0, scanStatusSequence = 0;
let historySearchJobId = 'private-capability'; let historyTranscriptNextCursor = 'private-cursor';
let historyTranscriptPageNumber = 5, remotePolicyLoadedEnabled = true, remotePolicySavePending = true;
let HISTORY_SECURITY_MODE = 'strict'; let healthPoll = 10; let scanPoll = 11;
const stopped = []; function clearInterval(id) {stopped.push(id);}
const nodes = new Map();
const input = {value: 'secret-fixture', checked: true};
const area = {inert: false};
const document = {querySelectorAll(selector) {return selector.includes('input') ? [input] : [area];},
    getElementById(id) {if (!nodes.has(id)) nodes.set(id, {textContent:'private-fixture', hidden:false,
        value:'secret-fixture', disabled:true, style:{}, classList: {add(){}, remove(){}},
        replaceChildren(){this.textContent = '';}, focus(){this.focused = true;}}); return nodes.get(id);}};
function renderOperatingStatus(data) {assert.equal(data, null);}
function clearCitedMemoryView() {}
lockSession();
assert.equal(AUTH_TOKEN, ''); assert.equal(sessionEpoch, 1);
assert.equal(historySearchJobId, null); assert.equal(historyTranscriptNextCursor, null);
assert.equal(remotePolicySavePending, false); assert.equal(remotePolicyLoadedEnabled, false);
assert.deepEqual(stopped, [10,11]); assert.equal(healthPoll, null); assert.equal(scanPoll, null);
assert.equal(input.value, ''); assert.equal(input.checked, false); assert.equal(area.inert, true);
assert.equal(nodes.get('history-transcript-page').textContent, '');
assert.equal(nodes.get('search-results').textContent, '');
assert.equal(nodes.get('profile-textarea').value, '');
assert.equal(nodes.get('modal-token-input').value, '');
assert.equal(nodes.get('stat-total').textContent, 'Unknown');
assert.equal(nodes.get('ingest-badge').textContent, '');
assert.equal(nodes.get('ingest-badge').style.display, 'none');
""")


def test_auth_handshake_latest_attempt_wins_and_lock_invalidates_same_token():
    source = function("async function acceptToken(token)", "async function initializeAuth()")
    run_js(source, r"""
let AUTH_TOKEN = ''; let API_BASE = 'http://fixture'; let sessionEpoch = 0; let authAttemptSequence = 0;
let unlocked = 0; let checks = 0;
const nodes = new Map();
const document = {querySelectorAll() {return [];}, getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {classList: {remove(){}, contains(){return false;}}});
    return nodes.get(id);
}};
const pending = [];
let fetch = async () => new Promise(resolve => pending.push(resolve));
function lockSession() {sessionEpoch++; authAttemptSequence++; AUTH_TOKEN = '';}
function checkHealth() {checks++;}
function startStatusPolling() {unlocked++;}
(async () => {
    const older = acceptToken('old'); const newer = acceptToken('new');
    pending[1]({ok:true}); assert.equal(await newer, true);
    pending[0]({ok:true}); assert.equal(await older, false);
    assert.equal(AUTH_TOKEN, 'new'); assert.equal(unlocked, 1);
    const stale = acceptToken('new'); lockSession();
    pending[2]({ok:true}); assert.equal(await stale, false);
    assert.equal(AUTH_TOKEN, ''); assert.equal(unlocked, 1);
    const live = acceptToken('new'); pending[3]({ok:true});
    assert.equal(await live, true); assert.equal(AUTH_TOKEN, 'new'); assert.equal(unlocked, 2);
    assert.equal(checks, 2);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")
