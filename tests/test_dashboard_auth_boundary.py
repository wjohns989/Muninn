"""The anonymous dashboard must not hand local clients the API bearer."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from muninn.core import security


def test_anonymous_dashboard_never_contains_active_bearer(monkeypatch):
    sentinel = "dashboard-test-bearer-not-for-anonymous-readers"
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", sentinel)
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)

    response = TestClient(server.app).get("/")

    assert response.status_code == 200
    assert "Muninn Hub" in response.text
    assert sentinel not in response.text
    assert "{{MUNINN_TOKEN}}" not in response.text


def test_legacy_import_controls_start_hidden_until_health_confirms_legacy():
    response = TestClient(server.app).get("/")

    assert response.status_code == 200
    assert 'id="legacy-section" style="display:none"' in response.text
    assert "HISTORY_SECURITY_MODE === 'legacy'" in response.text


def test_dashboard_exposes_distinct_bounded_history_search_without_remote_assets():
    page = TestClient(server.app).get("/").text
    assert 'id="tab-history"' in page
    assert 'id="history-query"' in page
    assert "best-effort redaction" in page
    assert "Ordinary Search" in page
    assert "fonts.googleapis.com" not in page
    assert "/history/secure/search/jobs" in page
    assert "/history/secure/fetch" in page
    assert 'id="remote-policy-enabled"' in page
    assert 'id="remote-policy-override"' in page
    assert "/history/secure/remote-policy" in page
    assert "/history/secure/raw" not in page
    assert "/credentials/reveal" not in page
    assert "cache: 'no-store'" in page
    history_logic = page.split("function historyMessage(message)", 1)[1].split("async function handleSearch()", 1)[0]
    assert "excerpt.textContent" in history_logic
    assert "innerHTML" not in history_logic
    assert "copyToClipboard" not in history_logic
    assert "sequence !== historySearchSequence" in history_logic
    assert "if (queued.data?.job_id)" in history_logic
    assert "Sanitized excerpt — best-effort redaction" in history_logic


def test_dashboard_inline_javascript_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    page = Path(__file__).resolve().parents[1].joinpath("dashboard.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", page, re.DOTALL)
    assert len(scripts) == 1
    checked = subprocess.run([node, "--check", "-"], input=scripts[0].encode("utf-8"),
                             capture_output=True, timeout=15, check=False)
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")


def test_standard_search_renders_untrusted_memory_as_literal_text():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    root = Path(__file__).resolve().parents[1]
    page = root.joinpath("dashboard.html").read_text(encoding="utf-8")
    source = "async function handleSearch()" + page.split("async function handleSearch()", 1)[1].split(
        "async function handleIngest()", 1,
    )[0]
    css = root.joinpath("dashboard.css").read_text(encoding="utf-8")
    assert "content.innerHTML = r.content" not in source
    assert "white-space: pre-wrap;" in css
    assert "overflow-wrap: anywhere;" in css
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
const unsafeAssignments = [];
function element() {
    const node = {children: [], style: {}, textContent: '', value: '', checked: false,
        appendChild(child) { this.children.push(child); }};
    Object.defineProperty(node, 'innerHTML', {
        get() { return this.html || ''; },
        set(value) {
            if (String(value).includes('<img')) unsafeAssignments.push(value);
            this.html = value;
            if (value === '') this.children = [];
        }
    });
    return node;
}
const document = {
    getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
    },
    createElement: element,
    createTextNode(text) { return {textContent: text}; }
};
const context = vm.createContext({document, pulseNav() {}, log() {}});
vm.runInContext(__SOURCE__, context);
const attack = '<img src=x onerror="window.steal()">\n' + 'A'.repeat(200);
document.getElementById('search-input').value = 'test';
context.api = async () => ({data: [{id: 'memory-1', score: 1, memory_type: 'episodic',
    created_at: 1, content: attack}]});

vm.runInContext('handleSearch()', context).then(() => {
    const item = document.getElementById('search-results').children[0];
    assert.equal(item.children[1].textContent, attack);
    assert.deepEqual(unsafeAssignments, []);
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__SOURCE__", json.dumps(source))
    checked = subprocess.run([node, "-"], input=harness.encode("utf-8"),
                             capture_output=True, timeout=15, check=False)
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")


def test_history_ui_discards_stale_search_and_fetch_responses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    page = Path(__file__).resolve().parents[1].joinpath("dashboard.html").read_text(encoding="utf-8")
    source = page.split("function historyMessage(message)", 1)[1].split(
        "async function handleSearch()", 1,
    )[0]
    source = "let HISTORY_SECURITY_MODE = 'strict'; let historySearchSequence = 0; " + (
        "let historySearchJobId = null; function historyMessage(message)" + source
    )
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
function element() {
    return {
        children: [], textContent: '', value: '', listeners: {}, style: {},
        appendChild(child) { this.children.push(child); },
        replaceChildren(...children) { this.children = children; },
        addEventListener(name, handler) { this.listeners[name] = handler; }
    };
}
const document = {
    getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
    },
    createElement: element,
    addEventListener() {}
};
    const context = vm.createContext({document, setTimeout(callback) { callback(); }});
    vm.runInContext(__SOURCE__, context);

(async () => {
    let releaseOld;
    const oldResponse = new Promise(resolve => { releaseOld = resolve; });
    const calls = [];
    context.api = async (path, method, body) => {
        calls.push({path, method, body});
        if (path === '/history/secure/search/jobs' && body.query === 'old') return oldResponse;
        if (path === '/history/secure/search/jobs') return {data: {job_id: 'new-job'}};
        if (path === '/history/secure/search/jobs/new-job')
            return {data: {state: 'succeeded', result: {
                matches: [], ready: 1, total: 1, missing: 0, overflow: 0, complete: true
            }}};
        return {data: {}};
    };
    const query = document.getElementById('history-query');
    query.value = 'old';
    const first = vm.runInContext('handleEncryptedHistorySearch()', context);
    query.value = 'new';
    await vm.runInContext('handleEncryptedHistorySearch()', context);
    releaseOld({data: {job_id: 'old-job'}});
    await first;
    assert(calls.some(call => call.path === '/history/secure/search/jobs/old-job' && call.method === 'DELETE'));
    assert.match(document.getElementById('history-search-status').textContent, /Search complete/);

    let releaseFetch;
    context.api = async () => new Promise(resolve => { releaseFetch = resolve; });
    vm.runInContext(
        "renderHistoryMatches({matches: [{fetch_capability: 'opaque'}], " +
        "ready: 1, total: 1, missing: 0, overflow: 0, complete: true}, historySearchSequence)",
        context
    );
    const card = document.getElementById('history-results').children[0];
    const pendingFetch = card.children[2].listeners.click();
    vm.runInContext('historySearchSequence += 1', context);
    releaseFetch({data: {redacted_text: 'late sensitive response'}});
    await pendingFetch;
    assert.doesNotMatch(card.children[3].textContent, /late sensitive response/);

    // Exercise the terminal timeout path after a new search invalidates the old sequence.
    const status = document.getElementById('history-search-status');
    let statusText = '';
    let polls = 0;
    Object.defineProperty(status, 'textContent', {
        get() { return statusText; },
        set(value) {
            statusText = value;
            if (value === 'Local search running...' && ++polls === 60) {
                vm.runInContext('historySearchSequence += 1', context);
                statusText = 'Newer search is active';
            }
        }
    });
    context.api = async (path, method) => {
        if (path === '/history/secure/search/jobs' && method === 'POST')
            return {data: {job_id: 'timing-job'}};
        return {data: {state: 'running'}};
    };
    query.value = 'timing';
    await vm.runInContext('handleEncryptedHistorySearch()', context);
    assert.equal(status.textContent, 'Newer search is active');
})().catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__SOURCE__", json.dumps(source))
    checked = subprocess.run([node, "-"], input=harness.encode("utf-8"),
                             capture_output=True, timeout=15, check=False)
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")


def test_remote_policy_ui_discards_loads_during_or_before_save():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    page = Path(__file__).resolve().parents[1].joinpath("dashboard.html").read_text(encoding="utf-8")
    source = "async function loadRemotePolicy()" + page.split("async function loadRemotePolicy()", 1)[1].split(
        "function renderHistoryMatches(", 1,
    )[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: '', value: '', checked: false, disabled: false});
    return elements.get(id);
}};
const context = vm.createContext({document, window: {confirm() { return true; }}});
vm.runInContext('let remotePolicyLoadedEnabled = true; let remotePolicyLoadSequence = 0; ' +
    'let remotePolicySavePending = false; ' + __SOURCE__, context);
document.getElementById('remote-policy-enabled').checked = false;
document.getElementById('remote-daily-usd').value = '1';
document.getElementById('remote-monthly-usd').value = '20';

(async () => {
    let releasePost;
    let gets = 0;
    context.api = (path, method) => {
        if (method === 'POST') return new Promise(resolve => { releasePost = resolve; });
        gets++;
        return Promise.resolve({data: {enabled: true, daily_usd: 1, monthly_usd: 20,
            override_ceiling: false, generation: 1, source: 'managed'}});
    };
    const save = vm.runInContext('saveRemotePolicy()', context);
    await vm.runInContext('loadRemotePolicy()', context);
    assert.equal(gets, 0);
    releasePost({data: {enabled: false, generation: 2}});
    await save;
    assert.equal(vm.runInContext('remotePolicyLoadedEnabled', context), false);

    let releaseGet;
    context.api = (path, method) => {
        if (method === 'POST') return Promise.resolve({data: {enabled: false, generation: 3}});
        return new Promise(resolve => { releaseGet = resolve; });
    };
    const staleLoad = vm.runInContext('loadRemotePolicy()', context);
    await vm.runInContext('saveRemotePolicy()', context);
    releaseGet({data: {enabled: true, daily_usd: 1, monthly_usd: 20,
        override_ceiling: false, generation: 1, source: 'managed'}});
    await staleLoad;
    assert.equal(vm.runInContext('remotePolicyLoadedEnabled', context), false);
    assert.equal(document.getElementById('remote-policy-enabled').checked, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__SOURCE__", json.dumps(source))
    checked = subprocess.run([node, "-"], input=harness.encode("utf-8"),
                             capture_output=True, timeout=15, check=False)
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")


def test_health_reports_effective_history_security_mode(monkeypatch):
    class HealthyMemory:
        async def health(self):
            return {"status": "ok"}

    monkeypatch.setattr(server, "memory", HealthyMemory())
    client = TestClient(server.app)
    monkeypatch.delenv("MUNINN_HISTORY_SECURITY", raising=False)
    assert client.get("/health").json()["history_security_mode"] == "strict"
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY", "legacy")
    assert client.get("/health").json()["history_security_mode"] == "legacy"


def test_dashboard_token_check_rejects_wrong_bearer_without_disclosing_right_one(monkeypatch):
    sentinel = "dashboard-test-valid-bearer"
    monkeypatch.setattr(server, "is_security_enabled", lambda: True)
    monkeypatch.setattr(server, "core_verify_token", lambda token: token == sentinel)
    client = TestClient(server.app)

    missing = client.get("/auth/check")
    wrong = client.get("/auth/check", headers={"Authorization": "Bearer wrong"})
    valid = client.get("/auth/check", headers={"Authorization": f"Bearer {sentinel}"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert valid.status_code == 200
    assert valid.json() == {"authenticated": True}
    assert sentinel not in valid.text


def test_development_no_auth_mode_still_has_usable_dashboard(monkeypatch):
    monkeypatch.setattr(server, "is_security_enabled", lambda: False)
    client = TestClient(server.app)

    page = client.get("/")
    check = client.get("/auth/check")

    assert page.status_code == 200
    assert "let SECURITY_ENABLED = false;" in page.text
    assert check.status_code == 200


def test_generated_fallback_token_is_required_when_security_is_enabled(monkeypatch):
    for name in ("MUNINN_API_KEY", "MUNINN_AUTH_TOKEN", "MUNINN_SERVER_AUTH_TOKEN",
                 "MUNINN_NO_AUTH", "MUNINN_DEV_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", "generated-test-bearer")
    client = TestClient(server.app)

    assert client.get("/auth/check").status_code == 401
    assert client.get("/auth/check", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/auth/check", headers={"Authorization": "Bearer generated-test-bearer"}).status_code == 200


def test_generated_fallback_token_is_never_written_to_logs(monkeypatch, caplog):
    for name in ("MUNINN_API_KEY", "MUNINN_AUTH_TOKEN", "MUNINN_SERVER_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", None)
    with caplog.at_level("WARNING", logger="Muninn.security"):
        generated = security.initialize_security()

    assert generated not in caplog.text


def test_mimir_api_auth_does_not_bypass_generated_bearer(monkeypatch):
    for name in ("MUNINN_API_KEY", "MUNINN_AUTH_TOKEN", "MUNINN_SERVER_AUTH_TOKEN",
                 "MUNINN_NO_AUTH", "MUNINN_DEV_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "_GLOBAL_AUTH_TOKEN", "generated-test-bearer")

    assert security.verify_api_token(None) is False
    assert security.verify_api_token("wrong") is False
    assert security.verify_api_token("generated-test-bearer") is True
