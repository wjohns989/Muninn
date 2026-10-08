"""Sanitized UI fixtures: conspicuous local input, redaction, lock races."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_review_status_is_visible_and_delayed_responses_cannot_restore_after_lock():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable')
    page = Path(__file__).resolve().parents[1].joinpath('dashboard.html').read_text(encoding='utf-8')
    assert 'id="triage-attention"' in page and 'role="alert"' in page
    source = 'function renderCredentialTriageStatus' + page.split(
        'function renderCredentialTriageStatus', 1)[1].split('function checkAuthOnLoad', 1)[0]
    lock = 'function lockSession()' + page.split('function lockSession()', 1)[1].split(
        'function startStatusPolling()', 1)[0]
    assert 'triageStatusSequence++' in lock and 'renderCredentialTriageStatus(null, true)' in lock
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
const document = {getElementById(id) {
    if (!elements.has(id)) {
        const node = {textContent: '', hidden: true};
        Object.defineProperty(node, 'innerHTML', {set() {throw Error('HTML sink');}});
        elements.set(id, node);
    }
    return elements.get(id);
}};
const context = vm.createContext({document});
vm.runInContext("let AUTH_TOKEN='fixture'; let SECURITY_ENABLED=true; let sessionEpoch=0; let triageStatusSequence=0;" + __SOURCE__, context);
const run = code => vm.runInContext(code, context);
const message = () => document.getElementById('credential-triage-status').textContent;
const banner = () => document.getElementById('triage-attention');
(async () => {
    context.api = async path => { assert.equal(path, '/credentials/triage/status');
        return {data: {state:'awaiting_passphrase',input_needed:true,worker_count:1}}; };
    await run('loadCredentialTriageStatus()');
    assert.equal(banner().hidden, false);
    assert.match(message(), /existing local review window/);
    run("renderCredentialTriageStatus({state:'idle',input_needed:false,worker_count:0})");
    assert.equal(banner().hidden,true);
    assert.match(message(), /No active/);
    run("renderCredentialTriageStatus({state:'awaiting_passphrase',input_needed:true,worker_count:0})");
    assert.match(message(), /could not be verified/);
    run("renderCredentialTriageStatus({state:'previous_run_failed',input_needed:false,worker_count:0})");
    assert.equal(banner().hidden,false);
    assert.match(message(), /No automatic retry/);
    run("renderCredentialTriageStatus({state:'reviewing',counters:{page:3,rows:60,left_pending:12,secret:'<img>',model_calls:'<img>'}})");
    assert.match(message(), /page: 3, rows: 60, left pending: 12/);
    assert.doesNotMatch(message(), /img|secret/);
    run("renderCredentialTriageStatus({state:'<img>',input_needed:true,worker_count:1})");
    assert.doesNotMatch(message(), /img/);
    let resolveOld;
    context.api = () => new Promise(resolve => {resolveOld=resolve;});
    const pending = run('loadCredentialTriageStatus()');
    run("AUTH_TOKEN=''; sessionEpoch++; triageStatusSequence++; renderCredentialTriageStatus(null,true)");
    resolveOld({data:{state:'awaiting_passphrase',input_needed:true,worker_count:1}});
    await pending;
    assert.equal(banner().hidden,true);
    assert.equal(document.getElementById('triage-attention-message').textContent,'');
    assert.match(message(), /Session locked/);
    // Same token, newer request: an old response also cannot overwrite progress.
    run("AUTH_TOKEN='fixture'");
    let resolveFirst;
    context.api = () => new Promise(resolve => {resolveFirst=resolve;});
    const first = run('loadCredentialTriageStatus()');
    context.api = async () => ({data:{state:'reviewing',input_needed:false,worker_count:1}});
    await run('loadCredentialTriageStatus()');
    resolveFirst({data:{state:'awaiting_passphrase',input_needed:true,worker_count:1}});
    await first;
    assert.match(message(), /worker is running/);
    assert.equal(banner().hidden,true);
})().catch(error => {console.error(error);process.exitCode=1;});
""".replace('__SOURCE__', json.dumps(source))
    result = subprocess.run([node, '-'], input=harness.encode(), capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
