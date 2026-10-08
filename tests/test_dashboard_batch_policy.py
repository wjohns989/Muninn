"""Actual batch control functions; no token, private service or model use."""
from tests.test_dashboard_control_center import function, run_js, PAGE


def consumer():
    return function("function clearBatchPolicyView()", "async function loadRemotePolicy()")


SETUP = r"""
let AUTH_TOKEN = 'fixture', sessionEpoch = 1, batchPolicySequence = 0;
let batchPolicyGeneration = null, batchPolicyBusy = false;
const nodes = new Map();
const document = {getElementById(id) {if (!nodes.has(id)) nodes.set(id,
    {textContent:'', value:'', checked:false, disabled:false}); return nodes.get(id);}};
let confirms = 0; const window = {confirm(message) {confirms++; return true;}};
const policy = {enabled:true, generation:2, max_batches:10000, remaining_batches:9990};
"""


def test_read_preserves_quota_and_does_not_write_and_invalid_data_clears():
    run_js(SETUP + consumer(), r"""
let calls = []; let api = async (...args) => {calls.push(args); return {success:true, data:policy};};
(async () => {
    await loadBatchPolicy(); assert.equal(calls.length, 1);
    assert.equal(calls[0][1], 'GET'); assert.equal(confirms, 0);
    assert.equal(batchPolicyGeneration, 2);
    assert.equal(nodes.get('batch-policy-max').value, '10000');
    assert.match(nodes.get('batch-policy-status').textContent, /9,990.*10,000/);
    api = async () => ({data:{...policy, generation:'<img>'}});
    await loadBatchPolicy(); assert.equal(batchPolicyGeneration, null);
    assert.equal(nodes.get('batch-policy-save').disabled, true);
    assert.doesNotMatch(nodes.get('batch-policy-status').textContent, /<img>/);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_save_requires_explicit_confirmation_and_generation_and_no_retry():
    run_js(SETUP + consumer(), r"""
let calls = []; let api = async (...args) => {calls.push(args); return {data:policy};};
(async () => {
    await saveBatchPolicy(); assert.equal(calls.length, 0);
    await loadBatchPolicy();
    window.confirm = () => false; await saveBatchPolicy(); assert.equal(calls.length, 1);
    window.confirm = () => true;
    api = async (...args) => {calls.push(args); return {error:'PRIVATE_DETAIL'};};
    await saveBatchPolicy();
    assert.deepEqual(calls.at(-1).slice(0,3), ['/history/secure/batch-policy', 'POST',
        {enabled:true, max_batches:10000, expected_generation:2}]);
    assert.equal(batchPolicyGeneration, null);
    assert.match(nodes.get('batch-policy-status').textContent, /unknown.*refresh/i);
    assert.doesNotMatch(nodes.get('batch-policy-status').textContent, /PRIVATE_DETAIL/);
    await saveBatchPolicy(); assert.equal(calls.length, 2);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_lock_and_same_token_session_change_discard_pending_save():
    run_js(SETUP + consumer(), r"""
let resolve; let api = async () => ({data:policy});
(async () => {
    await loadBatchPolicy();
    api = () => new Promise(r => resolve = r);
    const pending = saveBatchPolicy();
    sessionEpoch++; clearBatchPolicyView();
    resolve({data:{...policy,generation:3}}); await pending;
    assert.equal(batchPolicyGeneration, null); assert.equal(batchPolicyBusy, false);
    assert.equal(nodes.get('batch-policy-save').disabled, true);
    assert.doesNotMatch(nodes.get('batch-policy-status').textContent, /Saved/);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_lock_wires_clear_and_no_storage_or_batch_deletion():
    assert "clearBatchPolicyView();" in function("function lockSession()", "function startStatusPolling()")
    source = consumer()
    assert "localStorage" not in source and "innerHTML" not in source
    assert "DELETE" not in source
    assert 'id="batch-policy-refresh"' in PAGE.read_text(encoding="utf-8")
