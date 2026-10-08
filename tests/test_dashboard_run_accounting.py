"""Actual read-only cost UI, sanitized counters and no live keys or providers."""
import json
import re

from tests.test_dashboard_control_center import function, run_js, PAGE


def consumer():
    return function("function clearRunAccounting(", "function historyMessage(message)")


SETUP = r"""
let AUTH_TOKEN = 'fixture', sessionEpoch = 1, runAccountingSequence = 0, runAccountingBusy = false;
const nodes = new Map();
const document = {getElementById(id) {if (!nodes.has(id)) {
    const node = {textContent:'', value:'', disabled:false};
    Object.defineProperty(node,'innerHTML',{set(){throw Error('HTML sink');}}); nodes.set(id,node);
} return nodes.get(id);}};
document.getElementById('run-accounting-start').value = '2026-10-04T22:55:42.000905Z';
document.getElementById('run-accounting-baseline');
const cutoff = 1791154542.000905;
const sample = {state:'observed', scope:'all_managed_admissions_since_start',
    rounding:'up_per_admission_micro_usd', since:cutoff, sampled_at:cutoff+86400,
    settled_cost_usd:.15, admission_states:{settled:2,released:1,reserved:0,unknown:1},
    settled_resolutions:{response:2,operator:0}, global_unresolved:1,
    batch_owned:{state:'known',settled_admissions:2,settled_cost_usd:.15}};
"""


def test_exact_fractional_cutoff_invalid_dates_and_get_only_unknown_billing():
    run_js(SETUP + consumer(), r"""
assert.equal(runAccountingCutoff('2026-10-04T22:55:42.000905Z'), cutoff);
for (const value of ['', '2026-02-30T00:00:00Z','2026-10-04','<img>', '2026-10-04T21:35:42.1234567Z'])
    assert.equal(runAccountingCutoff(value), null);
let calls = []; let api = async (...args) => {calls.push(args);return {data:sample};};
(async () => {
    assert.equal(calls.length,0); await loadRunAccounting(false);
    assert.equal(calls.length,1);assert.equal(calls[0][1],'GET');
    assert.match(calls[0][0],/since=1791154542\.000905$/);
    assert.match(nodes.get('run-accounting-ledger').textContent,/\$0\.150000.*2 settled admissions/);
    assert.match(nodes.get('run-accounting-ledger').textContent,/unknown.*1/);
    assert.match(nodes.get('run-accounting-comparison').textContent,/not checked/i);
    api = async () => ({data:{...sample,settled_cost_usd:null}});
    await loadRunAccounting(false);
    assert.match(nodes.get('run-accounting-ledger').textContent,/unknown/i);
    assert.doesNotMatch(nodes.get('run-accounting-ledger').textContent,/0\.150000/);
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_baseline_required_no_zero_assumption_and_month_rollover_is_not_reconciled():
    run_js(SETUP + consumer(), r"""
let calls=[]; let api=async (...args)=>{calls.push(args); return {data:sample};};
(async()=>{
    await loadRunAccounting(true); assert.equal(calls.length,0);
    assert.match(nodes.get('run-accounting-status').textContent,/baseline/i);
    nodes.get('run-accounting-baseline').value='0.00252758';
    api=async (...args)=>{calls.push(args); return args[0].includes('accounting/run')?{data:sample}:
        {data:{usage_monthly_usd:.15252758},sample_started_at:sample.sampled_at,sample_finished_at:sample.sampled_at+2};};
    await loadRunAccounting(true); assert.equal(calls.length,2);
    assert.match(nodes.get('run-accounting-comparison').textContent,/\$0\.150000/);
    assert.match(nodes.get('run-accounting-comparison').textContent,/not.*reconciled/i);
    api=async (path)=>path.includes('accounting/run')?{data:sample}:
        {data:{usage_monthly_usd:.15252758},sample_started_at:1793491201,sample_finished_at:1793491202};
    await loadRunAccounting(true);
    assert.match(nodes.get('run-accounting-comparison').textContent,/month.*cannot cover/i);
    assert.doesNotMatch(nodes.get('run-accounting-comparison').textContent,/\$0\.150000/);
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_form_edit_and_session_lock_discard_old_results_and_finally_state():
    run_js(SETUP + consumer(), r"""
let resolve;let api=()=>new Promise(r=>resolve=r);
(async()=>{
    let old=loadRunAccounting(false);
    nodes.get('run-accounting-start').value='2026-10-03T00:00:00Z';clearRunAccounting(false);
    resolve({data:sample});await old;
    assert.match(nodes.get('run-accounting-ledger').textContent,/unknown/i);
    nodes.get('run-accounting-start').value='2026-10-04T22:55:42.000905Z';
    old=loadRunAccounting(false);sessionEpoch++;AUTH_TOKEN='';clearRunAccounting(true);
    resolve({data:sample});await old;
    assert.equal(nodes.get('run-accounting-start').value,'');
    assert.equal(runAccountingBusy,false);
    assert.equal(nodes.get('run-accounting-read').disabled,true);
    assert.doesNotMatch(nodes.get('run-accounting-ledger').textContent,/0\.150000/);
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_baseline_exceeds_usage_bad_costs_and_static_errors_never_echo_data():
    run_js(SETUP + consumer(), r"""
nodes.get('run-accounting-baseline').value='0.2';
let api=async path=>path.includes('accounting/run')?{data:sample}:
    {data:{usage_monthly_usd:.1},sample_started_at:sample.sampled_at,sample_finished_at:sample.sampled_at+1};
(async()=>{
    await loadRunAccounting(true);
    assert.match(nodes.get('run-accounting-comparison').textContent,/baseline.*exceeds/i);
    api=async()=>({error:'PRIVATE_DETAIL<img>'});await loadRunAccounting(false);
    assert.doesNotMatch(nodes.get('run-accounting-status').textContent,/PRIVATE_DETAIL|<img>/);
    assert.match(nodes.get('run-accounting-ledger').textContent,/unknown/i);
    for(const damage of [{settled_cost_usd:-1},{global_unresolved:null},
        {settled_resolutions:{response:9,operator:0}}, {since:cutoff+1}]) {
        api=async()=>({data:{...sample,...damage}});await loadRunAccounting(false);
        assert.match(nodes.get('run-accounting-ledger').textContent,/unknown/i);
    }
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_control_wiring_is_explicit_and_lock_clears_its_state():
    page = PAGE.read_text(encoding="utf-8")
    assert 'id="run-accounting-read"' in page and 'id="run-accounting-compare"' in page
    assert 'clearRunAccounting(true);' in function('function lockSession()', 'function startStatusPolling()')
    source = consumer()
    assert 'localStorage' not in source and 'innerHTML' not in source and "'POST'" not in source


def test_stale_rejection_cannot_overwrite_new_session_status():
    run_js(SETUP + consumer(), r"""
let reject;let api=()=>new Promise((_,r)=>reject=r);
(async()=>{
    const old=loadRunAccounting(false);sessionEpoch++;AUTH_TOKEN='';clearRunAccounting(true);
    const before=nodes.get('run-accounting-status').textContent;
    reject(Error('PRIVATE_STALE'));await old;
    assert.equal(nodes.get('run-accounting-status').textContent,before);
    assert.match(nodes.get('run-accounting-comparison').textContent,/not checked/i);
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_entire_inline_dashboard_javascript_parses_without_execution():
    scripts = re.findall(r'<script\b[^>]*>(.*?)</script>', PAGE.read_text(encoding='utf-8'), re.S)
    assert scripts
    run_js("const vm = require('node:vm');", '\n'.join(
        'new vm.Script(' + json.dumps(script) + ');' for script in scripts))
