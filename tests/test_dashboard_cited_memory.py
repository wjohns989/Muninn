"""Real UI consumers; synthetic data only, no private service/model calls."""
from tests.test_dashboard_control_center import function, run_js, PAGE


def consumer():
    return function("function clearCitedMemoryView()", "function clearHistoryTranscript()")


SETUP = r"""
let AUTH_TOKEN = 'fixture', sessionEpoch = 1, citedReadSequence = 0;
let citedReviewCursor = null, citedReadBusy = false, historySearchSequence = 0;
const nodes = new Map();
function node() {return {textContent: '', children: [], disabled: false, hidden: false,
    appendChild(n) {this.children.push(n); return n;},
    replaceChildren(...n) {this.children = n; this.textContent = '';},
    addEventListener(event, fn) {this[event] = fn;}};}
const document = {createElement: () => node(), getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id);}};
let transcriptClears = 0, transcriptOpens = [];
function clearHistoryTranscript() {transcriptClears++;}
async function openHistoryTranscript(...args) {transcriptOpens.push(args);}
function allText(n) {return n.textContent + n.children.map(allText).join(' ');}
const ident = 'a'.repeat(64);
const memory = {id: ident, state: 'provisional', truth_status: 'model_inferred',
    epistemic_kind: 'model_interpretation', text: '<img src=x onerror=alert(1)>',
    quote: 'literal quote', source_ref: 'b'.repeat(64), project_ref: 'c'.repeat(64),
    time_basis: 'provider_record', event_at: '2026-10-01', project_basis: 'source',
    type: 'decision', proposal_origin: 'model'};
"""


def test_normalization_and_literal_rendering_do_not_emit_capabilities():
    run_js(SETUP + consumer(), r"""
assert.equal(citedMemoryView({...memory, id: '<bad>'}), null);
const safe = citedMemoryView({...memory, secret: 'MUST_NOT_RENDER',
    transcript_capability: 'PRIVATE_CAPABILITY'});
const card = renderCitedMemory(safe, () => true);
const text = allText(card);
assert.match(text, /<img src=x onerror=alert\(1\)>/);
assert.match(text, /model_inferred/);
assert.match(text, /provisional/);
assert.match(text, /provider_record/);
assert.doesNotMatch(text, /MUST_NOT_RENDER|PRIVATE_CAPABILITY/);
assert.equal(citedMemoryView({...memory, text: 'x'.repeat(2049)}).text, null);
""")


def test_source_withheld_invalid_identity_and_transcript_click_race():
    run_js(SETUP + consumer(), r"""
let reply = {success: true, data: {memory, context_state: 'withheld',
    context: 'MUST_NOT_RENDER', transcript_capability: 'fixture-cap'}};
let api = async (path, method, body) => {
    assert.equal(path, '/history/secure/memories/source');
    assert.equal(method, 'POST'); assert.equal(body.max_chars, 3000); return reply;
};
(async () => {
    await loadCitedMemory(ident, true, citedReadSequence);
    let detail = nodes.get('cited-memory-detail');
    assert.match(allText(detail), /withheld/i);
    assert.doesNotMatch(allText(detail), /MUST_NOT_RENDER/);
    reply.data.context_state = 'available'; reply.data.context = 'visible source';
    await loadCitedMemory(ident, true, citedReadSequence);
    assert.match(allText(detail), /visible source/);
    const button = detail.children.find(n => n.textContent === 'Read redacted transcript');
    assert.ok(button); button.click();
    assert.equal(transcriptOpens.length, 1);
    clearCitedMemoryView(); button.disabled = false; button.click();
    assert.equal(transcriptOpens.length, 1);
    reply.data.context = 'x'.repeat(3001);
    await loadCitedMemory(ident, true, citedReadSequence);
    assert.match(allText(detail), /withheld/i);
    assert.doesNotMatch(allText(detail), /x{100}/);
    reply.data.memory = {...memory, id: 'd'.repeat(64)};
    await loadCitedMemory(ident, true, citedReadSequence);
    assert.equal(detail.children.length, 0);
    assert.match(nodes.get('cited-memory-status').textContent, /unavailable/i);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_read_bounds_stale_same_token_session_and_review_subset():
    run_js(SETUP + consumer(), r"""
let calls = [], resolve;
let api = async (...args) => {calls.push(args); return new Promise(r => resolve = r);};
(async () => {
    nodes.set('cited-memory-query', {...node(), value: '界'.repeat(200)});
    await searchCitedMemories(); assert.equal(calls.length, 0);
    nodes.get('cited-memory-query').value = 'bounded fixture';
    const old = searchCitedMemories();
    assert.deepEqual(calls[0].slice(0, 3), ['/history/secure/memories/search', 'POST',
        {query: 'bounded fixture', limit: 10}]);
    sessionEpoch++; clearCitedMemoryView();
    resolve({success: true, data: {matches: [memory], total_matches: 1}}); await old;
    assert.equal(nodes.get('cited-memory-results').children.length, 0);
    api = async (...args) => {calls.push(args); return {success: true, data: {
        matches: [memory], has_more: true, next_cursor: 'private-cursor',
        credential_or_withheld_excluded: true}};};
    await browseCitedReviews(false);
    assert.deepEqual(calls.at(-1).slice(0, 3), ['/history/secure/memories/review-queue', 'POST', {limit: 6}]);
    assert.equal(citedReviewCursor, 'private-cursor');
    assert.match(nodes.get('cited-memory-status').textContent, /subset/i);
    assert.doesNotMatch(allText(nodes.get('cited-memory-results')), /private-cursor/);
    await browseCitedReviews(true);
    assert.deepEqual(calls.at(-1)[2], {limit: 6, cursor: 'private-cursor'});
    assert.equal(citedReviewCursor, null); // Repeated cursor cannot loop forever.
    assert.match(nodes.get('cited-memory-status').textContent, /invalid/i);
    clearCitedMemoryView(); assert.equal(citedReviewCursor, null);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_source_completion_after_lock_and_busy_read_do_not_repaint():
    run_js(SETUP + consumer(), r"""
let resolve, calls = 0;
let api = async () => {calls++; return new Promise(r => resolve = r);};
(async () => {
    const old = loadCitedMemory(ident, true, citedReadSequence);
    await loadCitedMemory(ident, true, citedReadSequence);
    assert.equal(calls, 1);
    sessionEpoch++; clearCitedMemoryView();
    resolve({success: true, data: {memory, context_state: 'available', context: 'OLD_PRIVATE',
        transcript_capability: 'OLD_CAPABILITY'}}); await old;
    assert.equal(nodes.get('cited-memory-detail').children.length, 0);
    api = async () => ({error: 'timeout with private detail MUST_NOT_RENDER'});
    await loadCitedMemory(ident, true, citedReadSequence);
    assert.match(nodes.get('cited-memory-status').textContent, /may still be running/);
    assert.doesNotMatch(nodes.get('cited-memory-status').textContent, /MUST_NOT_RENDER/);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")


def test_consumer_is_explicit_read_only_and_lock_clears_it():
    page = PAGE.read_text(encoding='utf-8')
    source = consumer()
    assert 'innerHTML' not in source
    assert 'localStorage' not in source
    assert "'DELETE'" not in source
    assert "clearCitedMemoryView();" in function('function lockSession()', 'function startStatusPolling()')
    assert 'clearCitedMemoryView();' in function('async function handleEncryptedHistorySearch()', 'async function handleSearch()')
    assert 'id="cited-memory-search"' in page
    assert 'id="cited-memory-reviews"' in page
    assert 'id="cited-memory-next"' in page


def test_missing_search_completeness_is_unknown_not_complete():
    run_js(SETUP + consumer(), r"""
const api = async () => ({success: true, data: {matches: []}});
(async () => {
    await readCitedMemories('search', {query: 'fixture', limit: 10});
    const status = nodes.get('cited-memory-status').textContent;
    assert.match(status, /Further matches unknown/);
    assert.doesNotMatch(status, /No further matches/);
})().catch(e => {console.error(e); process.exitCode = 1;});
""")
