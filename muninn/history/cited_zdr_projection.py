"""Screened original-coordinate view; never a batch admission.

Excluded text is replaced with spaces without shifting original coordinates.
Only unchanged, independently screened ranges can support a quote. Credentials
remain local; callers still need explicit consent and the exact-body ZDR gate.
"""
from __future__ import annotations

import re
from copy import deepcopy

from muninn.history.cited_analysis_source import CitedSourceError
from muninn.history.cited_redaction_ranges import unchanged_cited_ranges
from muninn.history.streaming_redaction import _SENSITIVE


def _safe_run(source, text):
    # A dangling retained label is not prose admission: after its removed value,
    # it could consume the next line as a value in the final serialized body.
    tail = re.search(r"([\w-]+)[\s'\":=]*$", text)
    dangling = tail and _SENSITIVE.search(tail[1].casefold().replace('-', '_'))
    return not dangling and source.ledger._screen({'text': text})


class CitedZDRProjection:
    @classmethod
    def from_source_view(cls, source, descriptor, view):
        """Recompute proof; authenticated storage alone is not a privacy proof."""
        if (not isinstance(view, dict) or set(view) != {'policy', 'window', 'ranges'}
                or view['policy'] != 'zdr-ranges-v1' or view['window'] != descriptor
                or not isinstance(view['ranges'], list) or len(view['ranges']) > 3000
                or any(not isinstance(r, dict) or set(r) != {'start', 'length'}
                       or type(r['start']) is not int or type(r['length']) is not int
                       or r['start'] < 0 or r['length'] < 1 for r in view['ranges'])):
            raise CitedSourceError('Invalid original-range source view')
        # Python equality aliases True/1 and 1.0/1; the source contract does not.
        source.validate_descriptor(descriptor)
        source.validate_descriptor(view['window'])
        proof = cls(source, descriptor)
        if proof.source_view() != view:
            raise CitedSourceError('Source view differs from canonical original ranges')
        return proof

    def source_view(self):
        return {'policy': 'zdr-ranges-v1', 'window': deepcopy(self._descriptor),
                'ranges': deepcopy(self._window['citation_ranges'])}

    def validate_page_proposal(self, page, proposal):
        """Map a ledger citation back to its original, non-crossable window."""
        descriptor = self._descriptor
        prefix = descriptor['prefix']
        if page == descriptor['page']:
            start = proposal['start'] - descriptor['offset'] + (prefix['length'] if prefix else 0)
        elif prefix and page == prefix['page']:
            start = proposal['start'] - prefix['offset']
        else:
            raise CitedSourceError('Citation is outside source-view pages')
        checked = self.validated_proposals(descriptor, [{**proposal, 'start': start}])
        if checked != [{'page': page, 'proposal': proposal}]:
            raise CitedSourceError('Citation is outside source-view coordinates')
        return start

    def __init__(self, source, descriptor, *, should_cancel=lambda: False):
        self._source = source
        self._descriptor = deepcopy(descriptor)
        self._should_cancel = should_cancel
        raw = source.reopen(descriptor)
        ranges = unchanged_cited_ranges(source, descriptor, should_cancel=should_cancel)
        admitted = []
        for span in ranges:
            text, start = span['text'], span['start']
            if _safe_run(source, text):
                admitted.append({'start': start, 'text': text})
                continue
            # A surviving label or private path must not poison unrelated
            # lines. Drop unsafe lines, never rewrite them into claimed quotes.
            offset = start
            for line in text.splitlines(keepends=True):
                content = line.rstrip('\r\n')
                if content and _safe_run(source, content):
                    admitted.append({'start': offset, 'text': content})
                offset += len(line)
        text = [' '] * len(raw['text'])
        for span in admitted:
            start, value = span['start'], span['text']
            text[start:start + len(value)] = value
        self._window = {**raw, 'text': ''.join(text),
                        'citation_ranges': [{'start': r['start'], 'length': len(r['text'])}
                                            for r in admitted],
                        'projection_policy': 'zdr-ranges-v1'}
        self._check(descriptor)

    def _check(self, descriptor):
        if self._should_cancel():
            raise CitedSourceError('Cited ZDR projection cancelled')
        if descriptor != self._descriptor:
            raise CitedSourceError('Cited ZDR projection descriptor changed')
        # Immutable raw input must still authenticate, even when using a view
        # already computed after whole-unit EOF. No plaintext cache is persisted.
        self._source.reopen(descriptor)

    def reopen(self, descriptor):
        self._check(descriptor)
        return deepcopy(self._window)

    def result_source_span(self, descriptor):
        """Local-only output scrub input; never part of the provider view."""
        self._check(descriptor)
        return self._source.reopen(descriptor)['text']

    def remote_input(self, descriptor):
        window = self.reopen(descriptor)
        content = {k: v for k, v in window.items() if k != 'project_ref'}
        return window if (window['text'].strip() and window['citation_ranges']
                          and self._source.ledger._screen(content)) else None

    def validated_proposals(self, descriptor, proposals):
        window = self.reopen(descriptor)
        if not isinstance(proposals, list) or len(proposals) > 12:
            raise CitedSourceError('Invalid bounded cited proposals')
        for p in proposals:
            start, quote = (p.get('start'), p.get('quote')) if isinstance(p, dict) else (None, None)
            if (type(start) is not int or not isinstance(quote, str) or not quote.strip()
                    or window['text'][start:start + len(quote)] != quote
                    or not any(r['start'] <= start and start + len(quote) <= r['start'] + r['length']
                               for r in window['citation_ranges'])):
                raise CitedSourceError('Model quote is outside admitted original ranges')
        return self._source.validated_proposals(descriptor, proposals)
