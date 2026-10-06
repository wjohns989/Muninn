"""Private synchronous-ZDR view, never a batch or public-read admission.

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
