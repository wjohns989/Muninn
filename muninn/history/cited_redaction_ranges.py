"""Private original-coordinate selection; NOT a model or public-read permission.

The complete immutable unit is authenticated and redacted before any result is
returned. Only unchanged runs inside the already authenticated bounded cited
window are retained. Separate ranges never imply continuity across excluded
text or page citation boundaries. No API, provider, ledger write or cache here.
"""
from __future__ import annotations

from muninn.history.cited_analysis_source import CitedSourceError
from muninn.history.memory_ledger import MemoryLedgerIntegrityError
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.streaming_redaction import redacted_fragments


def unchanged_cited_ranges(source, descriptor, *, should_cancel=lambda: False):
    """Return bounded exact ORIGINAL window positions after whole-unit EOF.

    An unchanged run can still contain sensitive labels, paths or undetected
    values. Any later provider use needs independent exact-body screening,
    consent, strict ZDR and billing/dispatch guards; batch admission and public
    reads retain their current whole-unit policy. This is not that admission.
    """
    if should_cancel():
        raise CitedSourceError('Cited range selection cancelled')
    window = source.reopen(descriptor)
    entry, data, _ = source._window(descriptor)
    unit, _ = source.ledger._source(entry, descriptor['version'], descriptor['attempt'], descriptor['page'])
    wanted = [(data['fragment'], descriptor['offset'], descriptor['length'], 0)]
    prefix = descriptor['prefix']
    if prefix is not None:
        previous, page = source.ledger._source(entry, descriptor['version'], descriptor['attempt'], prefix['page'])
        if previous != unit:
            raise CitedSourceError('Cited range source authentication failed')
        wanted = [(page['fragment'], prefix['offset'], prefix['length'], 0),
                  (data['fragment'], descriptor['offset'], descriptor['length'], prefix['length'])]
    selected = []
    ranges = []
    seen = set()

    def chunks():
        offset = 0
        parts = source.ledger.units.unit_fragments(entry, descriptor['version'], descriptor['attempt'], unit.ordinal)
        try:
            for fragment, part in enumerate(parts):
                if should_cancel():
                    raise CitedSourceError('Cited range selection cancelled')
                if part.unit != unit:
                    raise CitedSourceError('Cited range source authentication failed')
                for index, (target, start, length, window_start) in enumerate(wanted):
                    if fragment == target:
                        if part.final or start + length > len(part.text):
                            raise CitedSourceError('Cited range source authentication failed')
                        selected.append((index, offset + start, length, window_start))
                        seen.add(index)
                offset += len(part.text)
                yield part.text
        finally:
            parts.close()

    def unchanged(text, start, length):
        if should_cancel():
            raise CitedSourceError('Cited range selection cancelled')
        for index, original_start, selected_length, window_start in selected:
            left, right = max(start, original_start), min(start + length, original_start + selected_length)
            if left >= right:
                continue
            position = window_start + left - original_start
            value = text[left - start:right - start]
            if window['text'][position:position + len(value)] != value:
                raise CitedSourceError('Cited range source authentication failed')
            if ranges and ranges[-1][0] == index and ranges[-1][1] + len(ranges[-1][2]) == position:
                ranges[-1] = index, ranges[-1][1], ranges[-1][2] + value
            else:
                ranges.append((index, position, value))

    # Nothing escapes this function until the generator, including its sealed
    # final unit marker, succeeds. Late errors discard the speculative list.
    stream = chunks()
    try:
        for _ in redacted_fragments(stream, on_unchanged=unchanged):
            pass
    except (ProjectionIntegrityError, MemoryLedgerIntegrityError) as exc:
        raise CitedSourceError('Cited range source authentication failed') from exc
    finally:
        stream.close()
    if seen != set(range(len(wanted))):
        raise CitedSourceError('Cited range source authentication failed')
    if should_cancel():
        raise CitedSourceError('Cited range selection cancelled')
    return [{'start': start, 'text': text} for _, start, text in ranges]
