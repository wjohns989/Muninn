"""Private, authenticated, bounded model windows and exact memory citations.

Descriptors belong only in encrypted job state. This module never dispatches
models, returns raw text through an endpoint, or claims whole-source extraction.
"""
from __future__ import annotations

import hashlib
import hmac

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError, TYPES, _json
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import UnsupportedTranscript
from muninn.history.transcript_units import PARSER_VERSION

_FIELDS = {"format", "blob", "sha256", "version", "attempt", "page", "offset",
           "length", "parser_version", "input_sha256", "boundary_hit", "prefix"}


class CitedSourceError(ValueError):
    """A source or proposed citation is invalid; never include private context."""


class CitedAnalysisSource:
    def __init__(self, archive, *, read_only=False):
        self.archive = archive
        self.ledger = MemoryLedger(archive, read_only=read_only)

    @staticmethod
    def _digest(window):
        # Canonical structured encoding separates metadata/text unambiguously.
        return hashlib.sha256(b"muninn-cited-input-v1\0" + _json(window)).hexdigest()

    def prepare(self, capability, *, should_cancel=lambda: False):
        entry, version, grant = SecureHistoryBlindIndex(self.archive)._entry_for_capability(capability)
        try:
            attempt = self.ledger.units.build_snapshot(entry, version, should_cancel=should_cancel)
            selected, carry, current, previous = None, "", None, None
            term = grant["term"]
            for page, part in enumerate(self.ledger.units.fragments(entry, version, attempt)):
                if should_cancel():
                    raise CitedSourceError("Cited analysis preparation cancelled")
                if current != part.unit.ordinal:
                    carry, current, previous = "", part.unit.ordinal, None
                if selected is None and part.text and not part.final:
                    folded = part.text.casefold()
                    position = folded.find(term)
                    crossing = position < 0 and term in (carry + folded[:len(term)])
                    if position >= 0 or crossing:
                        # Case folding can expand Unicode scalars; translate the
                        # matched folded coordinate back to the ORIGINAL page.
                        raw_position, consumed = 0, 0
                        if position >= 0:
                            for raw_position, char in enumerate(part.text):
                                width = len(char.casefold())
                                if consumed + width > position:
                                    break
                                consumed += width
                        offset = max(0, raw_position - 1000) if position >= 0 else 0
                        prefix = None
                        if crossing:
                            if previous is None:
                                raise CitedSourceError("Cited boundary context is unavailable")
                            prefix_length = min(1000, len(previous[1]))
                            prefix = {"page": previous[0], "offset": len(previous[1]) - prefix_length,
                                      "length": prefix_length}
                        selected = {"format": 1, "blob": entry["blob"], "sha256": entry["sha256"],
                                    "version": version, "attempt": attempt, "page": page,
                                    "offset": offset, "length": min(3000 - (prefix["length"] if prefix else 0),
                                                                      len(part.text) - offset),
                                    "parser_version": PARSER_VERSION, "boundary_hit": crossing, "prefix": prefix}
                    carry = (carry + folded)[-len(term):]
                    previous = page, part.text
                if part.final:
                    carry = ""
                # Authenticate every sealed page, even after an early hit.
            if selected is None:
                return None
            window = self._window(selected)[2]
            selected["input_sha256"] = self._digest(window)
            return selected
        except (MemoryLedgerIntegrityError, ProjectionIntegrityError) as exc:
            raise CitedSourceError("Cited source authentication failed") from exc
        except (StreamingJSONError, UnsupportedTranscript):
            # Legacy unsupported sources retain raw capture/search access, but
            # cannot acquire fabricated structured citations or coverage.
            return None

    @staticmethod
    def validate_descriptor(value):
        if (not isinstance(value, dict) or set(value) != _FIELDS
                or value["format"] != 1 or type(value["format"]) is not int
                or type(value["parser_version"]) is not int or value["parser_version"] != PARSER_VERSION
                or any(type(value[k]) is not int or value[k] < 0 for k in ("version", "page", "offset"))
                or type(value["length"]) is not int or not 1 <= value["length"] <= 3000
                or value["offset"] + value["length"] > 4096
                or type(value["boundary_hit"]) is not bool
                or not isinstance(value["blob"], str) or len(value["blob"]) != 32
                or any(c not in "0123456789abcdef" for c in value["blob"])
                or not isinstance(value["attempt"], str) or len(value["attempt"]) != 32
                or any(c not in "0123456789abcdef" for c in value["attempt"])
                or not MemoryLedger._hex(value["sha256"]) or not MemoryLedger._hex(value["input_sha256"])):
            raise CitedSourceError("Invalid cited source descriptor")
        prefix = value["prefix"]
        if prefix is not None:
            if (not isinstance(prefix, dict) or set(prefix) != {"page", "offset", "length"}
                    or any(type(prefix[k]) is not int or prefix[k] < 0 for k in prefix)
                    or not 1 <= prefix["length"] <= 1000
                    or prefix["page"] + 1 != value["page"]
                    or prefix["offset"] + prefix["length"] > 4096
                    or prefix["length"] + value["length"] > 3000
                    or value["offset"] != 0 or not value["boundary_hit"]):
                raise CitedSourceError("Invalid cited boundary descriptor")
        elif value["boundary_hit"]:
            raise CitedSourceError("Cited boundary descriptor is incomplete")
        return value

    def _window(self, descriptor):
        try:
            entry = self.ledger._entries[(descriptor["blob"], descriptor["version"])]
            if entry["sha256"] != descriptor["sha256"]:
                raise CitedSourceError("Cited snapshot is unavailable")
            unit, page = self.ledger._source(entry, descriptor["version"], descriptor["attempt"],
                                            descriptor["page"])
            offset, length = descriptor["offset"], descriptor["length"]
            if offset + length > len(page["text"]):
                raise CitedSourceError("Cited source window is unavailable")
            text, ranges = page["text"][offset:offset + length], [{"start": 0, "length": length}]
            prefix = descriptor["prefix"]
            if prefix is not None:
                earlier, earlier_page = self.ledger._source(entry, descriptor["version"], descriptor["attempt"],
                                                           prefix["page"])
                if earlier != unit or prefix["offset"] + prefix["length"] != len(earlier_page["text"]):
                    raise CitedSourceError("Cited boundary context changed")
                text = earlier_page["text"][prefix["offset"]:] + text
                ranges = [{"start": 0, "length": prefix["length"]},
                          {"start": prefix["length"], "length": length}]
            window = self.content_window(unit, text, boundary_hit=descriptor["boundary_hit"],
                                         citation_ranges=ranges)
            return entry, page, window
        except (KeyError, TypeError, MemoryLedgerIntegrityError, ProjectionIntegrityError) as exc:
            raise CitedSourceError("Cited source authentication failed") from exc

    def content_window(self, unit, text, *, boundary_hit=False, citation_ranges=None):
        """Canonical private window encoding, shared by search and coverage plans."""
        project = (hmac.new(self.ledger._key, b"project\0" + unit.cwd.encode("utf-8"),
                            hashlib.sha256).hexdigest() if unit.cwd else None)
        return {"text": text, "provider": unit.provider,
                "role": unit.role.casefold() if unit.role else None,
                "event_at": unit.event_at, "time_basis": unit.time_basis,
                "project_ref": project, "project_basis": unit.project_basis,
                "boundary_hit": boundary_hit,
                "citation_ranges": citation_ranges if citation_ranges is not None
                                   else [{"start": 0, "length": len(text)}]}

    def reopen(self, descriptor):
        self.validate_descriptor(descriptor)
        window = self._window(descriptor)[2]
        if not hmac.compare_digest(self._digest(window), descriptor["input_sha256"]):
            raise CitedSourceError("Cited analysis input changed")
        return window

    def remote_input(self, descriptor):
        window = self.reopen(descriptor)
        entry = self._window(descriptor)[0]
        # This drains/authenticates the whole unit, not just the chosen excerpt.
        approved = self.ledger.remote_input(entry, descriptor["version"], descriptor["attempt"],
                                            descriptor["page"])
        # Opaque project refs are not supplied to the content sanitizer as prose.
        content = {k: v for k, v in window.items() if k != "project_ref"}
        return window if approved is not None and self.ledger._screen(content) else None

    def validated_proposals(self, descriptor, proposals):
        window = self.reopen(descriptor)
        if not isinstance(proposals, list) or len(proposals) > 12:
            raise CitedSourceError("Invalid bounded cited proposals")
        checked = []
        for proposal in proposals:
            if (not isinstance(proposal, dict) or set(proposal) != {"type", "text", "quote", "start"}
                    or not isinstance(proposal["type"], str)
                    or proposal["type"] not in TYPES
                    or any(not isinstance(proposal[k], str) or not 1 <= len(proposal[k]) <= 2048
                           for k in ("text", "quote"))
                    or type(proposal["start"]) is not int or proposal["start"] < 0
                    or window["text"][proposal["start"]:proposal["start"] + len(proposal["quote"])]
                    != proposal["quote"]):
                raise CitedSourceError("Model quote is not supported by its cited input")
            prefix = descriptor["prefix"]
            prefix_length = prefix["length"] if prefix else 0
            start, end = proposal["start"], proposal["start"] + len(proposal["quote"])
            if prefix and end <= prefix_length:
                page, source_start = prefix["page"], prefix["offset"] + start
            elif start >= prefix_length:
                page, source_start = descriptor["page"], descriptor["offset"] + start - prefix_length
            else:
                raise CitedSourceError("Proposed quote crosses its authenticated citation range")
            checked.append({"page": page, "proposal": {**proposal, "start": source_start}})
        return checked

    def record_proposals(self, descriptor, proposals, *, model_identity, source_view=None):
        if source_view is not None and (not isinstance(source_view, dict)
                                       or source_view.get('window') != descriptor):
            raise CitedSourceError('Publication view does not match its extraction window')
        checked = self.validated_proposals(descriptor, proposals)
        if not MemoryLedger._hex(model_identity):
            raise CitedSourceError("Invalid extraction model identity")
        if not checked:
            if source_view is not None:
                from muninn.history.cited_zdr_projection import CitedZDRProjection
                CitedZDRProjection.from_source_view(self, descriptor, source_view)
            return []
        entry = self._window(descriptor)[0]
        return self.ledger.record_batch(entry, descriptor["version"], descriptor["attempt"], checked,
                                        model_identity=model_identity, source_view=source_view)

    def expected_refs(self, descriptor, proposals, *, model_identity, source_view=None):
        """Compute stage-bound identities without publishing any candidate."""
        if source_view is not None and (not isinstance(source_view, dict)
                                       or source_view.get('window') != descriptor):
            raise CitedSourceError('Publication view does not match its extraction window')
        checked = self.validated_proposals(descriptor, proposals)
        if not MemoryLedger._hex(model_identity):
            raise CitedSourceError("Invalid extraction model identity")
        entry = self._window(descriptor)[0]
        return [ref for ref, _ in self.ledger._prepare_batch(
            entry, descriptor["version"], descriptor["attempt"], checked,
            model_identity=model_identity, source_view=source_view)]
