"""Credential-only projections for ordinary memory; never a vault/unlock path.

Detection is conservative best-effort, not a classifier for arbitrary unlabeled
secrets. Projections are ephemeral: callers must not replace stored originals
with them. Hashes, paths and credential-location/status metadata remain useful.
"""
from __future__ import annotations

import re
from typing import Any


REDACTED = "[REDACTED_CREDENTIAL_VALUE]"
_NAME = re.compile(
    r"(?i)(?:^|_)(?:api_key|access_key|secret_key|private_key|access_token|"
    r"refresh_token|auth_token|client_secret|token|secret|password|passwd|"
    r"credential|credentials|authorization|bearer|cookie)$"
)
_LABEL = re.compile(
    r"(?P<label>[A-Za-z_][A-Za-z0-9_]*(?:[ -][A-Za-z_][A-Za-z0-9_]*){0,4})"
    r"[\"']?[ \t]*[:=][ \t]*"
)
_PREFIX = re.compile(
    r"(?<![\w])(?:sk-|ghp_|gho_|github_pat_|hf_|xoxb-|xoxp-)"
    r"[^\s\"'`<>]+", re.IGNORECASE
)
_BEARER = re.compile(r"\b(?:Bearer|Basic)[ \t]+[^\s\"'`<>]+", re.IGNORECASE)
_URL_AUTH = re.compile(r"\b(?P<scheme>[a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)
_PEM = re.compile(
    r"-----BEGIN (?P<kind>(?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED) )?PRIVATE KEY)-----"
    r"[\s\S]*?(?:-----END (?P=kind)-----|\Z)"
)
_STATUS = frozenset({"stored", "configured", "missing", "available", "absent",
                     "present", "unknown", "unset", "redacted", "issued_at"})
_MARKERS = (REDACTED, "[REDACTED]", "[REDACTED_SENSITIVE_VALUE]")
_LOCATION_FIELDS = frozenset({"status", "present", "source_path", "source_hint", "env_file",
                             "service", "credential_name", "name", "kind", "project",
                             "id", "record_id", "vault_ref", "created_at", "last_seen"})


class CredentialMemoryError(ValueError):
    def __init__(self):
        super().__init__("Credential values require the authenticated local vault workflow; "
                         "ordinary memory cannot store or rewrite them")


def _sensitive(name: str) -> bool:
    return bool(_NAME.search(name.lower().replace("-", "_").replace(" ", "_")))


def _marker_or_status(value: str) -> bool:
    return value in _MARKERS or value.casefold() in _STATUS


def _value_end(text: str, start: int, label: str) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return start
    if text[start] in "\"'`":
        quote = text[start]
        delimiter = quote * 3 if text.startswith(quote * 3, start) else quote
        index = start + len(delimiter)
        while index < len(text):
            if text[index] == "\\":
                index += 2
            elif text.startswith(delimiter, index):
                return index + len(delimiter)
            else:
                index += 1
        return len(text)  # incomplete quoted value: no suffix is released
    # Unquoted passwords can contain spaces and punctuation. Suppress the whole
    # value line rather than release a tail after its first lexical token.
    newline = text.find("\n", start)
    return len(text) if newline < 0 else newline


def _text(text: str) -> str:
    text = _PEM.sub(REDACTED, text)
    pieces, cursor = [], 0
    for match in _LABEL.finditer(text):
        if match.start() < cursor or not _sensitive(match.group("label")):
            continue
        start = match.end()
        # Existing transcript markers must remain idempotent, including commas
        # following quoted JSON values. They carry no original credential.
        marker = next((m for m in _MARKERS if text.startswith(m, start)), None)
        if marker:
            continue
        end = _value_end(text, start, match.group("label"))
        value = text[start:end]
        unquoted = value.strip().strip("\"'`")
        # A provenance description such as "fresh token: issued_at 0.5d" is
        # not an authentication value; retain this narrow status form.
        status_description = (match.group("label").lower().endswith("token")
                              and unquoted.startswith("issued_at "))
        if not value or _marker_or_status(unquoted) or status_description:
            continue
        pieces.extend((text[cursor:start], REDACTED))
        cursor = end
    pieces.append(text[cursor:])
    projected = _BEARER.sub(REDACTED, _PREFIX.sub(REDACTED, "".join(pieces)))
    return _URL_AUTH.sub(lambda m: m["scheme"] + REDACTED + "@", projected)


def _structured_secret(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value if _marker_or_status(value) or not value else REDACTED
    if isinstance(value, dict):
        return {_text(str(k)): (project_credentials(v) if k in _LOCATION_FIELDS
                                else _structured_secret(v)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_structured_secret(v) for v in value)
    return REDACTED


def project_credentials(value: Any) -> Any:
    """Return a copy safe for ordinary consumers, never persist this projection."""
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, dict):
        return {_text(str(k)): (_structured_secret(v) if _sensitive(str(k))
                                else project_credentials(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(project_credentials(v) for v in value)
    return value


def require_credential_free(*values: Any) -> None:
    """Fail before telemetry, inference or ordinary persistence; errors are static."""
    if any(project_credentials(value) != value for value in values):
        raise CredentialMemoryError()
