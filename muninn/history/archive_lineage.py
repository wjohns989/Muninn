"""Authenticated capture-location aliases, not project or event-time authority.

Canonical file keys and historical entries are immutable. Format 2 is emitted
only on the first proven relocation; format-1 writers must reject it. Aliases
are direct references, never redirects through another alias.
"""
from __future__ import annotations

import re
from pathlib import Path

from muninn.history.credential_crypto import VaultIntegrityError

_SESSION = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)


def native_identity(path: str, provider: str, kind: str):
    if provider not in {"codex", "claude_code"} or kind != "transcript":
        return None
    matches = _SESSION.findall(Path(path).name)
    # Do not choose an arbitrary identity from a multi-UUID filename.
    return (provider, kind, matches[0].lower()) if len(matches) == 1 else None


def canonical_source(manifest: dict, source: str) -> str:
    alias = manifest.get("aliases", {}).get(source)
    return alias["anchor"] if alias is not None else source


def validate_lineage(manifest: dict) -> None:
    format_version = manifest.get("format")
    if type(format_version) is not int or format_version not in {1, 2}:
        raise VaultIntegrityError("History archive lineage format is unsupported")
    files = manifest["files"]
    if any(not isinstance(anchor, str) or not isinstance(entries, list) or not entries
           or any(not isinstance(entry, dict) for entry in entries)
           for anchor, entries in files.items()):
        raise VaultIntegrityError("History archive lineage anchor is invalid")
    if format_version == 1:
        if "aliases" in manifest or any(
                "observed_source" in entry for entries in files.values() for entry in entries):
            raise VaultIntegrityError("History archive lineage requires format 2")
        return
    aliases = manifest.get("aliases")
    if not isinstance(aliases, dict):
        raise VaultIntegrityError("History archive aliases are invalid")
    identities = {}
    for anchor, entries in files.items():
        latest = entries[-1]
        identity = native_identity(anchor, latest.get("provider"), latest.get("kind"))
        if identity is not None:
            identities.setdefault(identity, []).append(anchor)
    for source, alias in aliases.items():
        if (not isinstance(source, str) or source in files or not isinstance(alias, dict)
                or set(alias) != {"anchor", "size", "mtime_ns", "sha256"}
                or not isinstance(alias["anchor"], str) or alias["anchor"] not in files
                or type(alias["size"]) is not int or alias["size"] < 0
                or type(alias["mtime_ns"]) is not int or alias["mtime_ns"] < 0):
            raise VaultIntegrityError("History archive alias reference is invalid")
        anchor = alias["anchor"]
        entries = files[anchor]
        latest = entries[-1]
        identity = native_identity(anchor, latest.get("provider"), latest.get("kind"))
        if (identity is None or native_identity(source, *identity[:2]) != identity
                or identities.get(identity) != [anchor]
                or not any(entry.get("size") == alias["size"]
                           and entry.get("sha256") == alias["sha256"] for entry in entries)):
            raise VaultIntegrityError("History archive alias origin is invalid")
    for anchor, entries in files.items():
        for entry in entries:
            observed = entry.get("observed_source")
            if "observed_source" in entry and (
                    not isinstance(observed, str) or observed not in aliases
                    or aliases[observed]["anchor"] != anchor
                    or native_identity(observed, entry.get("provider"), entry.get("kind"))
                    != native_identity(anchor, entry.get("provider"), entry.get("kind"))):
                raise VaultIntegrityError("History archive observed location is invalid")


def capture_anchor(manifest: dict, source: str, provider: str, kind: str) -> str:
    """Choose by native identity only; the caller must still prove latest bytes."""
    files = manifest["files"]
    if source in files:
        if (any(alias["anchor"] == source for alias in manifest.get("aliases", {}).values())
                and any(files[source][-1].get(key) != value
                        for key, value in (("provider", provider), ("kind", kind)))):
            raise ValueError("History aliased origin cannot change provider or kind")
        return source
    identity = native_identity(source, provider, kind)
    known = manifest.get("aliases", {}).get(source)
    if known is not None:
        latest = files[known["anchor"]][-1]
        if identity != native_identity(known["anchor"], latest["provider"], latest["kind"]):
            raise ValueError("History relocation origin changed")
    if identity is None:
        return source
    matches = [anchor for anchor, entries in files.items()
               if native_identity(anchor, entries[-1]["provider"], entries[-1]["kind"]) == identity]
    if len(matches) > 1:
        raise ValueError("History relocation has conflicting native lineages")
    return matches[0] if matches else source


def record_alias(manifest: dict, source: str, anchor: str, entry: dict, mtime_ns: int) -> None:
    if source == anchor:
        return
    manifest["format"] = 2
    manifest.setdefault("aliases", {})[source] = {
        "anchor": anchor, "size": entry["size"], "mtime_ns": mtime_ns,
        "sha256": entry["sha256"],
    }


def source_signatures(manifest: dict) -> dict[str, tuple[int, int]]:
    signatures = {path: (entries[-1]["size"], entries[-1]["mtime_ns"])
                  for path, entries in manifest["files"].items() if entries}
    for path, alias in manifest.get("aliases", {}).items():
        latest = manifest["files"][alias["anchor"]][-1]
        if alias["sha256"] == latest["sha256"] and alias["size"] == latest["size"]:
            signatures[path] = (alias["size"], alias["mtime_ns"])
    return signatures
