"""Choose a duplicate native-session locator by bytes, never discovery order.

Selection is read-only and pinned to the caller's authenticated manifest. It
does not grant capture authority; the archive writer repeats its current-latest
continuity and opened/path identity checks before publishing anything.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from muninn.history.archive_lineage import native_identity

_CHUNK = 1024 * 1024


def _fingerprint(details):
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _digests(chunks, boundaries):
    digest, size = hashlib.sha256(), 0
    prefixes = {boundary: hashlib.sha256() for boundary in set(boundaries)}
    for block in chunks:
        digest.update(block)
        for boundary, prefix in prefixes.items():
            if size < boundary:
                prefix.update(block[:min(len(block), boundary - size)])
        size += len(block)
    return size, digest.hexdigest(), {boundary: prefix.hexdigest()
                                      for boundary, prefix in prefixes.items()}


def _read_path(path, boundaries=(), *, expected=None):
    before = _fingerprint(path.stat())
    if expected is not None and before != expected:
        raise RuntimeError("Capture selection source changed")

    def checked():
        with path.open("rb") as source:
            if path.resolve(strict=True) != path or _fingerprint(os.fstat(source.fileno())) != before:
                raise RuntimeError("Capture selection source changed")
            yield from iter(lambda: source.read(_CHUNK), b"")
            if _fingerprint(os.fstat(source.fileno())) != before:
                raise RuntimeError("Capture selection source changed")
        if path.resolve(strict=True) != path or _fingerprint(path.stat()) != before:
            raise RuntimeError("Capture selection source changed")

    size, digest, prefixes = _digests(checked(), boundaries)
    if size != before[2]:
        raise RuntimeError("Capture selection source changed")
    return {"path": path, "size": size, "sha256": digest,
            "prefixes": prefixes, "fingerprint": before}


def select_capture_locator(archive, manifest, paths, provider):
    """Return one byte-preserving locator, None for already captured stale copies.

    Different viable branches or unverified copies require explicit review.
    Only groups with multiple physical locators need these content reads. Each
    is read once; the chosen largest needs at most one extra prefix-proof pass.
    """
    paths = sorted(set(paths), key=str)
    identity = native_identity(str(paths[0]), provider, "transcript")
    if identity is None or any(native_identity(str(path), provider, "transcript") != identity
                               for path in paths):
        raise ValueError("Capture selection native origins conflict")
    anchors = [(anchor, entries) for anchor, entries in manifest["files"].items()
               if native_identity(anchor, entries[-1]["provider"], entries[-1]["kind"]) == identity]
    if len(anchors) > 1:
        raise ValueError("Capture selection native lineages require review")
    latest = anchors[0][1][-1] if anchors else None
    observations = [_read_path(path, (latest["size"],) if latest else ()) for path in paths]
    viable, unresolved = [], []
    prior = {(entry["size"], entry["sha256"]) for entry in anchors[0][1]} if anchors else set()
    for item in observations:
        if latest is None or (item["size"] >= latest["size"]
                              and item["prefixes"][latest["size"]] == latest["sha256"]):
            viable.append(item)
        elif (item["size"], item["sha256"]) not in prior:
            unresolved.append(item)
    if unresolved:
        # A shorter, previously uncaptured physical copy may still be a proven
        # prefix of the latest immutable archive. Authenticate the entire blob.
        if latest is None or any(item["size"] > latest["size"] for item in unresolved):
            raise ValueError("Capture selection branches require review")
        _, _, prefixes = _digests(archive._iter_verified_entry(latest),
                                  (item["size"] for item in unresolved))
        if any(prefixes[item["size"]] != item["sha256"] for item in unresolved):
            raise ValueError("Capture selection branches require review")
    if not viable:
        return None
    preferred = latest.get("observed_source", anchors[0][0]) if latest else None
    viable.sort(key=lambda item: (-item["size"], str(item["path"]) != preferred, str(item["path"])))
    chosen = viable[0]
    if any((item["size"], item["sha256"]) != (chosen["size"], chosen["sha256"])
           for item in viable[1:]):
        proved = _read_path(chosen["path"], (item["size"] for item in viable[1:]),
                            expected=chosen["fingerprint"])
        if proved["sha256"] != chosen["sha256"]:
            raise RuntimeError("Capture selection source changed")
        if any(proved["prefixes"][item["size"]] != item["sha256"] for item in viable[1:]):
            raise ValueError("Capture selection branches require review")
    return chosen["path"]
