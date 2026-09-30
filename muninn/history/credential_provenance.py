"""Local-only provenance lookup for already scanned transcript findings.

This index is built from the archive's authenticated manifest in memory. It
must not be serialized to an ordinary search index or returned through MCP:
source paths are private, and capture time is not a message event time.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from muninn.history.credential_store import source_fingerprint

_BLOB = re.compile(r"[a-f0-9]{32}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class ArchiveSourceEvidence:
    source_path: str
    version: int
    provider: str
    kind: str
    captured_at: float
    source_mtime_ns: int
    event_at: None = None  # The manifest cannot prove a message event time.


def archive_source_index(archive: Any) -> dict[str, ArchiveSourceEvidence | None]:
    """Join existing queue source hashes to exact immutable archive snapshots.

    `None` marks a nonunique derived key; callers must treat that as unverified
    rather than selecting whichever source was encountered first.
    """
    manifest = archive._load_manifest()
    files = manifest.get("files") if isinstance(manifest, dict) else None
    vault_id = getattr(archive, "vault_id", None)
    if not isinstance(files, dict) or not isinstance(vault_id, str) or not vault_id:
        raise ValueError("Invalid archive source provenance")
    found: dict[str, ArchiveSourceEvidence | None] = {}
    for source_path, versions in files.items():
        if not isinstance(source_path, str) or not source_path or not isinstance(versions, list):
            raise ValueError("Invalid archive source provenance")
        for version, entry in enumerate(versions):
            if not isinstance(entry, dict):
                raise ValueError("Invalid archive source provenance")
            blob, sha = entry.get("blob"), entry.get("sha256")
            captured, mtime = entry.get("captured_at"), entry.get("mtime_ns")
            provider, kind = entry.get("provider"), entry.get("kind")
            if (not isinstance(blob, str) or not _BLOB.fullmatch(blob)
                    or not isinstance(sha, str) or not _SHA256.fullmatch(sha)
                    or type(captured) not in (int, float) or not math.isfinite(captured)
                    or captured < 0 or type(mtime) is not int or mtime < 0
                    or not isinstance(provider, str) or not provider
                    or not isinstance(kind, str) or not kind):
                raise ValueError("Invalid archive source provenance")
            key = source_fingerprint(f"{vault_id}:{blob}:{sha}")
            evidence = ArchiveSourceEvidence(source_path, version, provider, kind,
                                             float(captured), mtime)
            if key in found:
                found[key] = None
            else:
                found[key] = evidence
    return found
