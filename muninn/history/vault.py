"""History vault: Muninn's own copy of every local AI conversation, kept after the apps delete theirs.

Claude Code deletes transcripts older than ``cleanupPeriodDays`` (30 by
default), Gemini CLI can expire sessions, and deleting a Codex thread removes
its rollout. The vault mirrors each file into Muninn's data directory
(gzip-compressed, owner-only permissions) and never deletes anything:

- a file that grew is re-copied; a file that shrank or was rewritten keeps
  the previous copy as a numbered version;
- a file that disappeared stays in the vault and is marked ``missing_since``;
- the manifest lives next to the copies (``manifest.db``), so the vault
  folder is self-contained and can be backed up as a unit.

Memories are built from the vault copies, so they can be rebuilt at any time.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from muninn.history.locations import (
    NEVER_READ,
    HistorySource,
    app_data_dirs,
    export_candidates,
    history_homes,
    history_sources,
)

MANIFEST = """
CREATE TABLE IF NOT EXISTS vault_files (
    source_path   TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    vault_rel     TEXT NOT NULL,
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    sha256        TEXT NOT NULL,
    versions      INTEGER NOT NULL DEFAULT 0,
    first_seen    REAL NOT NULL,
    last_synced   REAL NOT NULL,
    missing_since REAL
);
"""

_STORED_COMPRESSED = (".zst", ".zip", ".gz")


@dataclass
class VaultFile:
    source_path: str
    provider: str
    kind: str
    path: Path
    size: int
    mtime: float
    missing_since: Optional[float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _zstd_reader(raw: bytes) -> bytes:
    try:  # Python 3.14+
        from compression import zstd  # type: ignore

        return zstd.decompress(raw)
    except ImportError:
        pass
    try:
        import zstandard  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading .zst Codex rollouts needs the 'zstandard' package (muninn-mcp[ingestion])") from exc
    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()


def version_path(path: Path, number: int) -> Path:
    """Earlier copy number ``number``: the version goes before the extension so it stays readable."""
    name = path.name
    for suffix in (".gz", ".zst", ".zip"):
        if name.endswith(suffix):
            return path.with_name(f"{name[:-len(suffix)]}.v{number}{suffix}")
    return path.with_name(f"{name}.v{number}")


def read_bytes(path: Path) -> bytes:
    """Contents of a vault or source file, decompressed."""
    raw = path.read_bytes()
    name = path.name
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
        name = name[:-3]
    if name.endswith(".zst"):
        raw = _zstd_reader(raw)
    return raw


def read_text(path: Path) -> str:
    return read_bytes(path).decode("utf-8", errors="replace")


class HistoryVault:
    def __init__(self, root: Path, home: Optional[Path] = None):
        self.root = Path(root)
        self.home = home
        _private_dir(self.root)
        self._db = sqlite3.connect(str(self.root / "manifest.db"), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(MANIFEST)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # --- collecting sources --------------------------------------------------

    def _candidates(self, extra_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for number, home in enumerate(history_homes(self.home)):
            # Extra homes get their own folder so identical relative paths never collide.
            prefix = Path() if number == 0 else Path("homes") / hashlib.sha1(str(home).encode()).hexdigest()[:8]
            found: List[Dict[str, Any]] = []
            for source in history_sources(home):
                if source.exists:
                    found.extend(self._source_files(source))
            for directory in app_data_dirs(home):
                meta_root = directory / "Claude" / "claude-code-sessions"
                if meta_root.is_dir():
                    for path in meta_root.glob("**/*.json"):
                        found.append({"path": path, "provider": "claude_desktop", "kind": "desktop_session",
                                      "rel": Path("claude_desktop") / path.relative_to(meta_root)})
            for path in export_candidates(home):
                tag = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
                found.append({"path": path, "provider": "export", "kind": "export",
                              "rel": Path("exports") / f"{tag}-{path.name}"})
            for item in found:
                item["rel"] = prefix / item["rel"]
            items.extend(found)
        for raw in extra_paths or []:
            path = Path(raw).expanduser()
            if path.is_file():
                tag = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
                items.append({"path": path, "provider": "export", "kind": "export",
                              "rel": Path("exports") / f"{tag}-{path.name}"})
        return items

    @staticmethod
    def _source_files(source: HistorySource) -> Iterator[Dict[str, Any]]:
        seen = set()
        for pattern in source.patterns:
            for path in source.home.glob(pattern):
                if path.name in NEVER_READ or not path.is_file() or path in seen:
                    continue
                seen.add(path)
                yield {"path": path, "provider": source.provider, "kind": "transcript",
                       "rel": Path(source.provider) / path.relative_to(source.home)}
        for name in source.prompt_histories:
            path = source.home / name
            if path.is_file():
                yield {"path": path, "provider": source.provider, "kind": "prompt_history",
                       "rel": Path(source.provider) / name}
        for db in source.extras.get("state_db", [])[:1]:
            yield {"path": db, "provider": source.provider, "kind": "state_db",
                   "rel": Path(source.provider) / db.name}

    # --- syncing ---------------------------------------------------------------

    def sync(self, extra_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Copy new and changed history into the vault; never remove anything."""
        now = time.time()
        report: Dict[str, Any] = {"new": 0, "updated": 0, "unchanged": 0, "versions_kept": 0,
                                  "newly_missing": 0, "errors": [], "by_provider": {}}
        present = set()
        for item in self._candidates(extra_paths):
            path: Path = item["path"]
            key = str(path.resolve())
            present.add(key)
            try:
                outcome = self._sync_file(key, path, item, now)
            except OSError as exc:
                report["errors"].append(f"{item['provider']}: {path.name}: {exc}")
                continue
            if outcome == "updated_with_version":
                report["versions_kept"] += 1
                outcome = "updated"
            report[outcome] += 1
            counts = report["by_provider"].setdefault(item["provider"], {"files": 0, "changed": 0})
            counts["files"] += 1
            counts["changed"] += outcome != "unchanged"
        for row in self._db.execute("SELECT source_path FROM vault_files WHERE missing_since IS NULL").fetchall():
            if row["source_path"] not in present and not Path(row["source_path"]).exists():
                self._db.execute("UPDATE vault_files SET missing_since = ? WHERE source_path = ?",
                                 (now, row["source_path"]))
                report["newly_missing"] += 1
        self._db.commit()
        report["totals"] = self.status()["totals"]
        return report

    def _sync_file(self, key: str, path: Path, item: Dict[str, Any], now: float) -> str:
        stat = path.stat()
        row = self._db.execute("SELECT * FROM vault_files WHERE source_path = ?", (key,)).fetchone()
        if row and row["size"] == stat.st_size and row["mtime"] == stat.st_mtime:
            if row["missing_since"] is not None:
                self._db.execute("UPDATE vault_files SET missing_since = NULL WHERE source_path = ?", (key,))
            return "unchanged"
        rel: Path = item["rel"]
        if not rel.name.endswith(_STORED_COMPRESSED) and item["kind"] != "state_db":
            rel = rel.with_name(rel.name + ".gz")
        target = self.root / rel
        _private_dir(target.parent)
        versions = row["versions"] if row else 0
        outcome = "new" if row is None else "updated"
        if row is not None and stat.st_size < row["size"] and target.exists():
            # Rewritten smaller (edited, compacted, truncated): keep what we had.
            versions += 1
            shutil.copy2(target, version_path(target, versions))
            outcome = "updated_with_version"
        self._copy(path, target, item["kind"])
        digest = _sha256(path)
        self._db.execute(
            "INSERT INTO vault_files (source_path, provider, kind, vault_rel, size, mtime, sha256, versions, "
            "first_seen, last_synced, missing_since) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL) "
            "ON CONFLICT(source_path) DO UPDATE SET vault_rel=excluded.vault_rel, size=excluded.size, "
            "mtime=excluded.mtime, sha256=excluded.sha256, versions=excluded.versions, "
            "last_synced=excluded.last_synced, missing_since=NULL",
            (key, item["provider"], item["kind"], str(rel), stat.st_size, stat.st_mtime, digest, versions, now, now),
        )
        return outcome

    @staticmethod
    def _copy(source: Path, target: Path, kind: str) -> None:
        tmp = target.with_name(target.name + ".tmp")
        if kind == "state_db":
            # A live SQLite database (WAL) must be copied through the backup API.
            src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
            dst = sqlite3.connect(str(tmp))
            try:
                src.backup(dst)
            finally:
                src.close()
                dst.close()
        elif target.name.endswith(".gz") and not source.name.endswith(".gz"):
            with source.open("rb") as src_handle, gzip.open(tmp, "wb", compresslevel=6) as dst_handle:
                shutil.copyfileobj(src_handle, dst_handle, 1 << 20)
        else:
            shutil.copyfile(source, tmp)
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, target)

    # --- reading -----------------------------------------------------------------

    def files(self, provider: Optional[str] = None, kind: Optional[str] = None) -> List[VaultFile]:
        query, params = "SELECT * FROM vault_files WHERE 1=1", []
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        return [
            VaultFile(row["source_path"], row["provider"], row["kind"], self.root / row["vault_rel"],
                      row["size"], row["mtime"], row["missing_since"])
            for row in self._db.execute(query + " ORDER BY mtime", params).fetchall()
        ]

    def status(self) -> Dict[str, Any]:
        rows = self._db.execute(
            "SELECT provider, kind, COUNT(*) AS files, SUM(size) AS bytes, "
            "SUM(missing_since IS NOT NULL) AS preserved_after_deletion, MAX(last_synced) AS last_synced "
            "FROM vault_files GROUP BY provider, kind ORDER BY provider, kind"
        ).fetchall()
        groups = [dict(row) for row in rows]
        totals = {
            "files": sum(g["files"] for g in groups),
            "source_bytes": sum(g["bytes"] or 0 for g in groups),
            "preserved_after_deletion": sum(g["preserved_after_deletion"] or 0 for g in groups),
        }
        return {"root": str(self.root), "groups": groups, "totals": totals}
