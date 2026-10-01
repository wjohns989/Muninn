"""Conservative, bounded local project-file credential discovery.

This module never prints or indexes values. The caller supplies an explicit
project root and an unlocked portable vault; a failed source is rolled back by
``CredentialStore.scan_source`` and reported as incomplete, not silently skipped.
"""

from __future__ import annotations

import codecs
import json
import os
import queue
import re
import stat
import threading
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, Iterable, Iterator

from muninn.history.credential_store import (
    AmbiguousCandidate,
    CredentialStore,
    _valid_project_label,
    _validated_source_hint,
    source_fingerprint,
)
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive

_ASSIGN = re.compile(
    r"(?m)(?:\A|\n)[ \t]{0,16}(?:export[ \t]+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]{2,63})[ \t]{0,16}=[ \t]{0,16}"
)
_INLINE_ASSIGN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z_][A-Za-z0-9_]{2,63})"
    r"[ \t]{0,16}(?:=|\\?\"[ \t]{0,16}:)[ \t]{0,16}\\?\"?"
)
_VALUE = re.compile(r'(?P<value>[A-Za-z0-9_./+=:@-]{8,512})(?=$|[ \t\r\n#"\'\\,;}\]])')
_PROJECT_ASSIGN = re.compile(
    r"(?<![A-Za-z0-9_])[\"']?(?P<name>[A-Za-z_][A-Za-z0-9_]{2,63})[\"']?"
    r"[ \t]{0,16}(?:=|:)[ \t]{0,16}[\"']?"
)
_SECRET_NAME = re.compile(
    r"(?i)(?:^|_)(?:API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|ACCESS_TOKEN|AUTH_TOKEN|"
    r"REFRESH_TOKEN|TOKEN|SECRET|PASSWORD|PASSWD)$"
)
_PLACEHOLDER = re.compile(r"(?i)(?:example|sample|placeholder|changeme|replace|your[_-]|dummy|not[_-]real)")
_EXCLUDED_ENV = {".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"}
_EXCLUDED_DIRS = {
    ".git", ".worktrees", ".muninn_runtime", ".venv", "venv", "node_modules",
    "__pycache__", ".tox", "dist", "build",
}
_PROJECT_TEXT_SUFFIXES = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".jsonc",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".ps1", ".sh", ".bash", ".zsh", ".md", ".txt", ".rst", ".sql",
    ".xml", ".http", ".ipynb",
}
_PROJECT_TEXT_NAMES = {"dockerfile", "makefile", "config", "settings"}
_OVERLAP = 1024
_CHUNK = 64 * 1024
_BINARY_CONTROLS = str.maketrans("", "", "".join(
    chr(codepoint) for codepoint in range(32) if codepoint not in (9, 10, 13)
))
_SGR_BYTES = re.compile(rb"\x1b\[[0-9;:]{0,64}m")
_SGR_PARTIAL = re.compile(rb"\x1b(?:\[[0-9;:]{0,64})?$")


@dataclass(repr=False)
class ExtractionStats:
    examined: int = 0
    ambiguous: int = 0
    accepted: int = 0
    ambiguous_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.ambiguous += 1
        self.ambiguous_reasons[reason] = self.ambiguous_reasons.get(reason, 0) + 1


class ArchiveVerificationError(RuntimeError):
    """An authenticated archive blob failed verification; no source details escape."""


class CredentialScanMetadataError(ValueError):
    """An archive entry cannot be represented as safe credential metadata."""


class CredentialScanPathError(ValueError):
    """A selected project source is not a safe in-root regular file."""


class CredentialScanSourceChangedError(RuntimeError):
    """A selected source changed before its scan could commit."""


class CredentialScanBinaryError(ValueError):
    """A selected project text file is binary-like and cannot be safely interpreted."""


def _value_rejection(value: str) -> str | None:
    if _PLACEHOLDER.search(value):
        return "placeholder_like"
    if len(set(value)) < 4:
        return "low_diversity"
    return None


def _bounded_candidate(window: str, start: int) -> str:
    """Keep only a short candidate token, never a surrounding transcript span."""
    fragment = window[start:start + 128]
    if any(0xD800 <= ord(char) <= 0xDFFF or
           (ord(char) < 32 and char not in "\t\r\n") for char in fragment):
        # A legacy byte/surrogate or binary control is not safe to serialize
        # into a UTF-8 vault row. Keep the unresolved event with no fragment.
        return ""
    if fragment.startswith("${") and "}" in fragment:
        end = fragment.index("}") + 1
        if end == len(fragment) or fragment[end].isspace() or fragment[end] in ',;"\'\\]':
            return fragment[:end]
    token = re.split(r"[\s,;\"'\\\]\[]", fragment, maxsplit=1)[0]
    # Never let a truncated prefix look like a complete candidate to a model.
    return "" if len(token) == len(fragment) == 128 else token


def _strip_sgr_chunks(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Discard only bounded ANSI color codes, including split codes.

    Other escapes and incomplete or overlong codes remain in the byte stream,
    where the binary and per-assignment guards can reject them. The fast path
    leaves ordinary text chunks unchanged.
    """
    pending = b""
    for raw in chunks:
        if not isinstance(raw, bytes):
            raise TypeError("Credential scan input is not bytes")
        if not pending and b"\x1b" not in raw:
            yield raw
            continue
        data = pending + raw
        match = _SGR_PARTIAL.search(data)
        if match is None:
            pending = b""
            body = data
        else:
            pending = data[match.start():]
            body = data[:match.start()]
        yield _SGR_BYTES.sub(b"", body)
    if pending:
        yield pending


def _iter_findings(chunks: Iterable[bytes], source_hint: str,
                   stats: ExtractionStats, assignment: re.Pattern[str], *,
                   encoding: str = "utf-8-sig", errors: str = "strict",
                   reject_binary: bool = False, include_ambiguous: bool = False,
                   include_context: bool = False,
                   ) -> Iterator[tuple[str, str, str] | AmbiguousCandidate]:
    """Scan decoded text with fixed overlap; yield only unambiguous assignments.

    The strict env/transcript path rolls back on invalid UTF-8. Project files
    may preserve undecodable bytes as surrogates, but never accept them inside
    an assignment. A late binary-like block rolls back the whole source.
    """
    decoder = codecs.getincrementaldecoder(encoding)(errors)
    tail = ""
    total = 0
    total_lines = 0
    last_start = -1

    def examine(window: str, base: int, safe_end: int
                ) -> Iterator[tuple[str, str, str] | AmbiguousCandidate]:
        nonlocal last_start
        def ambiguity(name: str, reason: str, candidate: str, start: int) -> AmbiguousCandidate:
            line_start = window.rfind("\n", 0, start) + 1
            line_end = window.find("\n", start)
            if line_end < 0:
                line_end = len(window)
            return AmbiguousCandidate(
                name, reason, candidate, source_hint,
                total_lines - tail.count("\n") + window.count("\n", 0, start) if include_context else None,
                window[max(line_start, start - 256):min(line_end, start + 1024)] if include_context else "",
            )
        for match in assignment.finditer(window):
            absolute = base + match.start()
            if absolute <= last_start or (base > 0 and match.start() == 0):
                continue
            # Defer near-edge assignments until enough following text arrives.
            if match.start() > safe_end:
                continue
            last_start = absolute
            name = match.group("name")
            if not _SECRET_NAME.search(name):
                continue
            stats.examined += 1
            value_match = _VALUE.match(window, match.end())
            if value_match is None:
                stats.reject("unparsed_value")
                if include_ambiguous:
                    yield ambiguity(name, "unparsed_value",
                                    _bounded_candidate(window, match.end()), match.start())
                continue
            rejection = _value_rejection(value_match.group("value"))
            if rejection is not None:
                stats.reject(rejection)
                if include_ambiguous:
                    yield ambiguity(name, rejection, value_match.group("value"), match.start())
                continue
            if reject_binary:
                line_start = window.rfind("\n", 0, match.start()) + 1
                context = window[line_start:value_match.end()]
                if any(0xD800 <= ord(char) <= 0xDFFF or
                       (ord(char) < 32 and char not in "\t\r\n") for char in context):
                    stats.reject("unsafe_context")
                    if include_ambiguous:
                        yield ambiguity(name, "unsafe_context", value_match.group("value"), match.start())
                    continue
            stats.accepted += 1
            yield name, value_match.group("value"), source_hint

    for raw in chunks:
        if not isinstance(raw, bytes):
            raise TypeError("Credential scan input is not bytes")
        decoded = decoder.decode(raw)
        if reject_binary and decoded:
            controls = len(decoded) - len(decoded.translate(_BINARY_CONTROLS))
            if controls / len(decoded) > 0.02:
                raise CredentialScanBinaryError("Unsupported binary-like project source")
        window = tail + decoded
        base = total - len(tail)
        safe_end = max(0, len(window) - _OVERLAP)
        yield from examine(window, base, safe_end)
        total += len(decoded)
        total_lines += decoded.count("\n")
        tail = window[-_OVERLAP:]
    final = decoder.decode(b"", final=True)
    if reject_binary and final:
        controls = len(final) - len(final.translate(_BINARY_CONTROLS))
        if controls / len(final) > 0.02:
            raise CredentialScanBinaryError("Unsupported binary-like project source")
    window = tail + final
    yield from examine(window, total - len(tail), len(window))


def iter_env_findings(chunks: Iterable[bytes], source_hint: str,
                      stats: ExtractionStats, *, include_ambiguous: bool = False
                      ) -> Iterator[tuple[str, str, str] | AmbiguousCandidate]:
    return _iter_findings(chunks, source_hint, stats, _ASSIGN,
                          include_ambiguous=include_ambiguous)


def iter_project_findings(chunks: Iterable[bytes], source_hint: str,
                          stats: ExtractionStats, *, include_ambiguous: bool = False
                          ) -> Iterator[tuple[str, str, str] | AmbiguousCandidate]:
    """Find safe ASCII assignments in UTF-8, legacy 8-bit, or BOM-marked UTF-16 text."""
    source = iter(chunks)
    prefix_parts = []
    prefix_size = 0
    while prefix_size < 4096:
        try:
            part = next(source)
        except StopIteration:
            break
        if not isinstance(part, bytes):
            raise TypeError("Credential scan input is not bytes")
        prefix_parts.append(part)
        prefix_size += len(part)
    prefix = b"".join(prefix_parts)
    encoding = "utf-8-sig"
    errors = "surrogateescape"
    if prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding, errors = "utf-16", "strict"
    elif prefix.startswith(codecs.BOM_UTF8):
        # Some exported Markdown has a UTF-8 BOM followed by a UTF-16LE body.
        sample = prefix[3:4099]
        high = sample[1::2]
        low = sample[::2]
        if (len(high) >= 8 and high.count(0) / len(high) >= 0.6
                and low.count(0) / len(low) <= 0.2):
            try:
                decoded = sample[:len(sample) - len(sample) % 2].decode("utf-16le")
            except UnicodeError:
                decoded = ""
            if decoded and sum(char.isprintable() or char in "\t\r\n" for char in decoded) / len(decoded) >= 0.9:
                prefix = prefix[3:]
                encoding, errors = "utf-16le", "strict"
    text_chunks = chain((prefix,), source)
    if Path(source_hint).suffix.lower() == ".txt":
        text_chunks = _strip_sgr_chunks(text_chunks)
    yield from _iter_findings(text_chunks, source_hint, stats,
                              _PROJECT_ASSIGN, encoding=encoding, errors=errors,
                              reject_binary=True, include_ambiguous=include_ambiguous)


def iter_transcript_findings(chunks: Iterable[bytes],
                             stats: ExtractionStats, *, include_ambiguous: bool = False,
                             include_context: bool = False,
                             ) -> Iterator[tuple[str, str, str] | AmbiguousCandidate]:
    """Historical observations only; never claim that a captured key is current."""
    return _iter_findings(chunks, "", stats, _INLINE_ASSIGN,
                          include_ambiguous=include_ambiguous, include_context=include_context)


def _is_link_or_junction(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        # The caller's walk/source check reports an inaccessible entry.
        return False
    # Some Windows directory reparse tags (e.g. inaccessible WSL links) are
    # neither Path.is_symlink() nor Path.is_junction(). Never descend into one.
    if getattr(info, "st_file_attributes", 0) & 0x400:
        return True
    return stat.S_ISLNK(info.st_mode) or (hasattr(path, "is_junction") and path.is_junction())


def _env_files(root: Path, *, all_project_text: bool,
               onerror: Callable[[OSError], None]) -> Iterator[Path]:
    for directory, children, files in os.walk(root, followlinks=False, onerror=onerror):
        folder = Path(directory)
        children[:] = sorted(
            name for name in children
            if name.lower() not in _EXCLUDED_DIRS and not _is_link_or_junction(folder / name)
        )
        for name in sorted(files):
            lowered = name.lower()
            is_env = (lowered == ".env" or lowered.startswith(".env.")) and lowered not in _EXCLUDED_ENV
            is_text = (Path(lowered).suffix in _PROJECT_TEXT_SUFFIXES
                       or lowered in _PROJECT_TEXT_NAMES)
            if is_env or (all_project_text and is_text):
                yield folder / name


def scan_project_env(root: Path, store: CredentialStore, *, passphrase: str) -> dict[str, int | bool]:
    """Scan only explicitly selected project .env sources, with no values in the report."""
    return _scan_project(root, store, passphrase=passphrase, all_project_text=False)


def scan_project_files(root: Path, store: CredentialStore, *, passphrase: str,
                       progress: Callable[[dict[str, int | bool]], None] | None = None
                       ) -> dict[str, int | bool]:
    """Stream supported project text formats, never following linked directories."""
    return _scan_project(root, store, passphrase=passphrase, all_project_text=True,
                         progress=progress)


def _scan_project(root: Path, store: CredentialStore, *, passphrase: str,
                  all_project_text: bool,
                  progress: Callable[[dict[str, int | bool]], None] | None = None
                  ) -> dict[str, int | bool]:
    report: dict[str, int | bool] = {
        "files": 0, "succeeded": 0, "errors": 0, "walk_errors": 0, "ambiguous": 0,
        "ambiguous_reasons": {},
        "candidates": 0, "queued": 0, "inserted": 0, "updated": 0, "stale": 0,
        "complete": False, "ambiguity_free": False,
        "error_categories": {name: 0 for name in (
            "root", "walk", "path", "metadata", "utf8", "unsupported_binary",
            "io", "source_changed", "other"
        )},
    }
    try:
        root = Path(root).absolute()
        if _is_link_or_junction(root) or not root.is_dir():
            raise ValueError("Credential scan root must be a real directory")
        root = root.resolve(strict=True)
    except (OSError, ValueError, RuntimeError):
        # A missing/inaccessible selected root is a coverage gap, not a reason
        # to abort other selected roots or the independent archive phase.
        report["walk_errors"] = report["errors"] = 1
        report["error_categories"]["root"] = 1
        if progress is not None:
            progress({key: report[key] for key in (
                "files", "succeeded", "errors", "walk_errors", "ambiguous", "queued",
                "error_categories"
            )})
        return report
    project_names: dict[Path, str] = {root: root.name}

    def project_name(folder: Path) -> str:
        trail = []
        while folder not in project_names:
            trail.append(folder)
            marker = folder / ".git"
            if not _is_link_or_junction(marker) and (marker.is_dir() or marker.is_file()):
                project_names[folder] = folder.name
                break
            folder = folder.parent
        name = project_names[folder]
        for part in trail:
            project_names[part] = name
        return name

    def walk_error(_error: OSError) -> None:
        # os.walk skips an inaccessible directory after calling this callback.
        # Count the coverage gap without exposing its path or exception text.
        report["walk_errors"] += 1
        report["errors"] += 1
        report["error_categories"]["walk"] += 1

    for path in _env_files(root, all_project_text=all_project_text, onerror=walk_error):
        report["files"] += 1
        try:
            if _is_link_or_junction(path):
                raise CredentialScanPathError("Credential source link is unsupported")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise CredentialScanPathError("Credential source left approved root")
            try:
                hint = path.relative_to(root).as_posix()
                _validated_source_hint(hint)
            except (ValueError, VaultIntegrityError):
                raise CredentialScanPathError("Invalid credential source path") from None
            project = project_name(path.parent)
            if not _valid_project_label(project):
                raise CredentialScanMetadataError("Invalid credential project label")
            stats = ExtractionStats()

            def findings() -> Iterator[tuple[str, str, str]]:
                before = path.stat(follow_symlinks=False)
                with path.open("rb") as handle:
                    opened = os.fstat(handle.fileno())
                    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                        raise CredentialScanSourceChangedError("Credential source identity changed")
                    def chunks() -> Iterator[bytes]:
                        while block := handle.read(_CHUNK):
                            yield block
                    scanner = iter_env_findings if path.name.lower().startswith(".env") else iter_project_findings
                    yield from scanner(chunks(), hint, stats, include_ambiguous=True)
                    after = os.fstat(handle.fileno())
                current = path.stat(follow_symlinks=False)
                if (_is_link_or_junction(path) or path.resolve(strict=True) != resolved
                        or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                        != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)):
                    raise CredentialScanSourceChangedError("Credential source changed during scan")

            counts = store.scan_source(
                passphrase=passphrase, source_hash=source_fingerprint(str(resolved)),
                project=project, origin="project", findings=findings(),
            )
            report["succeeded"] += 1
            report["ambiguous"] += stats.ambiguous
            for reason, count in stats.ambiguous_reasons.items():
                report["ambiguous_reasons"][reason] = report["ambiguous_reasons"].get(reason, 0) + count
            report["candidates"] += stats.accepted
            report["queued"] += counts.get("ambiguities", 0)
            report["inserted"] += counts["created"]
            report["updated"] += counts["rotated"]
            report["stale"] += counts["staled"]
        except (OSError, UnicodeError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            # Do not put a path, value, decoder excerpt, or exception text in
            # agent-visible status or logs. Other files may still be scanned.
            report["errors"] += 1
            category = ("path" if isinstance(exc, CredentialScanPathError)
                        else "metadata" if isinstance(exc, CredentialScanMetadataError)
                        else "source_changed" if isinstance(exc, CredentialScanSourceChangedError)
                        else "unsupported_binary" if isinstance(exc, CredentialScanBinaryError)
                        else "utf8" if isinstance(exc, UnicodeError)
                        else "io" if isinstance(exc, OSError)
                        else "other")
            report["error_categories"][category] += 1
        if progress is not None and report["files"] % 100 == 0:
            progress({key: report[key] for key in (
                "files", "succeeded", "errors", "walk_errors", "ambiguous", "queued",
                "error_categories"
            )})
    # Coverage completion does not mean that every credential-looking string
    # could be interpreted. Keep the two claims separate for callers.
    report["complete"] = report["errors"] == 0
    report["ambiguity_free"] = report["complete"] and report["ambiguous"] == 0
    if progress is not None:
        progress({key: report[key] for key in (
            "files", "succeeded", "errors", "walk_errors", "ambiguous", "queued",
            "error_categories"
        )})
    return report


def _verified_archive_chunks(archive: SecureHistoryArchive,
                             entry: dict) -> Iterator[bytes]:
    """Bridge the archive's verifier callback into a bounded iterator.

    The producer validates the whole blob, including its final digest. A late
    error is raised into the consumer's vault transaction before it can commit.
    """
    chunks: queue.Queue[bytes | object] = queue.Queue(maxsize=2)
    finished = object()
    stopped = threading.Event()
    errors: list[BaseException] = []

    def submit(block: bytes) -> None:
        while not stopped.is_set():
            try:
                chunks.put(block, timeout=0.1)
                return
            except queue.Full:
                continue
        raise RuntimeError("Credential archive scan stopped")

    def produce() -> None:
        try:
            archive._verify_entry(entry, collect=False, on_chunk=submit)
        except VaultIntegrityError:
            errors.append(ArchiveVerificationError("Archive verification failed"))
        except BaseException as exc:
            errors.append(exc)
        finally:
            while not stopped.is_set():
                try:
                    chunks.put(finished, timeout=0.1)
                    break
                except queue.Full:
                    continue

    worker = threading.Thread(target=produce, name="muninn-credential-archive-read", daemon=True)
    worker.start()
    try:
        while True:
            item = chunks.get()
            if item is finished:
                if errors:
                    raise errors[0]
                return
            assert isinstance(item, bytes)
            yield item
    finally:
        stopped.set()
        worker.join(timeout=10)
        if worker.is_alive():
            raise RuntimeError("Credential archive reader did not stop")


def scan_archive(archive: SecureHistoryArchive, store: CredentialStore, *,
                 passphrase: str, offset: int = 0,
                 max_snapshots: int | None = None,
                 expected_generation: int | None = None,
                 progress: Callable[[dict[str, int | bool]], None] | None = None) -> dict[str, int | bool]:
    """Scan encrypted snapshots as historical credential observations only.

    Each snapshot is authenticated through the streaming verifier inside its
    atomic vault transaction. A late invalid blob or UTF-8 error rolls that
    source back; it can never produce a success receipt.
    """
    if offset < 0 or (max_snapshots is not None and max_snapshots < 1):
        raise ValueError("Invalid credential archive scan range")
    manifest = archive._load_manifest()
    if offset and expected_generation is None:
        raise ValueError("Archive scan resume requires a generation")
    if expected_generation is not None and expected_generation != manifest["generation"]:
        raise ValueError("Archive generation changed; restart the scan at offset zero")
    entries = [entry for _, versions in sorted(manifest["files"].items()) for entry in versions]
    total = len(entries)
    if offset > total:
        raise ValueError("Archive scan offset exceeds snapshot count")
    end = total if max_snapshots is None else min(total, offset + max_snapshots)
    report: dict[str, int | bool] = {
        "generation": manifest["generation"], "snapshots_total": total,
        "attempted": 0, "succeeded": 0, "skipped": 0, "errors": 0, "ambiguous": 0,
        "ambiguous_reasons": {}, "snapshots_not_evaluated": 0,
        "candidates": 0, "queued": 0, "inserted": 0, "updated": 0,
        "next_offset": end, "complete": False, "ambiguity_free": False,
        "changed_during_scan": False,
        "error_categories": {name: 0 for name in (
            "archive_integrity", "utf8", "io", "metadata", "vault", "other"
        )},
    }
    for entry in entries[offset:end]:
        report["attempted"] += 1
        try:
            stats = ExtractionStats()
            provider = entry["provider"]
            receipt_identity = json.dumps([
                archive.vault_id, entry["blob"], entry["sha256"], provider,
            ], separators=(",", ":"))
            if not _valid_project_label(provider) or len(receipt_identity) > 512:
                raise CredentialScanMetadataError("Invalid credential scan metadata")
            identity = source_fingerprint(
                f"{archive.vault_id}:{entry['blob']}:{entry['sha256']}"
            )
            counts = store.scan_source(
                passphrase=passphrase, source_hash=identity,
                project=provider, origin="transcript",
                findings=iter_transcript_findings(_verified_archive_chunks(archive, entry), stats,
                                                  include_ambiguous=True),
                receipt_identity=receipt_identity,
            )
            if counts.get("skipped", 0):
                report["skipped"] += 1
                report["snapshots_not_evaluated"] += 1
            else:
                report["succeeded"] += 1
            report["ambiguous"] += stats.ambiguous
            for reason, count in stats.ambiguous_reasons.items():
                report["ambiguous_reasons"][reason] = report["ambiguous_reasons"].get(reason, 0) + count
            report["candidates"] += stats.accepted
            report["queued"] += counts.get("ambiguities", 0)
            report["inserted"] += counts["created"]
            report["updated"] += counts["rotated"]
        except (OSError, ValueError, RuntimeError, UnicodeError, TypeError, KeyError) as exc:
            report["errors"] += 1
            category = ("archive_integrity" if isinstance(exc, ArchiveVerificationError)
                        else "utf8" if isinstance(exc, UnicodeError)
                        else "io" if isinstance(exc, OSError)
                        else "vault" if isinstance(exc, VaultIntegrityError)
                        else "metadata" if isinstance(exc, CredentialScanMetadataError)
                        else "other")
            report["error_categories"][category] += 1
        if progress is not None and (report["attempted"] % 100 == 0 or report["attempted"] == end - offset):
            progress({key: report[key] for key in (
                "generation", "snapshots_total", "attempted", "skipped", "errors",
                "queued", "ambiguous", "error_categories"
            )})
    end_generation = archive._load_manifest()["generation"]
    report["generation_at_end"] = end_generation
    report["changed_during_scan"] = end_generation != manifest["generation"]
    report["complete"] = report["errors"] == 0 and end == total and not report["changed_during_scan"]
    # A receipt-only scan did not re-evaluate its old rejected assignments.
    report["ambiguity_free"] = (report["complete"] and report["ambiguous"] == 0
                                and report["snapshots_not_evaluated"] == 0)
    return report
