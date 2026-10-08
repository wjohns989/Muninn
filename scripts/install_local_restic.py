"""Install a pinned, checksum-verified Windows backup utility; no PATH changes."""
from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

from muninn.history.private_acl import create_private_directory, create_private_file
from muninn.history.recovery_pool import durable_publish, private_file, unlinked

VERSION = "0.19.1"
ASSET = f"restic_{VERSION}_windows_amd64.zip"
ASSET_SHA = "da948ad707ed690426473aaba2046cd61f8f90f6f0e7dab6be0d5796531de67d"
SUMS_SHA = "fb520966ee01d2a3d4219c66762c66efa56300833b6f639f36082b7462f91cb8"
BASE = f"https://github.com/restic/restic/releases/download/v{VERSION}/"


def download(name, expected, maximum):
    request = urllib.request.Request(BASE + name, headers={"User-Agent": "Muninn-local-backup-installer"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Pinned upstream download checksum differs")
    return raw


def install(destination):
    if os.name != "nt":
        raise RuntimeError("This pinned binary is Windows-only")
    destination = unlinked(destination)
    if destination.exists():
        raise FileExistsError("Installation destination must be new")
    sums = download("SHA256SUMS", SUMS_SHA, 16384).decode("ascii")
    expected = dict(line.split(maxsplit=1)[::-1] for line in sums.splitlines() if line.strip())
    if expected.get(ASSET) != ASSET_SHA:
        raise ValueError("Upstream checksum manifest differs")
    raw = download(ASSET, ASSET_SHA, 32 * 1024 * 1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as package:
        entries = package.infolist()
        if (len(entries) != 1 or entries[0].is_dir()
                or not entries[0].filename.lower().endswith(".exe")
                or entries[0].file_size > 64 * 1024 * 1024):
            raise ValueError("Unexpected upstream binary package")
        executable = package.read(entries[0])
    missing = []
    current = destination
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        create_private_directory(directory)
    temporary = destination / ".restic.download"
    create_private_file(temporary)
    with temporary.open("wb") as handle:
        handle.write(executable)
        handle.flush()
        os.fsync(handle.fileno())
    target = destination / "restic.exe"
    durable_publish(temporary, target)
    private_file(target)
    receipt = {"version": VERSION, "asset_sha256": ASSET_SHA,
               "binary_sha256": hashlib.sha256(executable).hexdigest()}
    marker = destination / "verified-upstream.json"
    create_private_file(marker)
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.destination)))
