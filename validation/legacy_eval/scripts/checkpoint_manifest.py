#!/usr/bin/env python3
"""Create a metadata manifest suitable for before/after immutability checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from common import atomic_json, stable_hash


HASH_LIMIT = 16 * 1024 * 1024


def manifest(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"checkpoint is not a directory: {root}")
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        stat = path.stat()
        record = {
            "path": str(path.relative_to(root)),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= HASH_LIMIT:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        files.append(record)
    identity = stable_hash(files)
    return {
        "root": str(root),
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "identity_sha256": identity,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    atomic_json(args.output, manifest(Path(args.checkpoint)))


if __name__ == "__main__":
    main()
