#!/usr/bin/env python3
"""Populate a fresh pinned dataset cache without recording sample content."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

from common import atomic_json, load_config, stable_hash
from pinned_datasets import install


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pins = {entry["path"]: entry["revision"] for entry in config["datasets"].values()}
    original = install(pins, offline=False)

    records = {}
    for key, entry in config["datasets"].items():
        variants = {}
        for name in entry["configs"]:
            dataset = original(
                path=entry["path"],
                name=name,
                revision=entry["revision"],
                token=os.environ.get("HF_TOKEN") or None,
            )
            splits = {
                split: {"count": len(value), "fingerprint": value._fingerprint}
                for split, value in sorted(dataset.items())
            }
            variant = "__default__" if name is None else name
            observed_counts = {split: value["count"] for split, value in splits.items()}
            expected_counts = entry["expected_splits"][variant]
            if observed_counts != expected_counts:
                raise RuntimeError(
                    f"dataset split counts differ for {key}/{variant}: "
                    f"expected {expected_counts}, got {observed_counts}"
                )
            variants[variant] = splits
        records[key] = {
            "path": entry["path"],
            "revision": entry["revision"],
            "private": bool(entry.get("private", False)),
            "variants": variants,
        }
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cache_root": str(Path(os.environ["HF_HOME"])),
        "datasets": records,
    }
    payload["identity_sha256"] = stable_hash(payload["datasets"])
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
