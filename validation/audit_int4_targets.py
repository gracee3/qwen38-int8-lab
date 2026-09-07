#!/usr/bin/env python3
"""Audit the expanded-400 INT4 logical scope against a local checkpoint."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml

from validation.inspect_model import inspect_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-shards", action="store_true")
    args = parser.parse_args()

    recipe = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, complete = inspect_checkpoint(args.source, instantiate_meta=True)
    if not complete:
        raise SystemExit("source checkpoint is incomplete: " + "; ".join(report["errors"]))

    expected = {item["suffix"]: (item["count"], tuple(item["shape"])) for item in recipe["target_scope"]["families"]}
    targets = []
    failures = []
    source_revision = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    shard_hashes = {}
    if args.hash_shards:
        for shard in report['checkpoint']['shards']:
            path=args.source/shard['name']
            metadata=args.source/'.cache/huggingface/download'/(shard['name']+'.metadata')
            lines=metadata.read_text().splitlines()
            actual=sha256(path)
            if lines[0]!=source_revision or actual!=lines[1]:
                failures.append('source revision/hash mismatch: '+shard['name'])
            shard_hashes[shard['name']]=actual
            print('source_sha256_verified='+shard['name'],flush=True)
    for item in report["runtime_meta_model"]["linear_modules"]:
        name = item
        if re.fullmatch(r"model\.language_model\.layers\.\d+\..+", name) and any(name.endswith('.' + suffix) for suffix in expected):
            suffix = next(s for s in expected if name.endswith(s))
            targets.append({"name": name, "shape": None, "suffix": suffix})

    # Runtime module names are paired with authoritative tensor shapes from the
    # Safetensors headers, avoiding a second full model load.
    header_by_name = {item["name"].removesuffix(".weight"): item["shape"] for item in report["metadata_inferred_linear_modules"]}
    for target in targets:
        target["shape"] = header_by_name.get(target["name"])
    counts = Counter(item["suffix"] for item in targets)
    for suffix, (count, shape) in expected.items():
        found = [item for item in targets if item["suffix"] == suffix]
        if len(found) != count:
            failures.append(f"{suffix}: expected {count}, found {len(found)}")
        wrong = [item["name"] for item in found if tuple(item["shape"] or ()) != shape]
        if wrong:
            failures.append(f"{suffix}: wrong shape for {wrong[:3]}")
    if len(targets) != recipe["target_scope"]["expected_logical_targets"]:
        failures.append(f"total targets: expected 400, found {len(targets)}")

    quantized_weights = sum(math.prod(item["shape"]) for item in targets if item["shape"])
    if quantized_weights != 24326963200:
        failures.append(f"weight count mismatch: {quantized_weights}")
    scale_bytes = math.ceil(quantized_weights / recipe["quantization"]["weight_group_size"]) * 2
    report_out = {
        "status": "passed" if not failures else "failed",
        "source": {
            "path": str(args.source.resolve()),
            "revision": source_revision,
            "shard_hashes_verified": args.hash_shards and bool(shard_hashes),
            "shard_sha256": shard_hashes,
            "config_sha256": sha256(args.source / "config.json"),
            "index_sha256": sha256(args.source / "model.safetensors.index.json"),
            "shards": report["checkpoint"]["shards"],
        },
        "architecture": report["architecture"],
        "runtime_model_class": report["runtime_meta_model"]["model_class"],
        "target_count": len(targets),
        "target_families": {suffix: {"count": counts[suffix], "shape": list(shape)} for suffix, (count, shape) in expected.items()},
        "targets": sorted(targets, key=lambda item: item["name"]),
        "quantized_weight_count": quantized_weights,
        "estimated_packed_weight_bytes": (quantized_weights + 1) // 2,
        "estimated_scale_bytes": scale_bytes,
        "estimated_quantized_storage_gib": ((quantized_weights + 1) // 2 + scale_bytes) / 2**30,
        "estimated_complete_checkpoint_gib": (report["checkpoint"]["declared_weight_bytes"] - 2 * quantized_weights + (quantized_weights + 1) // 2 + scale_bytes) / 2**30,
        "preserved_mtp_tensors": report["notable_tensors"]["mtp"],
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report_out, handle, indent=2, sort_keys=True)
    print(json.dumps({key: report_out[key] for key in ("status", "target_count", "quantized_weight_count", "estimated_quantized_storage_gib", "failures")}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
