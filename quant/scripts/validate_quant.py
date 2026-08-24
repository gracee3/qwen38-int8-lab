#!/usr/bin/env python3
"""Validate compressed-tensors metadata, shard integrity, and optional vLLM dispatch logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from inspect_model import inspect_checkpoint


DISPATCH_PATTERN = re.compile(
    r"Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"
)
PROCESSOR_CONFIG_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def find_quantization_config(config: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("quantization_config", "compression_config"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return None


def validate_w8a8(quant_config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if quant_config.get("quant_method") != "compressed-tensors":
        errors.append("quant_method is not compressed-tensors")
    if quant_config.get("format") != "int-quantized":
        errors.append("compressed-tensors format is not int-quantized")
    if quant_config.get("quantization_status") != "compressed":
        errors.append("quantization_status is not compressed")

    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        return errors + ["quantization metadata has no config_groups"]
    for name, group in groups.items():
        weights = group.get("weights") if isinstance(group, dict) else None
        activations = group.get("input_activations") if isinstance(group, dict) else None
        expected_weights = {
            "type": "int",
            "num_bits": 8,
            "strategy": "channel",
            "dynamic": False,
            "symmetric": True,
        }
        expected_activations = {
            "type": "int",
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
        }
        for field, expected in expected_weights.items():
            if not isinstance(weights, dict) or weights.get(field) != expected:
                errors.append(f"{name} weights.{field} is not {expected!r}")
        for field, expected in expected_activations.items():
            if not isinstance(activations, dict) or activations.get(field) != expected:
                errors.append(f"{name} input_activations.{field} is not {expected!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Permit an intentionally non-production smoke artifact")
    parser.add_argument("--vllm-log", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    report, complete = inspect_checkpoint(args.model_path, instantiate_meta=False)
    if not complete:
        errors.extend(report.get("errors", []))
    config_path = args.model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    quant_config = find_quantization_config(config)
    if quant_config is None:
        errors.append("config.json has neither quantization_config nor compression_config")
    else:
        errors.extend(validate_w8a8(quant_config))

    mtp_names = report.get("notable_tensors", {}).get("mtp", [])
    if not args.synthetic and len(mtp_names) != 15:
        errors.append(f"production output must preserve 15 MTP tensors; found {len(mtp_names)}")

    processor_configs: dict[str, dict[str, Any]] = {}
    if not args.synthetic:
        for name in PROCESSOR_CONFIG_FILES:
            path = args.model_path / name
            if not path.is_file():
                errors.append(f"required processor config is missing: {name}")
                continue
            try:
                value = json.loads(path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid processor config {name}: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"processor config {name} is not a JSON object")
            raw = path.read_bytes()
            processor_configs[name] = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    target_modules: list[str] = []
    if not args.synthetic:
        index_path = args.model_path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            target_modules = sorted(
                name.removesuffix(".weight_scale")
                for name in index.get("weight_map", {})
                if name.endswith(".weight_scale")
            )
        if len(target_modules) != 256:
            errors.append(f"expected 256 quantized target modules; found {len(target_modules)}")

    dispatch_verified = None
    if args.vllm_log:
        log_text = args.vllm_log.read_text(encoding="utf-8", errors="replace")
        dispatch_verified = bool(DISPATCH_PATTERN.search(log_text))
        if not dispatch_verified:
            errors.append("vLLM log does not prove CUTLASS W8A8 INT8 dispatch")

    result = {
        "model_path": str(args.model_path.resolve()),
        "synthetic": args.synthetic,
        "checkpoint_complete": complete,
        "quantization_config": quant_config,
        "mtp_tensor_count": len(mtp_names),
        "processor_configs": processor_configs,
        "quantized_target_count": len(target_modules),
        "cutlass_w8a8_dispatch_verified": dispatch_verified,
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
