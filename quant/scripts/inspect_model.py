#!/usr/bin/env python3
"""Inspect a Hugging Face Safetensors checkpoint without loading tensor data."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import inspect
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path.name}: missing 8-byte Safetensors header length")
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length <= 0 or header_length > 100_000_000:
            raise ValueError(f"{path.name}: implausible header length {header_length}")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"{path.name}: truncated Safetensors header")
    return json.loads(raw_header), header_length


def tensor_kind(name: str, shape: list[int]) -> str:
    if name.endswith("embed_tokens.weight"):
        return "Embedding"
    if name == "lm_head.weight":
        return "Linear (lm_head)"
    if ".norm" in name or "layernorm" in name:
        return "Normalization"
    if name.endswith("conv1d.weight"):
        return "Depthwise Conv1d"
    if "patch_embed" in name:
        return "Vision patch projection"
    linear_suffixes = (
        ".q_proj.weight",
        ".k_proj.weight",
        ".v_proj.weight",
        ".o_proj.weight",
        ".gate_proj.weight",
        ".up_proj.weight",
        ".down_proj.weight",
        ".out_proj.weight",
        ".in_proj_qkv.weight",
        ".in_proj_z.weight",
        ".in_proj_a.weight",
        ".in_proj_b.weight",
        ".linear_fc1.weight",
        ".linear_fc2.weight",
        ".qkv.weight",
        ".attn.proj.weight",
        ".fc.weight",
    )
    if len(shape) == 2 and name.endswith(linear_suffixes):
        return "Linear"
    return "Parameter"


def component(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name == "lm_head.weight":
        return "lm_head"
    if name.endswith("embed_tokens.weight"):
        return "embeddings"
    if ".linear_attn." in name:
        return "recurrent_gdn"
    if ".self_attn." in name:
        return "full_attention"
    if ".mlp." in name:
        return "text_mlp"
    if ".norm" in name or "layernorm" in name:
        return "normalization"
    return "other"


def inspect_runtime(model_path: Path, architecture: str | None) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from accelerate import init_empty_weights
        from transformers import AutoConfig
    except ImportError as exc:
        return {"available": False, "error": str(exc)}

    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    architecture = architecture or (config.architectures or [None])[0]
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        return {
            "available": True,
            "config_class": type(config).__name__,
            "architecture": architecture,
            "model_class_resolved": False,
        }

    with init_empty_weights(include_buffers=True):
        model = model_class(config)

    module_types: dict[str, str] = {}
    linear_names: list[str] = []
    norm_implementations: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_types[name] = module_type
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
        if "Norm" in module_type and module_type not in norm_implementations:
            try:
                source = inspect.getsource(type(module))
            except (OSError, TypeError):
                source = ""
            norm_implementations[module_type] = {
                "zero_centered_weight": "1.0 + self.weight" in source or "1 + self.weight" in source,
                "parameter_dtype": str(next(module.parameters()).dtype) if list(module.parameters()) else None,
            }

    return {
        "available": True,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "config_class": type(config).__name__,
        "architecture": architecture,
        "model_class": model_class.__name__,
        "model_class_resolved": True,
        "module_count": len(module_types),
        "module_type_counts": dict(sorted(collections.Counter(module_types.values()).items())),
        "linear_module_count": len(linear_names),
        "linear_modules": linear_names,
        "normalization": norm_implementations,
    }


def inspect_checkpoint(model_path: Path, instantiate_meta: bool) -> tuple[dict[str, Any], bool]:
    model_path = model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    single_shard_path = model_path / "model.safetensors"
    errors: list[str] = []
    if not config_path.is_file():
        errors.append("config.json is missing")
    if not index_path.is_file() and not single_shard_path.is_file():
        errors.append("neither model.safetensors.index.json nor model.safetensors is present")
    if errors:
        return {"model_path": str(model_path), "errors": errors}, False

    config = read_json(config_path)
    index_present = index_path.is_file()
    if index_present:
        index = read_json(index_path)
        weight_map: dict[str, str] = index.get("weight_map", {})
    else:
        single_header, _ = safetensors_header(single_shard_path)
        weight_map = {name: single_shard_path.name for name in single_header if name != "__metadata__"}
        index = {"metadata": {}, "weight_map": weight_map}
    shards = sorted(set(weight_map.values()))
    top_level_shards = sorted(path.name for path in model_path.glob("*.safetensors"))
    missing_shards = [name for name in shards if not (model_path / name).is_file()]
    extra_shards = sorted(set(top_level_shards) - set(shards))
    if missing_shards:
        errors.append(f"missing {len(missing_shards)} referenced shard(s)")

    actual_map: dict[str, str] = {}
    duplicate_tensors: list[str] = []
    dtype_counts: collections.Counter[str] = collections.Counter()
    dtype_bytes: collections.Counter[str] = collections.Counter()
    kind_counts: collections.Counter[str] = collections.Counter()
    component_counts: collections.Counter[str] = collections.Counter()
    component_bytes: collections.Counter[str] = collections.Counter()
    linear_modules: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []

    for shard_name in shards:
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            continue
        try:
            header, header_length = safetensors_header(shard_path)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        tensor_entries = {name: meta for name, meta in header.items() if name != "__metadata__"}
        max_end = 0
        for name, meta in tensor_entries.items():
            if name in actual_map:
                duplicate_tensors.append(name)
            actual_map[name] = shard_name
            dtype = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            max_end = max(max_end, end)
            tensor_bytes = end - start
            expected_bytes = math.prod(shape) * DTYPE_BYTES.get(dtype, 0)
            if expected_bytes and tensor_bytes != expected_bytes:
                errors.append(f"{name}: offsets describe {tensor_bytes} bytes; shape/dtype describe {expected_bytes}")
            dtype_counts[dtype] += 1
            dtype_bytes[dtype] += tensor_bytes
            kind = tensor_kind(name, shape)
            comp = component(name)
            kind_counts[kind] += 1
            component_counts[comp] += 1
            component_bytes[comp] += tensor_bytes
            if kind.startswith("Linear"):
                linear_modules.append(
                    {"name": name.removesuffix(".weight"), "type": kind, "shape": shape, "component": comp}
                )
        data_region_bytes = shard_path.stat().st_size - 8 - header_length
        if max_end != data_region_bytes:
            errors.append(f"{shard_name}: tensor data ends at {max_end}, file data region is {data_region_bytes}")
        shard_reports.append(
            {
                "name": shard_name,
                "size_bytes": shard_path.stat().st_size,
                "header_bytes": header_length,
                "tensor_count": len(tensor_entries),
            }
        )

    missing_index_tensors = sorted(set(weight_map) - set(actual_map))
    unindexed_tensors = sorted(set(actual_map) - set(weight_map)) if index_present else []
    mapping_mismatches = sorted(name for name in set(weight_map) & set(actual_map) if weight_map[name] != actual_map[name])
    if missing_index_tensors:
        errors.append(f"{len(missing_index_tensors)} indexed tensor(s) absent from shard headers")
    if unindexed_tensors:
        errors.append(f"{len(unindexed_tensors)} shard tensor(s) absent from index")
    if mapping_mismatches:
        errors.append(f"{len(mapping_mismatches)} tensor-to-shard mapping mismatch(es)")
    if duplicate_tensors:
        errors.append(f"{len(duplicate_tensors)} duplicate tensor name(s) across shards")

    text = config.get("text_config", config)
    vision = config.get("vision_config")
    layer_types = text.get("layer_types", [])
    report: dict[str, Any] = {
        "inspected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_path": str(model_path),
        "architecture": (config.get("architectures") or [None])[0],
        "model_type": config.get("model_type"),
        "configured_dtype": text.get("dtype", config.get("dtype")),
        "transformers_version_recorded": config.get("transformers_version"),
        "checkpoint": {
            "index_present": index_present,
            "declared_weight_bytes": int(index.get("metadata", {}).get("total_size", 0)),
            "shard_bytes": sum(item["size_bytes"] for item in shard_reports),
            "directory_bytes": sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file()),
            "shard_count": len(shards),
            "tensor_count": len(weight_map),
            "shards": shard_reports,
            "missing_shards": missing_shards,
            "extra_shards": extra_shards,
            "missing_index_tensors": missing_index_tensors,
            "unindexed_tensors": unindexed_tensors,
            "mapping_mismatches": mapping_mismatches,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "dtype_bytes": dict(sorted(dtype_bytes.items())),
        },
        "architecture_summary": {
            "text_layers": text.get("num_hidden_layers"),
            "layer_type_counts": dict(sorted(collections.Counter(layer_types).items())),
            "hidden_size": text.get("hidden_size"),
            "intermediate_size": text.get("intermediate_size"),
            "vocab_size": text.get("vocab_size"),
            "max_position_embeddings": text.get("max_position_embeddings"),
            "vision_present": vision is not None or component_counts["vision"] > 0,
            "vision_layers": vision.get("depth") if vision else None,
            "mtp_layers": text.get("mtp_num_hidden_layers", 0),
            "gdn_recurrent_present": component_counts["recurrent_gdn"] > 0,
            "rms_norm_eps": text.get("rms_norm_eps"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
        },
        "tensor_components": {
            name: {"tensor_count": component_counts[name], "bytes": component_bytes[name]}
            for name in sorted(component_counts)
        },
        "parameter_kind_counts": dict(sorted(kind_counts.items())),
        "metadata_inferred_linear_modules": sorted(linear_modules, key=lambda item: item["name"]),
        "notable_tensors": {
            "embeddings": [name for name in weight_map if name.endswith("embed_tokens.weight")],
            "lm_head": [name for name in weight_map if name == "lm_head.weight"],
            "mtp": sorted(name for name in weight_map if name.startswith("mtp.")),
        },
        "errors": errors,
        "complete": not errors,
    }
    if instantiate_meta:
        report["runtime_meta_model"] = inspect_runtime(model_path, report["architecture"])
    return report, not errors


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def print_summary(report: dict[str, Any]) -> None:
    checkpoint = report.get("checkpoint", {})
    arch = report.get("architecture_summary", {})
    print(f"model_path: {report['model_path']}")
    print(f"architecture: {report.get('architecture')} ({report.get('model_type')})")
    print(f"configured_dtype: {report.get('configured_dtype')}")
    print(
        "checkpoint: "
        f"{checkpoint.get('shard_count', 0)} shards, {checkpoint.get('tensor_count', 0)} tensors, "
        f"{human_bytes(checkpoint.get('shard_bytes', 0))}"
    )
    print(f"complete: {report.get('complete')}")
    print(
        "text: "
        f"{arch.get('text_layers')} layers {arch.get('layer_type_counts')}; "
        f"hidden={arch.get('hidden_size')} intermediate={arch.get('intermediate_size')}"
    )
    print(
        "special_components: "
        f"vision={arch.get('vision_present')}({arch.get('vision_layers')} layers) "
        f"mtp={arch.get('mtp_layers')} gdn/recurrent={arch.get('gdn_recurrent_present')}"
    )
    print(f"parameter_kinds: {report.get('parameter_kind_counts')}")
    print(f"metadata_inferred_linear_modules: {len(report.get('metadata_inferred_linear_modules', []))}")
    runtime = report.get("runtime_meta_model")
    if runtime:
        print(
            "runtime_meta_model: "
            f"resolved={runtime.get('model_class_resolved')} class={runtime.get('model_class')} "
            f"linear_modules={runtime.get('linear_module_count')}"
        )
        print(f"normalization: {runtime.get('normalization')}")
    for error in report.get("errors", []):
        print(f"ERROR: {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", default=os.environ.get("SOURCE_MODEL", "/models/source"))
    parser.add_argument("--json-out", type=Path, help="Write the complete inspection report as JSON")
    parser.add_argument(
        "--instantiate-meta",
        action="store_true",
        help="Instantiate the Transformers model on the meta device to report exact module classes",
    )
    args = parser.parse_args()
    report, ok = inspect_checkpoint(Path(args.model_path), args.instantiate_meta)
    print_summary(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"json_report: {args.json_out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
