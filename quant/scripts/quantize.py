#!/usr/bin/env python3
"""Plan or execute the guarded Qwen3.8 W8A8 quantization workflow."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from inspect_model import inspect_checkpoint


IMPORTANT_PACKAGES = (
    "torch",
    "transformers",
    "llmcompressor",
    "compressed-tensors",
    "accelerate",
    "datasets",
    "safetensors",
    "huggingface-hub",
)


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in IMPORTANT_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def git_revision() -> str:
    revision = os.environ.get("GIT_COMMIT")
    if revision:
        return revision
    try:
        return subprocess.check_output(
            ["git", "-C", "/app", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pattern_matches(pattern: str, name: str) -> bool:
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], name) is not None
    return name == pattern or name.endswith("." + pattern)


def resolve_policy(model_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    architecture = (config.architectures or [None])[0]
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        raise RuntimeError(f"Transformers cannot resolve architecture {architecture!r}")
    with init_empty_weights(include_buffers=True):
        model = model_class(config)

    ignores = policy["ignore"]
    linear_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    ignored: dict[str, list[str]] = {pattern: [] for pattern in ignores}
    included: list[str] = []
    for name in linear_names:
        matching = [pattern for pattern in ignores if pattern_matches(pattern, name)]
        if matching:
            for pattern in matching:
                ignored[pattern].append(name)
        else:
            included.append(name)

    unresolved_ignores = [pattern for pattern, names in ignored.items() if not names and not pattern.startswith("re:^mtp")]
    components = {
        "text_mlp": [name for name in included if ".mlp." in name],
        "full_attention": [name for name in included if ".self_attn." in name],
        "other": [name for name in included if ".mlp." not in name and ".self_attn." not in name],
    }
    return {
        "architecture": architecture,
        "model_class": model_class.__name__,
        "linear_module_count": len(linear_names),
        "included_count": len(included),
        "included": included,
        "included_components": {name: len(names) for name, names in components.items()},
        "ignored": {pattern: names for pattern, names in ignored.items()},
        "ignored_counts": {pattern: len(names) for pattern, names in ignored.items()},
        "unresolved_ignores": unresolved_ignores,
    }


class PeakMonitor:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes = 0
        self.peak_gpu_mib: dict[str, int] = {}
        self.samples = 0

    def __enter__(self) -> "PeakMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _read_rss(self) -> int:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0

    def _read_gpus(self) -> dict[str, int]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {}
        values: dict[str, int] = {}
        for line in output.splitlines():
            index, used = (part.strip() for part in line.split(",", 1))
            values[index] = int(used)
        return values

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._read_rss())
            for index, used in self._read_gpus().items():
                self.peak_gpu_mib[index] = max(self.peak_gpu_mib.get(index, 0), used)
            self.samples += 1
            self.stop_event.wait(self.interval)

    def report(self) -> dict[str, Any]:
        return {
            "peak_process_rss_bytes": self.peak_rss_bytes,
            "peak_gpu_memory_mib": self.peak_gpu_mib,
            "samples": self.samples,
            "sample_interval_seconds": self.interval,
        }


def ensure_gpus(minimum: int = 2) -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count()
    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(count)
    ]
    if not torch.cuda.is_available() or count < minimum:
        raise RuntimeError(f"Expected at least {minimum} CUDA GPUs, found {count}")
    if any(device["capability"] != [8, 6] for device in devices[:minimum]):
        raise RuntimeError(f"Expected RTX 3090-class SM86 devices, found {devices}")
    return {"torch_cuda": torch.version.cuda, "device_count": count, "devices": devices}


def make_modifier(settings: dict[str, Any], policy: dict[str, Any], target: str | None = None):
    from llmcompressor.modifiers.quantization import GPTQModifier

    return GPTQModifier(
        targets=target or policy["target"],
        scheme=settings["scheme"],
        ignore=policy["ignore"],
        block_size=settings["block_size"],
        dampening_frac=settings["dampening_frac"],
    )


def synthetic_config():
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    text_config = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=0,
        dtype="bfloat16",
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=128,
        num_position_embeddings=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    return Qwen3_5Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=500,
        video_token_id=501,
        vision_start_token_id=502,
        vision_end_token_id=503,
    )


def run_synthetic_smoke(
    root_config: dict[str, Any], profile: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForConditionalGeneration

    torch.manual_seed(root_config["calibration"]["seed"])
    config = synthetic_config()
    model = Qwen3_5ForConditionalGeneration(config).to(device="cuda:0", dtype=torch.bfloat16)
    length = profile["max_seq_length"]
    samples = []
    for sample_index in range(profile["num_samples"]):
        ids = [1 + ((sample_index * 17 + position) % 480) for position in range(length)]
        samples.append({"input_ids": ids, "attention_mask": [1] * length})
    dataset = Dataset.from_list(samples)
    tokenizer_backend = Tokenizer(
        WordLevel(
            vocab={"[PAD]": 0, "[UNK]": 1, **{f"token_{index}": index for index in range(2, 512)}},
            unk_token="[UNK]",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=length,
    )
    modifier = make_modifier(root_config["quantization"], root_config["policy"], profile["target"])

    started = time.monotonic()
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=[modifier],
        max_seq_length=length,
        num_calibration_samples=profile["num_samples"],
        pipeline=root_config["quantization"]["pipeline"],
        sequential_targets=root_config["quantization"]["sequential_targets"],
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        save_compressed=True,
        max_shard_size="512MB",
    )
    return {
        "elapsed_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
        "artifact_kind": "synthetic_qwen3_5_w8a8_smoke",
        "complete_production_model": False,
        "calibration_samples": profile["num_samples"],
        "max_seq_length": length,
        "target": profile["target"],
    }


def load_calibration_dataset(config: dict[str, Any], profile: dict[str, Any], tokenizer):
    from datasets import Dataset, load_dataset

    calibration = config["calibration"]
    count = profile["num_samples"]
    if profile.get("local_prompts"):
        rows = [json.loads(line) for line in Path(calibration["local_smoke_file"]).read_text().splitlines() if line]
        dataset = Dataset.from_list((rows * count)[:count])
    else:
        dataset = load_dataset(
            calibration["dataset"],
            split=calibration["split"],
            cache_dir=calibration["cache_dir"],
        ).shuffle(seed=calibration["seed"]).select(range(count))

    def preprocess(example: dict[str, Any]) -> dict[str, str]:
        messages = example.get("messages")
        text = tokenizer.apply_chat_template(messages, tokenize=False) if messages else example.get("text", "")
        return {"text": text}

    dataset = dataset.map(preprocess)

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        return tokenizer(
            example["text"],
            padding=False,
            truncation=True,
            max_length=profile["max_seq_length"],
            add_special_tokens=False,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def inject_mtp_tensors(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Restore top-level MTP tensors omitted by the Transformers model class."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_index_path = source_dir / "model.safetensors.index.json"
    output_index_path = output_dir / "model.safetensors.index.json"
    if not output_index_path.is_file():
        raise RuntimeError("Expected a sharded output index before MTP reinjection")
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    output_index = json.loads(output_index_path.read_text(encoding="utf-8"))
    mtp_map = {name: shard for name, shard in source_index["weight_map"].items() if name.startswith("mtp.")}
    if not mtp_map:
        raise RuntimeError("Source index contains no top-level MTP tensors")

    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in mtp_map.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        with safe_open(source_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)
    preserved_name = "model-mtp-preserved.safetensors"
    save_file(tensors, output_dir / preserved_name, metadata={"format": "pt", "preserved_from": str(source_dir)})
    for name in tensors:
        if name in output_index["weight_map"]:
            raise RuntimeError(f"Output unexpectedly already contains {name}")
        output_index["weight_map"][name] = preserved_name
    added_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    output_index.setdefault("metadata", {})["total_size"] = int(
        output_index.get("metadata", {}).get("total_size", 0)
    ) + added_bytes
    temporary = output_index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_index_path)
    return {"tensor_count": len(tensors), "bytes": added_bytes, "shard": preserved_name}


def run_full_model(
    config: dict[str, Any], profile_name: str, profile: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.utils import load_context
    from transformers import AutoConfig, AutoTokenizer

    source = Path(config["model"]["source"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.incomplete-{utc_stamp()}")
    if staging.exists():
        raise FileExistsError(staging)
    offload_dir = Path(config["memory"]["offload_dir"])
    offload_dir.mkdir(parents=True, exist_ok=True)

    hf_config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    architecture = hf_config.architectures[0]
    model_class = getattr(transformers, architecture)
    with load_context(model_class):
        model = model_class.from_pretrained(
            source,
            dtype="auto",
            local_files_only=True,
            trust_remote_code=False,
            device_map="auto_offload",
            max_memory={"cpu": f"{config['memory']['max_cpu_gib']}GiB"},
            offload_folder=offload_dir,
        )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    dataset = load_calibration_dataset(config, profile, tokenizer)
    modifier = make_modifier(config["quantization"], config["policy"])
    started = time.monotonic()
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=[modifier],
        max_seq_length=profile["max_seq_length"],
        num_calibration_samples=profile["num_samples"],
        pipeline=config["quantization"]["pipeline"],
        sequential_targets=config["quantization"]["sequential_targets"],
    )
    staging.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(staging, safe_serialization=True, save_compressed=True, max_shard_size="4GB")
    tokenizer.save_pretrained(staging)
    mtp = inject_mtp_tensors(source, staging)
    staging.rename(output_dir)
    return {
        "elapsed_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
        "artifact_kind": "qwen38_27b_w8a8_compressed_tensors",
        "complete_production_model": True,
        "profile": profile_name,
        "calibration_samples": profile["num_samples"],
        "max_seq_length": profile["max_seq_length"],
        "mtp_reinjection": mtp,
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "tiny_source", "small", "quality"), default="smoke")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-full", action="store_true", help="Required safety acknowledgement for non-synthetic work")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = read_yaml(args.config)
    source = Path(config["model"]["source"])
    source_report, complete = inspect_checkpoint(source, instantiate_meta=False)
    if not complete:
        raise RuntimeError(f"Source checkpoint failed completeness checks: {source_report['errors']}")
    expected = config["model"]["expected"]
    actual = source_report["checkpoint"]
    if actual["shard_count"] != expected["shards"] or actual["tensor_count"] != expected["tensors"]:
        raise RuntimeError(f"Source invariants changed: expected {expected}, got {actual}")

    gpu = ensure_gpus()
    policy_resolution = resolve_policy(source, config["policy"])
    if policy_resolution["unresolved_ignores"]:
        raise RuntimeError(f"Unresolved ignore patterns: {policy_resolution['unresolved_ignores']}")
    if policy_resolution["included_components"] != {"text_mlp": 192, "full_attention": 64, "other": 0}:
        raise RuntimeError(f"Unexpected target resolution: {policy_resolution['included_components']}")

    base_metadata: dict[str, Any] = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_revision(),
        "config": str(args.config),
        "profile": args.profile,
        "packages": package_versions(),
        "gpu": gpu,
        "source": {
            "path": str(source),
            "architecture": source_report["architecture"],
            "shards": actual["shard_count"],
            "tensors": actual["tensor_count"],
            "shard_bytes": actual["shard_bytes"],
        },
        "policy_resolution": policy_resolution,
    }
    console_metadata = dict(base_metadata)
    console_metadata["policy_resolution"] = {
        key: value
        for key, value in policy_resolution.items()
        if key not in {"included", "ignored"}
    }
    print(json.dumps(console_metadata, indent=2, sort_keys=True))
    if args.plan_only:
        return 0

    profile = config["calibration"]["profiles"][args.profile]
    synthetic = bool(profile.get("synthetic_model"))
    if not synthetic and not args.execute_full:
        raise SystemExit("Refusing real-checkpoint quantization without --execute-full")

    stamp = utc_stamp()
    output = args.output
    if output is None:
        output = (
            Path("/work/scratch") / f"quant-smoke-{stamp}"
            if synthetic
            else Path(config["model"]["output"])
        )
    metadata_path = Path("/work/results") / f"quant-{args.profile}-{stamp}.json"
    status = "failed"
    with PeakMonitor() as monitor:
        try:
            if synthetic:
                run = run_synthetic_smoke(config, profile, output)
            else:
                run = run_full_model(config, args.profile, profile, output)
            status = "passed"
        finally:
            base_metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            base_metadata["status"] = status
            base_metadata["peaks"] = monitor.report()
            if status == "passed":
                base_metadata["run"] = run
            write_metadata(metadata_path, base_metadata)
            print(f"run_metadata: {metadata_path}")
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
