#!/usr/bin/env python3
"""Prepare the agentic W8A8 v2 multi-source calibration corpus.

Downloads, selects, renders, and tokenizes the token-weighted mixture
specified in qwen38-27b-v2.yaml. No GPU or model load required.

Usage (inside the quant container):
    python /app/quant/scripts/prepare_v2_calibration.py \
        --config /app/quant/config/qwen38-27b-v2.yaml \
        --profile preflight [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


@dataclass
class SourceSpec:
    name: str
    repo: str
    config: str | None = None
    revision: str | None = None
    split: str | None = None
    token_share: float = 0.0
    target_tokens: int = 0


@dataclass
class SourceResult:
    name: str
    repo: str
    revision: str | None
    split: str | None
    rows_selected: int = 0
    tokens_rendered: int = 0
    token_share_actual: float = 0.0
    length_min: int = 0
    length_max: int = 0
    length_mean: float = 0.0
    errors: list[str] = field(default_factory=list)


def load_source_dataset(spec: SourceSpec, cache_dir: str):
    """Load a single HuggingFace dataset with pinned config/revision/split."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"cache_dir": cache_dir}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.split:
        kwargs["split"] = spec.split

    print(f"  Loading {spec.repo} (config={spec.config}, revision={spec.revision}, split={spec.split})...")
    started = time.monotonic()
    dataset = load_dataset(spec.repo, spec.config, **kwargs)

    if spec.split is None:
        if isinstance(dataset, dict):
            available = list(dataset.keys())
            print(f"    Available splits: {available}")
            for preferred in ("train", "train_sft", "train_sft_short"):
                if preferred in available:
                    dataset = dataset[preferred]
                    spec.split = preferred
                    break
            else:
                dataset = dataset[available[0]]
                spec.split = available[0]

    elapsed = time.monotonic() - started
    print(f"    Loaded {len(dataset)} rows in {elapsed:.1f}s")
    return dataset


def extract_text(row: dict[str, Any]) -> str | None:
    """Extract renderable text from a dataset row.

    Handles: 'messages' (chat list or JSON string), 'text' (plain),
    'conversations' (ShareGPT). Returns None if the row cannot be rendered.
    """
    messages = row.get("messages")
    if messages and isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (json.JSONDecodeError, TypeError):
            return messages if messages.strip() else None
    if messages and isinstance(messages, list):
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts) if parts else None

    text = row.get("text")
    if text and isinstance(text, str) and text.strip():
        return text

    conversations = row.get("conversations")
    if conversations and isinstance(conversations, list):
        parts = []
        for conv in conversations:
            role = conv.get("from", conv.get("role", "unknown"))
            value = conv.get("value", conv.get("content", ""))
            if value:
                parts.append(f"{role}: {value}")
        return "\n".join(parts) if parts else None

    return None


def select_and_tokenize(
    dataset: Any,
    tokenizer: Any,
    target_tokens: int,
    length_range: tuple[int, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select rows to approximate target_tokens within length_range."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    selected: list[dict[str, Any]] = []
    total_tokens = 0
    min_len, max_len = length_range

    for idx in indices:
        if total_tokens >= target_tokens:
            break
        row = dataset[idx]
        text = extract_text(row)
        if text is None:
            continue
        tokens = tokenizer(
            text, padding=False, truncation=True,
            max_length=max_len, add_special_tokens=False,
        )
        token_count = len(tokens["input_ids"])
        if token_count < min_len:
            continue
        selected.append({"input_ids": tokens["input_ids"], "source_index": idx})
        total_tokens += token_count

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", default="preflight")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    calibration = config["calibration"]
    corpus_dir = Path(calibration["corpus_dir"])
    seed = calibration["seed"]
    random.seed(seed)

    print(f"=== Agentic W8A8 v2 Calibration Prep ({utc_stamp()}) ===")
    print(f"Profile: {args.profile}")
    print(f"Corpus dir: {corpus_dir}")
    print(f"Target tokens: {calibration['target_total_tokens']}")
    print(f"Sources: {len(calibration['sources'])}")

    if args.dry_run:
        print("\n[DRY RUN] Would load and tokenize the following sources:")
        for src in calibration["sources"]:
            print(f"  - {src['name']}: {src['repo']} "
                  f"({src['token_share']:.0%}, target {src['target_tokens']:,} tokens)")
        print("\n[DRY RUN] No data downloaded or written.")
        return

    from transformers import AutoTokenizer

    source_model = Path(config["model"]["source"])
    print(f"\nLoading tokenizer from {source_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        source_model, local_files_only=True, trust_remote_code=False
    )

    length_alloc = calibration["length_allocation"]
    short_range = tuple(length_alloc["short"]["range"])
    long_range = tuple(length_alloc["long"]["range"])
    short_share = length_alloc["short"]["share"]
    long_share = length_alloc["long"]["share"]

    results: list[SourceResult] = []
    all_tokenized: list[dict[str, Any]] = []
    total_tokens = 0
    for src_cfg in calibration["sources"]:
        spec = SourceSpec(
            name=src_cfg["name"],
            repo=src_cfg["repo"],
            config=src_cfg.get("config"),
            revision=src_cfg.get("revision"),
            split=src_cfg.get("split"),
            token_share=src_cfg["token_share"],
            target_tokens=src_cfg["target_tokens"],
        )
        print(f"\n--- {spec.name} ({spec.repo}) ---")
        result = SourceResult(
            name=spec.name, repo=spec.repo,
            revision=spec.revision, split=spec.split,
        )
        try:
            dataset = load_source_dataset(spec, calibration["cache_dir"])
            result.split = spec.split
            short_target = int(spec.target_tokens * short_share)
            long_target = int(spec.target_tokens * long_share)
            short_sel = select_and_tokenize(
                dataset, tokenizer, short_target, short_range, seed
            )
            long_sel = select_and_tokenize(
                dataset, tokenizer, long_target, long_range, seed + 1
            )
            source_tokens = short_sel + long_sel
            token_counts = [len(s["input_ids"]) for s in source_tokens]
            result.rows_selected = len(source_tokens)
            result.tokens_rendered = sum(token_counts)
            if token_counts:
                result.length_min = min(token_counts)
                result.length_max = max(token_counts)
                result.length_mean = statistics.fmean(token_counts)
            all_tokenized.extend(source_tokens)
            total_tokens += result.tokens_rendered
            print(f"  Selected {result.rows_selected} rows, "
                  f"{result.tokens_rendered:,} tokens "
                  f"(min={result.length_min}, max={result.length_max})")
        except Exception as exc:
            result.errors.append(str(exc))
            print(f"  ERROR: {exc}")
        results.append(result)
    for r in results:
        if total_tokens > 0:
            r.token_share_actual = r.tokens_rendered / total_tokens

    corpus_dir.mkdir(parents=True, exist_ok=True)
    if all_tokenized:
        from datasets import Dataset
        corpus = Dataset.from_list(all_tokenized)
        parquet_path = corpus_dir / "calibration.parquet"
        corpus.to_parquet(str(parquet_path))
        print(f"\nCorpus saved: {len(all_tokenized)} examples, "
              f"{total_tokens:,} tokens -> {parquet_path}")

    manifest = {
        "timestamp": utc_stamp(),
        "total_tokens": total_tokens,
        "target_tokens": calibration["target_total_tokens"],
        "acceptable_range": calibration["acceptable_range"],
        "seed": seed,
        "sources": [
            {
                "name": r.name,
                "repo": r.repo,
                "revision": r.revision,
                "split": r.split,
                "rows_selected": r.rows_selected,
                "tokens_rendered": r.tokens_rendered,
                "token_share_actual": r.token_share_actual,
                "length_min": r.length_min,
                "length_max": r.length_max,
                "length_mean": r.length_mean,
                "errors": r.errors,
            }
            for r in results
        ],
    }
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

    lo, hi = calibration["acceptable_range"]
    if total_tokens < lo:
        print(f"\nWARNING: total tokens {total_tokens:,} below acceptable "
              f"minimum {lo:,}. Consider increasing selection or adding sources.")
    elif total_tokens > hi:
        print(f"\nWARNING: total tokens {total_tokens:,} above acceptable "
              f"maximum {hi:,}. Consider reducing selection.")
    else:
        print(f"\nOK: total tokens {total_tokens:,} within acceptable range "
              f"[{lo:,}, {hi:,}]")



if __name__ == "__main__":
    main()
