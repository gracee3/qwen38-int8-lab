#!/usr/bin/env python3
"""Fail-closed runtime log checks and content-free gate evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from common import atomic_json, load_config


CUTLASS_MARKER = (
    "Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"
)
TEXT_ONLY_MARKER = "running in text-only mode."
CHUNKED_PREFILL_RE = re.compile(
    r"Chunked prefill is enabled with max_num_batched_tokens=([0-9,]+)\."
)
KV_CAPACITY_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+) tokens")
TRUNCATION_RE = re.compile(
    r"exceeds model.s max length|left truncated|truncating to last|truncating context",
    re.IGNORECASE,
)
ALLOCATOR_OOM_WARNING_RE = re.compile(
    r"^.*(?:\bWARNING\b|\[W[0-9]+|CUDACachingAllocator|allocator warning).*\bOOM\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
ERROR_EXCEPTION_RE = re.compile(
    r"(?:\bERROR\b.*(?:\bexception\b|(?:Error|Exception):)|"
    r"^(?:[A-Za-z_][\w.]*)(?:Error|Exception):)",
    re.IGNORECASE | re.MULTILINE,
)


def failure_reason(log_text: str) -> str | None:
    if ALLOCATOR_OOM_WARNING_RE.search(log_text):
        return "allocator_oom_warning_detected"
    if ERROR_EXCEPTION_RE.search(log_text):
        return "runtime_exception_detected"
    if TRUNCATION_RE.search(log_text):
        return "runtime_truncation_detected"
    return None


def observed_kv_capacity(log_text: str) -> int:
    matches = [int(value.replace(",", "")) for value in KV_CAPACITY_RE.findall(log_text)]
    if not matches:
        raise ValueError("runtime did not report GPU KV cache capacity")
    if len(set(matches)) != 1:
        raise ValueError("runtime reported inconsistent GPU KV cache capacities")
    return matches[0]


def validate_runtime_gate(
    log_text: str, execution: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    reason = failure_reason(log_text)
    if reason:
        raise ValueError(reason)
    if CUTLASS_MARKER not in log_text:
        raise ValueError("native CUTLASS W8A8 dispatch was not observed")
    if TEXT_ONLY_MARKER not in log_text:
        raise ValueError("text-only runtime activation was not observed")
    chunk_matches = [int(value.replace(",", "")) for value in CHUNKED_PREFILL_RE.findall(log_text)]
    expected_runtime = config["models"]["w8a8"]["runtime"]
    expected_chunk = int(expected_runtime["max_num_batched_tokens"])
    if not chunk_matches or set(chunk_matches) != {expected_chunk}:
        raise ValueError("chunked prefill activation was not observed")
    capacity = observed_kv_capacity(log_text)
    minimum_capacity = int(config["runtime_gate"]["minimum_kv_capacity_tokens"])
    if capacity < minimum_capacity:
        raise ValueError(
            f"KV capacity {capacity} is below required minimum {minimum_capacity}"
        )
    if execution.get("status") != "passed":
        raise ValueError("runtime gate execution did not pass")
    if execution.get("runtime") != {
        **expected_runtime,
        "cpu_offload_gb": config["models"]["w8a8"]["cpu_offload_gb"],
    }:
        raise ValueError("runtime gate active configuration differs from the suite")
    expected_tokens = int(config["runtime_gate"]["maximum_request_tokens"])
    if execution.get("maximum_request_tokens") != expected_tokens:
        raise ValueError("runtime gate did not execute the suite maximum request")
    if execution.get("prompt_tokens_returned") != expected_tokens:
        raise ValueError("runtime gate prompt was truncated")
    if execution.get("prompt_logprob_tokens_complete") != expected_tokens - 1:
        raise ValueError("runtime gate prompt logprobs are incomplete")
    if not execution.get("continuation_loglikelihood_finite"):
        raise ValueError("runtime gate continuation loglikelihood is not finite")
    if execution.get("sample_content_logged") is not False:
        raise ValueError("runtime gate did not assert content-safe logging")
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed",
        "maximum_request_tokens": expected_tokens,
        "minimum_kv_capacity_tokens": minimum_capacity,
        "observed_kv_capacity_tokens": capacity,
        "native_cutlass_w8a8_dispatch": True,
        "text_only_active": True,
        "chunked_prefill_active": True,
        "prompt_logprobs_complete": True,
        "runtime": execution["runtime"],
        "sample_content_logged": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--failure-only", action="store_true")
    parser.add_argument("--execution")
    parser.add_argument("--output")
    args = parser.parse_args()
    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    if args.failure_only:
        reason = failure_reason(log_text)
        if reason:
            print(reason)
            raise SystemExit(1)
        return
    if not args.execution or not args.output:
        parser.error("--execution and --output are required unless --failure-only is set")
    with Path(args.execution).open(encoding="utf-8") as handle:
        execution = json.load(handle)
    payload = validate_runtime_gate(log_text, execution, load_config())
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
