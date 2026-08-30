#!/usr/bin/env python3
"""Benchmark cold/warm prefill and steady-state decode through vLLM's HTTP API."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def headers(api_key: str | None) -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def request_json(url: str, payload: dict[str, Any] | None, timeout: float, api_key: str | None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers=headers(api_key))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def streaming_request(url: str, payload: dict[str, Any], timeout: float, api_key: str | None) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers(api_key))
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, int] | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text") and first_token_at is None:
                first_token_at = time.perf_counter()
    finished = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream did not include generated text and final usage")
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    ttft = first_token_at - started
    decode_seconds = max(finished - first_token_at, 1e-9)
    return {
        "time_to_first_token_seconds": ttft,
        "total_seconds": finished - started,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "approx_prefill_tokens_per_second": prompt_tokens / max(ttft, 1e-9),
        "decode_seconds_after_first_token": decode_seconds,
        "decode_tokens_per_second": max(completion_tokens - 1, 0) / decode_seconds,
    }


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "population_standard_deviation": statistics.pstdev(values),
    }


def summarize(runs: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    return distribution([float(run[metric]) for run in runs])


def make_prompt(tokenize_url: str, model: str, target: int, nonce: str, timeout: float, api_key: str | None) -> str:
    prefix = f"Benchmark nonce {nonce}. Read the following synthetic input, then answer with one word.\n"
    unit = "Local inference throughput measurement data. "
    low, high = 0, max(target, 1)
    best = prefix
    while low <= high:
        middle = (low + high) // 2
        candidate = prefix + unit * middle
        count = int(request_json(tokenize_url, {"model": model, "prompt": candidate}, timeout, api_key)["count"])
        if count <= target:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {"model": model, "prompt": prompt, "temperature": 0, "seed": 42, "max_tokens": max_tokens,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen38-w8a8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 8192, 32768, 60000])
    parser.add_argument("--prefill-runs", type=int, default=3)
    parser.add_argument("--decode-runs", type=int, default=7)
    parser.add_argument("--decode-prompt-length", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key")
    args = parser.parse_args()
    if args.prefill_runs < 1 or args.decode_runs < 1 or args.decode_tokens < 2:
        parser.error("run counts must be positive and --decode-tokens must be at least 2")

    root_url = args.base_url.removesuffix("/v1")
    models = request_json(f"{args.base_url}/models", None, args.timeout, args.api_key)
    model_record = next((item for item in models.get("data", []) if item.get("id") == args.model), None)
    if model_record is None:
        raise RuntimeError(f"{args.model!r} was not returned by /models")
    max_model_len = int(model_record.get("max_model_len", 0))
    if max(args.prompt_lengths + [args.decode_prompt_length]) + args.decode_tokens > max_model_len:
        raise RuntimeError("requested prompt plus generation exceeds the advertised model context")

    completions_url, tokenize_url = f"{args.base_url}/completions", f"{root_url}/tokenize"
    prefill: list[dict[str, Any]] = []
    for target in args.prompt_lengths:
        warm_prompt = make_prompt(tokenize_url, args.model, target, f"warm-{target}", args.timeout, args.api_key)
        # Several forced tokens avoid an all-special-token completion with no
        # client-visible text, which would make HTTP TTFT unobservable.
        warm_payload = payload(args.model, warm_prompt, 4)
        streaming_request(completions_url, warm_payload, args.timeout, args.api_key)
        warm_runs = [streaming_request(completions_url, warm_payload, args.timeout, args.api_key) for _ in range(args.prefill_runs)]
        cold_runs = []
        for index in range(args.prefill_runs):
            prompt = make_prompt(tokenize_url, args.model, target, f"cold-{target}-{index}-{uuid.uuid4().hex}", args.timeout, args.api_key)
            run = streaming_request(completions_url, payload(args.model, prompt, 4), args.timeout, args.api_key)
            run["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            cold_runs.append(run)
        prefill.append({"target_prompt_tokens": target, "cold_runs": cold_runs, "warm_runs": warm_runs,
                        "cold_summary": summarize(cold_runs, "approx_prefill_tokens_per_second"),
                        "warm_summary": summarize(warm_runs, "approx_prefill_tokens_per_second"),
                        "cold_ttft_summary": summarize(cold_runs, "time_to_first_token_seconds"),
                        "warm_ttft_summary": summarize(warm_runs, "time_to_first_token_seconds")})

    decode_prompt = make_prompt(tokenize_url, args.model, args.decode_prompt_length, "decode-steady-state", args.timeout, args.api_key)
    decode_payload = payload(args.model, decode_prompt, args.decode_tokens)
    streaming_request(completions_url, decode_payload, args.timeout, args.api_key)
    decode_runs = [streaming_request(completions_url, decode_payload, args.timeout, args.api_key) for _ in range(args.decode_runs)]
    result = {
        "schema_version": 2, "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": {"base_url": args.base_url, "model": model_record},
        "settings": {"prompt_length_targets": args.prompt_lengths, "prefill_runs_per_condition": args.prefill_runs,
                     "decode_runs": args.decode_runs, "decode_prompt_length_target": args.decode_prompt_length,
                     "decode_tokens": args.decode_tokens, "temperature": 0, "seed": 42, "ignore_eos": True},
        "measurement_notes": {
            "cold_prefill": "Each recorded prompt has a unique leading nonce, preventing prefix-cache reuse.",
            "warm_prefill": "An identical prompt is submitted once before recorded runs, measuring configured prefix-cache behavior.",
            "prefill_rate": "Prompt tokens divided by client-observed time to first generated token; includes HTTP, scheduling, and first-token sampling overhead.",
            "decode_rate": "Completion tokens after the first divided by elapsed time after the first token; fixed-length generation is forced with ignore_eos."},
        "prefill": prefill,
        "decode": {"runs": decode_runs, "tokens_per_second_summary": summarize(decode_runs, "decode_tokens_per_second"),
                   "ttft_summary": summarize(decode_runs, "time_to_first_token_seconds")}}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
