#!/usr/bin/env python3
"""Measure HTTP time-to-first-token and approximate prefill/decode throughput."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def streaming_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, int] | None = None
    text_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                text = choices[0].get("text", "")
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(text)
    finished = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream did not include token data and final usage")
    ttft = first_token_at - started
    decode_seconds = max(finished - first_token_at, 1e-9)
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    return {
        "time_to_first_token_seconds": ttft,
        "total_seconds": finished - started,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "approx_prefill_tokens_per_second": prompt_tokens / max(ttft, 1e-9),
        "decode_tokens_per_second": max(completion_tokens - 1, 0) / decode_seconds,
        "generated_text_preview": "".join(text_parts)[:160],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen38-w8a8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    prompt = ("Explain one practical consideration when quantizing language models for deployment. " * 96).strip()
    payload = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for _ in range(args.warmups):
        streaming_request(f"{args.base_url}/completions", payload, args.timeout)
    runs = [streaming_request(f"{args.base_url}/completions", payload, args.timeout) for _ in range(args.runs)]
    result = {
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "settings": {
            "runs": args.runs,
            "warmups": args.warmups,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 42,
        },
        "measurement_note": "Prefill rate is prompt tokens divided by client-observed time to first token; it includes HTTP and scheduler overhead.",
        "runs": runs,
        "summary": {
            "median_ttft_seconds": statistics.median(run["time_to_first_token_seconds"] for run in runs),
            "median_approx_prefill_tokens_per_second": statistics.median(
                run["approx_prefill_tokens_per_second"] for run in runs
            ),
            "median_decode_tokens_per_second": statistics.median(run["decode_tokens_per_second"] for run in runs),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
