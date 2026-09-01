#!/usr/bin/env python3
"""Run a deterministic multi-depth retrieval probe against an OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


NEEDLES = (
    "EARLY-CODE-7319",
    "MIDDLE-CODE-2846",
    "LATE-CODE-9052",
)
FILLER = (
    "Repository archive filler. Keep remote access safe, preserve unrelated "
    "files, validate changes, and report only the requested evidence.\n"
)


def request_json(url: str, api_key: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def tokenize(base_url: str, api_key: str, content: str, timeout: int) -> int:
    errors = []
    # llama.cpp names this field `content`; vLLM names it `prompt`.
    for field in ("content", "prompt"):
        try:
            result = request_json(
                f"{base_url.rstrip('/')}/tokenize",
                api_key,
                {field: content, "add_special": False},
                timeout,
            )
        except RuntimeError as exc:
            errors.append(f"{field}: {exc}")
            continue
        tokens = result.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"unexpected tokenize response keys: {sorted(result)}")
        return len(tokens)
    raise RuntimeError("tokenize failed for content and prompt fields: " + "; ".join(errors))


def build_prompt(repeats: int) -> str:
    segment = FILLER * repeats
    return (
        "Read the archive and return all three access codes exactly, in order. "
        "Do not explain.\n\n"
        + segment
        + f"Early access code: {NEEDLES[0]}\n"
        + segment
        + f"Middle access code: {NEEDLES[1]}\n"
        + segment
        + f"Late access code: {NEEDLES[2]}\n"
        + segment
        + "\nReturn the early, middle, and late access codes now."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen35-27b-q4km")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", "local-qwen-only"))
    parser.add_argument("--target-tokens", type=int, default=120_000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 4_096 <= args.target_tokens <= 160_000:
        raise SystemExit("--target-tokens must be between 4096 and 160000")

    filler_tokens = tokenize(args.base_url, args.api_key, FILLER, args.timeout)
    if filler_tokens < 1:
        raise RuntimeError("filler unexpectedly tokenized to zero tokens")
    repeats = max(1, args.target_tokens // (4 * filler_tokens))
    prompt = build_prompt(repeats)
    prompt_tokens = tokenize(args.base_url, args.api_key, prompt, args.timeout)

    # One proportional correction keeps the request near the requested size
    # without repeatedly sending very large tokenize requests.
    if abs(prompt_tokens - args.target_tokens) > max(512, args.target_tokens // 100):
        repeats = max(1, round(repeats * args.target_tokens / prompt_tokens))
        prompt = build_prompt(repeats)
        prompt_tokens = tokenize(args.base_url, args.api_key, prompt, args.timeout)

    started = time.monotonic()
    response = request_json(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        args.api_key,
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 96,
            "seed": 42,
        },
        args.timeout,
    )
    elapsed = time.monotonic() - started
    answer = response["choices"][0]["message"]["content"]
    found = {needle: needle in answer for needle in NEEDLES}
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "requested_prompt_tokens": args.target_tokens,
        "observed_prompt_tokens": prompt_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "needle_found": found,
        "passed": all(found.values()),
        "usage": response.get("usage"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
