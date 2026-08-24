#!/usr/bin/env python3
"""Run two deterministic OpenAI-compatible chat requests against vLLM."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 180.0) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen38-w8a8")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    started = time.monotonic()
    result: dict[str, Any] = {
        "base_url": args.base_url,
        "requested_model": args.model,
        "deterministic": False,
        "coherence_reviewed": False,
    }
    try:
        models = request_json(f"{args.base_url}/models", timeout=args.timeout)
        available = [item["id"] for item in models.get("data", [])]
        if args.model not in available:
            raise RuntimeError(f"{args.model!r} not present in server model list: {available}")
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "Answer precisely and briefly."},
                {"role": "user", "content": "What is 17 multiplied by 23? Show one short calculation."},
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        first = request_json(f"{args.base_url}/chat/completions", payload, args.timeout)
        second = request_json(f"{args.base_url}/chat/completions", payload, args.timeout)
        first_text = first["choices"][0]["message"]["content"]
        second_text = second["choices"][0]["message"]["content"]
        if not first_text.strip():
            raise RuntimeError("vLLM returned an empty generation")
        if "391" not in first_text:
            raise RuntimeError(f"vLLM response did not contain the correct result 391: {first_text!r}")
        result.update(
            {
                "status": "passed",
                "available_models": available,
                "deterministic": first_text == second_text,
                "text": first_text,
                "usage": first.get("usage"),
            }
        )
        if not result["deterministic"]:
            raise RuntimeError("temperature=0, seed=42 requests produced different text")
    except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, IndexError) as exc:
        result.update({"status": "failed", "error": str(exc)})
    result["elapsed_seconds"] = time.monotonic() - started
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
