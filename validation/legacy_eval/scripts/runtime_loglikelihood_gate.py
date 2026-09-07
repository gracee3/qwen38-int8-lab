#!/usr/bin/env python3
"""Execute the suite maximum privately and retain only content-free evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from common import atomic_json, load_config


def redacted_error(error: BaseException) -> str:
    """Return a failure marker without an exception message or request content."""
    return f"runtime_gate_status=failed exception_type={type(error).__name__}"


def token_ids_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def active_runtime(llm: Any) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    multimodal = config.model_config.multimodal_config
    return {
        "language_model_only": bool(
            multimodal is not None and multimodal.language_model_only
        ),
        "enable_chunked_prefill": bool(
            config.scheduler_config.enable_chunked_prefill
        ),
        "max_num_batched_tokens": int(
            config.scheduler_config.max_num_batched_tokens
        ),
        "kv_cache_memory_bytes": int(config.cache_config.kv_cache_memory_bytes or 0),
        "cpu_offload_gb": float(config.offload_config.uva.cpu_offload_gb),
    }


def verify_prompt_logprobs(
    token_ids: list[int], context_tokens: int, output: Any
) -> dict[str, Any]:
    observed_prompt = list(output.prompt_token_ids)
    if observed_prompt != token_ids:
        raise RuntimeError("runtime prompt token sequence was incomplete or changed")
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
        raise RuntimeError("runtime did not return a complete prompt-logprob sequence")
    if prompt_logprobs[0] is not None:
        raise RuntimeError("first prompt token unexpectedly has a conditional logprob")

    scored = 0
    continuation_scored = 0
    continuation_total = 0.0
    for index, (token_id, values) in enumerate(
        zip(token_ids[1:], prompt_logprobs[1:], strict=True), start=1
    ):
        if values is None or token_id not in values:
            raise RuntimeError("prompt logprob is missing the observed token")
        value = float(getattr(values[token_id], "logprob", values[token_id]))
        if not math.isfinite(value):
            raise RuntimeError("prompt logprob is non-finite")
        scored += 1
        if index >= context_tokens:
            continuation_scored += 1
            continuation_total += value

    expected_continuation = len(token_ids) - context_tokens
    if continuation_scored != expected_continuation or not math.isfinite(
        continuation_total
    ):
        raise RuntimeError("continuation loglikelihood is incomplete or non-finite")
    return {
        "prompt_tokens_returned": len(observed_prompt),
        "prompt_logprob_tokens_complete": scored,
        "continuation_tokens_scored": continuation_scored,
        "continuation_loglikelihood_finite": True,
    }


def execute(request_path: Path, model_path: str, output_path: Path) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    config = load_config()
    with request_path.open(encoding="utf-8") as handle:
        request = json.load(handle)
    token_ids = [int(item) for item in request["token_ids"]]
    context_tokens = int(request["context_tokens"])
    expected_tokens = int(config["runtime_gate"]["maximum_request_tokens"])
    if request.get("request_type") != "loglikelihood":
        raise ValueError("private gate request has an unexpected type")
    if len(token_ids) != expected_tokens or int(request["token_count"]) != expected_tokens:
        raise ValueError("private gate request has an unexpected length")
    if request.get("token_ids_sha256") != token_ids_hash(token_ids):
        raise ValueError("private gate request identity is invalid")
    if not 1 <= context_tokens < len(token_ids):
        raise ValueError("private gate request has an invalid context boundary")

    protocol = config["protocol"]
    expected_runtime = {
        **config["models"]["w8a8"]["runtime"],
        "cpu_offload_gb": config["models"]["w8a8"]["cpu_offload_gb"],
    }
    llm = LLM(
        model=model_path,
        dtype=protocol["dtype"],
        tensor_parallel_size=int(protocol["tensor_parallel_size"]),
        max_model_len=int(protocol["context_length"]),
        kv_cache_dtype=protocol["kv_cache_dtype"],
        seed=int(protocol["seed"]),
        enforce_eager=bool(protocol["enforce_eager"]),
        enable_prefix_caching=bool(protocol["enable_prefix_caching"]),
        cpu_offload_gb=float(config["models"]["w8a8"]["cpu_offload_gb"]),
        **config["models"]["w8a8"]["runtime"],
    )
    observed_runtime = active_runtime(llm)
    if observed_runtime != expected_runtime:
        raise RuntimeError("active runtime configuration differs from the suite")

    results = llm.generate(
        [TokensPrompt(prompt_token_ids=token_ids)],
        SamplingParams(
            temperature=0,
            prompt_logprobs=1,
            max_tokens=1,
            detokenize=False,
        ),
        use_tqdm=False,
    )
    if len(results) != 1:
        raise RuntimeError("runtime returned an unexpected number of results")
    logprob_evidence = verify_prompt_logprobs(
        token_ids, context_tokens, results[0]
    )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "request_type": "loglikelihood",
        "maximum_request_tokens": len(token_ids),
        "runtime": observed_runtime,
        **logprob_evidence,
        "sample_content_logged": False,
    }
    atomic_json(output_path, payload)
    print(
        "runtime_gate_status=passed "
        f"maximum_request_tokens={len(token_ids)} "
        f"prompt_logprob_tokens_complete={logprob_evidence['prompt_logprob_tokens_complete']} "
        "sample_content_logged=false"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        execute(Path(args.request), args.model, Path(args.output))
    except BaseException as error:
        print(redacted_error(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
