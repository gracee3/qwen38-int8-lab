#!/usr/bin/env python3
"""Render every suite request with both tokenizers and prove zero truncation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from common import atomic_json, dataset_pins, load_config
from pinned_datasets import install


@dataclass
class Summary:
    count: int = 0
    maximum_tokens: int = 0
    digest: Any = None

    def __post_init__(self) -> None:
        self.digest = hashlib.sha256()

    def add(
        self, task: str, request_type: str, rendered: str, token_ids: list[int], reserve: int
    ) -> None:
        total = len(token_ids) + reserve
        self.count += 1
        self.maximum_tokens = max(self.maximum_tokens, total)
        self.digest.update(task.encode())
        self.digest.update(b"\0")
        self.digest.update(request_type.encode())
        self.digest.update(b"\0")
        self.digest.update(rendered.encode())
        self.digest.update(b"\0")
        self.digest.update(len(token_ids).to_bytes(4, "big"))
        self.digest.update(reserve.to_bytes(4, "big"))
        for token_id in token_ids:
            self.digest.update(int(token_id).to_bytes(4, "big", signed=False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.count,
            "maximum_prompt_plus_generation_tokens": self.maximum_tokens,
            "rendered_request_and_token_sha256": self.digest.hexdigest(),
        }


def apply_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=not add_generation_prompt,
        enable_thinking=False,
    )


def encode_request(tokenizer: Any, request: Any) -> tuple[str, list[int], int, str]:
    request_type = request.request_type
    if request_type == "loglikelihood":
        context, continuation = request.arguments
        ids = tokenizer.encode(context + continuation, add_special_tokens=False)
        return context + continuation, ids, 0, request_type
    if request_type == "generate_until":
        context, generation = request.arguments
        ids = tokenizer.encode(context, add_special_tokens=False)
        reserve = int(generation.get("max_gen_toks", 256))
        return context, ids, reserve, request_type
    raise ValueError(f"unexpected request type: {request_type}")


def build_for(
    tokenizer: Any, task_config: dict[str, Any], context_length: int
) -> tuple[Summary, dict[str, dict[str, int]], dict[str, int]]:
    from lm_eval.tasks import TaskManager

    random.seed(42)
    np.random.seed(42)
    summary = Summary()
    per_task: dict[str, dict[str, int]] = {}
    manager = TaskManager()
    group_counts = {}
    for group_name, group_config in task_config.items():
        loaded = manager.load([group_config["harness_task"]])
        group_count = 0
        for task_name, task in sorted(loaded["tasks"].items()):
            task.build_all_requests(
                limit=None,
                rank=0,
                world_size=1,
                cache_requests=False,
                rewrite_requests_cache=False,
                system_instruction=None,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                chat_template=lambda messages, add_generation_prompt=True: apply_chat(
                    tokenizer, messages, add_generation_prompt
                ),
                tokenizer_name="checkpoint-tokenizer-preflight",
            )
            maximum = 0
            documents = len(task.eval_docs)
            group_count += documents
            for request in task.instances:
                rendered, ids, reserve, request_type = encode_request(tokenizer, request)
                total = len(ids) + reserve
                runtime_limit = (
                    context_length - 1
                    if request_type == "loglikelihood"
                    else context_length
                )
                if total > runtime_limit:
                    raise RuntimeError(
                        f"request would truncate at runtime: task={task_name} "
                        f"doc_id={request.doc_id} total={total} limit={runtime_limit}"
                    )
                summary.add(task_name, request_type, rendered, ids, reserve)
                maximum = max(maximum, total)
            per_task[task_name] = {"documents": documents, "maximum_tokens": maximum}
        expected = int(group_config["expected_eval_documents"])
        if group_count != expected:
            raise RuntimeError(
                f"evaluation document count differs for {group_name}: "
                f"expected {expected}, got {group_count}"
            )
        group_counts[group_name] = group_count
    return summary, per_task, group_counts


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    return {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config()
    install(dataset_pins(config), offline=True)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    candidate = AutoTokenizer.from_pretrained(args.candidate, local_files_only=True)
    source = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    candidate_identity = tokenizer_identity(candidate)
    source_identity = tokenizer_identity(source)
    if candidate_identity != source_identity:
        raise RuntimeError("checkpoint tokenizer special-token or chat-template identities differ")
    if candidate.add_bos_token or source.add_bos_token:
        raise RuntimeError("tokenizer would add a BOS token contrary to protocol")
    expected_special = config["protocol"]["special_tokens"]
    observed_special = {
        "bos_token_id": candidate.bos_token_id,
        "eos_token_id": candidate.eos_token_id,
        "pad_token_id": candidate.pad_token_id,
    }
    if observed_special != expected_special:
        raise RuntimeError(
            f"unexpected special-token IDs: expected {expected_special}, got {observed_special}"
        )

    context_length = int(config["protocol"]["context_length"])
    candidate_summary, candidate_tasks, candidate_groups = build_for(
        candidate, config["tasks"], context_length
    )
    source_summary, source_tasks, source_groups = build_for(
        source, config["tasks"], context_length
    )
    if (
        candidate_summary.as_dict() != source_summary.as_dict()
        or candidate_tasks != source_tasks
        or candidate_groups != source_groups
    ):
        raise RuntimeError("candidate and source rendered requests/token IDs differ")
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed",
        "context_limit": context_length,
        "runtime_truncation_allowed": False,
        "tokenizer_identity": candidate_identity,
        "requests": candidate_summary.as_dict(),
        "tasks": candidate_tasks,
        "groups": candidate_groups,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
