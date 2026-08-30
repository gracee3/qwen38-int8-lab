#!/usr/bin/env python3
"""Aggregate private harness outputs into content-free retention evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import atomic_json, load_config


def metric_value(metrics: dict[str, Any], name: str) -> float:
    matches = [value for key, value in metrics.items() if key == name or key.startswith(f"{name},")]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} metric, found {len(matches)}")
    value = float(matches[0])
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric {name}")
    return value


def find_result(stage: Path) -> tuple[Path, dict[str, Any]]:
    matches = sorted(stage.rglob("results_*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one result JSON below {stage}, found {len(matches)}")
    with matches[0].open(encoding="utf-8") as handle:
        return matches[0], json.load(handle)


def headline(raw: dict[str, Any], harness_task: str, metric: str) -> float:
    values = raw.get("groups", {}).get(harness_task)
    if values is None:
        values = raw.get("results", {}).get(harness_task)
    if values is None:
        raise ValueError(f"missing task/group result: {harness_task}")
    return metric_value(values, metric)


def samples(stage: Path, metric: str) -> dict[str, dict[int, tuple[str, float]]]:
    strata: dict[str, dict[int, tuple[str, float]]] = {}
    for path in sorted(stage.rglob("samples_*.jsonl")):
        task_name = path.name.removeprefix("samples_").split("_20", 1)[0]
        task = strata.setdefault(task_name, {})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("filter") != "none":
                    continue
                value = row[metric]
                if isinstance(value, (list, tuple)):
                    value = value[0]
                doc_id = int(row["doc_id"])
                if doc_id in task:
                    raise ValueError(f"duplicate sample {task_name}/{doc_id}")
                task[doc_id] = (row["doc_hash"], float(value))
    if not strata:
        raise ValueError(f"no samples below {stage}")
    return strata


def paired_values(candidate: Path, source: Path, metric: str) -> dict[str, np.ndarray]:
    left, right = samples(candidate, metric), samples(source, metric)
    if left.keys() != right.keys():
        raise ValueError("paired task strata differ")
    paired: dict[str, np.ndarray] = {}
    for task in sorted(left):
        if left[task].keys() != right[task].keys():
            raise ValueError(f"paired document IDs differ for {task}")
        values = []
        for doc_id in sorted(left[task]):
            left_hash, left_value = left[task][doc_id]
            right_hash, right_value = right[task][doc_id]
            if left_hash != right_hash:
                raise ValueError(f"paired document hashes differ for {task}/{doc_id}")
            values.append(left_value - right_value)
        paired[task] = np.asarray(values, dtype=np.float64)
    return paired


def stratified_bootstrap(
    strata: dict[str, np.ndarray], replicates: int, seed: int
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    total = sum(len(values) for values in strata.values())
    if not total:
        raise ValueError("empty bootstrap input")
    draws = np.empty(replicates, dtype=np.float64)
    ordered = [strata[key] for key in sorted(strata)]
    for index in range(replicates):
        numerator = 0.0
        for values in ordered:
            numerator += float(values[rng.integers(0, len(values), size=len(values))].sum())
        draws[index] = numerator / total
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), total


def aggregate(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    def optional_json(name: str) -> Any:
        path = run_root / name
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    git_identity = optional_json("git-identity.json")
    evaluation_scope = (git_identity or {}).get("evaluation_scope", "paired")
    if evaluation_scope not in {"paired", "candidate-only"}:
        raise ValueError(f"unexpected evaluation scope: {evaluation_scope}")
    task_results: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    missing_candidate = []
    missing_bf16 = []
    for name, task in config["tasks"].items():
        candidate_stage = run_root / "stages" / f"w8a8-{name}"
        if not candidate_stage.is_dir():
            missing_candidate.append(name)
            continue
        candidate_path, candidate_raw = find_result(candidate_stage)
        candidate_score = headline(candidate_raw, task["harness_task"], task["headline_metric"])
        task_results[name] = {
            "headline_metric": task["headline_metric"],
            "w8a8": candidate_score,
            "candidate_result": str(candidate_path),
        }
        if not task["paired"]:
            continue
        bf16_stage = run_root / "stages" / f"bf16-{name}"
        if not bf16_stage.is_dir():
            missing_bf16.append(name)
            continue
        bf16_path, bf16_raw = find_result(bf16_stage)
        bf16_score = headline(bf16_raw, task["harness_task"], task["headline_metric"])
        delta = candidate_score - bf16_score
        strata = paired_values(candidate_stage, bf16_stage, task["headline_metric"])
        low, high, count = stratified_bootstrap(
            strata,
            int(config["acceptance"]["bootstrap_replicates"]),
            int(config["acceptance"]["bootstrap_seed"]),
        )
        task_results[name].update({"bf16": bf16_score, "delta": delta, "bf16_result": str(bf16_path)})
        paired[name] = {
            "delta": delta,
            "confidence_interval_95": [low, high],
            "paired_documents": count,
            "strata": {key: len(value) for key, value in sorted(strata.items())},
        }

    paired_names = [name for name, task in config["tasks"].items() if task["paired"]]
    blocker = None
    decision = "incomplete"
    macro_delta = None
    if missing_candidate:
        blocker = "missing candidate groups: " + ", ".join(missing_candidate)
    elif evaluation_scope == "candidate-only":
        decision = "candidate_scores_only_no_retention_or_deployment_recommendation"
    elif missing_bf16:
        blocker = "BF16 comparison incomplete: " + ", ".join(missing_bf16)
        decision = "candidate_scores_only_no_retention_or_deployment_recommendation"
    elif set(paired) == set(paired_names):
        macro_delta = float(np.mean([paired[name]["delta"] for name in paired_names]))
        macro_ok = macro_delta >= float(config["acceptance"]["macro_min_delta"])
        individual_ok = all(
            paired[name]["delta"] >= float(config["acceptance"]["individual_min_delta"])
            for name in paired_names
        )
        decision = "retention_accepted" if macro_ok and individual_ok else "retention_rejected"

    return {
        "schema_version": 1,
        "run_id": run_root.name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": config["scope"],
        "evaluation_scope": evaluation_scope,
        "harness": config["harness"],
        "git": git_identity,
        "image": optional_json("image-identity.json"),
        "packages": optional_json("package-identity.json"),
        "models": {
            "before": optional_json("model-manifests-before.json"),
            "after": optional_json("model-manifests-after.json"),
        },
        "datasets": optional_json("dataset-preflight.json"),
        "request_preflight": optional_json("request-preflight.json"),
        "protocol": config["protocol"],
        "runtime": optional_json("runtime-identity.json"),
        "tasks": task_results,
        "paired_retention": {
            "groups": paired,
            "macro_delta": macro_delta,
            "acceptance": config["acceptance"],
        },
        "telemetry": optional_json("telemetry.json"),
        "evidence_root": str(run_root),
        "decision": decision,
        "blocker": blocker,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Standardized W8A8 accuracy evaluation — {payload['run_id']}",
        "",
        "## Outcome",
        "",
        f"Decision: `{payload['decision']}`.",
    ]
    if payload["blocker"]:
        lines.append(f"Blocker: {payload['blocker']}.")
    runtime = payload.get("runtime")
    if runtime:
        configured = runtime["configured"]
        lines += [
            "",
            "## W8A8 runtime",
            "",
            f"Text-only loading: `{str(configured['language_model_only']).lower()}`; chunked prefill: `{str(configured['enable_chunked_prefill']).lower()}` with `{configured['max_num_batched_tokens']:,}` batched tokens; explicit KV allocation: `{configured['kv_cache_memory_bytes']:,}` bytes per GPU; observed KV capacity: `{runtime['observed_kv_capacity_tokens']:,}` tokens.",
        ]
    lines += ["", "## Headline metrics", "", "| Group | Metric | W8A8 | BF16 | Delta |", "| --- | --- | ---: | ---: | ---: |"]
    for name, result in payload["tasks"].items():
        bf16 = "—" if "bf16" not in result else f"{result['bf16']:.6f}"
        delta = "—" if "delta" not in result else f"{result['delta']:+.6f}"
        lines.append(f"| {name} | {result['headline_metric']} | {result['w8a8']:.6f} | {bf16} | {delta} |")
    lines += [
        "",
        "## Scope",
        "",
        payload["scope"] + f". A passing result applies only to non-thinking text under the {payload['protocol']['context_length']:,}-token protocol; it does not cover multimodal, very-long-context, safety, or task-specific production quality.",
        "",
        f"Private evidence: `{payload['evidence_root']}`. No gated sample content is included in this report.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    payload = aggregate(Path(args.run_root), load_config())
    atomic_json(args.output, payload, mode=0o644)
    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
