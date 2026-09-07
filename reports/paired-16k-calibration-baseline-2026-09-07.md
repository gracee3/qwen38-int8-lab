# INT4-v1 vs INT8-v2 paired calibration baseline

Date: 2026-09-07  
Source: local-agent-evals run `20260907T204532Z-dd2ef2503084`  
Protocol: deterministic, non-thinking, 16K scored calibration

This compact report records the reviewed baseline for future model comparisons,
including Q5/Q6 candidates. Raw prompts, responses, generated code, caches,
weights, and host-private paths remain outside Git.

## Serving profiles

The reproducible native presets are:

- `serving/profiles/int4-v1-16k-fp8-tp1.yaml`: TP1 on GPU 0, 16K context,
  FP8 KV, 0.75 GiB KV reservation, eager execution.
- `serving/profiles/int8-v2-16k-bf16-tp2.yaml`: TP2 across both GPUs, 16K
  context, BF16 KV, 0.75 GiB KV reservation, eager execution.

Both use one sequence, non-thinking text-only loading, 1,024-token chunked
prefill, no prefix cache, and vLLM 0.27.1. The host has two RTX 3090 GPUs with
24 GiB VRAM each.

## Reviewed scores

Eight identical frozen examples were scored per benchmark per model.

| Benchmark | Metric | INT4-v1 TP1 | INT8-v2 TP2 | Delta |
|---|---|---:|---:|---:|
| IFEval | prompt-level strict accuracy | 0.7500 (6/8) | 0.8750 (7/8) | +0.1250 |
| HumanEval+ | pass@1 | 0.8750 (7/8) | 1.0000 (8/8) | +0.1250 |
| BBH | answer-choice likelihood accuracy | 0.7500 (6/8) | 0.7500 (6/8) | +0.0000 |
| MMLU-Pro | answer-choice likelihood accuracy | 0.6250 (5/8) | 0.7500 (6/8) | +0.1250 |

All 64 planned examples were scored with no infrastructure errors. INT4 had one
HumanEval+ execution failure and one IFEval truncation. INT8 had two IFEval
truncations. These are directional calibration results, not leaderboard scores.
The small denominator does not justify a single aggregate intelligence number.

Future comparisons should reuse the same task pins, seed, profile family, and
frozen IDs before increasing the sample size.
