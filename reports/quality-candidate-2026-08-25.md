# Self-contained W8A8 quality candidate — 2026-08-25

## Outcome

The guarded tiny → small → quality sequence passed from merged implementation commit `44e4742a46f63830ea180122d0ae293c0fbefc46` (PR #8; implementation commit `8589f786020a338e67ac5144e03d4c70582bfa30`). The final checkpoint was published atomically at `/data/models/Qwen3.8-27B-W8A8-INT8` and occupies 36,798,458,455 bytes across 45 files.

This is a functional quality candidate, not a standardized accuracy result. Text loading, native W8A8 dispatch, deterministic generation, and performance measurement passed. Standardized accuracy and vision/video inference remain separate work.

## Calibration gates

| Gate | Samples × cap | Calibration result | Strict/direct vLLM result |
| --- | ---: | --- | --- |
| Tiny | 2 × 128 | Passed in 1,067.43 s | Passed; deterministic `391` |
| Small | 32 × 512 | Passed in 1,156.10 s | Passed; deterministic `391` |
| Quality | 512 × 2,048 | Passed in 4,124.52 s | Passed; deterministic `391`; benchmark completed |

Every stage used a fresh destination and passed its own clean-repository, memory, GPU, disk, destination-absence, dataset-revision, and protected-disk preflight. Swappiness was captured as `60`, set and verified as `1` only around each calibration, then restored and verified as `60` before validation. No safety interrupt fired and no stage was retried.

## Dataset identity

The small and quality gates used `HuggingFaceH4/ultrachat_200k`, split `train_sft`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, and seed 42. The quality selection recorded source fingerprint `5f27a15160f0c616` and tokenized fingerprint `4a9ff3305e8ec705`. Its 512 token lengths had minimum 245, median 1,131.5, mean 1,193.25390625, and maximum 2,048; 55 samples reached the cap. No sample content was logged.

## Artifact validation

Strict validation reported:

- Complete Safetensors index/shard integrity.
- `compressed-tensors` / `int-quantized` / `compressed` W8A8 metadata.
- Exactly 256 intended quantized targets.
- All 15 preserved MTP tensors (849,398,784 bytes).
- Byte-for-byte processor files: `preprocessor_config.json` (390 bytes, SHA-256 `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`) and `video_preprocessor_config.json` (385 bytes, SHA-256 `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13`).
- No `EXPERIMENTAL_NON_PRODUCTION.json` marker.

vLLM 0.27.1 loaded the artifact directly with TP2, 2,048-token context, 0.88 GPU utilization, BF16 KV cache, seed 42, eager mode, disabled prefix caching/chunked prefill/speculation, and disabled FlashInfer sampling. It emitted `Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8` and used both RTX 3090 GPUs. Two identical non-thinking smoke requests returned the same correct response containing `391`.

## Performance and telemetry

Three measured 1,153-prompt-token / 128-output-token requests, after one warmup, produced medians of:

- TTFT: 0.607919702 seconds.
- Approximate client-observed prefill: 1,896.632 tokens/s.
- Decode: 10.409792 tokens/s.

During quality calibration, peak process RSS was 83,402,051,584 bytes, peak process `VmSwap` was zero, minimum host `MemAvailable` was 27,226,578,944 bytes, host swap growth was 23,494,656 bytes, and minimum disk availability was 375,536,308,224 bytes. Peak GPU memory was 12,542 MiB on GPU 0 and 266 MiB on GPU 1. Memory PSI deltas were 2,917,516 µs `some` and 2,912,660 µs `full`; swap-I/O deltas were 12,689,408 bytes in and 23,953,408 bytes out.

## Evidence and postflight

- Supervisor: `/data/qwen38-int8-lab/quality-gate-supervisor-20260824T232337Z.sh`
- Supervisor log/result: `/data/qwen38-int8-lab/logs/quality-supervisor-20260824T232337Z.log`, `/data/qwen38-int8-lab/results/quality-supervisor-20260824T232337Z.json`
- Quality run metadata: `/data/qwen38-int8-lab/results/quant-quality-20260825T001009Z.json`
- Strict validation: `/data/qwen38-int8-lab/results/strict-quality-20260825T011912Z.json`
- Smoke/benchmark: `/data/qwen38-int8-lab/results/smoke-quality-20260825T011912Z.json`, `/data/qwen38-int8-lab/results/benchmark-quality-20260825T011912Z.json`
- Complete server log: `/data/qwen38-int8-lab/logs/vllm-quality-20260825T011912Z.log`

Postflight found swappiness restored to 60, both GPUs idle at 1 MiB with no compute processes, port 8000 and the validation container released, SSH/Docker/containerd active, the default route still on `wlx00c0cab51e69`, 375,283,175,424 bytes free, and protected Samsung 990 PRO serial `S7KHNU0X722442H` read-only and unmounted.

## Next action

Run a separately designed standardized text-accuracy evaluation against this immutable candidate before making an accuracy or deployment recommendation. Vision/video validation is also still required for multimodal use.
