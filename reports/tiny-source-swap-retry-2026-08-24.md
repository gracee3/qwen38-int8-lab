# Tiny-source serialization retry — 2026-08-24

## Outcome

The controlled two-sample W8A8 retry passed from merged implementation commit `6a95fc5ac714a3c5dbe4cebceac36e84c51fdee7` (PR #6; implementation commit `ad9b3d6ceb6750a7169f9bb94ed0b459efd8cc8a`). Quantization completed all 65 sequential subgraphs and 256 intended projections, serialized the checkpoint, restored all 15 MTP tensors, and atomically published the experimental output.

The strict index/W8A8/MTP validator passed. vLLM 0.27.1 then loaded the weights with TP2 on both RTX 3090s, emitted the required `Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8` line, and returned deterministic coherent text from two identical seed-42 requests. The server stopped normally with exit status 0.

This remains a two-sample experimental artifact, not a quality checkpoint. The 512-sample × 2,048-token run was not started.

## Controlled host setting

- Wrapper: `/data/qwen38-int8-lab/retry-tiny-source-swappiness-20260824T215125Z.sh`
- Wrapper log: `/data/qwen38-int8-lab/logs/swappiness-retry-20260824T215125Z.log`
- tmux session / initial pane PID: `qwen38-retry-swap1-20260824T215125Z` / `545879`
- Captured swappiness: `60`; set and verified: `1`; restored and verified: `60`
- Command / wrapper status: `0` / `0`
- Quantization log: `/data/qwen38-int8-lab/logs/quant-tiny-20260824T215947Z.log`
- Run metadata: `/data/qwen38-int8-lab/results/quant-tiny_source-20260824T215952Z.json`

Preflight recorded 95,021,166,592 bytes available RAM, 1/1 MiB GPU usage with no compute processes, 527,341,260,800 bytes free on `/data`, a clean synchronized `main`, no new destination, and the unchanged prior staging evidence.

## Serialization and pressure evidence

The configured 1 GB limit produced 35 regular shards plus the preserved-MTP shard. Two regular files are about 2.54 GB because a single embedding or head tensor cannot be split across shards. The final directory contains 44 files and occupies 36,798,457,954 bytes; its index maps 1,455 tensors to 36 Safetensors files and declares 36,778,108,384 tensor bytes.

- Elapsed quantization/serialization: 1,022.873 seconds
- Peak process RSS / `VmSwap`: 76,613,160,960 bytes / 0 bytes
- Minimum host `MemAvailable`: 35,390,631,936 bytes
- Initial / maximum host swap used: 33,968,128 / 185,065,472 bytes
- Maximum swap growth: 151,097,344 bytes
- `pswpin`: 4,429 pages / 18,141,184 bytes
- `pswpout`: 26,857 pages / 110,006,272 bytes
- Memory PSI `some` / `full` delta: 3,279,712 / 3,234,335 microseconds
- GPU peaks: 13,736 / 266 MiB
- Minimum free disk: 490,554,281,984 bytes
- Safety trigger: none

The original failed staging directory remains unchanged at 14,367,972,286 bytes.

## Structural and runtime validation

Strict validation at `/data/qwen38-int8-lab/results/quant-tiny-source-vllm-validation-20260824T223200Z.json` reports a complete index, valid compressed-tensors W8A8 metadata, 15 MTP tensors, and verified CUTLASS dispatch.

The requested vLLM settings were TP2, context 2,048, GPU utilization 0.88, seed 42, BF16 KV cache, `speculative_config=None`, no prefix caching, no chunked prefill, and eager mode with compilation/CUDA graphs disabled. FlashInfer sampling was also disabled because the minimal runtime image deliberately has no CUDA compiler. Runtime evidence at `/data/qwen38-int8-lab/results/vllm-runtime-evidence-20260824T223200Z.txt` recorded TP worker PIDs 613627/613628 on GPUs 0/1 at 21,284 MiB each, health 200, one required dispatch match, and a clean server exit.

The coherent deterministic smoke result is `/data/qwen38-int8-lab/results/inference-smoke-tiny-source-coherent-20260824T223500Z.json`; both requests returned `$17 \times 23 = 391$`. The complete server log is `/data/qwen38-int8-lab/logs/vllm-tiny-source-final-20260824T223200Z.log`.

## Remaining boundary

The artifact is not independently vLLM-loadable because `tokenizer.save_pretrained` omitted `preprocessor_config.json` and `video_preprocessor_config.json`. Runtime validation used a read-only symlink view: every checkpoint file resolved to the unchanged artifact, while only those two processor JSON files resolved to the unchanged source. No checkpoint bytes were modified.

Before requesting the quality run, update serialization to save the multimodal processor files inside hidden staging and make the serving launcher explicitly disable FlashInfer sampling in this runtime-only image. Then validate a self-contained artifact without the read-only view. Because those gaps remain, this record does not recommend quality-run authorization yet.

Postflight confirmed swappiness 60, idle GPUs, no compute processes, active SSH/Docker/containerd, the unchanged default Wi-Fi route, 490,484,756,480 bytes free, and the protected secondary Samsung 990 PRO (currently enumerated as `/dev/nvme0n1`) read-only and unmounted.
