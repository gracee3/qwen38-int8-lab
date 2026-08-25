# Qwen3.8-27B W8A8 INT8 Lab

Reproducible infrastructure for turning the official Qwen3.8-27B BF16 checkpoint into a calibrated W8A8 INT8 `compressed-tensors` checkpoint and serving it with vLLM tensor parallelism on two RTX 3090 GPUs.

```text
Qwen3.8-27B BF16
        ↓ calibrated GPTQ W8A8 (conservative text targets)
compressed-tensors / Safetensors
        ↓ vLLM tensor parallel = 2
RTX 3090 native INT8 / CUTLASS
```

The self-contained quality candidate has loaded directly through native `CompressedTensorsW8A8Int8 → CutlassInt8ScaledMMLinearKernel` execution. Standardized accuracy remains blocked as described below.

## Scope and hardware

The lab targets one Linux host with two 24 GB RTX 3090 cards, roughly 96 GB RAM, Docker with NVIDIA Container Toolkit, and a slow Internet connection. Quantization and inference intentionally use independent images because LLM Compressor and vLLM currently resolve different `compressed-tensors` versions.

No CI/CD or GitHub Actions are configured. The repository holds code, configuration, small reports, and curated benchmark summaries only.

## Data boundary

The complete source checkpoint currently lives at:

```text
/home/emmy/workspace/qwen3.8-27b-download/model
```

It is mounted read-only at `/models/source`; it is never modified or copied into the repository. The interrupted `/data/models/Qwen3.8-27B` download is unrelated and remains untouched.

```text
/data/models/Qwen3.8-27B-W8A8-INT8/   eventual output
/data/qwen38-int8-lab/
    scratch/                           offload and disposable smoke artifacts
    cache/                             pip, Hugging Face, datasets, vLLM compile cache
    calibration/                       prepared calibration material
    logs/                              timestamped run logs
    results/                           machine-readable run/benchmark results
```

Containers see `/models`, `/work`, and read-only project code at `/app`. Inference mounts `/data/models` read-only. Defensive ignore rules reject common model-weight and cache formats.

## Initial quantization policy

The checked-in policy is intentionally narrower than “all Linear modules”:

- Quantize ordinary text MLP projections and full-attention Q/K/V/O projections with calibrated GPTQ W8A8.
- Use per-channel INT8 weights and dynamic per-token INT8 activations, serialized by `compressed-tensors`.
- Preserve embeddings and the untied `lm_head` for output quality.
- Preserve the vision tower because it is more sensitive and is not needed for the first text-serving gate.
- Preserve all Gated DeltaNet/recurrent (`linear_attn`) projections, convolution, state parameters, and gated normalization until mixed-path quality is measured.
- Preserve the top-level MTP tensors. Transformers does not instantiate these tensors, so a complete serializer must copy them back byte-for-byte before an output can be called complete.
- Do not apply SmoothQuant yet. Qwen3.8 uses zero-centered RMSNorm parameters applied as `(1 + weight)`, and generic smoothing mappings have not been validated for that implementation.

The complete machine-readable policy and reasons are in `quant/config/qwen38-27b.yaml`.

## Workflow

Prerequisites are `docker`, NVIDIA container support, and `just`.

```bash
just info
just build-quant
just gpu
just inspect-model
just quant-plan
just quant-smoke
just load-trace   # complete BF16 load + two-prompt sequential trace; no GPTQ/output
just quant-tiny   # two-sample real-source experimental artifact under /work/scratch
just dataset-preflight quality  # pin/select/tokenize all 512 rows; no model load
just quant-small  # 32-sample x 512-token scaling candidate under /work/scratch
just quant        # authorized 512-sample x 2,048-token quality candidate

just build-vllm
just validate
just serve       # foreground server on port 8000
just smoke       # from a second shell
just bench

# Standardized non-thinking text accuracy (lm-eval v0.4.12)
just build-eval
# Run the exact pushed feature commit inside a labeled tmux session:
just eval-standardized <exact-pushed-commit>
```

Paths and image names are overridable with `MODEL_ROOT`, `WORK_ROOT`, `SOURCE_MODEL`, `OUTPUT_MODEL`, `QUANT_IMAGE`, `VLLM_IMAGE`, and `PORT`.

`just quant-smoke` is designed to quantize a tiny synthetic Qwen3.5/3.8-shaped model with the real package APIs and serialization path. It is not a partially quantized 27B checkpoint. Experimental real-source profiles require an explicit destination below `/work/scratch` and are marked non-production. The guarded `just quant` command is reserved for the measured quality run and refuses to overwrite an existing output. The external dataset is pinned to commit `8049631c405ae6576f93f445c6b8166f76f5505a`; run metadata records both dataset fingerprints, seed, count, and aggregate token lengths, never sample text.

`just load-trace` is the production-faithful, non-executing real-source safety gate. It loads the complete BF16 checkpoint with the same loader, tokenizer, local two-prompt dataset, 80 GiB CPU ceiling, and `auto_offload` mapping used by `just quant-tiny`, then invokes LLM Compressor's sequential tracer without GPTQ or serialization. `just quant-tiny` always creates a timestamped, explicitly non-production checkpoint below `/work/scratch`; neither command writes to the production model destination.

Real-checkpoint serialization uses the configured 1 GB shard limit, hidden incomplete staging, and an atomic final rename. Before that rename, it restores 15 MTP tensors and copies both processor JSON files byte-for-byte from the source after validating that they are in-tree JSON objects. Strict validation requires complete index/shard integrity, W8A8 metadata, exactly 256 quantized targets, 15 MTP tensors, and both processor files. Run metadata includes process RSS and `VmSwap`, host swap-I/O deltas, memory PSI totals, available RAM, swap growth, GPU memory, free disk, and any sustained safety-trigger reason. The interrupt thresholds remain 8 GiB available RAM or more than 4 GiB swap growth for 10 seconds.

The supported serving command uses TP2, a 2,048-token context, 0.88 GPU-memory utilization, BF16 KV cache, seed 42, eager mode, and disables prefix caching, chunked prefill, speculation, and the FlashInfer sampler. `just smoke` sends the same non-thinking request twice with a 128-token allowance and requires a deterministic response containing `391`.

For the complete guarded sequence, install a timestamped copy of `scripts/quality_gate_supervisor.sh` under `/data/qwen38-int8-lab`, set `EXPECTED_COMMIT` to the merged `main` commit, and launch it in a newly named tmux session. The supervisor performs fresh host/repository/storage gates before tiny, small, and quality calibration, changes swappiness only around each calibration child process group, restores it exactly, validates direct vLLM loading, and halts without retry on the first failure.

Passing this workflow means the artifact is a functional quality candidate: it is self-contained, loads through the intended native W8A8 path, produces the deterministic arithmetic answer, and completes performance measurement. It is not a standardized accuracy result; accuracy evaluation and vision/video inference remain separate work.

The standardized accuracy workflow is defined in `eval/config/leaderboard-v2.yaml`. It pins the Open LLM Leaderboard v2 task group and all six dataset revisions, prefetches into a fresh private cache, proves identical request rendering with the BF16 and W8A8 tokenizers, and then runs offline. It evaluates all six groups on W8A8 and the four multiple-choice groups on BF16, using TP2 and the fixed 8,192-token non-thinking protocol. Raw samples—including gated GPQA rows—remain mode-restricted under `/data/qwen38-int8-lab/evaluations`; only reviewed aggregates belong in Git. Limited smoke results are gates and must never be reported as accuracy.

## Reproducibility

Docker bases are pinned by digest, direct requirements are version-pinned, and successfully resolved transitive environments are captured in lock files. Run metadata records the Git commit, package versions, recipe, source invariants, timestamps, and observed process/GPU peaks. Dataset caches and vLLM compilation caches remain under `/work/cache` for fast iteration without redownloading.

Benchmark scripts write JSON under `/data/qwen38-int8-lab/results`; only reviewed, compact results belong in Git. No benchmark numbers are pre-populated.

## Current status

As of 2026-08-24:

- The local source is complete: 18 Safetensors shards, 1,199 tensors, about 51.75 GiB of shard files.
- Metadata identifies `Qwen3_5ForConditionalGeneration`: 64 text layers (48 recurrent/linear-attention and 16 full-attention), a 27-layer vision tower, and one MTP layer.
- The host has two visible RTX 3090 GPUs and an NVIDIA-capable Docker runtime.
- Both pinned Docker images build and see both SM86 GPUs. The exact resolved environments are checked in.
- The sequential synthetic gate passes calibration, GPTQ W8A8 compression, Safetensors serialization, index inspection, and quantization-metadata validation. It recorded about 2.0 GiB peak process RSS and 364/266 MiB peak GPU memory.
- The guarded 32-sample scaling gate and 512-sample quality calibration passed; the self-contained candidate is at `/data/models/Qwen3.8-27B-W8A8-INT8`.
- The standardized accuracy infrastructure is published in draft PR #10. GPQA access is accepted and all six pinned datasets now prefetch and validate offline. Exact-commit run `20260825T031746Z` then stopped at the mandatory zero-truncation gate because a task-defined GPQA Extended request renders to 12,314 tokens under the fixed 8,192-token protocol. No model was loaded and no score or deployment/retention decision is available.

See `reports/smoke-test-2026-08-24.md` for the measured gates and remaining boundary.
