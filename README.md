# Qwen3.8-27B W8A8 INT8 Lab

Reproducible infrastructure for turning the official Qwen3.8-27B BF16 checkpoint into a calibrated W8A8 INT8 `compressed-tensors` checkpoint and serving it with vLLM tensor parallelism on two RTX 3090 GPUs.

```text
Qwen3.8-27B BF16
        ↓ calibrated GPTQ W8A8 (conservative text targets)
compressed-tensors / Safetensors
        ↓ vLLM tensor parallel = 2
RTX 3090 native INT8 / CUTLASS
```

The final kernel-dispatch line is a validation target, not a completed claim. Native `CompressedTensorsW8A8Int8 → CutlassInt8ScaledMMLinearKernel` execution has not yet been observed for this checkpoint.

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

just build-vllm
just validate
just serve       # foreground server on port 8000
just smoke       # from a second shell
just bench
```

Paths and image names are overridable with `MODEL_ROOT`, `WORK_ROOT`, `SOURCE_MODEL`, `OUTPUT_MODEL`, `QUANT_IMAGE`, `VLLM_IMAGE`, and `PORT`.

`just quant-smoke` is designed to quantize a tiny synthetic Qwen3.5/3.8-shaped model with the real package APIs and serialization path. It is not a partially quantized 27B checkpoint. The guarded `just quant` command is reserved for the measured quality run and refuses to overwrite an existing output.

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
- vLLM 0.27.1 registers the Qwen architecture and, on this host, selects `CutlassInt8ScaledMMLinearKernel` for the intended `CompressedTensorsW8A8Int8` scheme. Loading and dispatching the eventual 27B checkpoint remain untested and are not claimed.
- No full 27B quantization has been started.

See `reports/smoke-test-2026-08-24.md` for the measured gates and remaining boundary.
