# Qwen3.8-27B W8A8 INT8 Lab

Reproducible infrastructure for turning the official Qwen3.8-27B BF16 checkpoint into a calibrated W8A8 INT8 `compressed-tensors` checkpoint and serving it with vLLM tensor parallelism on two RTX 3090 GPUs.

```text
Qwen3.8-27B BF16
        ↓ calibrated GPTQ W8A8 (conservative text targets)
compressed-tensors / Safetensors
        ↓ vLLM tensor parallel = 2
RTX 3090 native INT8 / CUTLASS
```

The self-contained quality candidate has loaded directly through native `CompressedTensorsW8A8Int8 → CutlassInt8ScaledMMLinearKernel` execution. Candidate-only standardized accuracy uses the guarded workflow described below.

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
just serve       # 64K, CUDA-graph server on loopback port 8000
just smoke       # from a second shell
just bench

# Standardized non-thinking text accuracy (lm-eval v0.4.12)
just build-eval
# Run the exact pushed feature commit inside a labeled tmux session:
just eval-standardized <exact-pushed-commit>
# Candidate scores only; intentionally omits the impractical local BF16 baseline:
just eval-candidate-only <exact-pushed-commit>
```

Paths and image names are overridable with `MODEL_ROOT`, `WORK_ROOT`, `SOURCE_MODEL`, `OUTPUT_MODEL`, `QUANT_IMAGE`, `VLLM_IMAGE`, and `PORT`. Serving also accepts `VLLM_API_KEY`, `INFERENCE_CONTEXT`, and `INFERENCE_KV_CACHE_BYTES` overrides.

`just quant-smoke` is designed to quantize a tiny synthetic Qwen3.5/3.8-shaped model with the real package APIs and serialization path. It is not a partially quantized 27B checkpoint. Experimental real-source profiles require an explicit destination below `/work/scratch` and are marked non-production. The guarded `just quant` command is reserved for the measured quality run and refuses to overwrite an existing output. The external dataset is pinned to commit `8049631c405ae6576f93f445c6b8166f76f5505a`; run metadata records both dataset fingerprints, seed, count, and aggregate token lengths, never sample text.

`just load-trace` is the production-faithful, non-executing real-source safety gate. It loads the complete BF16 checkpoint with the same loader, tokenizer, local two-prompt dataset, 80 GiB CPU ceiling, and `auto_offload` mapping used by `just quant-tiny`, then invokes LLM Compressor's sequential tracer without GPTQ or serialization. `just quant-tiny` always creates a timestamped, explicitly non-production checkpoint below `/work/scratch`; neither command writes to the production model destination.

Real-checkpoint serialization uses the configured 1 GB shard limit, hidden incomplete staging, and an atomic final rename. Before that rename, it restores 15 MTP tensors and copies both processor JSON files byte-for-byte from the source after validating that they are in-tree JSON objects. Strict validation requires complete index/shard integrity, W8A8 metadata, exactly 256 quantized targets, 15 MTP tensors, and both processor files. Run metadata includes process RSS and `VmSwap`, host swap-I/O deltas, memory PSI totals, available RAM, swap growth, GPU memory, free disk, and any sustained safety-trigger reason. The interrupt thresholds remain 8 GiB available RAM or more than 4 GiB swap growth for 10 seconds.

The default serving command is the measured interactive profile: TP2 across two RTX 3090s, a 65,536-token engineering window, an explicit 2.5 GiB BF16 KV allocation per GPU, CUDA graphs, text-only loading, non-thinking chat, prefix caching, 2,048-token chunked prefill, one concurrent sequence, native CUTLASS W8A8 kernels, and loopback-only authenticated HTTP. MTP speculation is disabled because the local checkpoint produced zero accepted draft tokens in the compatibility trial. `just smoke` sends the same non-thinking request twice and requires a deterministic response containing `391`.

`just bench` measures cold prefill, warm prefix-cache reuse, and steady-state decode separately. The reviewed dual-3090 result is 39.053 median decode tok/s over seven forced 256-token generations. Cold client-observed prefill medians range from 1,778.7 to 2,089.5 tok/s across approximately 1K, 8K, 32K, and 60K prompts. See `reports/inference-performance-2026-08-30.md` for definitions, dispersion, raw-evidence paths, and claim boundaries.

For the complete guarded sequence, install a timestamped copy of `scripts/quality_gate_supervisor.sh` under `/data/qwen38-int8-lab`, set `EXPECTED_COMMIT` to the merged `main` commit, and launch it in a newly named tmux session. The supervisor performs fresh host/repository/storage gates before tiny, small, and quality calibration, changes swappiness only around each calibration child process group, restores it exactly, validates direct vLLM loading, and halts without retry on the first failure.

Passing this workflow means the artifact is a functional quality candidate: it is self-contained, loads through the intended native W8A8 path, produces the deterministic arithmetic answer, and completes performance measurement. It is not a standardized accuracy result; accuracy evaluation and vision/video inference remain separate work.

The standardized accuracy workflow is defined in `eval/config/leaderboard-v2.yaml`. It pins the Open LLM Leaderboard v2 task group and all six dataset revisions, prefetches into a fresh private cache, proves identical request rendering with the BF16 and W8A8 tokenizers, and then runs offline. The default `paired` scope evaluates all six groups on W8A8 and the four multiple-choice groups on BF16. The explicit `candidate-only` scope evaluates all six W8A8 groups but omits BF16 inference; its aggregate must remain `candidate_scores_only_no_retention_or_deployment_recommendation`. Both use TP2, batch size 1, and the fixed 16,384-token non-thinking protocol. W8A8 evaluation loads only the language model from the unchanged multimodal checkpoint, enables 1,024-token chunked prefill, and reserves exactly 805,306,368 KV-cache bytes per GPU with BF16 KV and no CPU offload. Before smoke, a private 12,314-token log-likelihood gate requires at least 16,384 observed KV tokens, complete prompt log-probabilities, and native CUTLASS W8A8 dispatch. Raw samples—including gated GPQA rows and the private gate request—remain mode-restricted under `/data/qwen38-int8-lab/evaluations`; only reviewed aggregates belong in Git. Limited smoke results are gates and must never be reported as accuracy.

## Reproducibility

Docker bases are pinned by digest, direct requirements are version-pinned, and successfully resolved transitive environments are captured in lock files. Run metadata records the Git commit, package versions, recipe, source invariants, timestamps, and observed process/GPU peaks. Dataset caches and vLLM compilation caches remain under `/work/cache` for fast iteration without redownloading.

Benchmark scripts write full JSON under `/data/qwen38-int8-lab/results`; only reviewed, compact summaries belong in Git.

## Current status

As of 2026-08-30:

- The local source is complete: 18 Safetensors shards, 1,199 tensors, about 51.75 GiB of shard files.
- Metadata identifies `Qwen3_5ForConditionalGeneration`: 64 text layers (48 recurrent/linear-attention and 16 full-attention), a 27-layer vision tower, and one MTP layer.
- The host has two visible RTX 3090 GPUs and an NVIDIA-capable Docker runtime.
- Both pinned Docker images build and see both SM86 GPUs. The exact resolved environments are checked in.
- The sequential synthetic gate passes calibration, GPTQ W8A8 compression, Safetensors serialization, index inspection, and quantization-metadata validation. It recorded about 2.0 GiB peak process RSS and 364/266 MiB peak GPU memory.
- The guarded 32-sample scaling gate and 512-sample quality calibration passed; the self-contained candidate is at `/data/models/Qwen3.8-27B-W8A8-INT8`.
- The standardized accuracy infrastructure is published in draft PR #10. GPQA access is accepted and all six pinned datasets prefetch and validate offline. Exact-commit run `20260825T031746Z` showed that one zero-shot GPQA Extended document renders four 12,314-token choice requests, exceeding the former 8,192-token protocol. A complete audit found 12,314 tokens is the suite maximum, so the reviewed follow-up protocol uses a uniform 16,384-token context without truncation.
- A former full-group attempt exhausted GPU memory while producing prompt log-probabilities. The candidate-only retry therefore uses text-only loading, chunked prefill, and an explicit bounded KV allocation, with the suite maximum executed as a mandatory runtime gate before smoke.
- The revised preflight, 12,314-token runtime log-likelihood gate, and limited W8A8 smoke passed. MMLU-Pro was manually paused at 5,811/113,990 requests to prioritize interactive inference; there is no reportable accuracy score, and later resumption restarts that group from zero.
- A loopback-only vLLM server is running the candidate with the measured 64K CUDA-graph profile. Native Rust Goose and standalone Qwen Code both completed real local tool calls. Qwen Code fits the larger window but spends roughly 10K tokens on its fresh built-in prompt/tool envelope; Goose remains the lower-overhead interactive option.

See `reports/smoke-test-2026-08-24.md` for the measured gates and remaining boundary.
See `reports/evaluation-and-agent-status-2026-08-29.md` for the complete attempt ledger, dataset map, agent results, public W8A8 comparison, and next evaluation ladder.
