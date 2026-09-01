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

# Optional Qwen3.5-27B Q4_K_M llama.cpp path with GPU 1 ASR headroom
just serve-llama       # 131,072-token candidate
just serve-llama-160k  # 163,840-token experiment; not yet promoted
just smoke-llama
just probe-llama 120000

# Standardized non-thinking text accuracy (lm-eval v0.4.12)
just build-eval
# Run the exact pushed feature commit inside a labeled tmux session:
just eval-standardized <exact-pushed-commit>
# Candidate scores only; intentionally omits the impractical local BF16 baseline:
just eval-candidate-only <exact-pushed-commit>
```

Paths and image names are overridable with `MODEL_ROOT`, `WORK_ROOT`, `SOURCE_MODEL`, `OUTPUT_MODEL`, `QUANT_IMAGE`, `VLLM_IMAGE`, and `PORT`. Serving also accepts `VLLM_API_KEY`, `INFERENCE_CONTEXT`, and `INFERENCE_KV_CACHE_BYTES` overrides.

The optional GGUF serving path is documented in
[`inference/llama-gguf.md`](inference/llama-gguf.md). It pins llama.cpp and the
Q4_K_M source revision, uses compatible layer splitting with quantized KV, and
keeps the 160K profile experimental until long-context, Qwen Code, soak, and
native-ASR co-residency gates pass.

`just quant-smoke` is designed to quantize a tiny synthetic Qwen3.5/3.8-shaped model with the real package APIs and serialization path. It is not a partially quantized 27B checkpoint. Experimental real-source profiles require an explicit destination below `/work/scratch` and are marked non-production. The guarded `just quant` command is reserved for the measured quality run and refuses to overwrite an existing output. The external dataset is pinned to commit `8049631c405ae6576f93f445c6b8166f76f5505a`; run metadata records both dataset fingerprints, seed, count, and aggregate token lengths, never sample text.

`just load-trace` is the production-faithful, non-executing real-source safety gate. It loads the complete BF16 checkpoint with the same loader, tokenizer, local two-prompt dataset, 80 GiB CPU ceiling, and `auto_offload` mapping used by `just quant-tiny`, then invokes LLM Compressor's sequential tracer without GPTQ or serialization. `just quant-tiny` always creates a timestamped, explicitly non-production checkpoint below `/work/scratch`; neither command writes to the production model destination.

Real-checkpoint serialization uses the configured 1 GB shard limit, hidden incomplete staging, and an atomic final rename. Before that rename, it restores 15 MTP tensors and copies both processor JSON files byte-for-byte from the source after validating that they are in-tree JSON objects. Strict validation requires complete index/shard integrity, W8A8 metadata, exactly 256 quantized targets, 15 MTP tensors, and both processor files. Run metadata includes process RSS and `VmSwap`, host swap-I/O deltas, memory PSI totals, available RAM, swap growth, GPU memory, free disk, and any sustained safety-trigger reason. The interrupt thresholds remain 8 GiB available RAM or more than 4 GiB swap growth for 10 seconds.

The default serving command is the measured interactive profile: TP2 across two RTX 3090s, a 65,536-token engineering window, an explicit 2.5 GiB BF16 KV allocation per GPU, CUDA graphs, text-only loading, non-thinking chat, prefix caching, 2,048-token chunked prefill, one concurrent sequence, native CUTLASS W8A8 kernels, and loopback-only authenticated HTTP. `just smoke` sends the same non-thinking request twice and requires a deterministic response containing `391`.

### Measured inference performance

The reviewed default-profile result is **39.053 median decode tok/s** across seven forced 256-token generations, with a very narrow 39.050–39.060 tok/s range. This is single-sequence HTTP inference on two 24 GiB RTX 3090s with vLLM 0.27.1, TP2, BF16 KV, and CUDA graphs.

| Prompt target | Actual cold prompt | Cold approximate prefill | Cold median TTFT | Warm cached effective prefill | Warm median TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,020–1,022 | 1,903.7 tok/s | 0.5368 s | 5,992.9 tok/s | 0.1700 s |
| 8,192 | 8,187–8,191 | 2,089.5 tok/s | 3.9181 s | 30,954.0 tok/s | 0.2646 s |
| 32,768 | 32,765–32,767 | 1,935.5 tok/s | 16.9289 s | 58,355.6 tok/s | 0.5615 s |
| 60,000 | 59,997–60,000 | 1,778.7 tok/s | 33.7324 s | 91,461.5 tok/s | 0.6560 s |

Each cold condition contains three requests with unique leading nonces, preventing prefix-cache reuse. Each warm condition contains one unrecorded cache-populating request followed by three identical recorded requests. Cold prefill is prompt tokens divided by client-observed TTFT and therefore includes HTTP, scheduler, and first-token overhead. Warm results measure effective application throughput from KV-cache reuse; the 91K figure is not fresh-compute prefill throughput.

CUDA graphs produced the major decode improvement. In the earlier matched 1,153-input/128-output tuning probe, eager mode decoded at 10.688 tok/s and CUDA-graphs-only mode decoded at 39.124 tok/s: **3.66× throughput, or about a 266% increase**. The full suite above independently confirmed the final default at 39.053 tok/s over longer generations.

MTP weights remain preserved in the checkpoint, but speculative MTP serving is **disabled**. The compatibility trial accepted 0 of 1,524 drafted tokens, so it provided no valid speculative-token benefit and is excluded from the defaults and headline claim. Thinking/reasoning is also **disabled** through the server's default chat-template arguments; the benchmark uses deterministic non-thinking completions. These choices avoid wasted draft/reasoning work and keep the measured profile focused on responsive coding-agent use.

Run `just bench` to reproduce the cold, warm, and decode suite. Full methodology, dispersion, and evidence paths are in `reports/inference-performance-2026-08-30.md`; the complete raw run is `/data/qwen38-int8-lab/results/benchmark-suite-defaults-20260830T004833Z.json` on the measured host. The result does not establish concurrent-serving throughput, vision performance, standardized accuracy, or 64K context quality.

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
- The standardized accuracy infrastructure is implemented and validated. GPQA access is accepted and all six pinned datasets prefetch and validate offline. Exact-commit run `20260825T031746Z` showed that one zero-shot GPQA Extended document renders four 12,314-token choice requests, exceeding the former 8,192-token protocol. A complete audit found 12,314 tokens is the suite maximum, so the reviewed follow-up protocol uses a uniform 16,384-token context without truncation.
- A former full-group attempt exhausted GPU memory while producing prompt log-probabilities. The candidate-only retry therefore uses text-only loading, chunked prefill, and an explicit bounded KV allocation, with the suite maximum executed as a mandatory runtime gate before smoke.
- The revised preflight, 12,314-token runtime log-likelihood gate, and limited W8A8 smoke passed. MMLU-Pro was manually paused at 5,811/113,990 requests to prioritize interactive inference; there is no reportable accuracy score, and later resumption restarts that group from zero.
- A loopback-only vLLM server is running the candidate with the measured 64K CUDA-graph profile. Native Rust Goose and standalone Qwen Code both completed real local tool calls. Qwen Code fits the larger window but spends roughly 10K tokens on its fresh built-in prompt/tool envelope; Goose remains the lower-overhead interactive option.

See `reports/smoke-test-2026-08-24.md` for the measured gates and remaining boundary.
See `reports/evaluation-and-agent-status-2026-08-29.md` for the complete attempt ledger, dataset map, agent results, public W8A8 comparison, and next evaluation ladder.

## Next steps

The immediate next milestone is the complete candidate-only standardized accuracy run. Stop the interactive vLLM server so the evaluator has exclusive use of both GPUs, use the exact merged `main` commit, and launch `just eval-candidate-only <exact-main-commit>` in its labeled tmux session. The supervisor reruns preflight, the private 12,314-token maximum-request gate, and smoke before scoring.

The scored group order is MMLU-Pro from zero, followed by BBH, GPQA, MATH Level 5, IFEval, and MuSR. Each group is preserved atomically before the next begins. Do not report the old partial MMLU-Pro samples or combine them with the restart. The completed candidate-only aggregate may report W8A8 scores, but it cannot support a BF16-retention or deployment recommendation because the BF16 comparison is intentionally omitted.

After the standardized run, the next practical gates are pinned code-generation canaries, 64K context-retrieval quality, and small repository-agent tasks. Throughput and serving capacity are already measured; these follow-ups determine whether the model is useful and reliable at the advertised window.
