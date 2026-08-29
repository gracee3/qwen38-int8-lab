# Qwen3.8-27B W8A8 evaluation and local-agent status — 2026-08-29

## Executive status

The local artifact at `/data/models/Qwen3.8-27B-W8A8-INT8` is a
self-contained, calibrated Qwen3.8-27B W8A8 INT8 quality candidate. It has
passed checkpoint-integrity validation, native dual-RTX-3090 CUTLASS dispatch,
deterministic text generation, latency/throughput measurement, the complete
16,384-token request-render audit, the private 12,314-token runtime
log-likelihood gate, and a limited non-reportable leaderboard smoke.

It does **not** yet have a reportable standardized accuracy score. MMLU-Pro was
manually paused to prioritize interactive inference after 5,811 of 113,990
log-likelihood requests (5.0977%). The partial samples remain private and are
not an accuracy result. Resuming later requires restarting MMLU-Pro from zero.

The standing decision is:

```text
candidate_scores_only_no_retention_or_deployment_recommendation
```

No result here qualifies BF16 retention, multimodal behavior, thinking mode,
production use, safety, or context beyond 16,384 tokens.

## Artifact and quantization evidence

| Item | Recorded evidence |
| --- | --- |
| Source | Complete official Qwen3.8-27B BF16 checkpoint; 18 shards, 1,199 tensors, about 51.75 GiB |
| Candidate | `/data/models/Qwen3.8-27B-W8A8-INT8`; 36,798,458,455 bytes across 45 files |
| Calibration | `HuggingFaceH4/ultrachat_200k`, `train_sft`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, seed 42 |
| Quality profile | 512 sequences capped at 2,048 tokens; median 1,131.5; mean 1,193.254; maximum 2,048; no sample text logged |
| Quantized paths | Exactly 256 intended text projections: 192 MLP and 64 full-attention projections |
| Preserved paths | Embeddings, untied `lm_head`, all Gated DeltaNet/recurrent projections, vision tower, and all 15 MTP tensors remain BF16 |
| Format | `compressed-tensors`, static symmetric per-channel INT8 weights, dynamic symmetric per-token INT8 activations |
| Structural gates | Complete Safetensors index/shards, W8A8 metadata, 256 target count, 15 MTP tensors, processor-file hashes, no experimental marker |
| Native runtime | vLLM 0.27.1, TP2, both RTX 3090s, `CompressedTensorsW8A8Int8` using `CutlassInt8ScaledMMLinearKernel` |
| Determinism | Two identical seed-42 non-thinking requests returned the correct arithmetic result `391` |
| Performance | 1,153 input + 128 output tokens: median TTFT 0.6079 s, client-observed prefill 1,896.6 tok/s, decode 10.41 tok/s |

The successful quality calibration took 4,124.52 seconds. Its peak process RSS
was 83,402,051,584 bytes, process swap remained zero, and the safety supervisor
restored host swappiness and verified the protected NVMe, services, free space,
and idle GPUs after the run. See
[`quality-candidate-2026-08-25.md`](quality-candidate-2026-08-25.md) for the full
telemetry and artifact hashes.

## Quantization attempt ledger

1. The synthetic sequential smoke initially used `Linear` as both the modifier
   and sequential boundary. LLM Compressor's AST tracer failed closed with
   `KeyError: forward`. Retaining `Linear` as the quantization target while
   changing the sequential boundary to `Qwen3_5DecoderLayer` passed.
2. The complete BF16 load/trace gate passed: 64 decoder-layer targets became 65
   sequential subgraphs and all 256 intended modules were found without model
   output or disk offload.
3. The first real two-sample W8A8 run completed calibration and compression but
   was interrupted while serializing shard 4 of 10. Host swap growth remained
   above the 4 GiB guard for 10 seconds and reached 6.677 GiB. No final artifact
   was published.
4. A controlled retry temporarily set swappiness to 1, used 1 GB sharding, and
   passed quantization, serialization, MTP restoration, strict validation,
   native TP2 loading, and deterministic inference. It remained experimental
   because two multimodal processor files were absent from the artifact and had
   to be supplied through a read-only validation view.
5. The serializer was fixed to copy and verify those processor files before the
   atomic rename. Fresh tiny (2 × 128), small (32 × 512), and quality
   (512 × 2,048) stages all passed without retry, producing the current
   self-contained candidate.

The interrupted and experimental checkpoints were never promoted over the
quality candidate. Detailed records are in
[`real-source-gates-2026-08-24.md`](real-source-gates-2026-08-24.md) and
[`tiny-source-swap-retry-2026-08-24.md`](tiny-source-swap-retry-2026-08-24.md).

## Standardized-evaluation attempt ledger

All rows below are engineering gates or failures, not model scores.

| Run | Result |
| --- | --- |
| `20260825T023823Z` | Stopped during dataset prefetch because GPQA access had not yet been granted. |
| `20260825T031746Z` | GPQA access and all dataset pins passed. Request preflight caught GPQA Extended document 356 at 12,314 tokens versus the former 8,191-token request limit; no model was loaded. |
| `20260825T034510Z` | The 16K smoke stopped because root-owned shard permissions made `model-00001-of-00035.safetensors` unavailable inside the read-only mount. |
| `20260825T040045Z` | The 16K W8A8 smoke OOMed while prompt log-probabilities were being produced; the allocator tried to reserve 1.14 GiB with 956.31 MiB free. |
| `20260825T041803Z` | W8A8 smoke passed. The impractical BF16 smoke spent 983.59 seconds loading/profiling and was manually terminated; no scored group ran. |
| `20260825T100526Z` | Candidate-only smoke passed. MMLU-Pro then OOMed during prompt rendering/scheduling when the allocator requested 2.30 GiB with about 1.01 GiB free. |
| `20260829T161438Z` | The revised candidate-only preflight, maximum-request runtime gate, and smoke passed. MMLU-Pro reached 5,811/113,990 requests and was manually paused for interactive inference. |

The latest configuration removes the former vision and allocator pressure
without modifying the immutable multimodal checkpoint:

```text
language_model_only=true
enable_chunked_prefill=true
max_num_batched_tokens=1024
kv_cache_memory_bytes=805306368 per GPU
cpu_offload_gb=0
```

`gpu_memory_utilization` is intentionally absent when the explicit KV allocation
is active. The runtime gate observed 21,162 KV tokens, required at least 16,384,
executed the suite's 12,314-token maximum request with complete prompt
log-probabilities, confirmed text-only and chunked-prefill activation and native
CUTLASS W8A8 dispatch, and found no sample logging or OOM condition.

The pause record is private evidence at
`/data/qwen38-int8-lab/evaluations/20260829T161438Z/manual-pause.json`.

## Leaderboard datasets already pinned and rendered

The local protocol uses lm-evaluation-harness 0.4.12, TP2, batch size 1, seed
42, BF16 KV, a 16,384-token context, the official chat template, non-thinking
mode, and zero truncation. Preflight rendered 154,531 requests; the maximum was
12,314 tokens.

| Group | Dataset and evaluated documents | Headline metric | What it covers |
| --- | --- | --- | --- |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro`, 12,032 | `acc` | Broad difficult multidisciplinary knowledge and reasoning |
| BBH | `SaylorTwift/bbh`, 24 tasks, 5,761 | `acc_norm` | Algorithmic, symbolic, commonsense, and multi-step reasoning |
| GPQA | `Idavidrein/gpqa`, diamond/extended/main, 1,192 | `acc_norm` | Graduate-level science; gated rows and samples remain private |
| MATH Level 5 | `DigitalLearningGmbH/MATH-lighteval`, 7 subjects, 1,324 | `exact_match` | Difficult competition mathematics |
| IFEval | `wis-k/instruction-following-eval`, 541 | `prompt_level_strict_acc` | Verifiable instruction following |
| MuSR | `TAUR-Lab/MuSR`, 3 tasks, 756 | `acc_norm` | Long-form multi-step soft reasoning |

This is the Open LLM Leaderboard v2 task group under a locally qualified
protocol; it is not an official leaderboard submission.

## Useful next evaluations

The most efficient order is to add signal in layers rather than immediately
launch another multi-day suite.

1. **Fast language sanity:** run pinned subsets of ARC-Challenge, HellaSwag,
   Winogrande, GSM8K, and TruthfulQA. These are cheap regression indicators,
   not a complete quality claim.
2. **Code generation:** run HumanEval+ and MBPP+ with generated code isolated in
   disposable, network-disabled containers. EvalPlus expands the original test
   suites substantially and is a better basic code gate than a handful of chat
   prompts: <https://github.com/evalplus/evalplus>.
3. **Serving behavior:** measure TTFT, inter-token latency, decode throughput,
   and failure rate at 1K, 4K, 8K, 12K, and 16K input lengths with concurrency
   fixed at one. Record agent turn count and tool-call success separately from
   token speed.
4. **Context quality:** run RULER at 4K, 8K, and 16K before changing the runtime
   window. RULER includes retrieval, multi-hop tracing, aggregation, and QA
   rather than only a needle test: <https://github.com/NVIDIA/RULER>.
5. **Repository agents:** start with a deterministic five-instance
   SWE-bench Verified canary, then 25 instances, then the full 500 only if the
   harness/tool loop is stable. The official evaluator applies patches inside
   reproducible Docker environments: <https://github.com/SWE-bench/SWE-bench>.
6. **Terminal agents:** use a small pinned Terminal-Bench 2.x slice after the
   local shell/edit loop is reliable: <https://github.com/harbor-framework/terminal-bench>.

For quantization-specific diagnosis, a separate optional experiment can compare
W8A8 and BF16 log-probabilities on a pinned, non-calibration holdout and report
perplexity ratio, top-1 token agreement, and log-probability/KL deltas. That is a
diagnostic comparison only; it is not a BF16-retention decision and is outside
the current candidate-only execution scope.

## Interactive inference and coding agents

The current local server is bound only to `127.0.0.1:8000` and serves
`qwen38-w8a8` from the immutable candidate. It uses the same 16K text-only,
non-thinking, TP2, chunked-prefill, explicit-KV profile as the successful
runtime gate, plus the Qwen XML tool parser. Request-content and access logging
are disabled.

Three harness paths were tested:

| Harness | Result |
| --- | --- |
| Goose 1.48.0 | Native Rust CLI/TUI. A real read-only developer-tool request correctly listed the one root Markdown file in 6.67 s; TTFT 0.92 s. This is the recommended interactive harness. Tool mode is `approve`. |
| Qwen Code 0.22.3 | Standalone Qwen-native TUI. A real `glob` tool call correctly found `README.md`, but the three-call interaction took 114.18 s and consumed 31,618 aggregate tokens. Its standalone binary embeds Node internally; no system Node/npm was installed. |
| Codex CLI 0.151.0 | The vLLM Responses endpoint passed a direct 19-input/5-output-token request. Codex reached the endpoint, but its complete local prompt/tool/skill envelope exceeded the 16K server window at no fewer than 16,385 input tokens. Its model-catalog refresh also expects a different `/v1/models` schema. No Codex agent turn completed. |

Attach to the working TUIs with:

```bash
tmux attach -t qwen38-goose
tmux attach -t qwen38-qwen-code
```

The server is in `qwen38-agent-vllm`. Goose is the better fit for this host
today: it is Rust-native, supports vLLM Chat Completions directly, required much
less overhead in the observed tool test, and can require approval for commands.
Qwen Code remains useful as a Qwen-specific comparison.

Direct Codex remains experimental. Codex custom providers officially use the
Responses wire API, which this vLLM build provides, but current public Qwen3.x
reports also document `apply_patch` argument/schema failures even after endpoint
connectivity succeeds: <https://github.com/openai/codex/issues/33405>. Raising
the server window may clear the first-turn size blocker, but it would not by
itself prove reliable Codex editing.

## Public W8A8 comparison download

The strongest relevant public comparison found is
[`RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP`](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP),
pinned here to revision `d4680bb71d0369f3eacbeb2bf75cad9481125e7a`.

It is a 29.1 GiB compressed-tensors W8A8 build designed specifically for two
RTX 3090s. It quantizes the same 192 MLP and 64 full-attention projections as
the local candidate and additionally quantizes 144 Gated DeltaNet linear
projections; vision, MTP, `lm_head`, and selected recurrent paths remain BF16.
Its 512 roughly 2K-token calibration sequences came from public Python code,
with about half containing tool calls. The author reports MTP throughput and
very long FP8-KV configurations, but explicitly reports no MMLU or GSM8K
accuracy comparison against the base model.

The exact public revision is downloading to
`/data/models/RukaRat-Qwen3.8-27B-INT8-W8A8-imatrix-MTP` in tmux session
`qwen38-public-w8a8-download`. It is a comparison candidate, not a validated
replacement for the locally calibrated artifact.

## Repository validation

The documentation and evaluator were revalidated from this branch on
2026-08-29. Python compilation, Bash syntax, ShellCheck, YAML parsing, the
Justfile dry-runs for candidate-only evaluation, serving, and quantization, and
the evaluation image's `pip check` all passed. All 15 unit tests passed in the
pinned evaluation image. The host-only test invocation passed 12 tests and
skipped the three NumPy-dependent aggregation tests because NumPy is
intentionally absent from the host Python; the container invocation exercised
those three tests successfully without changing the host environment.

## Capacity boundary

The official BF16 model card describes a native 262,144-token model and reports
strong agentic results, but its published coding evaluations use thinking mode,
256K context, and a Claude Code harness. Those numbers cannot be transferred to
this conservative non-thinking 16K W8A8 runtime.

The local evidence currently supports only:

- text-only, non-thinking inference through 16,384 tokens;
- exact execution of the suite's 12,314-token maximum request;
- one successful tool-using Qwen Code turn and one successful Goose tool turn;
- no claim of state-of-the-art or long-horizon coding performance.

The observed KV capacity of 21,162 tokens leaves room for a separately labeled
20K engineering experiment, but not for a credible long-horizon claim. Longer
windows require a new memory protocol (for example, a validated lower-precision
KV configuration), request-render checks, context-quality evaluation, and
agent-loop testing before they can be reported.
