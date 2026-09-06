# Qwen3.8-27B Expanded-400 INT4 v1 — build handoff

Date: 2026-09-06  
Status: build specification; no INT4 quantization or runtime validation has been performed.  
Repository: `gracee3/qwen38-int8-lab`  
Suggested implementation branch: `checkpoint/expanded-400-int4-v1`

## 1. Objective and instructions to the implementing agent

Build a separate GPTQ W4A16 checkpoint directly from the official Qwen3.8-27B BF16 source. Quantize exactly 400 logical text projections, using the expanded target scope already proposed for agentic W8A8 v2. Serve the resulting model with vLLM and Marlin on one RTX 3090, leaving the other GPU available for quantization or validation work.

Implement, pilot, build, and validate this candidate using the host and repository access available to you. Inspect applicable repository instructions and current work before editing. Preserve the original BF16 source, existing W8A8 checkpoint, and v2 INT8 work. Use a separate branch/worktree and distinct output paths. Do not overwrite an existing output or change the working TP2 service defaults. An active service requiring interruption should be coordinated with the user unless the current session already authorizes that interruption.

The initial serving target is TP1, 32,768 total context tokens, one active sequence, text-only, FP8 KV, and no MTP. Start runtime diagnosis at 16,384 tokens if needed. A 65,536-token profile is a later capacity experiment, not a required v1 outcome. Do not attempt to retain the 262K TP2 window as a v1 requirement.

Success means a complete, reproducible checkpoint, verified intended kernel execution, measured single-card fit, and useful bounded coding/terminal/tool behavior. Startup alone is insufficient.

## 2. Source, paths, and isolation

| Item | Value / action |
| --- | --- |
| Official source | `Qwen/Qwen3.8-27B`; pin revision and hash local source files |
| Known host source | `/home/emmy/workspace/qwen3.8-27b-download/model` — verify, mount read-only |
| Existing repository | `/home/emmy/projects/qwen38-int8-lab` — verify current checkout |
| New final output | `/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1` |
| Container output | `/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1` |
| Working data | `/data/qwen38-int8-lab/` with dedicated INT4-v1 calibration, results, and scratch subdirectories |
| Serving name | `qwen38-27b-expanded400-int4-v1` |
| Model family | `Qwen3_5ForConditionalGeneration`; 64 text layers, 48 GDN and 16 full-attention |

Historical source inventory: 18 Safetensors shards, 1,199 tensors, 55,563,006,776 bytes of shard files. Treat pinned local tensor headers as authoritative if these differ. Do not re-quantize an INT8 checkpoint. Do not download another full source copy if the existing official source passes integrity checks.

Keep weights, datasets, raw task outputs, caches, and large logs outside Git. Commit recipes, target manifests without weight payloads, scripts, dependency locks, and compact reviewed reports.

Bind the serving container to exactly one physical GPU by UUID where supported; record the UUID-to-index mapping. The device may be renumbered to GPU 0 inside the container. Verify the process allocates no memory on the other card. Apply explicit device assignment to calibration/evaluation processes too; TP1 by itself does not isolate other processes.

## 3. Exact logical target scope

Resolve fully qualified names from the source and instantiated modules. Match explicit text decoder paths, not an unrestricted `Linear` or broad attention regex. Expand a manifest before quantization and require the exact expected names, dimensions, and count.

Expected matrix shape is `(output, input)`:

| Family | Shape | Count |
| --- | --- | ---: |
| Text MLP `gate_proj` | `(17408, 5120)` | 64 |
| Text MLP `up_proj` | `(17408, 5120)` | 64 |
| Text MLP `down_proj` | `(5120, 17408)` | 64 |
| Full-attention `q_proj` including output gate | `(12288, 5120)` | 16 |
| Full-attention `k_proj` | `(1024, 5120)` | 16 |
| Full-attention `v_proj` | `(1024, 5120)` | 16 |
| Full-attention `o_proj` | `(5120, 6144)` | 16 |
| GDN `linear_attn.in_proj_qkv` | `(10240, 5120)` | 48 |
| GDN `linear_attn.in_proj_z` | `(6144, 5120)` | 48 |
| GDN `linear_attn.out_proj` | `(5120, 6144)` | 48 |
| **Total** | | **400** |

The original 256 targets contain 18,790,481,920 weights. The additional 144 GDN targets contain 5,536,481,280 weights. Total intended quantized weights: **24,326,963,200**.

Preserve the source precision of embeddings, untied `lm_head`, norms, GDN `in_proj_a` and `in_proj_b`, convolutions, recurrent dynamics/control parameters, vision components, and MTP tensors. Leave runtime recurrent-state precision unchanged. This experiment quantizes the large projections around recurrent processing, not the recurrent state itself.

Retain all 15 source MTP tensors and processor files under the existing complete-checkpoint policy. No MTP execution is allowed in the v1 serving profile. Text-only/no-MTP loading must demonstrably skip the unused components; removing them from the artifact is unnecessary.

## 4. Quantization recipe and compatibility pilot

| Setting | Initial v1 choice |
| --- | --- |
| Algorithm | Calibrated GPTQ, from original BF16 |
| Weights | Symmetric INT4, groupwise |
| Weight group size | 128 along the input dimension |
| Activations | 16-bit floating point; begin with BF16 if supported by the pinned Marlin path |
| GPTQ processing block size | 128, a separate algorithm setting from weight group size |
| Dampening fraction | 0.01 |
| Activation ordering | Disabled initially (`desc_act=false` or supported equivalent) |
| SmoothQuant | Disabled; do not introduce generic zero-centered RMSNorm mappings |
| Sequential boundary | `Qwen3_5DecoderLayer` |
| Calibration seed | 42 |
| Format preference | Supported `compressed-tensors` W4A16 / Safetensors, reusing existing infrastructure where compatible |
| Alternative format | Standard GPTQ packing only if the preferred format cannot correctly use the intended Marlin path; document the change |

These are implementation choices to verify against installed APIs, not a claim that the existing W8A8 configuration can be changed by setting `bits=4` alone.

Before a full build, prove that the chosen quantizer visits every target family and the pinned vLLM release supports their shapes, packed storage, fused projections, and BF16 activation execution. In particular, inspect how vLLM combines GDN projections and whether excluded `a`/`b` paths remain correctly represented when adjacent projections are quantized. Do not infer compatibility from generic Ampere INT4 support.

Marlin W4A16 stores packed INT4 weights and dequantizes tiles for floating-point tensor-core multiplication. It is GPU-accelerated weight-only quantization, not the current native INT8 W8A8 arithmetic. Never report CUTLASS INT8 dispatch as proof of the new INT4 path.

If BF16 activations are unsupported in the selected kernel, report the limitation and evaluate FP16 as an explicitly separate runtime choice, including overflow/numerical checks. Do not silently relabel a fallback as the planned profile. Runtime casts do not justify modifying preserved checkpoint tensors.

## 5. Reuse the v2 calibration strategy

Use the same frozen public corpus as v2 if it has already been prepared and passes provenance/overlap checks. Otherwise implement the v2 sampling specification once and make it reusable by both builds. A different calibration corpus is not needed simply because weights are four-bit, but quality must be assessed separately.

| Dataset | Target share of actual rendered tokens |
| --- | ---: |
| `togethercomputer/CoderForge-Preview` | 35% |
| `nvidia/Nemotron-Terminal-Corpus` | 30% |
| `Fortytwo-Network/Strandset-Rust-v1` | 25% |
| `nvidia/Nemotron-Agentic-v1` | 10% |

Target 1.5 million actual rendered, non-padding tokens; acceptable first-run range 1.3–1.7 million. Prefer coherent 2K–4K sequences, with approximately 25% of tokens in 8K–16K sequences if the host memory pilot supports them. Pin revisions, splits, row/task IDs, rendering, tokenizer, order, seed, fingerprints, and achieved token proportions. Inspect actual schemas and dataset terms before using them.

Use existing public data only. Preserve valid tool-call/result associations and useful failed-command/recovery sequences. Exclude separately provided hidden-reasoning fields for the initial non-thinking arm. Do not invent missing tool outputs or collect new private user/Qwen sessions. Deduplicate tasks and near-duplicates; remove overlap with held-out evaluations. Do not confuse GPTQ calibration with fine-tuning or claim that it calibrates FP8 KV scales.

Pilot short samples and the longest planned sequence before the full build. Reuse host RAM/swap safeguards from the repository, including its documented sustained low-available-RAM and swap-growth limits. Measure actual peaks; a smaller output does not guarantee a cheaper GPTQ build.

## 6. Calculated memory budget

All values below are GiB (2^30 bytes), not decimal GB. These are architecture-derived estimates pending a local tensor audit and actual vLLM measurement.

Assumption for a quantized matrix with N weights: packed weight bytes = `N/2`; BF16 scales at group size 128 add approximately `2*N/128`. Format-specific zero-point storage, packing/alignment, metadata, and runtime repacking are additional. Source file size includes headers and is only an approximate baseline.

| Recipe | Estimated complete checkpoint |
| --- | ---: |
| Original 256 targets, INT8 | 34.3 GiB |
| Expanded 400 targets, INT8 | 29.1 GiB |
| Original 256 targets, INT4 with scales | 25.8 GiB |
| **Expanded 400 targets, INT4 with scales** | **18.1 GiB** |

Expanded INT4 calculated text-only weight inventory:

| Component | GiB |
| --- | ---: |
| Original 256 targets, packed INT4 | 8.750 |
| Additional 144 GDN targets, packed INT4 | 2.578 |
| Untied embeddings and output head, BF16 | 4.736 |
| Group scales for all 400 targets | 0.354 |
| Remaining small text tensors | ~0.049 |
| **Text-model weight subtotal** | **~16.47** |

Relative to conservative INT4, quantizing the added GDN targets saves approximately **7.65 GiB net of these scales**. This is the main reason to attempt the expanded profile. Conservative INT4 text weights alone are approximately 24.1 GiB and are not a practical all-GPU 3090 solution.

For TP1, 16 full-attention layers, 4 KV heads, head dimension 256, and FP8 K/V, the raw attention cache payload is `16 * 2 * 4 * 256 = 32,768 bytes/token`:

| Total window | Raw FP8 attention KV payload |
| --- | ---: |
| 16,384 | 0.5 GiB |
| 32,768 | 1.0 GiB |
| 65,536 | 2.0 GiB |

This is not vLLM's total cache allocation. Include recurrent-state storage/caching, hybrid allocator padding, prefix-cache policy, graph capture, activations, CUDA context, Marlin workspace/repacking, and transient load/prefill peaks. One FP32 GDN state per recurrent layer/sequence is roughly 0.14 GiB in aggregate, but runtime caching may retain more than one state. Do not budget that figure as the total hybrid state allocation.

At 32K, the ~16.5 GiB text weights plus ~1 GiB raw attention KV leave roughly 6.5 GiB on a nominal 24 GiB GPU for everything else. Treat this as a promising budget, not a fit guarantee. Report allocated, reserved, process GPU memory, free VRAM, and peak memory separately. Verify startup, cold prefill, near-window generation, and repeated requests.

## 7. Serving profile

| Setting | v1 target |
| --- | --- |
| Runtime | Pinned vLLM build with verified Marlin W4A16 dispatch |
| Device | One RTX 3090 / SM86, explicit GPU isolation |
| TP / PP | 1 / 1 |
| Loading | Text-only; no CPU offload for the passing profile |
| Context | 32,768 total input + output tokens |
| Concurrency | One sequence |
| KV | FP8; record subtype, backend, scales and fallback behavior |
| MTP/speculation | Disabled |
| Thinking | Non-thinking initial comparison, supported source template |
| Chunked prefill | Begin at 2,048 tokens; measure and tune separately |
| CUDA graphs | Enable after eager diagnosis passes; measure added memory |
| Prefix caching | Match intended agent workflow; record effect on hybrid state/cache capacity |
| API | Existing authenticated loopback pattern, distinct available port |

Capture the exact working host FP8 configuration before adapting it. FP8 cache scale behavior is separate from weight quantization. Do not assume a historical scale-1 fallback is quality-equivalent to calibrated scales. Keep tool parser, chat template, sampling, and agent scaffold fixed for matched comparisons. Reserve output space, e.g. at most 28,672 rendered input tokens plus 4,096 output tokens in the 32K profile.

The launch command must be generated and tested against the pinned runtime. This specification intentionally does not provide an unverified copy-paste command for the fused 400-target checkpoint.

## 8. Execution and validation gates

1. **Audit:** inspect current repo instructions/work; pin source and dependencies; enumerate 400 logical targets and preserved tensors; recompute bytes from headers. Publish a machine-readable target manifest and size report before quantization.
2. **Compatibility pilot:** use a small synthetic architecture-faithful model to exercise full-attention and GDN target families, GPTQ packing, serializer, vLLM loader, and actual Marlin dispatch. Then run a small real-source calibration pilot and the longest-sequence memory pilot. Stop on missing targets, incompatible fusion, or silent fallback.
3. **Implementation:** add dedicated INT4 configuration, output paths, validators, launch profile, and reporting. Retain existing 256/400 INT8 validators and defaults. Logical source targets may fuse into fewer runtime modules; validate the mapping rather than demanding 400 runtime modules.
4. **Build:** quantize from BF16 with the frozen corpus. Use the existing staged serialization and atomic final rename; retain 1 GB sharding where supported. Write incomplete artifacts only to disposable, distinctly named scratch locations.
5. **Integrity:** validate index/shards, source-to-output mapping, exactly 400 intended logical targets, INT4 packing/scales, preserved tensor hashes, all 15 MTP tensors, and processor files. Record exceptions explicitly; do not weaken completeness checks to make a load pass.
6. **Runtime:** load on one GPU, verify Marlin dispatch for all eligible target families including fused GDN paths, and confirm no unintended high-precision expanded weight copies or CPU offload. Measure both startup and steady-state memory. Confirm the other GPU is untouched by serving.
7. **Capacity:** test 16K and 32K actual rendered windows with output reservation, cold prefill, continued generation, repeated requests, and the intended prefix-cache/graph settings. Record OOMs, resets, NaNs/infinities, and malformed outputs. A max-length flag alone is not a capacity result.
8. **Quality:** compare against existing W8A8 at common context length with the same prompts, parser, template, sampling, and task budgets. Include deterministic instruction following, tool argument/ID correctness, failed-command recovery, small script/config changes, compilation/tests, and multi-turn continuation. Use held-out cases and record task outcomes rather than relying on fluent text.
9. **Soak and report:** run a 1–2 hour representative agent session/sequence of tasks, then report weights, memory, throughput, latency, quality, and limitations. Promote only the profile actually demonstrated.

Recommended evaluation reuse: MultiPL-E HumanEval Rust, EvalPlus HumanEval+, the fixed 100-case IFEval subset, and initial 10-task repository plus 10-task terminal canaries from the v2 plan. Use the existing 30-minute / 40-turn / 32K aggregate-generated-token agent budget. Validate suite prerequisites and report subset denominators. Full task suites are follow-on evidence; do not present a small canary as general BF16 quality retention.

For the initial pass decision, require no structural/kernel/isolation failures, repeatable 32K capacity, and no unexplained repeatable tool-schema or instruction-following regression on the frozen canary set. Publish paired changed outcomes; there is no pre-established numerical quality-retention guarantee. If 32K fails but 16K passes, report a 16K candidate rather than silently claiming the 32K target.

## 9. Failure handling and deliverables

If 400-target Marlin support fails, identify the exact family, runtime module, dtype, shape, or packing constraint. Do not silently omit targets, expand weights to BF16, switch to GGUF, or use both GPUs while labeling the result a passing v1. Propose a separately named follow-up candidate if needed. If quality regresses, isolate added projection families using the frozen corpus; a reduced-scope candidate is a different experiment.

Required handoff results from the implementing agent:

- Commit/branch, dependency and source revisions, build commands, and tested serve/stop commands.
- Explicit source and fused-runtime target mapping, preserved tensor checks, quantization metadata, and recalculated versus actual sizes.
- Immutable final checkpoint location and complete build provenance; existing models remain available.
- Measured single-GPU load/prefill/decode/soak memory and evidence of intended kernel execution.
- Cold/warm TTFT, decode tokens/sec, power settings, and task outcomes, with comparable conditions clearly identified.
- Highest tested context, known limitations, failed gates, and a concise promotion recommendation.

Do not promise a build duration from inference speed or the old W8A8 run. Measure the pilot and forecast from observed GPTQ, serialization, loading, and evaluation costs.

## 10. Reference material

- [Existing v2 build specification and expanded target rationale](https://github.com/gracee3/qwen38-int8-lab/blob/checkpoint/agentic-w8a8-v2/AGENTIC-W8A8-V2.md). File blob observed during this handoff: `689df782085a7147d05a51bb970d395ac1ab7f52`; this is a file blob hash, not a repository commit. Resolve and pin the branch commit before implementation.
- [Repository README](https://github.com/gracee3/qwen38-int8-lab/blob/main/README.md).
- [Official Qwen3.8-27B source](https://huggingface.co/Qwen/Qwen3.8-27B) and [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
- [vLLM W4A16 documentation](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int4/).
- [Marlin kernel description](https://github.com/IST-DASLab/marlin).

These links establish source dimensions, project intent, and general kernel capability. They do not establish that this exact expanded-400 Qwen3.8 checkpoint has been built or validated. The local source audit, compatibility pilot, and measured gates provide that evidence.
