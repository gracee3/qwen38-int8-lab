# Agentic W8A8 v2 checkpoint recipe and evaluation plan

Status: proposed build specification, captured 2026-09-06.
Branch: `checkpoint/agentic-w8a8-v2`.

This document captures the next checkpoint's calibration, serving, evaluation,
and implementation requirements. It is not an executable configuration and
does not change the existing quantizer or launch a model. Dataset revisions,
selected rows, harness revisions, and host runtime settings must be resolved
and recorded before execution; none are represented as already validated.

## Goal and accepted decisions

Build anew from the original BF16 Qwen3.8-27B source for general agentic coding,
Rust, terminal work, and systems engineering through Qwen Code and vLLM.

- Existing public datasets only. No collection of new Qwen Code sessions,
  private conversations, or user repository examples is required.
- GPTQ W8A8: symmetric per-channel INT8 weights and dynamic per-token INT8
  activations, compressed-tensors / Safetensors.
- Two RTX 3090s, tensor parallelism 2; no pipeline parallelism.
- FP8 KV cache. User reports this is working on the host now.
- No MTP/speculative decoding.
- Requested native context target: 262,144 total tokens, including rendered
  instructions, tools, input, and generated output.
- Keep the existing checkpoint available for matched comparisons.

The repository README records a 163,840-token FP8 profile and a 150,037-token
retrieval probe. These are historical evidence, not a rejection of the user's
current working FP8 setup. Capture the actual working host command, image,
backend, FP8 format, scale behavior, and memory allocation before deriving
the new 262,144-token profile. The larger target still needs capacity and
quality verification. Do not add RoPE scaling to claim a native window;
verify the source configuration and runtime interpretation.

“No MTP” means no speculative runtime, draft model, or MTP generation.
Retain the 15 source MTP tensors for complete-checkpoint integrity under the
existing serializer and validator. Removing those tensors would be a separate
artifact-format change and is not part of this recipe.

## Quantization settings

Start from `quant/config/qwen38-27b.yaml` and retain:

| Setting | v2 initial choice |
| --- | --- |
| Source | Original BF16 source, mounted read-only |
| Quantized targets | Existing 256 projections: 192 MLP and 64 full-attention |
| Preserved precision | Embeddings, output head, norms, vision, recurrent/GDN paths, MTP tensors |
| GPTQ block size | 128; algorithm processing block, not a weight quantization group size |
| Dampening fraction | 0.01 |
| Sequential boundary | Qwen3_5DecoderLayer |
| SmoothQuant | Disabled; existing zero-centered norm compatibility concern remains |
| Serialization | Existing 1 GB shards, staging, atomic rename, processor preservation |
| Output | New host path: /data/models/Qwen3.8-27B-W8A8-INT8-Agentic-v2 |
| Container output | /models/Qwen3.8-27B-W8A8-INT8-Agentic-v2 |

Do not quantize an existing quant. Do not overwrite either the source or the
current checkpoint. Retain current RAM/swap safeguards and record runtime
versions, source hashes, recipe hash, and peak memory.

## Expanded recurrent-projection candidate

Added 2026-09-06: capture a separate v2-expanded candidate alongside the
conservative v2. This is a proposed experiment, not an executed quantization
or a claim of validated quality.

Both candidates start independently from the original BF16 source and use
the same frozen public calibration corpus, token order, seed, GPTQ settings,
and evaluation cases. Keep TP2, FP8 KV, no speculation/MTP execution, and the
262,144-token target unchanged. This isolates the additional layer targets
from the calibration-data change.

| Candidate | Targets | Distinct host output |
| --- | ---: | --- |
| v2 conservative | 256 existing projections | /data/models/Qwen3.8-27B-W8A8-INT8-Agentic-v2 |
| v2 expanded | 400: existing 256 plus 144 recurrent projections | /data/models/Qwen3.8-27B-W8A8-INT8-Agentic-v2-Expanded |

### Target scope and estimated savings

Add the following large Linear projections in each of the 48 recurrent
blocks. Confirm names and dimensions against the immutable local checkpoint
and instantiated modules before constructing an explicit target manifest.

| Projection family | Expected matrix shape per block (output, input) | Additional matrices | Estimated BF16-to-INT8 savings |
| --- | --- | ---: | ---: |
| linear_attn.in_proj_qkv | (10240, 5120) | 48 | 2.34375 GiB |
| linear_attn.in_proj_z | (6144, 5120) | 48 | 1.40625 GiB |
| linear_attn.out_proj | (5120, 6144) | 48 | 1.40625 GiB |
| Total | — | 144 | 5.15625 GiB |

Calculation: sum of matrix element counts times (2 - 1) bytes, divided by
2^30. Shapes are architecture-derived estimates pending local tensor audit;
INT8 scales and metadata slightly reduce the net saving.

The historical artifact is approximately 34.27 GiB; the equivalent expanded
artifact is estimated at approximately 29.1 GiB. Expected loaded weight
savings under balanced TP2 are roughly 2.58 GiB per GPU. These are not measured
VRAM results: verify sharding, quantized/fused storage, workspaces, and actual
post-load memory. Do not convert the estimate directly into a promised
context capacity or throughput increase.

Replace the blanket recurrent exclusion with precise matching for these
three families, rather than allowing every recurrent component to quantize.
Continue preserving in_proj_a, in_proj_b, convolutions, normalization,
recurrent dynamics parameters, embeddings, output head, vision, and MTP.
Leave runtime recurrent-state precision unchanged. Quantizing projections
does not imply quantizing the recurrent state or the FP8 KV scales.

### Implementation and validation

1. Audit actual source tensor names, shapes, dtypes, and bytes. Produce
   predicted savings and explicit per-candidate target manifests.
2. Verify installed GPTQ calibration hooks visit every intended module and
   retain Qwen3_5DecoderLayer sequential boundaries. Pilot the added families
   before a full expanded build.
3. Implement candidate-specific serializer/validator expectations: exactly
   256 targets for conservative and 400 for expanded, plus exact names and
   shapes. Do not relax the shared validator to accept arbitrary counts.
4. Verify preserved tensor hashes, processor files, shard/index integrity,
   scale metadata, and INT8 storage for each added family.
5. Verify vLLM compatibility for fused projections and TP2 partitioning.
   Confirm intended native dispatch rather than inferring it from a successful
   load or silently accepting higher-precision fallback.
6. Compare conservative and expanded on the same function, repository, terminal,
   tool-use, and context suites. Include long prefill, continued generation,
   and multi-turn failure/recovery sequences because recurrent projection
   errors can affect information carried through subsequent tokens.
7. Record per-GPU weight/total memory, capacity, latency, and paired quality
   outcomes. Preserve both candidates until promotion is justified.

If expanded quality regresses, isolate projection families using the same
corpus. An out_proj-only arm adds 48 targets (304 total) and saves about
1.41 GiB overall. This is a convenient smaller experiment, not a proven
ordering of sensitivity. No intermediate arms are required if the full
expanded candidate is sufficiently validated.

### Public precedent and evidence limits

- [RukaRat Qwen3.8 W8A8](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP):
  the repository's prior audit recorded 144 additional recurrent projections
  and an approximately 29.1 GiB artifact. The audited revision was
  d4680bb71d0369f3eacbeb2bf75cad9481125e7a; see
  reports/evaluation-and-agent-status-2026-08-29.md.
- Its [published configuration](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP/blob/main/config.json)
  uses per-channel INT8 weights, dynamic per-token INT8 activations, and an
  imatrix-mse observer. This is relevant targeting/format precedent, not the
  same algorithm as our GPTQ recipe. Pin and inspect the actual revision
  before borrowing implementation details; main is mutable.
- No controlled BF16 quality-retention result was recorded in that audit.
  Working serving examples or speed claims do not establish Rust/agentic
  quality or long-context retention for our candidate. Its MTP claims do not
  apply to this no-MTP profile.
- [Official Qwen3.5-27B GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4)
  is related-architecture precedent, not validation of Qwen3.8 W8A8.
- [Qwen3.8 source configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
  provides the dimensions used for these estimates; local pinned tensor
  headers are authoritative for the build.

### Other preserved components

Embedding and untied output-head matrices each contain approximately
248320 * 5120 weights; BF16-to-INT8 would theoretically save about 1.18 GiB
per matrix before overhead. Leave both unchanged in these candidates:
embedding lookup requires supported quantized storage/execution, while the
output head directly affects token scores.

Compressing or removing vision/MTP tensors may reduce disk size without
material live VRAM savings when those components are already excluded from
text-only/no-MTP serving. Small control tensors offer little savings relative
to the large projections above.

### Additional time

The earlier 2–5 hour quantization estimate applies to the conservative build
and is not a measured forecast for 400 targets. The expanded candidate adds
a second source build, more GPTQ work, and its own runtime/evaluation passes.
Measure its pilot; do not scale time simply by 400/256 because matrix sizes,
Hessian costs, loading, and serialization differ. Reuse frozen data and
evaluation setup. For 20 agent tasks at a 30-minute cap, one additional
candidate alone adds up to 10 hours of task runtime, excluding setup.

## Calibration sources

Proposed token-weighted mixture; these proportions are an experimental
starting point, not a published optimum.

| Dataset | Share of actual rendered tokens | Target at 1.5M tokens | Selection |
| --- | ---: | ---: | --- |
| togethercomputer/CoderForge-Preview | 35% | 525,000 | Diverse repository investigation, edits, test feedback, and repair trajectories |
| nvidia/Nemotron-Terminal-Corpus | 30% | 450,000 | Software engineering, debugging, shell and relevant Linux workflows |
| Fortytwo-Network/Strandset-Rust-v1 | 25% | 375,000 | Diverse Rust tasks, balanced by task category and crate |
| nvidia/Nemotron-Agentic-v1 | 10% | 150,000 | Multi-turn tool selection, tool outputs, and goal completion |

Sources:
- https://huggingface.co/datasets/togethercomputer/CoderForge-Preview
- https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus
- https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1
- https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1

Use training-designated material where available; inspect each actual schema
and split before assigning it. Pin immutable revisions, subset/split names,
upstream task IDs, row IDs, source attribution, and applicable dataset terms.
Do not assume all four datasets load with the same message schema.

CoderForge is skewed toward bug fixing and uses its own agent scaffold.
Terminal-Corpus includes adapted upstream tasks; audit their origin before
selecting evaluation cases. Strandset is synthetic and requires sample
inspection; publisher validation is not a guarantee of every example.
These datasets' fine-tuning gains are not predictions for GPTQ calibration.

For terminal and repository sources, seek available examples of build and
dependency failures, shell scripting, logs, processes, filesystems,
configuration, networking, and data processing. Coverage of Kubernetes,
Ansible, Prometheus, or other named products has not been verified. Report
coverage actually found; do not label generic terminal data as product-specific.

Rust coverage should include ownership, lifetimes, traits, generics, errors,
async/cancellation, channels, shared state, serialization, subprocesses, and
tests. Balance by available data rather than manufacturing new examples.

## Sampling, lengths, and rendering

- Seed: 42. Target 1,500,000 actual rendered non-padding tokens, acceptable
  first-run range 1.3M–1.7M. Report achieved source shares; aim within 3
  percentage points of each target and explain deviations.
- Count system text and tool definitions in the budget. Do not use row count
  or maximum-length multiplication as the claimed token total.
- Prefer 2K–4K coherent examples. Allocate approximately 25% of the token
  budget to 8K–16K examples when the source data and memory pilot support it.
  This length allocation is independent of the source mixture.
- Use existing complete short trajectories or coherent segments with the
  required preceding context. Do not invent missing tool outputs, summaries,
  or synthetic Qwen sessions to repair an incomplete segment.
- Preserve useful failed-command / correction sequences, valid tool-call
  IDs, roles, arguments, and result associations.
- Normalize to the source model's supported chat/tool template; validate the
  resulting tokenization and semantics. Do not blindly rename tools to Qwen
  Code tools whose arguments or behavior differ.
- No requirement to repeat Qwen Code's entire system prompt in every sample.
  Measure repeated-prefix contribution; prevent boilerplate and large
  repetitive logs from dominating the corpus.
- Do not silently truncate the beginning of every trajectory. Select examples
  that fit, or explicit valid segments; reject broken tool pairs and empty
  content. Retain untruncated selection metadata.
- Deduplicate exact and near-duplicate task trajectories and code. Limit
  repeated snapshots and concentration from one task/repository.
- Match the intended thinking setting. Initial comparison inherits the
  existing non-thinking profile. Datasets with separate reasoning fields
  need explicit handling: exclude separate hidden-reasoning fields for this
  arm while retaining ordinary explanations, actions, and results. A
  thinking-enabled arm is a later separate experiment.
- Save dataset locks, selected IDs, token counts, length histograms, coverage,
  exclusion reasons, and hashes. Keep corpus payloads outside Git under
  /data/qwen38-int8-lab/calibration/agentic-v2/.

Implement mixed-source and variable-length support explicitly. Do not assume
the current single-dataset quality profile accepts this specification.

## Serving profile

| Setting | Requested/default choice |
| --- | --- |
| Serving | vLLM, text-only, native CUTLASS W8A8 |
| TP / PP | 2 / 1 |
| KV | FP8; inherit and record the exact working host format/backend/scales |
| Context | 262,144 total tokens |
| Speculation | Disabled, no MTP |
| Concurrency | One sequence initially |
| Prefix caching | Enabled; benchmark cold and warm separately |
| CUDA graphs | Enabled if validated with the new capacity |
| Chunked prefill | Start from the working host profile; tune independently |
| Thinking | Non-thinking initial comparison; separately evaluate any change |
| Power cap | Record host value and hold equal across comparisons |
| Endpoint | Existing authenticated loopback setup |

The GPTQ weight calibration does not itself calibrate FP8 KV scales. Record
the actual FP8 subtype, scale source (checkpoint, measured, or fallback),
attention backend, and relevant versions. If the working profile uses
scale 1.0, label that fact rather than calling it calibrated FP8 KV.

Measure explicit KV allocation per GPU and actual capacity after model load,
graphs, and recurrent state allocations. Do not assume a max-model-len flag
or the old 3 GiB allocation establishes 262K capacity. Reserve output space:
for example 258,048 rendered input tokens plus 4,096 output tokens is the
262,144 limit, not 262,144 input plus output.

Pin Qwen Code, tool schema/parser, system prompt, template, sampling settings,
and compaction behavior for agent evaluations. Record actual context used.
Compaction may keep an agent well below the server limit; an agent task is
not a full-window test unless its measured input demonstrates that.

## Evaluation plan

Freeze evaluation IDs and upstream provenance before calibration selection.
Remove task-level and near-duplicate overlap, including tasks transformed by
Terminal-Corpus or appearing in CoderForge. Prefer disjoint repositories for
repository evaluations. Publish subset IDs and denominator; never call a
selected subset the full official benchmark.

Compare existing W8A8 versus v2 under identical TP2/FP8/no-MTP settings at
common supported lengths. A separate shorter BF16-KV arm can diagnose KV
effects; do not conflate a cache change with a weight calibration improvement.
A BF16-weight comparison is optional and requires feasible hardware. Without
it, results compare two quants and do not establish BF16 quality retention.

| Stage | Evaluation | Initial scope | Measurements |
| --- | --- | --- | --- |
| 0 | Integrity/native dispatch | Existing full checkpoint gates | Shards, metadata, candidate-specific 256/400 target manifest, preserved hashes, processors, CUTLASS dispatch |
| 1 | MultiPL-E HumanEval Rust | Full Rust split at pinned revision | Compile rate, pass@1, runtime errors, generation time |
| 1 | EvalPlus HumanEval+ | Full pinned suite | pass@1; broader Python coding regression |
| 1 optional | EvalPlus MBPP+ | After primary function results | pass@1 |
| 2 | IFEval | 100 fixed held-out examples | Strict prompt-level score, explicit subset label |
| 3 | Multi-SWE-bench | 10 tasks initially, expand to 20–25; include Rust and other languages | Resolved fraction, tests, patch correctness, turns, tool failures, time |
| 3 | Terminal-Bench | 10 engineering tasks initially, expand to 20 | Task success, command/tool failures, recovery, elapsed time |
| 4 | RULER subset | Selected retrieval, tracing, and aggregation families; 10 cases per length initially | Per-family and per-length scores |
| 4 | Context-dependent repository tasks | Selected existing tasks with sufficient genuine supplied context | Correct references/edits/tests, missed constraints; label any adapted protocol |
| 5 | Serving performance and soak | Existing cold/warm suite plus long-input tests; 1–2 hour agent soak | TTFT, inter-token latency, decode rate, errors, memory, power, resets |

Evaluation sources:
- https://github.com/nuprl/MultiPL-E
- https://github.com/evalplus/evalplus
- https://github.com/multi-swe-bench/multi-swe-bench
- https://github.com/harbor-framework/terminal-bench
- https://github.com/google-research/google-research/tree/master/instruction_following_eval
- https://github.com/NVIDIA/RULER

Code-generation evaluations use their pinned generation protocols. Repository
and terminal evaluations use Qwen Code against vLLM and the benchmark's
evaluator, with an adapter if needed. Record scaffold deviations; do not
compare adapted runs to published scores as though the protocols match.
Use reproducible disposable execution environments for generated code.

Context ladder (total window, including output reservation):
8,192; 16,384; 32,768; 65,536; 131,072; 196,608; 262,144.
Generate/select actual tokenized inputs that fit the specified output budget.
Use multiple target positions and distractors. Test both long prefill and
continued generation near the boundary. A single needle or successful server
startup establishes neither coding reliability nor full context quality.

Recommended initial agent budget: 30 minutes, 40 model turns, and 32,768
aggregate generated tokens per task, whichever is reached first. Apply the
same budget to both quants and record timeout/budget failures separately.
These are local pilot limits, not official benchmark defaults.
Freeze suite-specific decoding/output limits and software revisions before
the first comparative run.

## Decision rules

- Structural integrity and native dispatch must pass.
- No new repeatable schema/tool-call failure or reproducible runtime failure
  is acceptable without investigation.
- Evaluate per-task paired outcomes, not only aggregate scores. Review
  changed passes/failures; rerun a small uncertain subset with fixed
  additional seeds before attributing a tiny difference to calibration.
- A 10-task agent suite is a canary, not proof of a general improvement.
- Prefer v2 only after useful coding/terminal outcomes are at least comparable,
  regressions are understood, and its intended serving profile passes.
- Report the highest context with demonstrated capacity and quality
  separately from the requested/native configuration limit.
- Keep existing full MMLU-Pro/GPQA/MATH work separate and optional; it is not
  the first promotion gate for this engineering checkpoint.

## Additional recipe experiments, after the first comparison

1. Thinking mode versus non-thinking with explicit output/time budgets.
2. FP8 scale handling, if the current kernel/model combination supports a
   validated alternative; this is distinct from GPTQ calibration.
3. Calibration length mix: same content mix and token budget, longer versus
   shorter samples.
4. Content ablation: same token budget, public agentic mixture versus the
   existing UltraChat source, to separate content and budget effects.
5. GPTQ sensitivity: a small dampening/ordering experiment only after checking
   installed LLM Compressor support. Record each change explicitly.
6. Chunked-prefill size, graph memory, and KV allocation for TTFT/capacity.
7. Recurrent/GDN projection quantization as the separate v2-expanded candidate
   specified above, with its own target manifest and matched quality tests.
8. ASR co-residency as a separate workload if later desired; do not assume
   full-window capacity and spare-GPU memory can both be retained.

Do not vary all of these simultaneously. The initial v2 changes calibration
content and budget while retaining the established quantized layer policy.

## Time budget and implementation sequence

Historical run: 4,124.52 seconds (68.7 minutes), about 611K actual tokens,
83,402,051,584 bytes peak RSS (about 77.7 GiB), no process swap.
Source: reports/evaluation-and-agent-status-2026-08-29.md.

Planning estimates for the same host, cached BF16/images, exclusive GPUs:
- Dataset selection/adapters/manifests: 2–6 hours plus downloads.
- Small and longest-sequence pilots: 0.5–1.5 hours.
- Full 1.3M–1.7M quantization: 2–5 hours, provisional.
- Integrity and serving gates: 20–45 minutes.
- Core paired function/instruction/context/performance evaluation: 4–10 hours.
- Twenty agent tasks per quant at the proposed 30-minute cap: up to 20 hours
  of task runtime across both checkpoints, plus setup/loading/compilation.
- Soak: 1–2 hours for each selected profile.

Allow roughly 2–4 days for a first implementation and meaningful comparison;
full-window sweeps, downloads, harness integration, or memory pressure may
extend this. Replace estimates with measured pilot timings. Sequential
quantization uses one GPU at a time in the current pipeline; inference tok/s
does not predict GPTQ speed.

Execution order:
1. Resolve dataset and evaluation provenance/locks, sample and render corpus.
2. Implement dedicated v2 configuration, dataset adapters, preflight, and
   evaluation entrypoints; retain existing defaults.
3. Capture working FP8 host profile; establish matched baseline eval settings.
4. Run current structural smoke, real-source small pilot, and long-sequence
   memory pilot using existing safeguards.
5. Build the distinct full checkpoint and validate serialization/native load.
6. Run paired quick evals, then agent tasks and context ladder.
7. Review paired failures, publish aggregates/manifests, and decide promotion.

No quantization, benchmark execution, deployment, or live-host changes were
performed when this specification was captured.
