# Completed INT4-v1 / INT8-v2 evaluation — 2026-09-08

Run `20260908T120552Z-20646223eda1` completed every selected example: 1,220 per
model, 2,440 model-example evaluations overall. This report publishes aggregate
results and a corrected IFEval analysis. Raw prompts, responses, generated code,
checkpoint weights, caches, host UUIDs, and private artifact paths stay outside Git.

## Results and scope

| Benchmark | Metric | INT4-v1 | INT8-v2 | INT8 minus INT4 |
| --- | --- | ---: | ---: | ---: |
| IFEval | Prompt-level strict accuracy | 208/256 = 81.25% | 204/256 = 79.69% | −1.5625 percentage points |
| HumanEval+ | pass@1 on base and additional tests | 154/164 = 93.90% | 155/164 = 94.51% | +0.6098 percentage points |
| BBH | Normalized answer-choice likelihood accuracy | 294/400 = 73.50% | 304/400 = 76.00% | +2.5000 percentage points |
| MMLU-Pro | Answer-choice likelihood accuracy | 256/400 = 64.00% | 260/400 = 65.00% | +1.0000 percentage point |

IFEval, BBH and MMLU-Pro are deterministic subsets, not full benchmark results.
HumanEval+ includes all 164 problems. Counts for BBH and MMLU-Pro are totals
across categories, not counts per task or subject. Seed 42 freezes the selections;
the same selected work is assigned to each model.

The measured differences are small: four IFEval prompts, one HumanEval+ problem,
ten BBH examples and four MMLU-Pro examples. No combined intelligence score is
computed. There is no BF16 run in this comparison, no estimate of quantization
retention, and no demonstrated general advantage of one bit width. The models
differ in quantization targets, calibration data, runtime kernels, KV dtype and
parallel execution. A coding test pass is not evidence of repository-agent or
long-horizon coding performance.

## Checkpoints and runtime

Recipe provenance remains in the existing manifests and milestone reports:

- [Expanded-400 INT4 v1](../recipes/int4-expanded400-v1/README.md): W4A16,
  group size 128, 400 projections, 423 retained calibration sequences.
- [Agentic INT8 v2](../recipes/int8-agentic-v2/README.md): W8A8,
  256 projections, 307 retained sequences. Its documented historical calibration
  provenance limitations still apply.

| Setting | INT4-v1 | INT8-v2 |
| --- | --- | --- |
| Named profile | `int4-v1-16k-fp8-tp1` | `int8-v2-16k-bf16-tp2` |
| GPU allocation | Two independent TP1 model copies, one per card | One TP2 model split across both cards |
| Replica work assignment | Stable disjoint halves of every benchmark | Full selection in one worker |
| KV cache dtype | FP8 | BF16 |
| KV reservation per GPU | 805,306,368 bytes (0.75 GiB) | 805,306,368 bytes (0.75 GiB) |
| Active model wall time | 10,177.64 s (2h 49m 38s) | 17,808.85 s (4h 56m 49s) |

Hardware: two RTX 3090 cards with 24 GiB VRAM each; Xeon Silver 4215R,
8 cores / 16 threads, approximately 92 GiB usable RAM. Both profiles use a
16,384-token context, BF16 runtime dtype, text-only loading, non-thinking mode,
eager execution, disabled prefix caching, 1,024-token chunked prefill, no MTP and
no CPU offload. Each worker admits one model request at a time; the native harness
commits batches of four examples. Replica caches are separate.

IFEval, BBH and MMLU-Pro share a model allocation within each replica. After the
native stages, HumanEval+ uses owned HTTP generation servers. All generation
servers stop before CPU grading, which remains serial. INT4 finishes before
INT8 starts. This is a comparison of complete deployments, not a controlled TP1
versus TP2 throughput test; the wall times must not be described as isolated
quantization speedups or proof of linear replica scaling.

## Evaluation protocol

| Component | Frozen setting |
| --- | --- |
| Native harness | lm-evaluation-harness 0.4.12, leaderboard tasks |
| Inference | vLLM 0.27.1 |
| Coding evaluator | EvalPlus 0.3.1; HumanEval+ dataset v0.1.10 |
| IFEval generation | 1,024 output tokens maximum |
| HumanEval+ generation | Greedy, one 2,048-token response; EvalPlus chat prompt and sanitizer |
| BBH / MMLU-Pro | Native leaderboard few-shot prompts and answer-choice likelihood scoring |
| Sample seed | 42 |
| Input truncation | Rejected rather than silently shortening the request |
| Completion policy | `run_to_completion: true`; stage/model/overall active cutoffs disabled |
| Retained guards | One-hour queue limit; explicit stop; RAM/swap checks; inference-error handling |
| Grading bounds | 120-second outer per-example deadline plus EvalPlus test timeouts |

Generated code is graded in disposable containers with no network, GPUs, model
mounts, credentials or Docker socket; read-only root filesystem; unprivileged
host UID; two CPUs; 4 GiB RAM without additional swap; and 128 processes.
These are container boundaries on a shared kernel, not VM isolation.

Dataset revision pins:

| Dataset | Revision |
| --- | --- |
| `wis-k/instruction-following-eval` | `5a5661c2a35488308556cf4453dc074d1eba91a0` |
| `SaylorTwift/bbh` | `b5306be6f827cfafbb545ff5a51f96916029b0fd` |
| `TIGER-Lab/MMLU-Pro` | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |

## Completion, truncation and failures

Every stage reports complete, with zero infrastructure errors, zero infrastructure
retries and zero scheduling timeouts. Total active wall time was 27,987.22 seconds
(7h 46m 27s); queue time was 0.12 seconds. The supervisor exited, owned evaluation
containers were removed and both GPUs returned to idle.

| Evidence | INT4-v1 | INT8-v2 |
| --- | ---: | ---: |
| IFEval responses reaching output limit | 18/256 | 14/256 |
| HumanEval+ responses reaching output limit | 0/164 | 0/164 |
| HumanEval+ execution/test failures | 10/164 | 9/164 |
| Infrastructure errors | 0 | 0 |

Execution/test failures count as incorrect answers; they are distinct from
infrastructure failures. A truncated IFEval response can still pass its checks;
do not subtract truncations from the scored denominator. BBH and MMLU-Pro use
likelihood scoring, so output-generation truncation is not applicable.

Earlier crashed, stopped and deadline-constrained runs are not pooled into this
report. This fresh run generated its own responses; the earlier eight-example
baseline is not part of these totals.

## IFEval pairing correction and interpretation

The original report's ID-based paired transition tables cannot be trusted.
Inspection of the pinned harness found that `Task.doc_iterator` selects rows in
dataset order while `evaluator.py` labels them with `indices[doc_id]` in the
supplied batch order. The runner supplies unsorted seeded batches and trusts
the returned `doc_id`. This permutes labels within batches; different replica
partitions create different permutations. The response, document and score stay
together, but the reported sample ID can refer to another selected document.

The separate read-only reconstruction paired IFEval by canonical full document:
prompt text, dataset key, ordered instruction IDs and constraint arguments.
It verified 256 unique documents on each side, exact overlap of all 256, and
identical actual model input arguments for every reconstructed pair. Only 44
pairs had equal originally reported IDs. Saved strict scores equal the conjunction
of their per-instruction checks; no inference or rescoring was performed.

| Corrected strict outcome | Prompts |
| --- | ---: |
| Both pass | 191 |
| Only INT4 passes | 17 |
| Only INT8 passes | 13 |
| Both fail | 35 |

The original 42 INT4-only / 38 INT8-only counts are superseded. Aggregate scores
remain 208/256 versus 204/256. On the corrected 30 disagreements, an exact
two-sided McNemar test gives p=0.5847. This is not persuasive evidence of a
general instruction-following advantage. Loose prompt scoring narrows the gap
to three prompts: INT4 217/256 (84.77%), INT8 214/256 (83.59%).

Selected instruction-level details:

| Constraint type | Checks | INT4 passes | INT8 passes |
| --- | ---: | ---: | ---: |
| Letter frequency | 16 | 13 | 10 |
| Postscript requirement | 18 | 15 | 17 |
| Number of highlighted sections | 19 | 16 | 18 |
| Capitalized-word frequency | 15 | 9 | 11 |
| Number of sentences | 29 | 22 | 23 |
| Number of words | 24 | 19 | 19 |
| All instruction occurrences | 397 | 347 | 343 |

These counts describe constraint occurrences, including repeated types within
one prompt; they are not independent benchmark samples. Prompt-level strict
scoring requires every constraint to pass, so instruction deltas do not add
directly to prompt deltas.

Concrete checks illustrate the narrow differences without publishing raw text:
one letter-count task required at least six instances and received seven from
INT4 versus five from INT8; another required at least 25 and received 33 versus
three. In two INT4-only wins, INT8 hit the output limit before satisfying JSON
or an exact-ending check. Six of the 30 disagreement pairs involved truncation
on at least one side.

One all-uppercase-English check failed INT8 despite its saved response being
uppercase. The pinned checker also requires `langdetect.detect(value) == 'en'`;
this is a checker/language-detection caveat, not evidence of lowercase output.
The detector's intermediate output was not saved. We retain the measured score.

BBH and MMLU-Pro use the same native harness path: their original ID-based paired
tables remain unaudited and are not reproduced here. Their completed aggregate
scores are reported separately. The next runner change should sort sample indices
or validate document identities, then audit all native paired reports. This
documentation change does not claim that the runner bug is fixed.

## Published model context, not a common leaderboard

Sources below were checked on 2026-09-08. These are historical external results,
not reruns on this host or current rankings. Different sample sizes, generation
budgets, prompts, precision and harnesses prevent a controlled head-to-head
comparison. The local scores above should not be turned into claims of beating
these models. No external numbers are substituted for an unmeasured local BF16
baseline.

### Instruction following and knowledge

Qwen's September 2024 instruction-tuned evaluation tables report:

| Published model | IFEval strict-prompt | MMLU-Pro |
| --- | ---: | ---: |
| Gemma2-27B-IT | 77.1% | 55.5% |
| Qwen2.5-14B-Instruct | 81.0% | 63.7% |
| Qwen2.5-32B-Instruct | 79.5% | 69.0% |

Source: [Qwen2.5-LLM, instruction-tuned evaluation](https://qwenlm.github.io/blog/qwen2.5-llm/#qwen-turbo--qwen25-14b-instruct--qwen25-32b-instruct-performance).
These are Qwen-published measurements, including the Gemma comparison, not
independently reproduced results here. Our 256/400-example subsets and local
scoring/generation settings are not established as protocol-matched to them.

### Code generation

The EvalPlus maintainers' published data provides these historical HumanEval+
pass@1 reference points:

| Published model label | HumanEval+ |
| --- | ---: |
| Qwen2.5-Coder-32B-Instruct | 87.2% |
| GPT-4-Turbo (April 2024) | 86.6% |
| GPT-4 (May 2023) | 79.3% |

Sources: [EvalPlus methodology notes](https://evalplus.github.io/leaderboard.html)
and [underlying results](https://evalplus.github.io/results.json), using the
`pass@1.humaneval+` field. The notes specify HumanEval+ v0.1.10 and greedy pass@1;
all three entries are marked chat-prompted. This makes them useful context, but
does not establish identical model snapshots, prompt templates, sanitization,
output limits or software versions. HumanEval+ is not original HumanEval, MBPP+,
or an average across coding tests. High scores on this small public set also
cannot rule out training-data contamination or establish coding-agent capability.

The [official Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
reports IFBench among its benchmarks. IFBench is a different benchmark from
IFEval and is not a numerical baseline for this table. Likewise, agentic and
thinking-mode results from that card are not transferred to these 16K,
non-thinking quantized profiles.

## Evidence and reproducibility

- Eval implementation commit:
  [`0dfbd55363c03ed1fd58baf84ed8450242057bf2`](https://github.com/gracee3/evals/commit/0dfbd55363c03ed1fd58baf84ed8450242057bf2).
- Frozen suite:
  [`paired-16k-distributed-complete.yaml`](https://github.com/gracee3/evals/blob/0dfbd55363c03ed1fd58baf84ed8450242057bf2/examples/paired-16k-distributed-complete.yaml).
  It contains host-specific GPU bindings; another host must resolve its own UUIDs.
- Preparation digest: `85023c597f06245ed7d4b6f148fcf32ef933b954a5ea912f03682ce7547376fe`.
- Runtime image: `sha256:1fdc456dc95eefd909e1a7a3e9aedf3eb28ec1b9be3c56363eee4abb6d68b537`.
- Frozen configuration file SHA-256: `e4f5af3af7fa2d7580c19bb1024f3fb3b5c379ef941e138567f78b166fa648c2`.
- Final original report JSON SHA-256: `1a4e4e473cb653e2c4d70918810b395327d7330bfcc1aa3043f1e5173a26ef41`.
  Its aggregate results are retained; its original native paired counts carry
  the correction described above.
- Corrected IFEval summary SHA-256: `f1991ce6b5915e761fae8f8b40ad78a8e9233b103fab813d4c16d511eeaa4818`.
  The private summary contains source hashes for all 512 records; its analysis
  script verified the source files were unchanged after reconstruction.
- External EvalPlus JSON SHA-256 at retrieval:
  `e6837dc3b94f32cb1da0ced012e52edc4a722007b684df5b63da029bbd3b46d2`.

Raw evidence is retained privately by run ID. This public report is deliberately
an aggregate summary, not a redistribution of prompts, responses or generated
solutions. Reproduction requires the retained checkpoints and pinned images;
it should incorporate the sample-ID correction before interpreting paired
transitions. No weights, dependencies, profiles or inference code were changed
to publish this report.
