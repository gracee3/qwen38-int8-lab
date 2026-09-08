# Qwen3.8 Lab

Build, validate, and serve three retained Qwen3.8-27B quantized checkpoints.
This repository owns the recipes, shared quantization implementation, pinned
container definitions, serving profiles, and reviewed evidence.
[evals](https://github.com/gracee3/evals) owns new benchmark runs.

This first layout release preserves the checkpoint recipes and dependency versions
from [`baseline-2026-09-07`](https://github.com/gracee3/qwen38-lab/tree/baseline-2026-09-07).
Existing local Docker images work without rebuilding, downloading, or retagging.
A source release does not include model weights or calibration payloads.

## Recipes and evidence

| Recipe | Quantization | Completed calibration | Evidence |
| --- | --- | --- | --- |
| [Original INT8 v1](recipes/int8-original-v1/README.md) | W8A8, 256 text projections | 512 UltraChat sequences | [Build and native CUTLASS validation](reports/quality-candidate-2026-08-25.md) |
| [Agentic INT8 v2](recipes/int8-agentic-v2/README.md) | W8A8, same 256 projections | 307 retained sequences | [Build milestone](reports/agentic-w8a8-v2-quantized-2026-09-06.md) |
| [Expanded-400 INT4 v1](recipes/int4-expanded400-v1/README.md) | W4A16, group 128, 400 projections | 423 retained sequences | [Build and Marlin validation](reports/int4-v1-milestone-2026-09-07.md) |

Each recipe includes its unchanged quantization policy and a release manifest
identifying the build commit, source revision, calibration evidence, and small
checkpoint metadata hashes. INT8-v2's historical corpus has incomplete upstream
provenance: use its retained payload to reproduce it; a new preparation run is
not claimed equivalent. [Recipe details](recipes/int8-agentic-v2/README.md).

## Completed local evaluation — 2026-09-08

Both retained candidates completed **2,440 model-example evaluations** in total,
using non-thinking, text-only 16K profiles on two RTX 3090s. INT4-v1 used two
independent TP1 replicas; INT8-v2 used one TP2 worker across both cards.

| Benchmark | Scored per model | INT4-v1 | INT8-v2 |
| --- | ---: | ---: | ---: |
| IFEval, strict prompt accuracy | 256 | 81.25% (208/256) | 79.69% (204/256) |
| HumanEval+, pass@1 | 164 (all problems) | 93.90% (154/164) | 94.51% (155/164) |
| BBH, answer-choice likelihood accuracy | 400 | 73.50% (294/400) | 76.00% (304/400) |
| MMLU-Pro, answer-choice likelihood accuracy | 400 | 64.00% (256/400) | 65.00% (260/400) |

All selected examples were scored in **7h 46m 27s**, with no infrastructure
errors, retries, or scheduling timeouts. These are local protocol-qualified
results; three benchmarks use subsets. They do not establish BF16 retention,
general model superiority, or an official leaderboard rank.

**Pairing correction:** a harness sample-ordering issue invalidated the original
ID-based paired counts. IFEval was reconstructed by exact document and input
identity: INT4 alone passed 17 prompts and INT8 alone passed 13. The four-prompt
advantage is small (exact paired test p=0.585); aggregate scores are unchanged.
Other native benchmarks' paired transition tables remain unaudited.

The [detailed results and comparison report](reports/paired-16k-complete-2026-09-08.md)
records protocol, provenance, truncations, execution failures, corrected IFEval
analysis, runtime limits, and separately labeled published model reference points.
The [earlier eight-example baseline](reports/paired-16k-calibration-baseline-2026-09-07.md)
remains historical evidence. Larger advertised contexts are capacity settings;
each profile carries its own evidence status.

## Discover and serve

Python 3.10+ and PyYAML 6.0.3 are enough to inspect profiles. GPU libraries are
inside the existing containers. Run from the repository root:

```sh
python3 serving/launch.py list
python3 serving/launch.py resolve int4-v1-96k-fp8-tp1
python3 serving/launch.py serve int4-v1-96k-fp8-tp1 --dry-run
```

`list`, `resolve`, and `--dry-run` do not contact Docker or load a model. Set
`VLLM_API_KEY` through your environment before actual serving; it is never included
in the printed command. With an idle GPU and an existing checkpoint/image:

```sh
python3 serving/launch.py serve int4-v1-96k-fp8-tp1 \
  --model /data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1 --gpus 0
```

The launcher binds HTTP to host loopback, mounts weights read-only, uses the exact
local image ID, and refuses to pull an image. FP8 profiles mount the existing CUDA
13.3 toolkit read-only. `just profiles`, `just serve-plan PROFILE`, and
`just serve PROFILE` are shortcuts to the same launcher. There is no implicit
profile default; select the model and memory envelope deliberately.

## Repository map

| Directory | Responsibility |
| --- | --- |
| `recipes/` | Versioned policies, artifact manifests, reproduction instructions |
| `calibration/` | Corpus preparation, rendering, deduplication, overlap screening |
| `quant/` | Quantizers, serialization/recovery, explicit INT4 stage runner |
| `validation/` | Integrity, target/kernel checks, smoke and performance probes |
| `serving/` | Discoverable profiles, common resolver, one serving launcher |
| `environments/` | Existing Dockerfiles, dependency locks, recorded local image IDs |
| `reports/` | Curated historical evidence; see the [index](reports/README.md) |
| `docs/` | Architecture, reproduction, eval integration, release procedure |
| `tests/` | CPU checks and explicit synthetic recovery integration |
| `tools/host/` | Optional host inspection and memory-policy installer |

[Reproduction](docs/reproduction.md) · [Serving contract](docs/profiles.md) ·
[Architecture and recovery](docs/architecture.md) ·
[Environment reuse](environments/README.md) · [Release procedure](docs/releasing.md) ·
[Layout migration](docs/migration.md) · [Contributing](CONTRIBUTING.md)
