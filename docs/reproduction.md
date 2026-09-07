# Reproducing and validating a recipe

This release preserves working build code and completed artifacts; applying the
layout does not require running any of these expensive commands. Use the recipe
README for the exact policy and historical limitations.

## Prerequisites and bindings

The recorded host has 96 GB nominal RAM, two RTX 3090 24 GB GPUs, Docker with
NVIDIA Container Toolkit, and CUDA 13.3. Full calibration needs substantial RAM
and temporary disk for source tensors, two rotating recovery generations, and
final serialization. INT4's wrapper requires 80 GiB available RAM and at least
96 GiB disk headroom before a real pilot/full build; live recovery checks impose
additional storage requirements. These floors are not a complete capacity estimate.

Use an existing verified BF16 source, not the interrupted `/data/models/Qwen3.8-27B`
download. The current host source is `/data/models/Qwen3.8-27B-source-download/model`.
Set `SOURCE_MODEL` to override that bind; containers always see `/models/source`
read-only. `MODEL_ROOT` defaults to `/data/models`, `WORK_ROOT` to
`/data/qwen38-int8-lab`. Data and old evidence remain in those locations; only
repository paths changed. Do not copy weights into the checkout.

Confirm idle GPUs, RAM/disk headroom, and the protected secondary disk's unmounted,
read-only state before real calibration. The existing real-quant wrapper locks
`$WORK_ROOT/quant-swappiness.lock`, temporarily applies swappiness 1, and restores
the prior value. It needs preconfigured noninteractive privilege; do not launch
through an unattended password prompt. No host policy is installed by serving.

## Original INT8

```sh
just quant-plan
just dataset-preflight quality
just quant-smoke
just quant-tiny
just quant-small
# Explicit production build only into an absent output:
just quant
# Resume a run created with this exact implementation/configuration:
just quant true
just validate
```

`quant-smoke` is synthetic; tiny/small use the real source and write experimental
scratch outputs. Real commands are explicit and resource gated. `OUTPUT_MODEL`
selects the original INT8 output basename beneath `MODEL_ROOT`; it is not an
arbitrary cross-filesystem destination. Existing production outputs are refused.

## Agentic INT8 v2

Use the retained `calibration/agentic-v2/calibration.parquet` payload below
`WORK_ROOT`; its release manifest records the exact SHA-256. The completed run
used 307 sequences and 1,371,502 tokens. `just v2-quant-small`, `just v2-quant`,
and `just v2-quant true` retain the original entrypoint and gates. The fixed
production basename is `Qwen3.8-27B-W8A8-INT8-Agentic-v2`.

`just v2-calibration-prep-dry` inspects the preparation configuration.
`just v2-calibration-prep` remains available for a deliberately new preparation,
but missing historical dataset revisions and the failed Agentic source mean it
must not be advertised as exact reconstruction of the completed corpus. Use a
separate `WORK_ROOT` to avoid overwriting retained calibration evidence.

## Expanded-400 INT4

Set `INT4_RUN_ROOT` to a fresh directory for a new reproduction and
`INT4_GPU_UUID` to the intended idle GPU. The existing wrapper retains its
host-specific default UUID and fixed final model path; normal serving profiles
instead use configurable bindings. The recipe's old `runtime` section describes
the original diagnostic policy; serving uses `serving/profiles/` exclusively.

Run stages explicitly in order:

```sh
just int4 audit
just int4 synthetic
just int4 runtime
just int4 corpus
just int4 screen-candidate
just int4 finalize
just int4 screen-final
just int4 real-pilot
```

Candidate screening may fetch pinned public held-out fixtures. Other preparation
stages use existing caches; cache and held-out mounts remain the original
host-specific paths in `quant/int4.sh`. Do not rerun completed stages in the
retained evidence directory. Full quantization additionally requires matching
successful pilot and overlap reports:

```sh
INT4_REAL_PILOT_REPORT=real-pilot-report.json \
INT4_OVERLAP_REPORT=overlap-clearance.json just int4 full
```

Use the actual pilot report filename, not the illustrative name above. The final
destination remains `/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1` and must
not exist. Append `--resume` to `real-pilot` or `full` only for the same frozen
implementation/source/corpus/config. Never manufacture passing gate reports.

## From checkpoint to evaluation

Run the corresponding integrity validator in the existing image with source and
checkpoint mounted read-only. `validation/validate_quant.py --help` and
`validation/validate_int4.py --help` describe their required paths and optional
log evidence. Select a compatible serving profile, inspect `--dry-run`, and only
then schedule a bounded smoke against idle hardware. Structural checks, kernel
dispatch, capacity, retrieval, and benchmark accuracy are distinct results.

New scored evaluations belong in local-agent-evals. Preserve its current image
and frozen runs until the direct catalog consumer is updated. See
[profiles](profiles.md) for pinning and override semantics and
[legacy evaluation](../validation/legacy_eval/README.md) for coverage not yet replaced.

## Troubleshooting

- Missing image: verify the local ID with `docker image inspect`; do not rebuild
  all images to resolve a renamed repository path.
- FP8 startup: check the existing CUDA toolkit mount and selected profile/image.
  A larger context may fail KV admission; do not silently lower it in a report.
- Resume mismatch: return to the original build commit; do not bypass identity checks.
- Existing output: retain it and choose an explicitly supported fresh destination;
  never remove a completed checkpoint to make a reproduction command proceed.
- Missing calibration provenance: use the recorded retained payload and state the
  limitation; do not fill historical revision fields with guesses.
