# Layout release validation — 2026-09-07

Baseline: `52de78d4c7fc2672a118802e02f742b701f02b37`
(`baseline-2026-09-07`). This report belongs to the layout PR; the Git commit
containing it identifies the reviewed source tree. The three recipe manifests
retain their original model-build commits separately.

## Checks completed

- 51 unit tests passed in the existing quant image, including the new catalog
  and frozen-layout contracts. The container had no network or GPU devices,
  two CPUs, and a 4 GiB memory limit. No tests skipped in that image.
- Tiny CPU W8A8 fresh-process interruption/resume exactly matched all 109
  uninterrupted model tensors; W4A16 exactly matched all 127. The existing image
  ran with two CPUs and an 8 GiB memory limit. Disposable tensors/logs remained
  inside removed containers.
- All 18 relocated quant/calibration/validation Python modules have identical
  ASTs after the declared import and path substitutions. Quantization algorithms
  were not retuned. The recovery adapter is byte-identical to the baseline.
- All dependency intent and lock files are byte-identical. Dockerfile differences
  are only relocated COPY source paths. Original local image IDs are recorded
  in `environments/images.lock.json`.
- The five original serving YAML profiles retain their effective values. The
  previous ordinary 64K serve configuration remains an explicit catalog entry.
  Existing 262K eval presets are imported with source provenance and separate
  evidence status; the INT4 262K preset is not marked validated.
- Existing historical reports retain their measurements and identities; only
  repository file references changed.
- ShellCheck, shell syntax, Python AST parsing, Justfile parsing, profile command
  previews, relative Markdown links, and whitespace checks passed.

## Scope

No Docker build, image download, image retag, full-model quantization, GPU
inference, system configuration change, or artifact cleanup was performed.
Completed weights, calibration payloads, raw evidence, recovery snapshots, and
the prior dirty checkout were retained. Small release manifests fingerprint
existing metadata and calibration payloads; they are not newly computed full
weight-hash manifests.

The new serving launcher is validated by argument/contract tests and previews.
Its resolved commands have not been GPU revalidated in this PR. Historical
checkpoint/kernel/capacity evidence remains as reported. INT8-v2's missing
upstream corpus revisions are documented rather than guessed.

The direct discovery interface is provided for local-agent-evals. That repository
still needs a consumer update pinned to the reviewed layout commit; existing
frozen runs and image ownership are unchanged. The legacy six-group evaluator
is retained until its additional coverage has a replacement.

The release tag is a reviewed publication step after this single PR; the existing
baseline tag is unchanged. See [release procedure](../docs/releasing.md).
