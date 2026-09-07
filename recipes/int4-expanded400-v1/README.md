# Expanded-400 INT4 v1

The completed GPTQ W4A16 checkpoint quantizes 400 logical targets: the original
256 text projections plus 144 recurrent `in_proj_qkv`, `in_proj_z`, and `out_proj`
projections. Weights use symmetric INT4 groups of 128, BF16 activations, block
size 128, dampening 0.01, and no activation ordering. Remaining recurrent state,
vision, embeddings, output head, and MTP tensors remain preserved.

The finalized calibration has **423 sequences and 1,537,643 tokens** from four
pinned public sources. Its corpus hash is in [manifest.json](manifest.json).
The final [overlap-clearance report](../../reports/int4-overlap-clearance-2026-09-06.json)
binds the corpus to frozen public/agent screens. Negative lexical and identity
screens do not prove absence of all semantic overlap.

[quant.yaml](quant.yaml) freezes the target policy and original diagnostic
settings. Normal serving uses the authoritative catalog instead of this file's
historical `runtime` section. Run the explicit `just int4` stages in
[reproduction](../../docs/reproduction.md); the shared memory, pilot, storage,
overlap, and recovery gates remain required. Do not overwrite the completed model.

The [milestone](../../reports/int4-v1-milestone-2026-09-07.md) records 638
byte-identical preserved tensors, 161 exact offset-norm round trips, all 15 MTP
tensors, and Marlin dispatch. [TP1/TP2 96K measurements](../../reports/int4-v1-inference-profiles-2026-09-07.md)
establish single-sequence capacity and throughput, not retrieval accuracy.
The [16K paired baseline](../../reports/paired-16k-calibration-baseline-2026-09-07.md)
is the small scored comparison. The imported 262K eval configuration remains
explicitly unvalidated and is not a stable-quality claim.
