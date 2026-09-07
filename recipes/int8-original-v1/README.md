# Original INT8 v1

The completed checkpoint quantizes 256 ordinary text projections using GPTQ
W8A8: per-channel INT8 weights and dynamic per-token INT8 activations. It
preserves recurrent/GDN paths, vision, embeddings, output head, and 15 MTP tensors.
SmoothQuant is disabled; block size is 128 and dampening is 0.01.

The build used 512 UltraChat train_sft samples at up to 2,048 tokens, seed 42,
revision `8049631c405ae6576f93f445c6b8166f76f5505a`. The source/build identities
and metadata hashes are in [manifest.json](manifest.json); executable settings
are in [quant.yaml](quant.yaml).

Use `just quant-plan`, `just quant-smoke`, `just quant-tiny`, `just quant-small`,
and then explicitly `just quant` under the prerequisites in
[reproduction](../../docs/reproduction.md). Do not overwrite the retained artifact.

The [quality-candidate report](../../reports/quality-candidate-2026-08-25.md)
establishes structural integrity and native CUTLASS execution. The measured
64K BF16 and recorded 262K FP8 profiles are discoverable through
`python3 serving/launch.py list`. The 262K profile must not inherit the older
report's smaller-window retrieval validation. No standardized accuracy or
vision/video equivalence is claimed.
