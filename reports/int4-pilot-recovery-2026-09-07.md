# Expanded-400 INT4 pilot export recovery

The five-sequence real-source pilot completed all 65 sequential stages and
exported 400 INT4 targets. Final validation rejected the first offset norm:
`model.language_model.layers.0.input_layernorm.weight`.

The pinned llmcompressor 0.13.0 `CalibrationOffsetNorm` converts weights to
`BF16(FP32(w) + 1)` during calibration and restores
`BF16(FP32(converted) - 1)`. This round trip changes source bytes through
rounding even with SmoothQuant disabled. The validator now accepts only this
exact transformation for the architecture's offset norms. GDN gated norms,
vision and MTP still require byte identity. This is not a general tolerance or
an exemption for arbitrary normalization changes.

Read-only validation of the saved export passed: 400 logical targets, 638
byte-identical preserved tensors, and 161 exact offset-norm round trips.
The complete shard inventory is 19,452,778,112 bytes including the MTP sidecar.
All 15 MTP tensors remain byte-identical. Packing, positive finite scales,
metadata, index completeness and processor preservation also passed.

The recovery command rechecks source shard hashes and corpus identity,
revalidates the saved export, adds the non-production marker, and renames it
to `real-pilot`. It writes a new report; the failed report is retained. It
does not modify weights or repeat GPTQ. Finalization requires exclusive access
via the shared quantization lock. The command is for fully serialized pilots,
not for recovering missing shards or resuming interrupted calibration.

Local evidence root: `/data/qwen38-int8-lab/int4-v1/`.

- Original metadata: `quant-real_source_short_and_long_pilot-20260907T004210Z.json`
- Validation: `revalidation-20260907.json`
- Recovered metadata: `quant-real_source_short_and_long_pilot-recovered-20260907.json`
- Pilot artifact: `real-pilot/`
- Log: `/data/qwen38-int8-lab/logs/int4-real-pilot-gpu0-20260907T003919Z.log`

Validation: two container tests compare the accepted conversion against the
pinned library itself and reject corruption, excluded norm families and
nonfinite inputs; seven existing corpus preparation tests pass. Shell syntax
and Git whitespace checks pass.

The pilot used the older `511e018` worktree, which did not include the merged
resume integration. It therefore has no rolling calibration snapshots. Future
runs must use the current integrated pipeline with `--resume` support and the
shared 8 GiB RAM / 32 GiB swap-growth / 10-second watchdog. The launcher now
defaults to physical GPU 0; `INT4_GPU_UUID` can explicitly override the UUID.

This remains a five-sequence compatibility/memory pilot. Real-checkpoint
vLLM loading, Marlin dispatch, capacity and quality have not been validated by
this recovery. No full-corpus INT4 build or serving deployment is implied.
