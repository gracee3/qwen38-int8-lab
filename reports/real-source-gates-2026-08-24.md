# Real-source safety gates — 2026-08-24

Implementation was run from exact `main` commit `0c507d6546a93500efc39ce7e14b72e2ea230498` after merging PRs #1–#4.

## Load/trace gate: passed

- The complete 18-shard, 1,199-tensor BF16 source and the 256-module policy invariants passed.
- The two local prompts tokenized to 70 and 59 tokens and were padded as one batch.
- LLM Compressor traced 64 `Qwen3_5DecoderLayer` targets into exactly 65 sequential subgraphs.
- Compressed-tensors recorded 1,345 `cpu->cpu` placements and no disk offload.
- No model output appeared. Peak RSS was 1.941 GiB, minimum host `MemAvailable` was 88.165 GiB, swap growth was zero, GPU peaks were 4/4 MiB, and minimum free disk was 504.545 GiB.

## Two-sample W8A8 experiment: safety-stopped

Detached tmux session `qwen38-quant-tiny-20260824T211053Z` (initial pane PID 481734) completed all 65 sequential subgraphs and compressed all 256 intended projections. During serialization, swap growth exceeded the 4 GiB limit for 10 seconds. The monitor intentionally interrupted the process while writing shard 4 of 10.

- Peak RSS: 65.241 GiB
- Minimum host `MemAvailable`: 34.176 GiB
- Swap growth: 7,169,265,664 bytes (6.677 GiB)
- GPU peaks: 13,736 MiB and 266 MiB
- Minimum free disk: 492.458 GiB
- Safety trigger: `unsafe host memory persisted: MemAvailable=43927826432, swap_growth=7169265664`

The hidden incomplete staging directory remains at `/data/qwen38-int8-lab/scratch/.Qwen3.8-27B-W8A8-tiny-source-20260824T211053Z.incomplete-20260824T211059Z`. Its four partial shards and small metadata files total 14,367,972,286 bytes. The full log remains at `/data/qwen38-int8-lab/logs/quant-tiny-20260824T211053Z.log`; machine metadata remains under `/data/qwen38-int8-lab/results`.

No final checkpoint was published, so strict checkpoint validation and vLLM TP2 were not run. This evidence does not support requesting authorization for the 512×2048 quality run.
