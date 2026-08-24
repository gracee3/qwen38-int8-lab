# Host baseline — 2026-08-24

Recorded at `2026-08-24T15:54:07-04:00` with `just info` before quantization work.

## Compute

- Ubuntu 26.04 LTS, kernel `7.0.0-29-generic`, 16 logical CPUs.
- 92 GiB visible RAM; about 90 GiB available at inspection time.
- 99 GiB swap, effectively unused.
- Two NVIDIA GeForce RTX 3090 GPUs, 24,576 MiB each, both idle in P8.
- NVIDIA driver `610.43.02`; host-reported CUDA UMD/toolkit `13.3`.
- Both cards report compute capability 8.6 (SM86); both project containers independently confirmed both devices.

## Docker

- Docker Engine client/server `29.7.2`.
- containerd `2.3.3`; Docker Buildx `0.36.1`; BuildKit `0.32.2`.
- `nvidia` runtime is registered and NVIDIA CDI devices 0, 1, and `all` are discoverable.
- No existing PyTorch, vLLM, or LLM Compressor image was present before this project build.
- Project images now use a shared digest-pinned PyTorch 2.13.0/CUDA 13.0 base; dependency layers remain separate.
- Existing unrelated containers and images were left running and unchanged.

## Storage and guardrails

- Root has about 527 GiB free at the recorded time.
- Live state differs from the older host baseline: `/dev/nvme0n1` is the unmounted read-only disk (`RO=1`), while `/dev/nvme1n1p3` backs the active encrypted root filesystem.
- No mount, filesystem, partition, or block-device changes were made.
- Experiment writes are confined to `/data/qwen38-int8-lab`; model output is reserved at `/data/models/Qwen3.8-27B-W8A8-INT8`.

## Local artifact reuse

- Complete source: `/home/emmy/workspace/qwen3.8-27b-download/model` (18/18 shards).
- Interrupted source attempt: `/data/models/Qwen3.8-27B` (preserved; never used as quantization input).
- Existing GGUF: `/data/models/Qwen3.8-27B-Q8_0` (preserved; not a W8A8 source).
- Existing Hugging Face and pip caches were inventoried. No second Qwen3.8 model download is part of this workflow.

Run `just info` for current state; this document is a dated record, not a live assertion.
