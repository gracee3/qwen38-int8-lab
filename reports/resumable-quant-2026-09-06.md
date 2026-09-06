# Shared quant recovery and host memory policy

## Implemented and verified

- Host `vm.swappiness=1` is installed persistently in
  `/etc/sysctl.d/90-qwen-memory.conf`; live value verified on 2026-09-06.
  Swap remains available. No network or service restart was required.
- Both real INT8 profiles use atomic rolling sequential-GPTQ snapshots at
  approximately 20% boundaries (13, 26, 39, 52, 65 for a 65-stage graph).
- A snapshot includes model/qparameter state, cached calibration batches and
  RNG state. Resume checks source hashes, token rows, recipe, software and code.
  A successful new generation replaces the pointer before the previous
  generation is removed. The latest generation remains after final export.
- Small Qwen tests with GDN and full-attention layers pass exact tensor equality
  against native uninterrupted sequential GPTQ, after fresh-process resumption.
  W8A8: 109 state tensors; expanded-target W4A16: 127. BF16 and CPU-offloaded
  registered parameter storage are covered, including final-stage recovery.
  Both schemes also passed on one actual RTX 3090, with CPU offload, at the
  same implementation commit. Initial snapshot and final-stage retries each
  matched the native uninterrupted CUDA baseline exactly. The guarded test
  wrapper captured and left swappiness at 1; no full quant was launched.
- Host tests: 24 cases, 21 passed, three skipped because host NumPy is absent.
  Installer/wrapper pass shell syntax and ShellCheck. Full 27B CUDA checkpoint
  I/O and power-loss recovery are not yet validated.

## Operational limits

The shared watchdog permits 32 GiB swap growth, retains an 8 GiB available-RAM
floor, and requires a sustained ten-second violation. A watchdog stop is logged
as `RESOURCE_SAFETY_ABORT`, not misattributed to a user keyboard interrupt.
These are guards, not a promise that every workload fits or cannot OOM.

Snapshots are large: roughly source-sized weights plus current activations.
Two generations coexist during publication; final export also needs room.
Each save checks actual tensor storage plus an 8 GiB disk reserve. Insufficient
space fails without replacing the previous complete snapshot. Incomplete
generations and final snapshots are intentionally retained for explicit review.
At verification the host had 188 GiB free disk, not 188 GiB free RAM.
Do not run both full quantizers concurrently or assume that retaining the INT8
snapshot leaves enough space for the INT4 run.

Old interrupted runs have no resumable state. The first run using this code
starts from the source. Use `just v2-quant true` only for a subsequent retry
of the same output/configuration. Pin the code and container until completion.

## Branch integration

Main already contains INT8 v2 preparation docs and corpus-parser corrections.
The memory guardrail change (PR 13) precedes the shared resumable-quant change.
Neither should silently merge an experimental INT4 recipe or alter serving
defaults. The separate INT4 integration must adopt the shared helper and
pipeline while retaining its corpus, target, pilot and held-out validation gates.
