# INT8 / expanded-400 INT4 coordination — 2026-09-06

## Published changes and merge order

1. PR 13, `fix/quant-memory-guardrails`, commit
   `18554c453e61aa1a9b18557db19b1ae3e709d634`: corrected resource watchdog.
2. PR 14, `feat/resumable-quant`, implementation commit
   `5094844644c4786daef8e23735739ebfb250f350`: shared rolling recovery,
   persistent swappiness installer, and INT8 entrypoint wiring.
3. PR 15, `integration/int4-shared-resume`: preserves INT4 preparation from
   original commit `511e0180efc90aef33ee89609c89eba0aca3bd14`, rebased onto
   shared fixes. Integration-only commit
   `1cbd7ebb74270b4eac621c23b9f3b6845b8b8914` wires real INT4 pilot/full into
   the shared memory policy and recovery engine. Corpus and launch gates remain.

All PRs are draft and require review; main has not been changed. Main already
contains earlier INT8 v2 documentation and corpus-parser fixes. Do not merge
the experimental INT4 recipe simply to share infrastructure.

## Other agent's next action

The original `/home/emmy/workspace/qwen38-int4-v1` worktree was left clean at
`511e018`; it has NOT automatically adopted the shared fixes. After saving any
new work and coordinating the branch update, either use the separate integrated
worktree `/home/emmy/workspace/qwen38-int4-resume-integration`, or rebase the
original INT4 branch onto `origin/feat/resumable-quant` and cherry-pick the
integration-only commit `1cbd7ebb74270b4eac621c23b9f3b6845b8b8914`.
Once shared PRs merge, rebase onto the resulting main instead. Do not blindly
cherry-pick the INT4 preparation again; it is already present on the agent branch.

## Behavior and evidence

- Host persistent and live swappiness are verified as 1. Swap remains enabled.
- Both real quant paths use a common exclusive lock, 8 GiB available-RAM floor,
  32 GiB swap-growth threshold and ten-second sustained watchdog.
- Real INT4 no longer has an earlier 84 GiB/no-swap cgroup ceiling. Small
  preparation/runtime containers remain bounded and without swap.
- Real calibrations checkpoint about every 20% of sequential stages. Publication
  is atomic and checksummed; only after publication is the prior generation
  removed. Last completed generation stays after final export.
- Retry INT8 with `just v2-quant true`; retry INT4 with
  `bash scripts/int4_prepare.sh real-pilot --resume` or `full --resume`.
  Same source, corpus, recipe and code are required. Old interrupted runs without
  snapshots must start from the source.
- Integrated host suite: 31 passed, three optional NumPy skips; ShellCheck,
  shell syntax, Python compilation and clean diff checks passed.
- CPU-offloaded BF16 synthetic W8A8/W4A16 fresh-process recovery exactly matches
  uninterrupted tensors, including retry from final calibration stage.

## Risks and immediate next step

Review the shared changes and run a real-source pilot before a full quant.
Full 27B checkpoint I/O and power-loss recovery have not been tested.
Disk is the tighter checkpoint constraint: 188 GiB was free at verification;
two large state/cache generations briefly coexist, and final export also needs
space. A completed INT8 snapshot may leave insufficient room for INT4.
Review retained snapshots explicitly after validating each artifact; do not
delete source models or historical candidate artifacts. Never run both full
quantizers concurrently. No full quant or serving restart was launched here.
