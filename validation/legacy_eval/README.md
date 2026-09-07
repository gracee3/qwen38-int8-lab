# Retained leaderboard workflow

This is the older, pinned Open LLM Leaderboard v2 runner, relocated intact with
path adjustments. Its dependencies remain in `environments/legacy-eval/`, using
the existing `qwen38-int8-lab/eval:0.1.0` image. It is not invoked by profile discovery
or normal serving. Commands are explicitly named `just legacy-eval-*`.

Local-agent-evals covers IFEval, HumanEval+, BBH, and MMLU-Pro. This older runner
also covers GPQA, MATH Level 5, and MuSR and supports the original BF16 comparison.
Those coverage differences prevent removing it during a layout-only freeze.
GPQA access, private fixture handling, pinned revisions, and offline checks remain
part of its existing protocol. No unfinished run is turned into a published score.

`config/leaderboard-v2.yaml` defines the six groups and dataset revisions. The
historical protocol uses 16,384-token context, BF16 KV, TP2, one request, eager
execution, no prefix cache, non-thinking, and a 12,314-token likelihood admission
gate. Candidate-only mode makes no BF16 retention/equivalence claim.

`supervisor.sh` is host-specific and must be supplied `REPO`, `EXPECTED_COMMIT`,
and `EXPECTED_BRANCH` matching an exact published checkout. It retains its
resource gates, model paths, run ownership, and private evidence policy. Update
source bindings explicitly for another host. The default branch expectation is
historical; pass the actual reviewed branch. Resume compatibility is not implied
across this layout change.

The old root-only `../quality_gate_supervisor.sh` is also retained solely for the
original tiny/small/quality reproduction. It requires an exact checkout via
`REPO`, an expected commit, live host gates, and the original privilege setup.
Its historical serving probe is fixed to the original gate protocol; use
`serving/launch.py` for normal serving.

See [historical status](../../reports/standardized-accuracy-eval-2026-08-25.md)
from the repository root via `reports/standardized-accuracy-eval-2026-08-25.md`.
