# Standardized W8A8 accuracy evaluation — 2026-08-25

## Outcome

Exact-commit run `20260825T023823Z` stopped at dataset prefetch before loading either model. The existing Hugging Face credential is present but has not accepted access to gated `Idavidrein/gpqa` at revision `633f5ee89ab8ad4522a9f850766b73f62147ffdd`.

No smoke or scored group ran. There are no W8A8 scores, BF16 scores, retention result, or deployment recommendation.

## Published implementation

Draft PR #10 contains implementation commit `19a133e92e39cef27fcd6089facb03f365589770`. The run used an evaluation image built from that exact commit, with image ID `sha256:10f14687071b3219599618d8d978b17962ae6a1c49c8d08e90dd22522faaa736`.

The image passed `pip check`, all 10 image-side unit tests, and the exact upstream leaderboard YAML hash gate. Python compilation, Bash syntax, ShellCheck, YAML parsing, and Justfile dry-runs passed. MMLU-Pro, BBH, MATH Level 5, IFEval, and MuSR task definitions validated offline from their pinned revisions. GPQA samples were neither available nor logged.

## Safety and evidence

The initial and dataset-stage host preflights passed with both GPUs idle. The protected NVMe remained read-only and unmounted, SSH/Docker/containerd remained active, and the default route remained on Wi-Fi. Before/after checkpoint metadata manifests were identical:

- W8A8: `fbc02d65e68d4c06c9f9b665624498cdac07bb7fbd46e5fe087a54cf7bbe207d`
- BF16: `30cb5077377a40d1b31667dc05e2a3bd3b229284f5289a9956691021e1352db6`

Private evidence is under `/data/qwen38-int8-lab/evaluations/20260825T023823Z`. The run contains no raw GPQA samples and no model outputs.

## Next action

Accept the GPQA dataset terms for the existing Hugging Face account, confirm that the current token can read the pinned revision, then start a new timestamped exact-commit supervisor run. Do not report limited smoke output as accuracy and do not make a retention or deployment decision unless all four BF16 comparisons complete.
