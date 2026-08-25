# Standardized W8A8 accuracy evaluation — 2026-08-25

## Outcome

GPQA access is now accepted and was verified at the pinned revision. Fresh exact-commit run `20260825T031746Z` prefetched and validated all six pinned datasets offline, then stopped at the mandatory request-length preflight before loading either model.

The first incompatible request was `leaderboard_gpqa_extended` document ID 356: its task-defined few-shot log-likelihood request renders to 12,314 tokens, exceeding the 8,191-token runtime request limit for the fixed 8,192-token context protocol. The supervisor did not truncate or retry it.

No smoke or scored group ran. There are no W8A8 scores, BF16 scores, retention result, or deployment recommendation.

## Published implementation

Draft PR #10 contains the exact run commit `86a3af64077927f1b486e5541654955da61b9e07`. The evaluation image was rebuilt from that commit and recorded as `sha256:c0d2de65c61fe66ba62e4cc9270d1e295e75a28f25249901e4e722fc166535e8`; its OCI revision label matches the commit.

The image passed `pip check` and all 10 unit tests. Python compilation, Bash syntax, ShellCheck, and YAML parsing passed. Dataset prefetch, expected split counts and fingerprints, upstream task validation, initial host preflight, and the preflight before each completed stage passed. The request preflight failed closed on the overlength GPQA request, as required.

## Safety and evidence

Both GPUs remained idle at 1 MiB and no compute process remained after shutdown. The protected NVMe remained read-only and unmounted. Neither model was loaded; their read-only pre-run checkpoint identities were:

- W8A8: `fbc02d65e68d4c06c9f9b665624498cdac07bb7fbd46e5fe087a54cf7bbe207d`
- BF16: `30cb5077377a40d1b31667dc05e2a3bd3b229284f5289a9956691021e1352db6`

Private evidence is under `/data/qwen38-int8-lab/evaluations/20260825T031746Z`. GPQA rows remain private; the committed JSON contains only revisions, hashes of the count/fingerprint manifests, and the blocker.

## Next action

Choose and review a protocol change before another run. The fixed Open LLM Leaderboard v2 GPQA group cannot be evaluated without truncation under the current 8,192-token context. Plausible changes are increasing the evaluation context enough for every rendered request or defining a different GPQA task/few-shot protocol; either changes the approved protocol and requires a new exact-commit run. Do not report partial smoke output as accuracy and do not make a retention or deployment decision unless all four BF16 comparisons complete.
