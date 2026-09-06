# Expanded-400 INT4 v1 preparation

This branch is preparation work, not a completed 27B INT4 checkpoint. Full
quantization has not run and is never automatically scheduled. The existing
INT8 code, checkpoint, corpus and serving defaults remain unchanged.

## Verified on 2026-09-06

- Source revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: all 18 shard
  SHA-256 hashes match the pinned local Hugging Face download metadata.
- Meta-instantiated source and tensor headers resolve exactly 400 projections,
  24,326,963,200 weights, and an estimated 18.1166 GiB complete checkpoint.
  The checked-in machine-readable source audit includes every target and hash.
- Synthetic 3-GDN/1-full-attention GPTQ gate: 25 explicit logical targets,
  symmetric INT4 groups of 128, BF16 activations, block size 128, dampening .01,
  `actorder=None` (explicitly overrides the quantizer's static default).
  All excluded synthetic tensors remained unchanged; packed metadata passed.
- vLLM 0.27.1 loaded the synthetic artifact with BF16 activations and BF16 KV,
  produced eight tokens, and profiled `_C::marlin_gemm` plus the CUDA Marlin
  kernel. All 16 fused quantized modules used `MarlinLinearKernel`; all three
  `in_proj_ba` modules used `UnquantizedLinearMethod`.
- The corrected monitored synthetic run used 18 monitor samples, 2,120,544,256
  bytes peak process RSS, 675 MiB GPU memory, and zero swap growth.
- Seven conversation-integrity tests and four existing quality-gate tests passed.
- Final separate corpus: 423 sequences and 1,537,643 tokens. All four source
  budgets and the global long-sequence budget passed. One candidate Rust sample
  was conservatively excluded after a normalized `fix-git` identity match.
- Final overlap clearance passed against 881 frozen cases: 156 MultiPL-E Rust,
  164 HumanEval+, all 541 IFEval cases (including fixed seed-42 evaluation 100),
  10 Multi-SWE-bench Rust/Go cases and 10 Terminal-Bench tasks. Screens include
  prompts, available canonical code/tests, repository patches and all selected
  upstream row identities. No unresolved matches remain. Negative lexical and
  identity checks cannot guarantee absence of every semantic transformation.

Runtime mapping: each GDN layer's `in_proj_qkv` + `in_proj_z` becomes
`in_proj_qkvz`; `in_proj_b` + `in_proj_a` becomes the preserved `in_proj_ba`.
MLP gate/up projections fuse; full-attention Q/K/V projections fuse. This is
why 25 synthetic logical targets become 16 runtime quantized modules.
These results do not prove real 27B load fit, production-shape execution,
FP8 KV quality, 32K capacity, task quality, or soak stability.

## Safe explicit stages

Use this worktree at `/home/emmy/workspace/qwen38-int4-v1`:

```bash
python3 quant/scripts/int4_readiness.py /data/qwen38-int8-lab/int4-v1
bash scripts/int4_prepare.sh audit
bash scripts/int4_prepare.sh synthetic
bash scripts/int4_prepare.sh runtime
bash scripts/int4_prepare.sh corpus
bash scripts/int4_prepare.sh screen-candidate
bash scripts/int4_prepare.sh finalize
bash scripts/int4_prepare.sh screen-final
# After the active INT8 build finishes and host headroom returns:
bash scripts/int4_prepare.sh real-pilot
```

Existing outputs are not overwritten. For a fresh reproduction, set
`INT4_RUN_ROOT` to a new directory under `/data/qwen38-int8-lab` and run stages
in the order above; candidate screening downloads small pinned public fixtures.
The prepared existing run already completed these stages; do not rerun them.
Every stage prints its timestamped log path. Small stages
use at most 6 GiB host memory, no container swap, and two CPUs. GPU stages
bind the physical GPU 1 UUID, which appears as device 0 in the container.
The runtime profiler permits trusted local callback serialization only inside
its network-disabled diagnostic container; it is not a serving configuration.

The real-source pilot requires at least 80 GiB available host RAM, no existing
GPU compute process, 64 GiB free disk, and the protected disk remaining read-only
and unmounted. It has an 84 GiB container memory ceiling and no container swap.
It selects four short samples and the longest actual finalized sequence and
uses the existing sequential/offload loader. It writes a separately marked
experimental checkpoint, verifies packed metadata and preserved tensor bytes,
then atomically renames the completed staging directory. Neither real pilot
nor full build was exercised while INT8 v2 was running.

`full` additionally requires matching successful real-pilot and reviewed
held-out overlap reports. Its only allowed final destination is
`/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1`. Do not create placeholder
passing reports to bypass these gates. A passing artifact is still a candidate;
single-GPU runtime/capacity/quality validation follows the build.

## Corpus boundary

The existing v2 manifest contains zero Nemotron-Agentic samples, null revisions,
and does not establish the required overlap checks. It is retained unchanged.
The separate preparer uses existing cached data pinned to four revisions,
records source IDs and rendered/tokenizer/corpus hashes, excludes separately
provided reasoning fields, normalizes JSON tool arguments, rejects orphan tool
results, and deduplicates task identities and similar user prompts.

Short sequences omit original system messages and tool definitions to avoid
boilerplate dominating the budget; remaining user/assistant/tool message bodies
and complete call/result associations are retained. Long sequences retain the
fuller setup. This deliberate rendering choice is recorded in the manifest.
The 25% long-token allocation is global: 210K CoderForge, 150K Terminal, 15K
Agentic, with Rust allocated coherent short examples. Each source retains its
35/30/25/10 total-token target. Candidate status is not overlap clearance.

The held-out fixtures are now frozen with immutable revisions and file hashes
in `heldout-public/`, `heldout-agents/` and the final screen reports. Final
selection follows that freeze. `overlap-clearance.json` binds the screens and
held-out identities to the final corpus SHA-256. Actual benchmark environment
execution remains a post-build evaluation gate, not a completed quality result.

The remaining pre-build gate is the real-source short/long pilot, which cannot
run safely alongside INT8 v2. Once that pilot passes, `full` can be explicitly
invoked with `INT4_REAL_PILOT_REPORT` set to its report filename and
`INT4_OVERLAP_REPORT=overlap-clearance.json`. There is no automatic launch.

## Evidence and corrected pilot issues

Evidence lives under `/data/qwen38-int8-lab/int4-v1/`:

- `target-audit-hashed.json`, `source-hash.log`
- `monitor-verified/synthetic/pilot-result.json`, `monitor-verified/synthetic.log`
- `runtime-result.json`, `runtime-final.log`
- `corpus-pinned.log`, `calibration-candidate/`
- `calibration-final/`, `overlap-clearance.json`, `clearance.log`
- `overlap-final-public.json`, `overlap-final-agents.json`, `origin-final-schema.json`
- `heldout-public/`, `heldout-agents/`

Earlier attempts are retained: the first synthetic run omitted the monitor's
required `/work/scratch` mount, so its memory report is invalid. The monitored
repeat above fixes this and asserts that the monitor thread remains active.
An initial runtime inspection required trusted callback serialization, and a
second read the scheme from `quant_method` instead of `layer.scheme`. The final
profile corrected both and passed. Corpus iterations corrected Transformers'
`BatchEncoding` return handling and JSON-string tool arguments; failed runs
are not passing gates.
