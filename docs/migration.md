# Layout migration

Baseline: `52de78d4c7fc2672a118802e02f742b701f02b37`, tagged
`baseline-2026-09-07`. This migration is one PR based on that remote main commit.
The prior feature checkout and its untracked handoff are preserved separately.

| Baseline area | Decision |
| --- | --- |
| `quant/config/` | Move each policy into a versioned recipe with manifest and README |
| `quant/scripts/` | Split by calibration, quantization/recovery, and validation responsibility |
| `inference/config/` | Convert to discoverable serving profiles; preserve effective settings |
| Ordinary `just serve` settings | Preserve as the explicit original 64K BF16 profile |
| Imported eval presets | Record existing values and image identity with external provenance |
| `docker/` | Move required environments; preserve dependency bytes and image tags |
| `eval/` | Retain as `validation/legacy_eval/` until coverage is replaced |
| `scripts/` | Move stage runners near their code; isolate optional host tools |
| Reports/tests | Retain evidence and coverage, update repository references/imports |
| Root handoffs/preparation proposals | Replace with maintained recipe/architecture/reproduction docs |
| Qwen3.5 GGUF/llama.cpp and PP2 launch experiments | Retire from current commands/tree; preserve Git history |

[layout-map.json](layout-map.json) records individual moved and retired paths.
Unlisted existing report/test/control files are retained. Dated raw evidence paths
remain historical even when host locations have since changed. Small report links
are updated; measured values and machine-readable historical reports are preserved.

There are no model, calibration, cache, or raw-run moves. No protected personal
files are touched. Dependency intent/locks are unchanged byte-for-byte; Dockerfile
edits only follow relocated COPY inputs. No image build, download, or retag is
necessary. The repository name is qwen38-lab; old image/data names intentionally
remain stable compatibility identifiers.

New code is confined to catalog discovery/resolution, the serving launcher, and
contract checks. Existing algorithms are relocated with imports/path bindings
updated. The common recovery adapter remains byte-identical. Historical snapshots
still require their original implementation hashes, as explained in
[architecture](architecture.md).

This is a frozen layout release candidate with inherited evidence. It does not
claim that newly resolved commands were GPU revalidated, that INT8-v2 upstream
provenance became complete, or that local-agent-evals has already switched to the
catalog. Those facts remain explicit in the release procedure and recipe docs.
