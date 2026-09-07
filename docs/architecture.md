# Architecture

Recipes bind the original BF16 checkpoint to a quantization policy, retained
calibration selection, build environment, and evidence. Every quant starts from
BF16; none quantizes another quant. The three `manifest.json` files identify the
original build commits separately from the layout release commit.

`calibration/` owns data preparation and overlap checks. Original INT8's simple
UltraChat loading remains inside the frozen shared INT8 quantizer; extracting
that behavior is deferred. INT8-v2 consumes its retained Parquet corpus. INT4
preparation uses pinned public sources, coherent tool conversations, task and
prompt deduplication, and frozen held-out screens. Its original manifest says
candidate pending review; the later overlap-clearance report is the approval
evidence bound to the final corpus hash. No report is rewritten to imply a
stronger gate than the historical run actually passed.

`quant/quantize.py` owns the shared offload loader, memory monitor, processor
copying, MTP reinjection, and atomic serialization. `quant/quantize_int4.py` uses
those helpers and `quant/resume_checkpoint.py`; it does not carry a second recovery
engine. Validators live in `validation/` and import the shared checkpoint inspector.
The implementation was moved with import/path adjustments, not retuned.

## Recovery and resource policy

Real quantization uses the existing exclusive lock and temporary swappiness wrapper.
The common monitor stops after ten sustained seconds below 8 GiB available host
RAM or above 32 GiB swap growth. This release does not install a persistent host
policy. The optional installer remains isolated in `tools/host/`.

Rolling recovery captures tensors, calibration intermediates, RNG state, and the
next stage around each 20% boundary. Checksums and atomic pointer publication
retain the previous verified generation until the new one is durable. The final
snapshot remains after successful export. Budget disk for two generations and
final output; never delete retained snapshots as part of repository cleanup.

Resume identity includes source, corpus, configuration, package versions, and
implementation hashes. **A snapshot produced before this layout move must be
resumed using its original build checkout**, not this release. Import edits change
code hashes even when numerical behavior is unchanged. We do not weaken that
check. Completed serving artifacts remain usable; they do not require rebuilding.

## Serving and evaluation

`serving/catalog.py` loads and validates every profile. `resolve()` separates
profile settings, explicit overrides, host bindings, and provenance. The CLI
and future eval consumer share the same engine argument conversion.

The existing local-agent-evals checkout still has copied profile dictionaries.
This PR provides its discovery contract without modifying a second repository.
The consumer update must pin this reviewed commit and replace those dictionaries;
it is not claimed deployed by this layout PR. See [integration](profiles.md).

The old six-group leaderboard runner remains under `validation/legacy_eval/`:
local-agent-evals does not replace GPQA, MATH, MuSR, or the old BF16 comparison.
The original quality supervisor is also retained for historical reproduction;
it has fixed host assumptions and explicit gates. Neither is the normal serving
entrypoint. Their pinned dependencies remain unchanged.
