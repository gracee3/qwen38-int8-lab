# Agentic INT8 v2

The completed checkpoint uses the original 256-target W8A8 policy, block size
128, dampening 0.01, and unchanged preservation/serialization rules. The new
calibration corpus changes the checkpoint; it does not add recurrent targets.
The abandoned expanded-INT8 proposal is not a supported recipe.

The actual retained corpus has **307 sequences and 1,371,502 tokens**, maximum
16,384 tokens. It contains CoderForge, Nemotron-Terminal, and Strandset-Rust.
The original manifest records null upstream revisions and **zero Nemotron-Agentic
samples**, following a preparation error. The four-source proportions in
[quant.yaml](quant.yaml) describe the original intent, not the observed mixture.
We preserve that configuration without rewriting the historical facts.

[manifest.json](manifest.json) records the retained Parquet and manifest hashes,
original build commit, source revision, and checkpoint metadata. Exact corpus
reuse requires the retained payload; rerunning preparation against current
upstream data is not claimed equivalent. These hashes identify local files,
not a redistributed dataset or a full weight-hash manifest.

Use the retained corpus with `just v2-quant-small` and, only when explicitly
scheduled into an absent output, `just v2-quant`. See
[reproduction](../../docs/reproduction.md) for bindings, resource limits, and
resume identity. The existing completed output remains available.

[Build validation](../../reports/agentic-w8a8-v2-quantized-2026-09-06.md),
[paired 16K results](../../reports/paired-16k-calibration-baseline-2026-09-07.md),
and the [262K bounded smoke](../../reports/int8-v2-262k-smoke-2026-09-07.md)
are separate evidence. The 262K eval preset retains 4.5 GiB KV per GPU and its
existing eval overlay identity. Neither a short smoke nor a large configured
window establishes broad long-context quality.
