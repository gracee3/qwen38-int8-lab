# Calibration implementation

Shared preparation and screening scripts live here; payloads do not. Retained
Parquet corpora and manifests stay under the external work root. Recipe manifests
record the hashes used for the release. `prepare_v2_calibration.py` retains its
historical behavior and provenance limitations; see the INT8-v2 recipe before
rerunning it. INT4 preparation, finalization, origin checks, and overlap clearance
are explicit stages of `quant/int4.sh`.

The `calibration/data/` directory is ignored for optional local payloads. Do not
commit datasets, prompts, held-out fixtures, tokenizer caches, or generated records.
