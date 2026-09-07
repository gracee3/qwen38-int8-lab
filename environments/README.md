# Frozen environments

No image was built, pulled, retagged, or deleted for this layout release.
`images.lock.json` records existing local content IDs on 2026-09-07. These are
Docker image IDs, not registry manifest digests or publicly downloadable images.
The launcher uses those IDs with `--pull never`. Original tags remain valid.

| Directory | Existing tag | Purpose |
| --- | --- | --- |
| `quant/` | `qwen38-int8-lab/quant:0.1.0` | Torch 2.13.0, LLM Compressor 0.13.0, compressed-tensors 0.18.0 |
| `serving/` | `qwen38-int8-lab/vllm:0.1.0` | vLLM 0.27.1, compressed-tensors 0.17.0 |
| `serving-fp8-sm86/` | `qwen38-int8-lab/vllm-fp8-sm86:0.1.0` | Existing compiler overlay; read-only host CUDA 13.3 mount |
| `legacy-eval/` | `qwen38-int8-lab/eval:0.1.0` | Existing lm-eval 0.4.12 dependency base |

The separately owned `local-agent-evals/runtime:0.1.0` overlay is recorded only
because existing 262K eval presets used it. Its source commit and owner are in
the lock; this repository does not copy or rebuild it. Moving all evaluation
dependencies to evals is deferred until its coverage/consumer migration.

Dependency intent and lock files are byte-identical to the baseline. Dockerfile
changes are limited to `COPY` source paths after relocation. Existing container
entrypoints, parent identities, versions, labels, and tags remain frozen. The
quant image still runs bind-mounted code from `/app`, so code relocation does
not require rebuilding its dependency layers.

Explicit future build commands remain `just build-quant`, `just build-vllm`,
`just build-vllm-fp8-sm86`, and `just build-legacy-eval`. They are never prerequisites
of list/resolve/serve/test. Do not run them to apply this layout migration.

A later local retag can point to an existing ID without a build. Do not replace
recorded content IDs merely because a tag name changes. A rebuild may produce a
different ID even with the same versions; record and revalidate that environment
in a separate change. Keep quant and serving environments separate because their
compressed-tensors versions differ.
