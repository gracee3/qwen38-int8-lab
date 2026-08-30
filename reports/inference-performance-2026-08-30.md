# Interactive inference performance — 2026-08-30

## Claim

On this host's two RTX 3090 GPUs, the local Qwen3.8-27B W8A8 checkpoint served by vLLM 0.27.1 with TP2 and CUDA graphs sustained **39.053 median decode tokens/s** across seven fixed 256-token generations. The observed range was 39.050–39.060 tok/s.

Cold client-observed approximate prefill medians were **1,903.7 tok/s at 1K**, **2,089.5 tok/s at 8K**, **1,935.5 tok/s at 32K**, and **1,778.7 tok/s at 60K**. These rates are prompt tokens divided by client-observed time to first visible generated token, so they include HTTP, scheduler, and first-token sampling overhead.

## Configuration

| Item | Value |
| --- | --- |
| Model | Local `Qwen3.8-27B-W8A8-INT8`, served as `qwen38-w8a8` |
| Runtime | vLLM 0.27.1, native `CutlassInt8ScaledMMLinearKernel` |
| GPUs | 2 × NVIDIA GeForce RTX 3090, 24,576 MiB each, driver 610.43.02 |
| Parallelism | Tensor parallel 2; maximum concurrent sequences 1 |
| Context | 65,536 tokens advertised; 2,684,354,560 BF16 KV bytes per GPU |
| Prefill | Chunked at 2,048 batched tokens; prefix caching enabled |
| Decode | CUDA graphs enabled; 256 forced tokens; temperature 0; seed 42 |
| Features | Text-only, non-thinking, no CPU offload, MTP disabled |
| Host kernel | Ubuntu kernel 7.0.0-29-generic |

## Results

| Prompt target | Actual cold tokens | Cold median tok/s | Cold median TTFT | Warm effective tok/s | Warm median TTFT |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | 1,020–1,022 | 1,903.7 | 0.5368 s | 5,992.9 | 0.1700 s |
| 8,192 | 8,187–8,191 | 2,089.5 | 3.9181 s | 30,954.0 | 0.2646 s |
| 32,768 | 32,765–32,767 | 1,935.5 | 16.9289 s | 58,355.6 | 0.5615 s |
| 60,000 | 59,997–60,000 | 1,778.7 | 33.7324 s | 91,461.5 | 0.6560 s |

Each cold condition has three recorded runs with a unique leading nonce, preventing prefix-cache reuse. Each warm condition has one unrecorded cache-populating request followed by three identical recorded requests. Warm rates are effective application throughput from KV reuse; they are not fresh-compute prefill throughput.

The seven decode observations have a mean of 39.0537 tok/s, median of 39.0534 tok/s, population standard deviation of 0.0032 tok/s, and range of 39.0497–39.0604 tok/s. Decode excludes TTFT and divides the 255 tokens following the first visible token by their elapsed streaming time. The 1,024-token decode prompt was warmed once before recording.

## Evidence and boundaries

The complete per-request JSON is `/data/qwen38-int8-lab/results/benchmark-suite-defaults-20260830T004833Z.json`; the authenticated deterministic smoke result is under `/data/qwen38-int8-lab/results/inference-smoke-defaults-20260830T004808Z.json`. The benchmark harness is `inference/scripts/benchmark.py`.

This supports a throughput claim only for the exact host, model artifact, runtime image, and single-sequence settings above. It does not establish concurrent serving throughput, vision performance, 64K retrieval quality, or standardized accuracy. The 65,536-token setting is a validated serving-capacity default, not a long-context quality result. A speculative MTP trial accepted 0 of 1,524 drafted tokens, so MTP is intentionally excluded from both the default and the performance claim.
