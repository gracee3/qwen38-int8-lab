# Smoke-test record — 2026-08-24

## Quantization gate

`just quant-smoke` passed with the quant image and a deliberately tiny synthetic `Qwen3_5ForConditionalGeneration`. It exercised the production sequential pipeline at `Qwen3_5DecoderLayer` boundaries, four 64-token calibration samples, GPTQ, compressed-tensors serialization, and the independent checkpoint validator.

Observed serialization metadata:

- `quant_method`: `compressed-tensors`
- format/status: `int-quantized` / `compressed`
- weights: symmetric 8-bit, per-channel, static scales
- input activations: symmetric 8-bit, dynamic per-token
- artifact size: about 1.7 MiB; explicitly marked non-production

Measured peaks for the committed-lock sequential run (`20260824T203507Z`, Git `5bfa399f561fcd935e8af883df1c58d8afe97610`):

- process RSS: 2,110,812,160 bytes (about 1.97 GiB)
- GPU 0: 364 MiB
- GPU 1: 266 MiB
- quantization elapsed time: 3.17 seconds

These measurements validate instrumentation and orchestration only. They are not estimates of the full 27B run.

The first sequential attempt used `Linear` as both the modifier target and the sequential subgraph target. It failed before calibration with `KeyError: forward` in LLM Compressor's AST tracer. Switching only the sequential boundary to `Qwen3_5DecoderLayer` passed; the modifier continues to target the explicitly filtered Linear modules.

Machine-readable run metadata and the disposable checkpoint are outside Git under `/data/qwen38-int8-lab/results` and `/data/qwen38-int8-lab/scratch`.

## Inference environment gate

The vLLM image passed `pip check`, saw both RTX 3090 GPUs as SM86, registered `Qwen3_5ForConditionalGeneration`, and directly exercised the installed W8A8 kernel selector. The emitted evidence was:

```text
Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

This is environment and kernel-eligibility evidence. It is not checkpoint-dispatch evidence: no complete 27B INT8 checkpoint exists yet, so TP2 loading, actual layer dispatch, deterministic generation, and benchmark gates remain open.
