# Qwen3.8-27B architecture and initial W8A8 policy

## Observed checkpoint

The official local checkpoint identifies itself as `Qwen3_5ForConditionalGeneration` / `qwen3_5`, configured for BF16. The complete Safetensors index contains 1,199 tensors in 18 shards and declares 55,562,855,904 tensor bytes.

The text tower has 64 decoder layers:

- 48 `linear_attention` layers implemented as Qwen3.5 Gated DeltaNet (GDN), with Q/K/V-style input projections, gate/time projections, a depthwise Conv1d, recurrent state parameters, gated RMSNorm, and an output projection.
- 16 conventional full-attention layers, one every fourth layer.
- Three MLP projections in every layer.

The checkpoint also contains a 27-layer vision tower, untied embeddings and `lm_head`, and 15 top-level tensors for one MTP layer. The Hugging Face conditional-generation class does not instantiate the top-level `mtp.*` tensors and explicitly tolerates them as unexpected checkpoint keys.

Qwen3.8's text RMSNorm is 1-centered: its stored parameter is initialized to zero and the forward path multiplies by `(1 + weight)`. That differs from a conventional RMSNorm parameter initialized to one.

## Initial production target

The first quality checkpoint uses `GPTQModifier` with `scheme="W8A8"`, sequential onloading at `Qwen3_5DecoderLayer` boundaries, per-channel INT8 weights, and dynamic per-token INT8 inputs. The resolved target set is expected to contain exactly:

- 192 text MLP linears (three per decoder layer);
- 64 full-attention projections (Q/K/V/O in 16 layers);
- no other linears.

This deliberately produces a mixed BF16/INT8 checkpoint. It is intended to establish correct native CUTLASS INT8 execution on ordinary text GEMMs before expanding coverage.

## Preserved components

| Component | Initial handling | Reason |
| --- | --- | --- |
| Token embeddings | BF16 | Accuracy-sensitive lookup table; not an INT8 GEMM target. |
| `lm_head` | BF16 | Untied 248,320-token output projection; preserve logits quality. |
| RMSNorm and gated norms | BF16 | Not Linear, and generic SmoothQuant assumptions are unverified for 1-centered parameters. |
| Vision tower and merger | BF16 | Sensitive multimodal path; not needed for the first text-serving gate. |
| GDN/recurrent path | BF16 | Architecture-sensitive projections, depthwise convolution, recurrent state, and gating should be evaluated before coverage expansion. |
| MTP layer | BF16, re-injected | Transformers omits the top-level tensors; serialization must explicitly restore all 15 before completion. MTP inference remains disabled initially. |

SmoothQuant is disabled in the first recipe. It can be reconsidered only with explicit Qwen3.8 norm/GDN mappings and an A/B quality result.

## Calibration progression

1. Four synthetic 64-token samples on a tiny instance of the exact Qwen architecture validate imports, target matching, the GPTQ API, and compressed serialization.
2. Two local 128-token prompts provide the first real-source staged calibration gate without a dataset download.
3. A 32-sample, 512-token cached experiment estimates quality and memory.
4. The quality candidate uses 512 representative chat samples at up to 2,048 tokens, cached under `/work/cache`.

The full quality run is not authorized by an ordinary script invocation: `quantize.py` requires `--execute-full`, refuses an existing output, serializes under an incomplete staging name, restores MTP, and only then renames to the final path. Non-quality real-source profiles additionally require an explicit output below `/work/scratch` and are recorded as experimental rather than production-complete.

## Runtime acceptance

The installed vLLM environment selected `CutlassInt8ScaledMMLinearKernel` when its W8A8 selector was exercised directly on SM86. Successful checkpoint loading is still necessary but insufficient: the actual model log must contain the same pairing for `CompressedTensorsW8A8Int8`, generation must be deterministic and coherent, and both GPUs must participate at tensor parallel size 2. Those checkpoint-level gates remain future work.

Targeting each `Linear` as a sequential subgraph is invalid for this multimodal wrapper and reproduced an LLM Compressor AST-tracing `KeyError: forward`. `Linear` remains the modifier target; `Qwen3_5DecoderLayer` is the tested sequential subgraph target.

References: [LLM Compressor INT8 W8A8 guide](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int8_w8a8/), [LLM Compressor memory guidance](https://docs.vllm.ai/projects/llm-compressor/en/stable/guides/memory/), and [vLLM quantization hardware matrix](https://docs.vllm.ai/en/stable/features/quantization/).
