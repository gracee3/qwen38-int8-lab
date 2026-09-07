# Expanded-400 INT4 v1 milestone

Date: 2026-09-07  
Checkpoint: `/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1`  
Build commit: `9628c4e45a60913dfa9f6e234a8c56ffd8b52eb8`

The full real-source GPTQ run completed all 423 finalized calibration
sequences and all 65 sequential stages. The result contains exactly 400
logical W4A16 targets with group size 128, BF16 activations, symmetric scales,
and no activation ordering. Integrity validation passed, including 638
byte-identical preserved tensors, 161 exact offset-norm round trips, all 15
MTP tensors, complete index/shard metadata, and processor files.

Peak process RSS was 69.5 GiB, peak GPU memory was 15.7 GiB, minimum available
host RAM was 23.9 GiB, and swap growth was 10.8 GiB. The resource guard did
not trigger. The final checkpoint was written atomically after validation; the
durable recovery snapshot remains retained separately.

The checkpoint passed a physical-GPU-0 vLLM smoke at 16K context with BF16 KV
and TP1. It generated `READY`, loaded in 16.84 GiB, and profiled `_C::marlin_gemm`
for the fused quantized modules while GPU 1 remained idle.

The local acceptance smoke then ran two fixed seed-42 examples per benchmark:
IFEval 2/2, HumanEval+ 2/2, BBH 2/2, and MMLU-Pro 2/2. All stages completed
with no retries, execution failures or infrastructure errors. One IFEval
response reached its local 1,024-token limit and still passed. This is a
runtime smoke result, not a representative quality score or promotion claim.

Raw evaluation data remains private under `/data/local-agent-evals`; no model
weights or generated code are committed.
