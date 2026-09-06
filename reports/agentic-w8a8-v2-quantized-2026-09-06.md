# Agentic W8A8 v2 quantization milestone

Date: 2026-09-06  
Source commit: `cecc0969478538511409ac052db0acb863f1b9a2`  
Output: `/data/models/Qwen3.8-27B-W8A8-INT8-Agentic-v2`

The full sequential GPTQ run completed successfully on the isolated physical
GPU 0. It processed 307 calibration sequences (maximum 16,384 tokens),
completed all 65 sequential stages, and wrote durable checkpoints at stages
13, 26, 39, 52, and 65. The final recovery snapshot remains retained at
`/data/models/.Qwen3.8-27B-W8A8-INT8-Agentic-v2.resume`.

Run duration was 12,214.6 seconds (about 3 h 24 m). The resource guard did not
trigger: peak process RSS was 71.1 GiB, minimum available host RAM was 32.2
GiB, peak GPU memory was 16.8 GiB, and swap growth was 4.55 GiB against the
32 GiB limit. Persistent host `vm.swappiness=1` was active throughout.

Structural validation passed:

- all 35 output Safetensors shards and the index are complete;
- compressed-tensors W8A8 metadata is valid;
- all 256 intended projection modules are quantized;
- all 15 MTP tensors are present in the preserved sidecar;
- processor configuration files are valid.

Preservation validation found 767 source tensors byte-identical, including all
15 MTP tensors. The only differences among non-quantized source tensors were
the 161 normalization tensors intentionally adjusted by the calibration
normalization pass. No unexpected non-normalization differences were found.

The vLLM smoke test loaded the artifact with TP2 across both RTX 3090s at a
16,384-token context and generated the exact requested response. The server
was stopped after the test. The narrow validator log pattern for CUTLASS
dispatch was not emitted by this vLLM build, so kernel dispatch should be
confirmed during the planned serving/evaluation run rather than claimed here.

This milestone establishes a usable quantized checkpoint, not an accuracy
result. Overnight evaluation remains the next gate: compare against the
existing W8A8 checkpoint, test Qwen Code tool calls, and then measure long
context behavior. Do not start the expanded INT4 quantization concurrently;
it competes for host RAM and the same exclusive quantization lock.

Machine-readable run metadata is in
`/data/qwen38-int8-lab/results/quant-quality-20260906T170139Z.json`; validation
summary is in `/data/qwen38-int8-lab/results/validation/summary-20260906.json`.
