# INT4 v1 inference profiles

The expanded-400 INT4 v1 checkpoint was measured with vLLM 0.27.1 and the
SM86 FP8 runtime image on the two RTX 3090 host. All requests were text-only,
non-thinking, single-sequence requests with prefix caching and chunked prefill.
The FP8 cache resolved to `float8_e4m3fn`; the checkpoint used Marlin W4A16.

## TP1 default

The single-GPU preset is [`serving/profiles/int4-v1-96k-fp8-tp1.yaml`](../serving/profiles/int4-v1-96k-fp8-tp1.yaml): physical GPU0, TP1, 98,304-token maximum context, and a 3.5 GiB FP8 KV reservation. vLLM reported 104,002 cache tokens and 1.06x maximum concurrency for a 98,304-token request. A 95,998-token prompt plus 64 generated tokens completed successfully.

Cold prefill was approximately 1,206 tok/s at 1K, 1,054 at 32K, 910 at 65K,
830 at 90K, and 813 at 96K. Short steady-state decode was 45.4 tok/s; decode
after a 96K prompt was 42.0 tok/s. GPU0 used 22.22 GiB and retained 1.91 GiB
free after the near-window request. This is a single-sequence maximum-context
preset with limited headroom; 32K and 64K remain better everyday choices.

Evidence: `/data/qwen38-int8-lab/int4-v1/fp8-96k-20260907T140218Z/`.

## TP2 comparison

The dual-GPU comparison preset is [`serving/profiles/int4-v1-96k-fp8-tp2.yaml`](../serving/profiles/int4-v1-96k-fp8-tp2.yaml). It uses the same 98,304-token context and 3.5 GiB FP8 reservation per GPU. vLLM reported 208,005 cache tokens and 2.12x maximum concurrency. A 95,999-token prompt plus 64 generated tokens completed successfully.

Short decode measured 68.37 tok/s, and the near-window request measured 67.74
tok/s. Cold prefill was approximately 1,532 tok/s at 1K, 1,459 at 32K, and
1,319 at 65K; the near-window prefill rate was 1,211 tok/s. Each GPU used
13.64 GiB and retained 10.49 GiB free after the request.

Evidence: `/data/qwen38-int8-lab/int4-v1/fp8-tp2-96k-20260907T142339Z/`.

These are runtime capacity and throughput measurements, not long-context
accuracy or retrieval-quality results. Both containers were stopped cleanly
after measurement and both GPUs returned to idle.
