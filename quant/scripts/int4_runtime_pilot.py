#!/usr/bin/env python3
"""Synthetic vLLM loader/kernel gate; BF16 KV isolates weight compatibility."""
import json
from pathlib import Path


def inspect_model(model):
    result = []
    for name, module in model.named_modules():
        method = getattr(module, 'quant_method', None)
        if method is not None:
            scheme = getattr(module, 'scheme', None)
            kernel = getattr(scheme, 'kernel', None)
            result.append({'name': name, 'method':type(method).__name__, 'scheme':type(scheme).__name__, 'kernel':type(kernel).__name__})
    return result


def begin_profile(model):
    import torch
    model._int4_profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    model._int4_profiler.__enter__()


def end_profile(model):
    import torch
    torch.cuda.synchronize()
    profiler = model._int4_profiler
    profiler.__exit__(None, None, None)
    names = sorted({event.name for event in profiler.events() if 'marlin' in event.name.lower()})
    del model._int4_profiler
    return names


def main():
    from vllm import LLM, SamplingParams
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU must be visible')
    llm = LLM(model='/run-int4/synthetic', dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=256, max_num_seqs=1, max_num_batched_tokens=256,
              gpu_memory_utilization=0.15, kv_cache_memory_bytes=64*1024**2,
              kv_cache_dtype='bfloat16', enforce_eager=True, language_model_only=True,
              enable_prefix_caching=False, enable_chunked_prefill=True, seed=42)
    mapping = llm.apply_model(inspect_model)
    print(json.dumps(mapping, indent=2), flush=True)
    llm.apply_model(begin_profile)
    output = llm.generate([{'prompt_token_ids':[2,3,4,5]}], SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
    assert len(output[0].outputs[0].token_ids)==8
    kernels = llm.apply_model(end_profile)
    assert kernels and all(kernels), 'No profiled Marlin execution'
    eligible = [m for m in mapping[0] if m['scheme']=='CompressedTensorsWNA16']
    assert len(eligible) == 16, eligible
    assert all(m['kernel']=='MarlinLinearKernel' for m in eligible), eligible
    controls = [m for m in mapping[0] if m['name'].endswith('in_proj_ba')]
    assert len(controls)==3 and all(m['method']=='UnquantizedLinearMethod' for m in controls), controls
    Path('/run-int4/runtime-result.json').write_text(json.dumps({'status':'passed', 'scope':'synthetic_only_bf16_kv', 'mapping':mapping, 'profiled_marlin_events':kernels, 'generated_tokens':8}, indent=2)+'\n')


if __name__ == '__main__':
    main()
