# Qwen3.5-27B GGUF long-context profile

This optional inference path keeps the calibrated Qwen3.8 W8A8/vLLM profile
unchanged. It serves the text-only `unsloth/Qwen3.5-27B-GGUF` Q4_K_M artifact
through a pinned CUDA-enabled llama.cpp build.

## Pinned inputs

- llama.cpp: `c845263f8b7d60113e213a3bd2d5cc6472ccf204`
- GGUF repository revision: `3221f178a6b842d04f1fb42f1c413534adcc0a6a`
- GGUF file: `Qwen3.5-27B-Q4_K_M.gguf`
- Expected size: `16,740,812,704` bytes
- Expected SHA-256:
  `84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806`
- Default model path:
  `/home/emmy/workspace/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf`

The checked-in launcher refuses a different llama.cpp commit or a GGUF that
does not match the pinned size and SHA-256. Model metadata and a complete load
must still pass before the downloaded artifact is accepted.

## Memory policy

The candidate profile uses one server slot, a 131,072-token context, Q8_0 K/V
cache, flash attention, and compatible layer splitting at `3,1`. Most model
layers and their cache live on GPU 0, leaving GPU 1 available for a separate
native-ASR process. Set `LLAMA_ALLOW_BUSY_GPUS=1` only during a deliberate
co-residency test.

Tensor split mode is excluded from the stable profile. It is experimental and
cannot currently be combined with quantized KV cache. The 163,840-token profile
is also experimental until it passes load, retrieval, multi-turn prompt-cache,
Qwen Code tool-use, soak, and ASR co-residency gates.

## Build and serve

Build outside the repository so generated objects stay under the workspace:

```bash
git -C /home/emmy/workspace/git/llama.cpp switch --detach c845263f8b7d60113e213a3bd2d5cc6472ccf204
cmake -S /home/emmy/workspace/git/llama.cpp \
  -B /home/emmy/workspace/git/llama.cpp/build-cuda-shared \
  -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /home/emmy/workspace/git/llama.cpp/build-cuda-shared \
  --config Release -j 8 --target llama-server llama-cli llama-bench
```

Start the 128K candidate on loopback port 8000:

```bash
just serve-llama
```

Start the separately labeled 160K experiment:

```bash
just serve-llama-160k
```

From another shell, run the ordinary smoke and then a multi-depth retrieval
probe near the 128K candidate boundary:

```bash
just smoke-llama
just probe-llama 120000
```

For the 160K experiment, use `just probe-llama 150000`. The probe records only
token counts, timing, usage, and pass/fail for fixed synthetic codes; it does
not persist the large generated prompt or response text.

The API key defaults to `local-qwen-only`, the model alias defaults to
`qwen35-27b-q4km`, and both can be overridden with `LLAMA_API_KEY` and
`LLAMA_MODEL_ALIAS`. `LLAMA_TENSOR_SPLIT` controls the layer allocation.

## Promotion gates

Do not call either advertised window stable based on allocation alone. Record:

1. a deterministic authenticated smoke request and native CUDA load evidence;
2. actual per-GPU idle and peak memory, including at least 10 GiB free on GPU 1;
3. retrieval checks at multiple depths through the advertised window;
4. incremental multi-turn prompt-cache behavior for the hybrid DeltaNet model;
5. a real Qwen Code read/edit/tool loop without malformed tool arguments;
6. at least a one-hour generation/prefill soak; and
7. a native-ASR co-residency run without OOM, reset, or material latency failure.

Promote 131K first. Keep 160K experimental until every gate passes.
