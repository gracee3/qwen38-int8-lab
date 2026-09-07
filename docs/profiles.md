# Serving profile contract v1

Profiles are YAML files under `serving/profiles/`. Their filenames equal their
declared IDs. `serving/catalog.py` is the executable schema: missing, duplicate,
unknown, incompatible, or unsupported settings fail validation.

| Field | Meaning |
| --- | --- |
| `schema_version`, `id` | Format version and stable catalog identity |
| `model_id`, `recipe`, `checkpoint_name` | Compatible checkpoint alias and recipe manifest |
| `image` | Key in `environments/images.lock.json`, including the existing content ID |
| `hardware` | GPU count, 24 GiB per card, recorded SM86 target |
| `runtime` | TP/PP, context, dtype, KV allocation, eager/caching/prefill settings |
| `generation` | Seed 42, non-thinking, vLLM generation defaults |
| `server` | API model name, Qwen XML tools, logging policy |
| `environment` | Recorded vLLM environment switches |
| `evidence` | Validation status, reports, original repository/commit/file |

KV memory is in bytes **per GPU**. Context is total rendered input plus output.
No speculative decoding, CPU offload, or concurrent sequences are enabled.
Sampling and output-token overrides required by benchmarks remain explicit
benchmark protocol settings; a serving default must not replace them silently.

## Local bindings and preview

```sh
python3 serving/launch.py serve int8-v2-16k-bf16-tp2 --dry-run \
  --model /data/models/Qwen3.8-27B-W8A8-INT8-Agentic-v2 --gpus 0,1
python3 serving/launch.py resolve int4-v1-96k-fp8-tp1 \
  --model-id int4-v1 --override max_model_len=65536
```

`MODEL_PATH`, `GPU_DEVICES`, `WORK_ROOT`, `PORT`, and `CUDA_TOOLKIT_ROOT` provide
host bindings; command arguments take precedence. GPU indices default to `0`
for TP1 and `0,1` for TP2. UUIDs are also accepted. Profiles carry no host UUID.
The launcher checks binding count, local image existence, model configuration
presence, and FP8 toolkit presence. Hardware requirements are declared evidence,
not a live resource scheduler: the operator must check idle GPUs and headroom.
It does not hash large weights or claim a path proves checkpoint identity.

Allowed runtime overrides are `max_model_len`, `kv_cache_memory_bytes`, and
`max_num_batched_tokens`. They are validated and recorded separately; an override
does not inherit the profile's measured quality/capacity claim. Changing dtype,
parallelism, or other switches requires a separately reviewed profile.
`--dry-run` has no Docker/filesystem mutation and never prints the API key.
Actual serving requires `VLLM_API_KEY` in the environment.

## Consuming directly from local-agent-evals

Use a clean qwen38-lab checkout at an exact reviewed commit. There is no package
installation or image build required for discovery; Python and PyYAML suffice:

```python
import sys
sys.path.insert(0, '/path/to/pinned/qwen38-lab')
from serving.catalog import discover, resolve, vllm_args, server_args

available = discover()
frozen = resolve('int4-v1-96k-fp8-tp1', model_id='int4-v1',
                 model_path='/data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1',
                 gpu_devices=['0'], overrides={'max_model_len': 16384})
assert not frozen['source']['dirty']
engine_kwargs = vllm_args(frozen)
http_arguments = server_args(frozen)
```

The consumer verifies the expected `source.commit`, discovers supported IDs,
selects one per model, and saves the entire resolved document. It additionally
freezes its own implementation, benchmark selections, generation overrides,
checkpoint weight hashes, and final evaluation overlay image ID. Harness adapters
translate `model`/`pretrained`, chat-template settings, and benchmark batching;
`batch_size` is a harness concern, not a serving profile field.

`source` includes the commit, dirty state, profile path/hash, recipe manifest hash,
and image-lock hash. The resolved `profile`, `bindings`, `image`, and `overrides`
remain distinct. Existing frozen eval runs keep their old implementation and
identities; do not silently reinterpret them through the new catalog.

Adding a compatible profile requires no model-specific dictionary in evals after
its consumer update. Unknown schema versions or unsupported backend settings must
fail rather than be dropped. This repository's CLI is ready; replacing evals'
current hardcoded loader remains a separate consumer change after this one PR.

## Evidence boundaries

The five former YAML presets retain their settings. The old ordinary `serve`
command is preserved as `int8-original-64k-bf16-tp2`; the 262K original preset
records operating configuration, without extending the report's retrieval claims.
The two 262K eval presets retain the values from the pinned eval checkout. INT8-v2
has a bounded smoke report; INT4's 262K preset is explicitly `imported_unvalidated`.
They retain the existing eval runtime image identity, whose Dockerfile is owned by
evals. This freeze does not pretend a different serving image was measured.
