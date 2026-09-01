set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

repo_root := justfile_directory()
model_root := env_var_or_default("MODEL_ROOT", "/data/models")
work_root := env_var_or_default("WORK_ROOT", "/data/qwen38-int8-lab")
source_model := env_var_or_default("SOURCE_MODEL", "/home/emmy/workspace/qwen3.8-27b-download/model")
output_model := env_var_or_default("OUTPUT_MODEL", model_root + "/Qwen3.8-27B-W8A8-INT8")
quant_image := env_var_or_default("QUANT_IMAGE", "qwen38-int8-lab/quant:0.1.0")
vllm_image := env_var_or_default("VLLM_IMAGE", "qwen38-int8-lab/vllm:0.1.0")
vllm_fp8_image := env_var_or_default("VLLM_FP8_IMAGE", "qwen38-int8-lab/vllm-fp8-sm86:0.1.0")
llama_image := env_var_or_default("LLAMA_IMAGE", "qwen38-int8-lab/llama:0.1.0")
eval_image := env_var_or_default("EVAL_IMAGE", "qwen38-int8-lab/eval:0.1.0")
eval_context := "16384"
port := env_var_or_default("PORT", "8000")
served_model := env_var_or_default("SERVED_MODEL", "qwen38-w8a8")
api_key := env_var_or_default("VLLM_API_KEY", "local-qwen-only")
inference_context := env_var_or_default("INFERENCE_CONTEXT", "65536")
inference_kv_cache_bytes := env_var_or_default("INFERENCE_KV_CACHE_BYTES", "2684354560")
cuda_toolkit_root := env_var_or_default("CUDA_TOOLKIT_ROOT", "/usr/local/cuda-13.3")
llama_api_key := env_var_or_default("LLAMA_API_KEY", "local-qwen-only")
llama_model_alias := env_var_or_default("LLAMA_MODEL_ALIAS", "qwen35-27b-q4km")
llama_gguf_model := env_var_or_default("LLAMA_GGUF_MODEL", "/home/emmy/workspace/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf")
llama_tensor_split := env_var_or_default("LLAMA_TENSOR_SPLIT", "3,1")

default:
    @just --list

info:
    SOURCE_MODEL="{{source_model}}" WORK_ROOT="{{work_root}}" "{{repo_root}}/scripts/host_info.sh"

build-quant:
    DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg VCS_REF="$(git -C "{{repo_root}}" rev-parse HEAD)" -t "{{quant_image}}" -f "{{repo_root}}/docker/quant/Dockerfile" "{{repo_root}}"

gpu:
    docker run --rm --gpus all --entrypoint python "{{quant_image}}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}"); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]'

shell-quant:
    docker run --rm -it --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e PIP_CACHE_DIR=/work/cache/pip -w /app "{{quant_image}}" /bin/bash

inspect-model:
    docker run --rm --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -w /app "{{quant_image}}" python /app/quant/scripts/inspect_model.py /models/source --instantiate-meta --json-out /work/results/source-model-inspection.json

quant-plan:
    docker run --rm --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile smoke --plan-only

quant-smoke:
    docker run --rm --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app --entrypoint /bin/bash "{{quant_image}}" -lc 'set -euo pipefail; python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile smoke; latest=$(find /work/scratch -maxdepth 1 -type d -name "quant-smoke-*" -printf "%f\n" | sort | tail -n 1); test -n "$latest"; python /app/quant/scripts/validate_quant.py "/work/scratch/$latest" --synthetic --json-out /work/results/quant-smoke-validation-latest.json'

load-trace:
    mkdir -p "{{work_root}}/logs" "{{work_root}}/results" "{{work_root}}/scratch"; stamp=$(date -u +%Y%m%dT%H%M%SZ); log="{{work_root}}/logs/load-trace-$stamp.log"; echo "load/trace log: $log"; docker run --rm --gpus all --ipc=host --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile tiny_source --load-trace-only 2>&1 | tee "$log"

quant-tiny:
    mkdir -p "{{work_root}}/logs" "{{work_root}}/results" "{{work_root}}/scratch"; stamp=$(date -u +%Y%m%dT%H%M%SZ); output="/work/scratch/Qwen3.8-27B-W8A8-tiny-source-$stamp"; log="{{work_root}}/logs/quant-tiny-$stamp.log"; echo "experimental non-production output: $output"; echo "quant log: $log"; docker run --rm --gpus all --ipc=host --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile tiny_source --output "$output" --execute-full 2>&1 | tee "$log"

dataset-preflight profile="quality":
    mkdir -p "{{work_root}}/results"; docker run --rm --gpus all --ipc=host --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile "{{profile}}" --dataset-preflight-only

quant-real profile output:
    mkdir -p "{{work_root}}/logs" "{{work_root}}/results" "{{work_root}}/scratch"; log="{{work_root}}/logs/quant-{{profile}}-$(date -u +%Y%m%dT%H%M%SZ).log"; echo "quant log: $log"; docker run --rm --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile "{{profile}}" --output "{{output}}" --execute-full 2>&1 | tee "$log"

quant-small:
    stamp=$(date -u +%Y%m%dT%H%M%SZ); just quant-real small "/work/scratch/Qwen3.8-27B-W8A8-small-$stamp"

quant:
    mkdir -p "{{work_root}}/logs"; log="{{work_root}}/logs/quant-$(date -u +%Y%m%dT%H%M%SZ).log"; echo "quant log: $log"; docker run --rm --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile quality --output /models/$(basename "{{output_model}}") --execute-full 2>&1 | tee "$log"

build-vllm:
    DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg VCS_REF="$(git -C "{{repo_root}}" rev-parse HEAD)" -t "{{vllm_image}}" -f "{{repo_root}}/docker/vllm/Dockerfile" "{{repo_root}}"

build-vllm-fp8-sm86:
    DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg VLLM_IMAGE="{{vllm_image}}" --build-arg VCS_REF="$(git -C "{{repo_root}}" rev-parse HEAD)" -t "{{vllm_fp8_image}}" -f "{{repo_root}}/docker/vllm-fp8-sm86/Dockerfile" "{{repo_root}}"

build-llama:
    DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg VCS_REF="$(git -C "{{repo_root}}" rev-parse HEAD)" -t "{{llama_image}}" -f "{{repo_root}}/docker/llama/Dockerfile" "{{repo_root}}"

build-eval:
    base_id=$(docker image inspect "{{vllm_image}}" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["Id"])'); test "$base_id" = "sha256:60508d8dcbbb0a985955e9cf2f66e561a66c3f1c99bd7ec8fa5020e991a0ef4d"; DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg VLLM_IMAGE="{{vllm_image}}" --build-arg VCS_REF="$(git -C "{{repo_root}}" rev-parse HEAD)" -t "{{eval_image}}" -f "{{repo_root}}/docker/eval/Dockerfile" "{{repo_root}}"

eval-dataset-preflight cache_root output:
    test ! -e "{{cache_root}}"; install -d -m 700 "{{cache_root}}"; output_parent=$(dirname "{{output}}"); install -d "$output_parent"; docker run --rm --user "$(id -u):$(id -g)" --mount type=bind,src="{{cache_root}}",dst=/cache --mount type=bind,src="$output_parent",dst=/out --mount type=bind,src="{{repo_root}}",dst=/app,readonly --env HF_TOKEN --env HF_HOME=/cache/huggingface --env HF_DATASETS_CACHE=/cache/huggingface/datasets --entrypoint python "{{eval_image}}" /app/eval/scripts/prefetch.py --output "/out/$(basename "{{output}}")"

eval-request-preflight cache_root output:
    output_parent=$(dirname "{{output}}"); install -d "$output_parent"; docker run --rm --user "$(id -u):$(id -g)" --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{source_model}}",dst=/models/bf16,readonly --mount type=bind,src="{{cache_root}}",dst=/cache --mount type=bind,src="$output_parent",dst=/out --mount type=bind,src="{{repo_root}}",dst=/app,readonly --env HF_HOME=/cache/huggingface --env HF_DATASETS_CACHE=/cache/huggingface/datasets --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1 --entrypoint python "{{eval_image}}" /app/eval/scripts/request_preflight.py --candidate "/models/$(basename "{{output_model}}")" --source /models/bf16 --output "/out/$(basename "{{output}}")" --maximum-request-output /out/maximum-loglikelihood-request.json

eval-smoke run_root:
    test -f "{{run_root}}/dataset-preflight.json"; docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{run_root}}",dst=/run --mount type=bind,src="{{repo_root}}",dst=/app,readonly --env HOME=/run/home --env HF_HOME=/run/cache/huggingface --env HF_DATASETS_CACHE=/run/cache/huggingface/datasets --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1 --env VLLM_USE_FLASHINFER_SAMPLER=0 --entrypoint python "{{eval_image}}" /app/eval/scripts/run_harness.py run --model vllm --model_args "pretrained=/models/$(basename "{{output_model}}"),dtype=bfloat16,tensor_parallel_size=2,max_model_len={{eval_context}},kv_cache_dtype=bfloat16,seed=42,enforce_eager=True,enable_prefix_caching=False,add_bos_token=False,enable_thinking=False,cpu_offload_gb=0,language_model_only=True,enable_chunked_prefill=True,max_num_batched_tokens=1024,kv_cache_memory_bytes=805306368" --tasks leaderboard --limit 2 --batch_size auto --max_batch_size 1 --seed 42 --apply_chat_template --fewshot_as_multiturn --log_samples --output_path /run/smoke

eval-validate cache_root:
    docker run --rm --user "$(id -u):$(id -g)" --mount type=bind,src="{{cache_root}}",dst=/cache --mount type=bind,src="{{repo_root}}",dst=/app,readonly --env HF_HOME=/cache/huggingface --env HF_DATASETS_CACHE=/cache/huggingface/datasets --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1 --entrypoint python "{{eval_image}}" /app/eval/scripts/run_harness.py validate --tasks leaderboard

eval-standardized expected_commit:
    EXPECTED_COMMIT="{{expected_commit}}" EVAL_IMAGE="{{eval_image}}" "{{repo_root}}/scripts/accuracy_eval_supervisor.sh"

eval-candidate-only expected_commit:
    EVAL_SCOPE=candidate-only EXPECTED_COMMIT="{{expected_commit}}" EVAL_IMAGE="{{eval_image}}" "{{repo_root}}/scripts/accuracy_eval_supervisor.sh"

shell-vllm:
    docker run --rm -it --gpus all --ipc=host --entrypoint /bin/bash --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -w /app "{{vllm_image}}"

validate:
    docker run --rm --gpus all --ipc=host --entrypoint python --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -w /app "{{vllm_image}}" /app/quant/scripts/validate_quant.py /models/$(basename "{{output_model}}")

serve:
    mkdir -p "{{work_root}}/logs"; log="{{work_root}}/logs/vllm-$(date -u +%Y%m%dT%H%M%SZ).log"; echo "vLLM log: $log"; docker run --rm --gpus all --ipc=host -p "127.0.0.1:{{port}}:8000" --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e VLLM_CACHE_ROOT=/work/cache/vllm -e VLLM_USE_FLASHINFER_SAMPLER=0 "{{vllm_image}}" /models/$(basename "{{output_model}}") --served-model-name "{{served_model}}" --api-key "{{api_key}}" --tensor-parallel-size 2 --max-model-len "{{inference_context}}" --kv-cache-memory-bytes "{{inference_kv_cache_bytes}}" --kv-cache-dtype bfloat16 --cpu-offload-gb 0 --seed 42 --language-model-only --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser qwen3_xml --default-chat-template-kwargs '{"enable_thinking":false}' --generation-config vllm --no-enable-log-requests --disable-uvicorn-access-log 2>&1 | tee "$log"

# Experimental: FP8 KV on SM86 uses FlashInfer JIT and the host CUDA toolkit.
# vLLM disables dynamic KV-scale calculation for this hybrid GDN model, so the
# resulting scale-1 cache must pass long-context quality gates before promotion.
serve-vllm-pp2-fp8-160k:
    test -x "{{cuda_toolkit_root}}/bin/nvcc"; "{{cuda_toolkit_root}}/bin/nvcc" --version | grep -F 'release 13.3'; mkdir -p "{{work_root}}/logs"; log="{{work_root}}/logs/vllm-pp2-fp8-160k-$(date -u +%Y%m%dT%H%M%SZ).log"; echo "vLLM PP2/FP8 log: $log"; docker run --rm --name qwen38-vllm-pp2-fp8-160k --gpus all --ipc=host -p "127.0.0.1:{{port}}:8000" --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly --mount type=bind,src="{{cuda_toolkit_root}}",dst=/usr/local/cuda,readonly -e CUDA_HOME=/usr/local/cuda -e VLLM_CACHE_ROOT=/work/cache/vllm -e VLLM_USE_FLASHINFER_SAMPLER=0 "{{vllm_fp8_image}}" /models/$(basename "{{output_model}}") --served-model-name qwen38-w8a8-pp2-fp8 --api-key "{{api_key}}" --tensor-parallel-size 1 --pipeline-parallel-size 2 --max-model-len 163840 --kv-cache-memory-bytes 3221225472 --kv-cache-dtype fp8 --cpu-offload-gb 0 --seed 42 --language-model-only --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser qwen3_xml --default-chat-template-kwargs '{"enable_thinking":false}' --generation-config vllm --no-enable-log-requests --disable-uvicorn-access-log 2>&1 | tee "$log"

serve-llama:
    docker run --rm --name qwen35-llama --gpus all --ipc=host -p "127.0.0.1:{{port}}:8000" --user "$(id -u):$(id -g)" --mount type=bind,src="{{llama_gguf_model}}",dst=/models/Qwen3.5-27B-Q4_K_M.gguf,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly --env LLAMA_ROOT=/opt/llama --env LLAMA_SERVER=/opt/llama/bin/llama-server --env LLAMA_GGUF_MODEL=/models/Qwen3.5-27B-Q4_K_M.gguf --env LLAMA_CONTEXT=131072 --env LLAMA_TENSOR_SPLIT="{{llama_tensor_split}}" --env LLAMA_HOST=0.0.0.0 --env PORT=8000 --env LLAMA_API_KEY="{{llama_api_key}}" --env LLAMA_MODEL_ALIAS="{{llama_model_alias}}" --env WORK_ROOT=/work --entrypoint /app/inference/scripts/serve_llama_gguf.sh "{{llama_image}}"

serve-llama-160k:
    docker run --rm --name qwen35-llama --gpus all --ipc=host -p "127.0.0.1:{{port}}:8000" --user "$(id -u):$(id -g)" --mount type=bind,src="{{llama_gguf_model}}",dst=/models/Qwen3.5-27B-Q4_K_M.gguf,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly --env LLAMA_ROOT=/opt/llama --env LLAMA_SERVER=/opt/llama/bin/llama-server --env LLAMA_GGUF_MODEL=/models/Qwen3.5-27B-Q4_K_M.gguf --env LLAMA_CONTEXT=163840 --env LLAMA_TENSOR_SPLIT="{{llama_tensor_split}}" --env LLAMA_HOST=0.0.0.0 --env PORT=8000 --env LLAMA_API_KEY="{{llama_api_key}}" --env LLAMA_MODEL_ALIAS="{{llama_model_alias}}" --env WORK_ROOT=/work --entrypoint /app/inference/scripts/serve_llama_gguf.sh "{{llama_image}}"

serve-llama-host:
    LLAMA_CONTEXT=131072 PORT="{{port}}" LLAMA_API_KEY="{{llama_api_key}}" LLAMA_MODEL_ALIAS="{{llama_model_alias}}" WORK_ROOT="{{work_root}}" "{{repo_root}}/inference/scripts/serve_llama_gguf.sh"

smoke-llama:
    INFERENCE_API_KEY="{{llama_api_key}}" python3 "{{repo_root}}/inference/scripts/smoke_test.py" --base-url "http://127.0.0.1:{{port}}/v1" --model "{{llama_model_alias}}" --output "{{work_root}}/results/llama-inference-smoke-$(date -u +%Y%m%dT%H%M%SZ).json"

probe-llama target_tokens="120000":
    LLAMA_API_KEY="{{llama_api_key}}" python3 "{{repo_root}}/inference/scripts/long_context_probe.py" --base-url "http://127.0.0.1:{{port}}" --model "{{llama_model_alias}}" --target-tokens "{{target_tokens}}" --output "{{work_root}}/results/llama-context-probe-{{target_tokens}}-$(date -u +%Y%m%dT%H%M%SZ).json"

smoke:
    python3 "{{repo_root}}/inference/scripts/smoke_test.py" --base-url "http://127.0.0.1:{{port}}/v1" --model "{{served_model}}" --api-key "{{api_key}}" --output "{{work_root}}/results/inference-smoke-$(date -u +%Y%m%dT%H%M%SZ).json"

bench:
    python3 "{{repo_root}}/inference/scripts/benchmark.py" --base-url "http://127.0.0.1:{{port}}/v1" --model "{{served_model}}" --api-key "{{api_key}}" --output "{{work_root}}/results/benchmark-suite-$(date -u +%Y%m%dT%H%M%SZ).json"

logs:
    @find "{{work_root}}/logs" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM  %s  %f\n' | sort
