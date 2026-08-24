set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

repo_root := justfile_directory()
model_root := env_var_or_default("MODEL_ROOT", "/data/models")
work_root := env_var_or_default("WORK_ROOT", "/data/qwen38-int8-lab")
source_model := env_var_or_default("SOURCE_MODEL", "/home/emmy/workspace/qwen3.8-27b-download/model")
output_model := env_var_or_default("OUTPUT_MODEL", model_root + "/Qwen3.8-27B-W8A8-INT8")
quant_image := env_var_or_default("QUANT_IMAGE", "qwen38-int8-lab/quant:0.1.0")
vllm_image := env_var_or_default("VLLM_IMAGE", "qwen38-int8-lab/vllm:0.1.0")
port := env_var_or_default("PORT", "8000")
served_model := env_var_or_default("SERVED_MODEL", "qwen38-w8a8")

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

quant:
    mkdir -p "{{work_root}}/logs"; log="{{work_root}}/logs/quant-$(date -u +%Y%m%dT%H%M%SZ).log"; echo "quant log: $log"; docker run --rm --gpus all --ipc=host --mount type=bind,src="{{model_root}}",dst=/models --mount type=bind,src="{{source_model}}",dst=/models/source,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e HF_HOME=/work/cache/huggingface -e HF_DATASETS_CACHE=/work/cache/huggingface/datasets -e GIT_COMMIT="$(git -C "{{repo_root}}" rev-parse HEAD)" -w /app "{{quant_image}}" python /app/quant/scripts/quantize.py --config /app/quant/config/qwen38-27b.yaml --profile quality --output /models/$(basename "{{output_model}}") --execute-full 2>&1 | tee "$log"

build-vllm:
    DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg VCS_REF="$(git -C "{{repo_root}}" rev-parse HEAD)" -t "{{vllm_image}}" -f "{{repo_root}}/docker/vllm/Dockerfile" "{{repo_root}}"

shell-vllm:
    docker run --rm -it --gpus all --ipc=host --entrypoint /bin/bash --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -w /app "{{vllm_image}}"

validate:
    docker run --rm --gpus all --ipc=host --entrypoint python --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -w /app "{{vllm_image}}" /app/quant/scripts/validate_quant.py /models/$(basename "{{output_model}}")

serve:
    mkdir -p "{{work_root}}/logs"; log="{{work_root}}/logs/vllm-$(date -u +%Y%m%dT%H%M%SZ).log"; echo "vLLM log: $log"; docker run --rm --gpus all --ipc=host -p "{{port}}:8000" --mount type=bind,src="{{model_root}}",dst=/models,readonly --mount type=bind,src="{{work_root}}",dst=/work --mount type=bind,src="{{repo_root}}",dst=/app,readonly -e VLLM_CACHE_ROOT=/work/cache/vllm "{{vllm_image}}" /models/$(basename "{{output_model}}") --served-model-name "{{served_model}}" --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 42 2>&1 | tee "$log"

smoke:
    python3 "{{repo_root}}/inference/scripts/smoke_test.py" --base-url "http://127.0.0.1:{{port}}/v1" --model "{{served_model}}" --output "{{work_root}}/results/inference-smoke-$(date -u +%Y%m%dT%H%M%SZ).json"

bench:
    python3 "{{repo_root}}/inference/scripts/benchmark.py" --base-url "http://127.0.0.1:{{port}}/v1" --model "{{served_model}}" --output "{{work_root}}/results/benchmark-$(date -u +%Y%m%dT%H%M%SZ).json"

logs:
    @find "{{work_root}}/logs" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM  %s  %f\n' | sort
