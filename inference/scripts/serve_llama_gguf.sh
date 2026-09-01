#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_COMMIT="c845263f8b7d60113e213a3bd2d5cc6472ccf204"
readonly MODEL_REVISION="3221f178a6b842d04f1fb42f1c413534adcc0a6a"
readonly MODEL_SIZE="16740812704"
readonly MODEL_SHA256="84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806"
readonly DEFAULT_LLAMA_ROOT="/home/emmy/workspace/git/llama.cpp"
readonly DEFAULT_MODEL="/home/emmy/workspace/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
readonly DEFAULT_WORK_ROOT="/data/qwen38-int8-lab"

llama_root="${LLAMA_ROOT:-$DEFAULT_LLAMA_ROOT}"
server="${LLAMA_SERVER:-$llama_root/build-cuda-shared/bin/llama-server}"
model="${LLAMA_GGUF_MODEL:-$DEFAULT_MODEL}"
work_root="${WORK_ROOT:-$DEFAULT_WORK_ROOT}"
context="${LLAMA_CONTEXT:-131072}"
tensor_split="${LLAMA_TENSOR_SPLIT:-3,1}"
port="${PORT:-8000}"
api_key="${LLAMA_API_KEY:-local-qwen-only}"
model_alias="${LLAMA_MODEL_ALIAS:-qwen35-27b-q4km}"

case "$context" in
    131072|163840) ;;
    *)
        echo "LLAMA_CONTEXT must be 131072 (candidate) or 163840 (experimental)" >&2
        exit 2
        ;;
esac

test -x "$server" || {
    echo "llama-server is missing or not executable: $server" >&2
    exit 1
}
test -f "$model" || {
    echo "GGUF model is missing: $model" >&2
    echo "Expected Hugging Face revision: $MODEL_REVISION" >&2
    exit 1
}
actual_size="$(stat -c %s "$model")"
test "$actual_size" = "$MODEL_SIZE" || {
    echo "GGUF size mismatch: expected $MODEL_SIZE bytes, found $actual_size" >&2
    exit 1
}
actual_sha256="$(sha256sum "$model" | cut -d ' ' -f 1)"
test "$actual_sha256" = "$MODEL_SHA256" || {
    echo "GGUF SHA-256 mismatch: expected $MODEL_SHA256, found $actual_sha256" >&2
    exit 1
}

actual_commit="$(git -C "$llama_root" rev-parse HEAD)"
test "$actual_commit" = "$LLAMA_COMMIT" || {
    echo "llama.cpp commit mismatch: expected $LLAMA_COMMIT, found $actual_commit" >&2
    exit 1
}

if [[ "${LLAMA_ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    test -z "$busy" || {
        echo "GPU compute processes are already active (PIDs: $(tr '\n' ' ' <<<"$busy"))" >&2
        echo "Set LLAMA_ALLOW_BUSY_GPUS=1 only for an intentional ASR co-residency test." >&2
        exit 1
    }
fi

mkdir -p "$work_root/logs"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$work_root/logs/llama-qwen35-${context}-${stamp}.log"
echo "llama.cpp commit: $LLAMA_COMMIT"
echo "model revision: $MODEL_REVISION"
echo "context: $context; layer split: $tensor_split; KV: q8_0/q8_0"
echo "log: $log"

exec > >(tee "$log") 2>&1
exec "$server" \
    --model "$model" \
    --alias "$model_alias" \
    --host 127.0.0.1 \
    --port "$port" \
    --api-key-file <(printf '%s\n' "$api_key") \
    --ctx-size "$context" \
    --parallel 1 \
    --n-gpu-layers all \
    --split-mode layer \
    --tensor-split "$tensor_split" \
    --main-gpu 0 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --flash-attn on \
    --jinja \
    --no-webui
