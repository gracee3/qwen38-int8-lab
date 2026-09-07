#!/usr/bin/env bash
set -Eeuo pipefail

# All stages are explicit. Full quantization is never launched automatically.
stage=${1:?usage: bash scripts/int4_prepare.sh audit|synthetic|runtime|corpus|screen-candidate|finalize|screen-final|real-pilot|full}
resume=${2:-}
[[ $# -le 2 && ( -z $resume || $resume == --resume ) ]] || { echo 'Only optional --resume is accepted' >&2; exit 2; }
[[ -z $resume || $stage == real-pilot || $stage == full ]] || { echo '--resume requires real-pilot or full' >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${INT4_RUN_ROOT:-/data/qwen38-int8-lab/int4-v1}
gpu=${INT4_GPU_UUID:-GPU-613c7d78-a76d-306b-05da-1db1f15a5032}
quant=sha256:7c0ac3089184466e7348dd98ba0219311e69fa32b7395a7674956467a9e02088
runtime=sha256:60508d8dcbbb0a985955e9cf2f66e561a66c3f1c99bd7ec8fa5020e991a0ef4d
mkdir -p "$run_root"
network=none
common=(--rm --cpus 2 --shm-size 256m
    --mount "type=bind,src=$repo,dst=/app,readonly"
    --mount "type=bind,src=$run_root,dst=/run-int4"
    --mount "type=bind,src=$run_root,dst=/work/scratch"
    --mount 'type=bind,src=/home/emmy/workspace/qwen3.8-27b-download/model,dst=/models/source,readonly'
    --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1
    --env "GIT_COMMIT=$(git -C "$repo" rev-parse HEAD)")
command=(); extra=(); image=$quant
memory_args=(--memory 6g --memory-swap 6g)
guard=()
case "$stage" in
    audit) command=(python /app/quant/scripts/audit_int4_targets.py /models/source /app/quant/config/qwen38-27b-int4-expanded400-v1.yaml --hash-shards --output /run-int4/target-audit-hashed.json);;
    synthetic) extra=(--gpus "device=$gpu"); command=(python /app/quant/scripts/int4_synthetic.py --output /run-int4/synthetic);;
    runtime)
        image=$runtime
        # Trusted local callback inspection only, in a network-disabled container.
        extra=(--gpus "device=$gpu" -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 -e VLLM_CACHE_ROOT=/run-int4/cache -e VLLM_USE_FLASHINFER_SAMPLER=0)
        command=(python /app/quant/scripts/int4_runtime_pilot.py);;
    corpus)
        extra=(--mount 'type=bind,src=/data/qwen38-int8-lab/cache/huggingface,dst=/hf,readonly')
        command=(python /app/quant/scripts/prepare_int4_corpus.py --cache /hf --source /models/source --output /run-int4/calibration-candidate);;
    screen-candidate|screen-final)
        extra=(--mount 'type=bind,src=/data/qwen38-int8-lab/cache/huggingface,dst=/hf,readonly'
            --mount 'type=bind,src=/data/qwen38-int8-lab/evaluations/20260825T100526Z/cache/huggingface/datasets/wis-k___instruction-following-eval/default/0.0.0/5a5661c2a35488308556cf4453dc074d1eba91a0,dst=/heldout-ifeval,readonly')
        if [[ $stage == screen-candidate ]]; then
            # Only this explicit stage may download pinned public held-out files.
            network=bridge
            command=(bash -ec 'python /app/quant/scripts/screen_int4_overlap.py; python /app/quant/scripts/freeze_int4_canaries.py; python /app/quant/scripts/review_int4_origins.py')
        else
            command=(bash -ec 'python /app/quant/scripts/screen_int4_overlap.py --corpus calibration-final --output overlap-final-public.json; python /app/quant/scripts/freeze_int4_canaries.py --corpus calibration-final --output overlap-final-agents.json; python /app/quant/scripts/review_int4_origins.py --corpus calibration-final --output origin-final-schema.json; python /app/quant/scripts/assemble_int4_clearance.py')
        fi;;
    finalize) command=(python /app/quant/scripts/finalize_int4_corpus.py);;
    real-pilot|full)
        # Entire host must have room for the BF16 source and GPTQ intermediates.
        available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        (( available >= 80*1024*1024 )) || { echo 'BLOCKED: real pilot requires 80 GiB available host RAM'; exit 2; }
        test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo 'BLOCKED: another GPU workload is active'; exit 2; }
        free_bytes=$(df -B1 --output=avail "$run_root" | tail -1)
        # A rolling snapshot needs another source-sized state plus activations.
        # Exact live tensor storage is checked before each save; this is a floor.
        (( free_bytes >= 96*1024*1024*1024 )) || { echo 'BLOCKED: require 96 GiB disk headroom for checkpoint rotation'; exit 2; }
        protected=$(lsblk -dnpo NAME,SERIAL | awk '$2=="S7KHNU0X722442H" {print $1}')
        test -n "$protected" && test "$(lsblk -dnro RO "$protected")" = 1
        test -z "$(lsblk -nrpo MOUNTPOINTS "$protected" | tr -d '[:space:]')"
        # Match real INT8: host RAM/swap watchdog, without an earlier cgroup OOM.
        # Small preparation/runtime stages retain their bounded no-swap containers.
        memory_args=()
        guard=(bash "$repo/scripts/with_quant_swappiness.sh")
        extra=(--gpus "device=$gpu")
        command=(python /app/quant/scripts/quantize_int4.py --audit /run-int4/target-audit-hashed.json --runtime-report /run-int4/runtime-result.json --corpus /run-int4/calibration-final)
        if [[ $stage == full ]]; then
            pilot_report=${INT4_REAL_PILOT_REPORT:?Set to the pilot report filename inside INT4_RUN_ROOT}
            overlap_report=${INT4_OVERLAP_REPORT:?Set to the reviewed overlap report filename inside INT4_RUN_ROOT}
            [[ $pilot_report != */* && $overlap_report != */* ]]
            test ! -e /data/models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1
            extra+=(--mount 'type=bind,src=/data/models,dst=/models')
            command+=(--full --output /models/Qwen3.8-27B-W4A16-INT4-Expanded400-v1 --real-pilot-report "/run-int4/$pilot_report" --overlap-report "/run-int4/$overlap_report")
        else
            command+=(--output /run-int4/real-pilot)
        fi
        if [[ $resume == --resume ]]; then command+=(--resume); fi;;
    *) echo "Unknown stage: $stage" >&2; exit 2;;
esac
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$run_root/$stage-$stamp.log"
echo "log=$log"
"${guard[@]}" docker run "${common[@]}" --network "$network" "${memory_args[@]}" "${extra[@]}" --entrypoint /usr/bin/env "$image" "${command[@]}" 2>&1 | tee "$log"
