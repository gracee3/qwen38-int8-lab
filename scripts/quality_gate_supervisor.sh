#!/usr/bin/env bash
set -Eeuo pipefail

# Root-only, fail-closed supervisor for the authorized tiny -> small -> quality run.
# Install a timestamped copy under /data/qwen38-int8-lab and set EXPECTED_COMMIT
# before launch. Existing outputs and evidence are never removed or reused.

readonly REPO=/home/emmy/qwen38-int8-lab
readonly WORK_ROOT=/data/qwen38-int8-lab
readonly MODEL_ROOT=/data/models
readonly SOURCE_MODEL=/home/emmy/workspace/qwen3.8-27b-download/model
readonly FINAL_MODEL=/data/models/Qwen3.8-27B-W8A8-INT8
readonly DATASET_REVISION=8049631c405ae6576f93f445c6b8166f76f5505a
readonly PROTECTED_SERIAL=S7KHNU0X722442H
readonly MIN_MEM_KIB=$((80 * 1024 * 1024))
readonly MIN_DISK_BYTES=$((100 * 1024 * 1024 * 1024))
readonly EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the merged main commit}
readonly RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
readonly SUPERVISOR_LOG=${WORK_ROOT}/logs/quality-supervisor-${RUN_ID}.log
readonly RESULT_JSON=${WORK_ROOT}/results/quality-supervisor-${RUN_ID}.json

CURRENT_CHILD_PGID=
CURRENT_CONTAINER=
ORIGINAL_SWAPPINESS=
SWAPPINESS_TOUCHED=0
CURRENT_STAGE=not_started
LAST_QUANT_METADATA=

if (( EUID != 0 )); then
    printf 'error=supervisor_requires_root\n' >&2
    exit 77
fi
if [[ -e "${SUPERVISOR_LOG}" || -e "${RESULT_JSON}" ]]; then
    printf 'error=run_id_already_has_evidence run_id=%s\n' "${RUN_ID}" >&2
    exit 78
fi
install -d -o emmy -g emmy "${WORK_ROOT}/logs" "${WORK_ROOT}/results" "${WORK_ROOT}/scratch"
touch "${SUPERVISOR_LOG}"
chown emmy:emmy "${SUPERVISOR_LOG}"
exec > >(/usr/bin/tee -a "${SUPERVISOR_LOG}") 2>&1

write_result() {
    local status=$1 reason=${2:-}
    python3 - "${RESULT_JSON}" "${RUN_ID}" "${status}" "${CURRENT_STAGE}" "${reason}" \
        "${EXPECTED_COMMIT}" "${SUPERVISOR_LOG}" <<'PY'
import datetime as dt, json, os, sys
path, run_id, status, stage, reason, commit, log = sys.argv[1:]
payload = {
    "run_id": run_id,
    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "last_stage": stage,
    "blocker": reason or None,
    "commit": commit,
    "supervisor_log": log,
}
with open(path, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    chown emmy:emmy "${RESULT_JSON}"
}

restore_swappiness() {
    local restored
    if (( SWAPPINESS_TOUCHED )); then
        /usr/sbin/sysctl -q -w "vm.swappiness=${ORIGINAL_SWAPPINESS}" || return 1
        restored=$(< /proc/sys/vm/swappiness)
        [[ "${restored}" == "${ORIGINAL_SWAPPINESS}" ]] || return 1
        printf 'stage=%s restored_swappiness=%s restoration_status=passed\n' "${CURRENT_STAGE}" "${restored}"
        SWAPPINESS_TOUCHED=0
    fi
}

cleanup() {
    local status=$?
    local reason="exit_status_${status}"
    trap - EXIT INT TERM
    if [[ -n "${CURRENT_CHILD_PGID}" ]]; then
        kill -TERM -- "-${CURRENT_CHILD_PGID}" 2>/dev/null || true
        wait "${CURRENT_CHILD_PGID}" 2>/dev/null || true
    fi
    if [[ -n "${CURRENT_CONTAINER}" ]]; then
        docker stop --time 30 "${CURRENT_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if ! restore_swappiness; then
        status=125
        reason=swappiness_restoration_failed
    fi
    if [[ ! -e "${RESULT_JSON}" ]]; then
        write_result failed "${reason}"
    fi
    printf 'supervisor_exit_status=%s\n' "${status}"
    exit "${status}"
}

forward_signal() {
    local signal_name=$1
    printf 'received_signal=%s stage=%s\n' "${signal_name}" "${CURRENT_STAGE}"
    [[ -z "${CURRENT_CHILD_PGID}" ]] || kill -"${signal_name}" -- "-${CURRENT_CHILD_PGID}" 2>/dev/null || true
    exit 128
}
trap cleanup EXIT
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

protected_disk_check() {
    local row name ro mounts
    row=$(lsblk -nrpo NAME,RO,MOUNTPOINTS,SERIAL | awk -v serial="${PROTECTED_SERIAL}" '$NF == serial {print; exit}')
    [[ -n "${row}" ]] || { printf 'error=protected_disk_not_found\n'; return 1; }
    name=$(awk '{print $1}' <<<"${row}")
    ro=$(awk '{print $2}' <<<"${row}")
    mounts=$(lsblk -nrpo MOUNTPOINTS "${name}" | sed '/^[[:space:]]*$/d')
    [[ "${ro}" == 1 && -z "${mounts}" ]] || {
        printf 'error=protected_disk_guard_failed device=%s ro=%s mounts=%q\n' "${name}" "${ro}" "${mounts}"
        return 1
    }
    printf 'protected_disk=%s protected_disk_ro=1 protected_disk_unmounted=true\n' "${name}"
}

preflight() {
    local destination=$1 current origin mem disk revision compute
    mapfile -t gpu_mem < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    current=$(sudo --user=emmy -- git -C "${REPO}" rev-parse HEAD)
    origin=$(sudo --user=emmy -- git -C "${REPO}" rev-parse origin/main)
    [[ "${current}" == "${EXPECTED_COMMIT}" && "${origin}" == "${EXPECTED_COMMIT}" ]]
    [[ $(sudo --user=emmy -- git -C "${REPO}" branch --show-current) == main ]]
    [[ -z $(sudo --user=emmy -- git -C "${REPO}" status --porcelain) ]]
    mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    (( mem >= MIN_MEM_KIB ))
    [[ ${#gpu_mem[@]} -eq 2 ]]
    (( gpu_mem[0] < 512 && gpu_mem[1] < 512 ))
    compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    [[ -z "${compute}" ]]
    disk=$(df -B1 --output=avail /data | tail -n 1 | tr -d ' ')
    (( disk >= MIN_DISK_BYTES ))
    [[ ! -e "${destination}" ]]
    revision=$(python3 - "${REPO}/quant/config/qwen38-27b.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["calibration"]["revision"])
PY
)
    [[ "${revision}" == "${DATASET_REVISION}" ]]
    protected_disk_check
    printf 'stage=%s preflight=passed mem_available_kib=%s gpu_memory_mib=%s,%s disk_available_bytes=%s destination=%s dataset_revision=%s\n' \
        "${CURRENT_STAGE}" "${mem}" "${gpu_mem[0]}" "${gpu_mem[1]}" "${disk}" "${destination}" "${revision}"
}

run_calibration() {
    local profile=$1 container_output=$2 before_metadata after_metadata
    before_metadata=$(find "${WORK_ROOT}/results" -maxdepth 1 -type f -name "quant-${profile}-*.json" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    ORIGINAL_SWAPPINESS=$(< /proc/sys/vm/swappiness)
    SWAPPINESS_TOUCHED=1
    /usr/sbin/sysctl -q -w vm.swappiness=1
    [[ $(< /proc/sys/vm/swappiness) == 1 ]]
    printf 'stage=%s captured_swappiness=%s set_swappiness=1\n' "${CURRENT_STAGE}" "${ORIGINAL_SWAPPINESS}"
    set +e
    setsid sudo --user=emmy --set-home -- env \
        MODEL_ROOT="${MODEL_ROOT}" WORK_ROOT="${WORK_ROOT}" SOURCE_MODEL="${SOURCE_MODEL}" \
        /home/emmy/.cargo/bin/just -f "${REPO}/Justfile" quant-real "${profile}" "${container_output}" &
    CURRENT_CHILD_PGID=$!
    wait "${CURRENT_CHILD_PGID}"
    local status=$?
    CURRENT_CHILD_PGID=
    set -e
    restore_swappiness || { printf 'error=swappiness_restoration_failed\n'; return 125; }
    (( status == 0 )) || { printf 'error=calibration_failed status=%s stage=%s\n' "${status}" "${CURRENT_STAGE}"; return "${status}"; }
    after_metadata=$(find "${WORK_ROOT}/results" -maxdepth 1 -type f -name "quant-${profile}-*.json" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    [[ -n "${after_metadata}" && "${after_metadata}" != "${before_metadata}" ]]
    python3 - "${after_metadata}" "${profile}" "${DATASET_REVISION}" <<'PY'
import json, sys
path, profile, revision = sys.argv[1:]
x=json.load(open(path, encoding="utf-8"))
expected={"tiny_source": (2, 128), "small": (32, 512), "quality": (512, 2048)}[profile]
assert x["status"] == "passed"
run=x["run"]
assert (run["calibration_samples"], run["max_seq_length"]) == expected
assert len(run["processor_configs"]) == 2
dataset=run["shared_load"]["calibration_dataset"]
assert dataset["sample_count"] == expected[0]
assert dataset["token_lengths"]["maximum"] <= expected[1]
if profile != "tiny_source":
    assert dataset["revision"] == revision
    assert dataset["source_fingerprint"] and dataset["tokenized_fingerprint"]
PY
    LAST_QUANT_METADATA=${after_metadata}
    printf 'stage=%s quant_metadata=%s\n' "${CURRENT_STAGE}" "${LAST_QUANT_METADATA}"
}

validate_and_serve() {
    local host_model=$1 do_benchmark=$2 stamp log smoke benchmark health_status
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    log="${WORK_ROOT}/logs/vllm-${CURRENT_STAGE}-${stamp}.log"
    smoke="${WORK_ROOT}/results/smoke-${CURRENT_STAGE}-${stamp}.json"
    benchmark="${WORK_ROOT}/results/benchmark-${CURRENT_STAGE}-${stamp}.json"
    sudo --user=emmy -- docker run --rm --entrypoint python \
        --mount type=bind,src="${host_model}",dst=/candidate,readonly \
        --mount type=bind,src="${WORK_ROOT}",dst=/work \
        --mount type=bind,src="${REPO}",dst=/app,readonly \
        qwen38-int8-lab/vllm:0.1.0 /app/quant/scripts/validate_quant.py /candidate

    CURRENT_CONTAINER="qwen38-quality-${RUN_ID}-${CURRENT_STAGE}"
    # The root supervisor intentionally opens the evidence log before dropping
    # privileges for Docker; no unprivileged redirection is expected here.
    # shellcheck disable=SC2024
    sudo --user=emmy -- docker run --rm --name "${CURRENT_CONTAINER}" --gpus all --ipc=host -p 8000:8000 \
        --mount type=bind,src="${host_model}",dst=/candidate,readonly \
        --mount type=bind,src="${WORK_ROOT}",dst=/work \
        -e VLLM_CACHE_ROOT=/work/cache/vllm -e VLLM_USE_FLASHINFER_SAMPLER=0 \
        qwen38-int8-lab/vllm:0.1.0 /candidate --served-model-name qwen38-w8a8 \
        --tensor-parallel-size 2 --max-model-len 2048 --gpu-memory-utilization 0.88 \
        --kv-cache-dtype bfloat16 --seed 42 --enforce-eager \
        --no-enable-prefix-caching --no-enable-chunked-prefill >"${log}" 2>&1 &
    local docker_pid=$!
    for _ in $(seq 1 180); do
        if ! kill -0 "${docker_pid}" 2>/dev/null; then
            wait "${docker_pid}" || true
            printf 'error=vllm_exited_before_health stage=%s log=%s\n' "${CURRENT_STAGE}" "${log}"
            return 1
        fi
        health_status=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
        [[ "${health_status}" == 200 ]] && break
        sleep 2
    done
    [[ "${health_status}" == 200 ]]
    grep -Fq 'Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8' "${log}"
    [[ $(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | sort -u | wc -l) -eq 2 ]]
    sudo --user=emmy -- docker run --rm --entrypoint python \
        --mount type=bind,src="${host_model}",dst=/candidate,readonly \
        --mount type=bind,src="${WORK_ROOT}",dst=/work \
        --mount type=bind,src="${REPO}",dst=/app,readonly \
        qwen38-int8-lab/vllm:0.1.0 /app/quant/scripts/validate_quant.py /candidate \
        --vllm-log "/work/logs/${log##*/}" \
        --json-out "/work/results/strict-${CURRENT_STAGE}-${stamp}.json"
    sudo --user=emmy -- python3 "${REPO}/inference/scripts/smoke_test.py" \
        --output "${smoke}"
    if [[ "${do_benchmark}" == true ]]; then
        [[ ! -e "${host_model}/EXPERIMENTAL_NON_PRODUCTION.json" ]]
        sudo --user=emmy -- python3 "${REPO}/inference/scripts/benchmark.py" --output "${benchmark}"
        python3 - "${benchmark}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x["runs"]
for run in x["runs"]:
    assert run["time_to_first_token_seconds"] > 0
    assert run["approx_prefill_tokens_per_second"] > 0
    assert run["decode_tokens_per_second"] > 0
PY
    fi
    sudo --user=emmy -- docker stop --time 30 "${CURRENT_CONTAINER}"
    wait "${docker_pid}"
    CURRENT_CONTAINER=
    if ss -ltn '( sport = :8000 )' | grep -q LISTEN; then
        printf 'error=port_8000_still_listening\n'
        return 1
    fi
    printf 'stage=%s inference_validation=passed log=%s smoke=%s benchmark=%s\n' \
        "${CURRENT_STAGE}" "${log}" "${smoke}" "$([[ "${do_benchmark}" == true ]] && printf '%s' "${benchmark}" || printf 'not_run')"
}

printf 'supervisor_started_at=%s run_id=%s expected_commit=%s\n' "$(date -u --iso-8601=seconds)" "${RUN_ID}" "${EXPECTED_COMMIT}"

CURRENT_STAGE=tiny
tiny_host="${WORK_ROOT}/scratch/Qwen3.8-27B-W8A8-tiny-source-${RUN_ID}"
preflight "${tiny_host}"
run_calibration tiny_source "/work/scratch/${tiny_host##*/}"
validate_and_serve "${tiny_host}" false

CURRENT_STAGE=small
small_host="${WORK_ROOT}/scratch/Qwen3.8-27B-W8A8-small-${RUN_ID}"
preflight "${small_host}"
run_calibration small "/work/scratch/${small_host##*/}"
validate_and_serve "${small_host}" false

CURRENT_STAGE=quality
preflight "${FINAL_MODEL}"
run_calibration quality "/models/${FINAL_MODEL##*/}"
validate_and_serve "${FINAL_MODEL}" true

CURRENT_STAGE=postflight
[[ $(< /proc/sys/vm/swappiness) == "${ORIGINAL_SWAPPINESS}" ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d') ]]
if ss -ltn '( sport = :8000 )' | grep -q LISTEN; then
    printf 'error=postflight_port_8000_still_listening\n'
    exit 1
fi
systemctl is-active --quiet ssh docker containerd
ip route show default
protected_disk_check
df -B1 --output=avail,target /data "${FINAL_MODEL}"
write_result passed
printf 'supervisor_status=passed\n'
trap - EXIT INT TERM
