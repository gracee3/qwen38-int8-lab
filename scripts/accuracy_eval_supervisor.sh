#!/usr/bin/env bash
set -Eeuo pipefail

# Non-root, fail-closed supervisor. Install/run the exact pushed version from a
# labeled tmux session. Completed stages are renamed atomically and never retried.

readonly REPO=/home/emmy/qwen38-int8-lab
readonly RUN_BASE=/data/qwen38-int8-lab/evaluations
readonly CANDIDATE=/data/models/Qwen3.8-27B-W8A8-INT8
readonly SOURCE=/home/emmy/workspace/qwen3.8-27b-download/model
readonly EVAL_IMAGE=${EVAL_IMAGE:-qwen38-int8-lab/eval:0.1.0}
readonly EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact pushed feature commit}
readonly EXPECTED_BRANCH=${EXPECTED_BRANCH:-feat/standardized-accuracy-eval}
readonly RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
readonly RUN_ROOT=${RUN_BASE}/${RUN_ID}
readonly PROTECTED_SERIAL=S7KHNU0X722442H
readonly MIN_MEM_KIB=$((80 * 1024 * 1024))
readonly STOP_MEM_KIB=$((8 * 1024 * 1024))
readonly MIN_DISK_BYTES=$((100 * 1024 * 1024 * 1024))
readonly MAX_SWAP_GROWTH_KIB=$((4 * 1024 * 1024))

CURRENT_STAGE=not_started
CURRENT_CONTAINER=
TELEMETRY_TSV=
FAILURE_REASON=

if (( EUID == 0 )); then
    printf 'error=supervisor_must_run_as_emmy\n' >&2
    exit 77
fi
if [[ $(id -un) != emmy ]]; then
    printf 'error=unexpected_user user=%s\n' "$(id -un)" >&2
    exit 77
fi
if [[ -e ${RUN_ROOT} ]]; then
    printf 'error=run_root_exists path=%s\n' "${RUN_ROOT}" >&2
    exit 78
fi
install -d -m 700 "${RUN_ROOT}" "${RUN_ROOT}/cache" "${RUN_ROOT}/home" "${RUN_ROOT}/stages"
TELEMETRY_TSV=${RUN_ROOT}/telemetry.tsv
touch "${TELEMETRY_TSV}"
chmod 600 "${TELEMETRY_TSV}"
exec > >(/usr/bin/tee -a "${RUN_ROOT}/supervisor.log") 2>&1

write_supervisor_result() {
    local status=$1 blocker=${2:-}
    python3 - "${RUN_ROOT}/supervisor-result.json" "${RUN_ID}" "${status}" "${CURRENT_STAGE}" "${blocker}" "${EXPECTED_COMMIT}" <<'PY'
import datetime as dt
import json
import os
import sys

path, run_id, status, stage, blocker, commit = sys.argv[1:]
temporary = path + ".tmp"
payload = {
    "run_id": run_id,
    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "last_stage": stage,
    "blocker": blocker or None,
    "commit": commit,
}
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n ${CURRENT_CONTAINER} ]]; then
        docker stop --time 30 "${CURRENT_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if [[ ! -e ${RUN_ROOT}/supervisor-result.json ]]; then
        [[ -n ${FAILURE_REASON} ]] || FAILURE_REASON="exit_status_${status}"
        if [[ -s ${TELEMETRY_TSV} && ! -e ${RUN_ROOT}/telemetry.json ]] && declare -F finalize_telemetry >/dev/null; then
            finalize_telemetry "${FAILURE_REASON}" || true
        fi
        write_supervisor_result failed "${FAILURE_REASON}"
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

protected_disk_check() {
    local row device ro mounts
    row=$(lsblk -nrpo NAME,RO,MOUNTPOINTS,SERIAL | awk -v serial="${PROTECTED_SERIAL}" '$NF == serial {print; exit}')
    [[ -n ${row} ]] || { FAILURE_REASON=protected_disk_not_found; return 1; }
    device=$(awk '{print $1}' <<<"${row}")
    ro=$(awk '{print $2}' <<<"${row}")
    mounts=$(lsblk -nrpo MOUNTPOINTS "${device}" | sed '/^[[:space:]]*$/d')
    [[ ${ro} == 1 && -z ${mounts} ]] || { FAILURE_REASON=protected_disk_guard_failed; return 1; }
}

host_preflight() {
    local commit branch remote_commit mem disk compute session_route
    commit=$(git -C "${REPO}" rev-parse HEAD)
    branch=$(git -C "${REPO}" branch --show-current)
    remote_commit=$(git ls-remote origin "refs/heads/${EXPECTED_BRANCH}" | awk '{print $1}')
    [[ ${commit} == "${EXPECTED_COMMIT}" && ${remote_commit} == "${EXPECTED_COMMIT}" ]] || {
        FAILURE_REASON=git_commit_not_exactly_pushed; return 1;
    }
    [[ ${branch} == "${EXPECTED_BRANCH}" && -z $(git -C "${REPO}" status --porcelain) ]] || {
        FAILURE_REASON=git_state_not_clean_expected_branch; return 1;
    }
    mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    (( mem >= MIN_MEM_KIB )) || { FAILURE_REASON=insufficient_available_memory; return 1; }
    mapfile -t gpu_mem < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [[ ${#gpu_mem[@]} -eq 2 ]] || { FAILURE_REASON=requires_exactly_two_gpus; return 1; }
    (( gpu_mem[0] < 512 && gpu_mem[1] < 512 )) || { FAILURE_REASON=gpus_not_idle; return 1; }
    compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    [[ -z ${compute} ]] || { FAILURE_REASON=gpu_compute_process_present; return 1; }
    disk=$(df -B1 --output=avail /data | tail -n 1 | tr -d ' ')
    (( disk >= MIN_DISK_BYTES )) || { FAILURE_REASON=insufficient_data_disk; return 1; }
    [[ -d ${CANDIDATE} && -d ${SOURCE} ]] || { FAILURE_REASON=model_missing; return 1; }
    systemctl is-active --quiet ssh docker containerd || { FAILURE_REASON=required_service_inactive; return 1; }
    docker info >/dev/null || { FAILURE_REASON=docker_unavailable; return 1; }
    session_route=$(ip route get 1.1.1.1 | head -n 1)
    [[ ${session_route} == *"dev wlx00c0cab51e69"* ]] || { FAILURE_REASON=unexpected_default_route; return 1; }
    protected_disk_check
    printf 'stage=%s preflight=passed mem_available_kib=%s disk_available_bytes=%s gpu_memory_mib=%s,%s\n' \
        "${CURRENT_STAGE}" "${mem}" "${disk}" "${gpu_mem[0]}" "${gpu_mem[1]}"
}

write_identities() {
    python3 - "${RUN_ROOT}/git-identity.json" "${EXPECTED_COMMIT}" "${EXPECTED_BRANCH}" <<'PY'
import json, os, sys
path, commit, branch = sys.argv[1:]
with open(path, "x", encoding="utf-8") as handle:
    json.dump({"commit": commit, "branch": branch}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.chmod(path, 0o600)
PY
    docker image inspect "${EVAL_IMAGE}" --format '{{json .}}' | IDENTITY_PATH="${RUN_ROOT}/image-identity.json" python3 -c '
import json, os, sys
x=json.load(sys.stdin)
payload={"id":x["Id"],"repo_digests":x.get("RepoDigests",[]),"labels":x.get("Config",{}).get("Labels",{})}
path=os.environ["IDENTITY_PATH"]
with open(path,"x",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
os.chmod(path,0o600)'
    docker run --rm --entrypoint python "${EVAL_IMAGE}" -c '
import json
from importlib.metadata import distributions
items=sorted((d.metadata["Name"],d.version) for d in distributions() if d.metadata["Name"])
print(json.dumps({"packages":[{"name":name,"version":version} for name,version in items]},sort_keys=True))' \
        >"${RUN_ROOT}/package-identity.json"
    chmod 600 "${RUN_ROOT}/package-identity.json"
}

write_model_manifests() {
    local when=$1 candidate_file source_file combined
    candidate_file=${RUN_ROOT}/.candidate-${when}.json
    source_file=${RUN_ROOT}/.source-${when}.json
    combined=${RUN_ROOT}/model-manifests-${when}.json
    python3 "${REPO}/eval/scripts/checkpoint_manifest.py" "${CANDIDATE}" --output "${candidate_file}"
    python3 "${REPO}/eval/scripts/checkpoint_manifest.py" "${SOURCE}" --output "${source_file}"
    python3 - "${candidate_file}" "${source_file}" "${combined}" <<'PY'
import json, os, sys
candidate, source, output = sys.argv[1:]
with open(candidate, encoding="utf-8") as f: c=json.load(f)
with open(source, encoding="utf-8") as f: s=json.load(f)
with open(output,"x",encoding="utf-8") as f: json.dump({"w8a8":c,"bf16":s},f,indent=2,sort_keys=True); f.write("\n")
os.chmod(output,0o600)
os.unlink(candidate); os.unlink(source)
PY
}

verify_manifests_unchanged() {
    python3 - "${RUN_ROOT}/model-manifests-before.json" "${RUN_ROOT}/model-manifests-after.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f: before=json.load(f)
with open(sys.argv[2], encoding="utf-8") as f: after=json.load(f)
for model in ("w8a8", "bf16"):
    if before[model]["identity_sha256"] != after[model]["identity_sha256"]:
        raise SystemExit(f"{model} model manifest changed")
PY
}

container_common=(
    --rm --gpus all --ipc=host
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=${CANDIDATE},dst=/models/w8a8,readonly"
    --mount "type=bind,src=${SOURCE},dst=/models/bf16,readonly"
    --mount "type=bind,src=${RUN_ROOT},dst=/run"
    --mount "type=bind,src=${REPO},dst=/app,readonly"
    --env HOME=/run/home
    --env HF_HOME=/run/cache/huggingface
    --env HF_DATASETS_CACHE=/run/cache/huggingface/datasets
    --env HF_HUB_DISABLE_TELEMETRY=1
    --env TOKENIZERS_PARALLELISM=false
)

prefetch_and_validate() {
    CURRENT_STAGE=dataset_prefetch
    host_preflight
    docker run "${container_common[@]}" --env HF_TOKEN --entrypoint python "${EVAL_IMAGE}" \
        /app/eval/scripts/prefetch.py --output /run/dataset-preflight.json
    chmod 600 "${RUN_ROOT}/dataset-preflight.json"
    CURRENT_STAGE=task_validation
    host_preflight
    docker run "${container_common[@]}" --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
        --entrypoint python "${EVAL_IMAGE}" /app/eval/scripts/run_harness.py validate --tasks leaderboard
    CURRENT_STAGE=request_preflight
    host_preflight
    docker run "${container_common[@]}" --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
        --entrypoint python "${EVAL_IMAGE}" /app/eval/scripts/request_preflight.py \
        --candidate /models/w8a8 --source /models/bf16 --output /run/request-preflight.json
    chmod 600 "${RUN_ROOT}/request-preflight.json"
}

process_memory_kib() {
    local root_pid=$1 pid rss=0 swap=0 value cursor=0 child
    local -a pids
    pids=("${root_pid}")
    while (( cursor < ${#pids[@]} )); do
        pid=${pids[cursor]}; cursor=$((cursor + 1))
        while read -r child; do [[ -z ${child} ]] || pids+=("${child}"); done < <(pgrep -P "${pid}" 2>/dev/null || true)
    done
    for pid in "${pids[@]}"; do
        [[ -r /proc/${pid}/status ]] || continue
        value=$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status"); rss=$((rss + ${value:-0}))
        value=$(awk '/^VmSwap:/ {print $2}' "/proc/${pid}/status"); swap=$((swap + ${value:-0}))
    done
    printf '%s %s\n' "${rss}" "${swap}"
}

run_eval_stage() {
    local model=$1 stage_name=$2 tasks=$3 limit=${4:-}
    local model_path max_batch cpu_offload temporary final log docker_pid status root_pid result_file
    local swap_free_start swap_free mem swap_growth breach=0 trigger='' gpu0 gpu1 disk rss proc_swap
    local psi_some psi_full pswpin pswpout
    CURRENT_STAGE=${stage_name}
    host_preflight
    final=${RUN_ROOT}/stages/${stage_name}
    temporary=${RUN_ROOT}/stages/.${stage_name}.tmp
    [[ ! -e ${final} && ! -e ${temporary} ]] || { FAILURE_REASON=stage_evidence_exists; return 1; }
    install -d -m 700 "${temporary}/results"
    log=${temporary}/harness.log
    if [[ ${model} == w8a8 ]]; then
        model_path=/models/w8a8; max_batch=4; cpu_offload=0
    else
        model_path=/models/bf16; max_batch=2; cpu_offload=8
    fi
    CURRENT_CONTAINER="qwen38-eval-${RUN_ID}-${stage_name}"
    model_args="pretrained=${model_path},dtype=bfloat16,tensor_parallel_size=2,max_model_len=8192,gpu_memory_utilization=0.88,kv_cache_dtype=bfloat16,seed=42,enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=False,add_bos_token=False,enable_thinking=False,cpu_offload_gb=${cpu_offload}"
    command=(
        docker run "${container_common[@]}" --name "${CURRENT_CONTAINER}"
        --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1
        --env VLLM_USE_FLASHINFER_SAMPLER=0 --env VLLM_CACHE_ROOT=/run/cache/vllm
        --entrypoint python "${EVAL_IMAGE}" /app/eval/scripts/run_harness.py run
        --model vllm --model_args "${model_args}" --tasks "${tasks}"
        --batch_size auto --max_batch_size "${max_batch}" --seed 42
        --apply_chat_template --fewshot_as_multiturn --log_samples
        --output_path "/run/stages/.${stage_name}.tmp/results"
    )
    [[ -z ${limit} ]] || command+=(--limit "${limit}")
    "${command[@]}" >"${log}" 2>&1 &
    docker_pid=$!
    swap_free_start=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    observed_gpu0=0; observed_gpu1=0
    while kill -0 "${docker_pid}" 2>/dev/null; do
        mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        swap_free=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
        swap_growth=$((swap_free_start - swap_free)); (( swap_growth < 0 )) && swap_growth=0
        mapfile -t gpu_mem < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        gpu0=${gpu_mem[0]:-0}; gpu1=${gpu_mem[1]:-0}
        (( gpu0 > 512 )) && observed_gpu0=1
        (( gpu1 > 512 )) && observed_gpu1=1
        disk=$(df -B1 --output=avail /data | tail -n 1 | tr -d ' ')
        root_pid=$(docker inspect --format '{{.State.Pid}}' "${CURRENT_CONTAINER}" 2>/dev/null || printf '0')
        rss=0; proc_swap=0
        if [[ ${root_pid} =~ ^[0-9]+$ && ${root_pid} -gt 0 ]]; then
            read -r rss proc_swap < <(process_memory_kib "${root_pid}")
            if docker inspect "${CURRENT_CONTAINER}" --format '{{range .Mounts}}{{if or (eq .Destination "/models/w8a8") (eq .Destination "/models/bf16")}}{{.Destination}}:{{.RW}} {{end}}{{end}}' | grep -q ':true'; then
                trigger=model_mount_not_read_only
            fi
        fi
        psi_some=$(awk '/^some/ {for(i=1;i<=NF;i++) if($i ~ /^total=/) {split($i,a,"="); print a[2]}}' /proc/pressure/memory)
        psi_full=$(awk '/^full/ {for(i=1;i<=NF;i++) if($i ~ /^total=/) {split($i,a,"="); print a[2]}}' /proc/pressure/memory)
        pswpin=$(awk '$1 == "pswpin" {print $2}' /proc/vmstat)
        pswpout=$(awk '$1 == "pswpout" {print $2}' /proc/vmstat)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%FT%TZ)" "${stage_name}" "${mem}" "${swap_growth}" "${rss}" \
            "${proc_swap}" "${gpu0},${gpu1}" "${disk}" "${psi_some}" "${psi_full}" \
            "${pswpin}" "${pswpout}" >>"${TELEMETRY_TSV}"
        if (( mem < STOP_MEM_KIB )); then
            trigger=mem_available_below_8_gib
        fi
        if (( swap_growth > MAX_SWAP_GROWTH_KIB )); then breach=$((breach + 1)); else breach=0; fi
        if (( breach >= 5 )); then trigger=swap_growth_over_4_gib_for_10_seconds; fi
        if [[ -n ${trigger} ]]; then
            FAILURE_REASON=${trigger}
            docker stop --time 30 "${CURRENT_CONTAINER}" >/dev/null 2>&1 || true
            break
        fi
        sleep 2
    done
    set +e
    wait "${docker_pid}"
    status=$?
    set -e
    CURRENT_CONTAINER=
    [[ -z ${trigger} ]] || return 1
    (( status == 0 )) || { FAILURE_REASON="harness_stage_failed_${stage_name}_status_${status}"; return 1; }
    [[ ${observed_gpu0} == 1 && ${observed_gpu1} == 1 ]] || { FAILURE_REASON=both_gpus_not_observed; return 1; }
    if [[ ${model} == w8a8 ]]; then
        grep -Fq 'Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8' "${log}" || {
            FAILURE_REASON=cutlass_w8a8_dispatch_missing; return 1;
        }
    else
        grep -Eiq 'cpu offload|CPUOffloading|offload' "${log}" || { FAILURE_REASON=bf16_cpu_offload_not_observed; return 1; }
    fi
    if grep -Eiq 'exceeds model.s max length|left truncated|truncating to last|truncating context' "${log}"; then
        FAILURE_REASON=runtime_truncation_detected; return 1
    fi
    [[ $(find "${temporary}/results" -type f -name 'results_*.json' | wc -l) -eq 1 ]] || {
        FAILURE_REASON=missing_or_duplicate_result_json; return 1;
    }
    result_file=$(find "${temporary}/results" -type f -name 'results_*.json')
    python3 - "${result_file}" "${REPO}/eval/config/leaderboard-v2.yaml" "${stage_name}" <<'PY'
import json, math, sys, yaml
result_path, config_path, stage = sys.argv[1:]
with open(result_path, encoding="utf-8") as f: result=json.load(f)
if stage == "smoke":
    if not result.get("results"):
        raise SystemExit("smoke result has no tasks")
    raise SystemExit(0)
group = stage.split("-", 1)[1]
with open(config_path, encoding="utf-8") as f: task=yaml.safe_load(f)["tasks"][group]
values=result.get("groups",{}).get(task["harness_task"]) or result.get("results",{}).get(task["harness_task"])
if values is None:
    raise SystemExit(f"missing result for {task['harness_task']}")
matches=[value for key,value in values.items() if key == task["headline_metric"] or key.startswith(task["headline_metric"] + ",")]
if len(matches) != 1 or not math.isfinite(float(matches[0])):
    raise SystemExit(f"missing, duplicate, or non-finite headline metric: {task['headline_metric']}")
PY
    chmod -R u=rwX,go= "${temporary}"
    mv "${temporary}" "${final}"
    for _ in $(seq 1 30); do
        [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d') ]] && break
        sleep 1
    done
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d') ]] || {
        FAILURE_REASON=gpu_process_remained_after_stage; return 1;
    }
    printf 'stage=%s model=%s status=passed evidence=%s\n' "${stage_name}" "${model}" "${final}"
}

finalize_telemetry() {
    local trigger=${1:-}
    python3 - "${TELEMETRY_TSV}" "${RUN_ROOT}/telemetry.json" "${trigger}" <<'PY'
import json, os, sys
from collections import defaultdict

source, output, trigger = sys.argv[1:]
stages=defaultdict(lambda:{"samples":0,"first_timestamp":None,"last_timestamp":None,"minimum_mem_available_kib":None,"maximum_swap_growth_kib":0,"maximum_process_rss_kib":0,"maximum_process_vmswap_kib":0,"maximum_gpu_memory_mib":[0,0],"minimum_disk_available_bytes":None,"_first_psi":None,"_last_psi":None,"_first_swap_io":None,"_last_swap_io":None})
with open(source,encoding="utf-8") as f:
    for line in f:
        timestamp,stage,mem,swap,rss,vmswap,gpus,disk,psi_some,psi_full,pswpin,pswpout=line.rstrip("\n").split("\t")
        row=stages[stage]; row["samples"]+=1
        mem,swap,rss,vmswap,disk,psi_some,psi_full,pswpin,pswpout=map(int,(mem,swap,rss,vmswap,disk,psi_some,psi_full,pswpin,pswpout)); gpu=list(map(int,gpus.split(",")))
        row["first_timestamp"] = row["first_timestamp"] or timestamp; row["last_timestamp"] = timestamp
        row["minimum_mem_available_kib"]=mem if row["minimum_mem_available_kib"] is None else min(row["minimum_mem_available_kib"],mem)
        row["maximum_swap_growth_kib"]=max(row["maximum_swap_growth_kib"],swap)
        row["maximum_process_rss_kib"]=max(row["maximum_process_rss_kib"],rss)
        row["maximum_process_vmswap_kib"]=max(row["maximum_process_vmswap_kib"],vmswap)
        row["maximum_gpu_memory_mib"]=[max(a,b) for a,b in zip(row["maximum_gpu_memory_mib"],gpu)]
        row["minimum_disk_available_bytes"]=disk if row["minimum_disk_available_bytes"] is None else min(row["minimum_disk_available_bytes"],disk)
        row["_first_psi"] = row["_first_psi"] or [psi_some,psi_full]; row["_last_psi"]=[psi_some,psi_full]
        row["_first_swap_io"] = row["_first_swap_io"] or [pswpin,pswpout]; row["_last_swap_io"]=[pswpin,pswpout]
for row in stages.values():
    row["memory_psi_delta_usec"]={"some":row["_last_psi"][0]-row["_first_psi"][0],"full":row["_last_psi"][1]-row["_first_psi"][1]}
    row["swap_io_delta_bytes"]={"in":(row["_last_swap_io"][0]-row["_first_swap_io"][0])*4096,"out":(row["_last_swap_io"][1]-row["_first_swap_io"][1])*4096}
    for key in ("_first_psi","_last_psi","_first_swap_io","_last_swap_io"): del row[key]
with open(output,"x",encoding="utf-8") as f: json.dump({"safety_trigger":trigger or None,"stages":dict(sorted(stages.items()))},f,indent=2,sort_keys=True); f.write("\n")
os.chmod(output,0o600)
PY
}

CURRENT_STAGE=initial_preflight
host_preflight
write_identities
write_model_manifests before
prefetch_and_validate
run_eval_stage w8a8 smoke leaderboard 2
for group in mmlu_pro bbh gpqa math_hard ifeval musr; do
    harness_task=$(python3 - "${REPO}/eval/config/leaderboard-v2.yaml" "${group}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f: config=yaml.safe_load(f)
print(config["tasks"][sys.argv[2]]["harness_task"])
PY
)
    run_eval_stage w8a8 "w8a8-${group}" "${harness_task}"
done
for group in mmlu_pro bbh gpqa musr; do
    harness_task=$(python3 - "${REPO}/eval/config/leaderboard-v2.yaml" "${group}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f: config=yaml.safe_load(f)
print(config["tasks"][sys.argv[2]]["harness_task"])
PY
)
    run_eval_stage bf16 "bf16-${group}" "${harness_task}"
done
CURRENT_STAGE=postflight
host_preflight
write_model_manifests after
verify_manifests_unchanged
finalize_telemetry ''
python3 "${REPO}/eval/scripts/aggregate.py" --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/standardized-result.json" --markdown "${RUN_ROOT}/standardized-result.md"
write_supervisor_result passed
printf 'run_status=passed run_root=%s\n' "${RUN_ROOT}"
