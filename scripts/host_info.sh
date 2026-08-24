#!/usr/bin/env bash
set -euo pipefail

source_model=${SOURCE_MODEL:-/home/emmy/workspace/qwen3.8-27b-download/model}
work_root=${WORK_ROOT:-/data/qwen38-int8-lab}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

date --iso-8601=seconds
uname -a
lsb_release -ds
free -h
df -h / "$work_root"
lsblk -o NAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINTS,RO,MODEL
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pstate,power.limit --format=csv
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}'
docker info --format 'runtimes={{range $name, $_ := .Runtimes}}{{$name}} {{end}}cdi={{json .CDISpecDirs}}'
python3 "$repo_root/quant/scripts/inspect_model.py" "$source_model"
