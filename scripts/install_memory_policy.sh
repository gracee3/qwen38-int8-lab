#!/usr/bin/env bash
# Apply only the dedicated memory policy; no services or networking are restarted.
set -Eeuo pipefail
(( EUID == 0 )) || { echo 'Run this installer with sudo' >&2; exit 77; }
source_file=$(dirname "$(readlink -f "$0")")/90-qwen-memory.conf
target=/etc/sysctl.d/90-qwen-memory.conf
if [[ -e $target ]] && ! cmp -s "$source_file" "$target"; then
    echo "Existing $target differs; refusing to overwrite it" >&2
    exit 78
fi
install -m 0644 "$source_file" "$target"
sysctl -p "$target"
[[ $(sysctl -n vm.swappiness) == 1 ]]
echo 'Persistent vm.swappiness=1 installed and verified'
