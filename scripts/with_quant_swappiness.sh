#!/usr/bin/env bash
# Host-only wrapper: serialize quant jobs, lower swappiness, restore on exit.
set -Eeuo pipefail
(( $# )) || { echo 'usage: with_quant_swappiness.sh command [args...]' >&2; exit 64; }
work_root=${WORK_ROOT:-/data/qwen38-int8-lab}
mkdir -p "${work_root}"
exec 9>"${work_root}/quant-swappiness.lock"
flock -n 9 || { echo 'Another guarded quant job is running' >&2; exit 75; }
original=$(sysctl -n vm.swappiness)
[[ $original =~ ^[0-9]+$ ]] || exit 77
changed=0
child=
write_setting() {
    if (( EUID == 0 )); then sysctl -q -w "vm.swappiness=$1";
    else sudo -n sysctl -q -w "vm.swappiness=$1"; fi
}
# Invoked by the EXIT trap.
# shellcheck disable=SC2329
cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n $child ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    if (( changed )); then
        if ! write_setting "$original" || [[ $(sysctl -n vm.swappiness) != "$original" ]]; then
            echo 'ERROR: failed to restore vm.swappiness' >&2
            status=125
        else
            echo "Restored vm.swappiness=$original"
        fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
if [[ $original != 1 ]]; then
    # Mark before writing so even an interrupted write is restored.
    changed=1
    write_setting 1 || { echo 'Cannot set swappiness; run with sysctl sudo access' >&2; exit 77; }
fi
[[ $(sysctl -n vm.swappiness) == 1 ]] || exit 77
echo "Quant swappiness: captured=$original active=1"
setsid -- "$@" &
child=$!
status=0
wait "$child" || status=$?
child=
exit "$status"
