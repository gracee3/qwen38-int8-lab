#!/usr/bin/env bash
set -uo pipefail

# Model shards are immutable root-owned 0600 files. Run evaluation as container
# root against read-only mounts, then return only the stage evidence to the
# unprivileged host supervisor with private permissions.

readonly OUTPUT_PATH=${EVAL_OUTPUT_PATH:?set EVAL_OUTPUT_PATH}
readonly OUTPUT_UID=${EVAL_OUTPUT_UID:?set EVAL_OUTPUT_UID}
readonly OUTPUT_GID=${EVAL_OUTPUT_GID:?set EVAL_OUTPUT_GID}

[[ ${OUTPUT_PATH} == /run/stages/.*.tmp ]] || {
    printf 'error=unsafe_eval_output_path path=%s\n' "${OUTPUT_PATH}" >&2
    exit 77
}
[[ ${OUTPUT_UID} =~ ^[0-9]+$ && ${OUTPUT_GID} =~ ^[0-9]+$ ]] || {
    printf 'error=invalid_eval_output_identity\n' >&2
    exit 77
}

cleanup() {
    local status=$?
    trap - EXIT
    if [[ -d ${OUTPUT_PATH} ]]; then
        chown -R "${OUTPUT_UID}:${OUTPUT_GID}" "${OUTPUT_PATH}"
        chmod -R u=rwX,go= "${OUTPUT_PATH}"
    fi
    exit "${status}"
}
trap cleanup EXIT

"$@"
