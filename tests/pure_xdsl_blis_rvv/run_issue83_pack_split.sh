#!/bin/bash
# ============================================================
# run_issue83_pack_split.sh -- issue #83: the instrumented pack/compute split, vector against scalar packer.
#
# 4 launches of compile_macro_bench.sh --instr --cpu 4 on config_issue83_pack_split.json, in the order
# vector, scalar, scalar, vector (A-B-B-A). Each row gives Vector_Pack_Time_ms (the timed loop's pack time, with
# whichever packer the launch runs) and Compute_Time_ms. DEBUG build: for the pack-time share only, not GOPs.
# Before each launch it waits while another user's job is busy on the board.
#
# Flags: --board B (required), --workspace W (board-side dir; needs its own checkout, shared NFS home),
#        --after PID (wait for that process to end first), --dry-run
# ============================================================
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD=""; WS=(); AFTER=""; DRY=false
die() { echo "Error: $*" >&2; exit 1; }
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --board) BOARD="$2"; shift ;;
        --workspace) WS=(--riscv-workspace "$2"); shift ;;
        --after) AFTER="$2"; shift ;;
        --dry-run) DRY=true ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[ -n "${BOARD}" ] || die "--board is required"

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"
busy_others() {  # other users' processes above 20% CPU on the board; unreadable counts as busy
    local out
    out=$(timeout 60 sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        "jlei@${REMOTE_HOST}" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 ${BOARD} 'ps -eo user=,pcpu=,comm='" 2>/dev/null) \
        || { echo "unreadable"; return; }
    echo "${out}" | awk '$1 != "jlei" && $1 != "root" && $2 > 20 {print $1 ":" $3 ":" $2}' | tr '\n' ' '
}

if [ -n "${AFTER}" ] && [ "${DRY}" != true ]; then
    echo "[$(date +%H:%M)] waiting for PID ${AFTER} to end"
    while kill -0 "${AFTER}" 2>/dev/null; do sleep 60; done
fi
n=0
for packer in vector scalar scalar vector; do
    n=$((n + 1))
    flag=(); [ "${packer}" = scalar ] && flag=(--scalar-pack)
    echo "[$(date +%H:%M)] launch ${n}/4: ${packer} packer, --instr --cpu 4 on ${BOARD}"
    if [ "${DRY}" = true ]; then continue; fi
    while b=$(busy_others); [ -n "${b}" ]; do echo "[$(date +%H:%M)] ${BOARD} busy (${b}); waiting 10 min"; sleep 600; done
    (cd "${HERE}/../.." && "${HERE}/compile_macro_bench.sh" --jump --riscv-ip "${BOARD}" "${WS[@]}" --no-nc-overrides \
        --instr --cpu 4 "${flag[@]}" --config "${HERE}/config_issue83_pack_split.json") \
        || { echo "launch ${n} (${packer}) failed"; exit 1; }
    echo "[$(date +%H:%M)] launch ${n}/4 done: $(ls -d "${HERE}"/logs/run_* | sort | tail -1)"
done
echo "[$(date +%H:%M)] issue #83 pack split finished."
