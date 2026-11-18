#!/bin/bash
# ============================================================
# run_issue85.sh -- issue #85: when does vector packing pay off? One board, one thread count, correctness first.
#
#   1. Correctness: config_issue85_correctness.json, oracle on, vector then scalar packer, --tmin 0.2.
#      Any row not PASS stops the series before timing.
#   2. M1 (GOPs): config_issue85_sweep.json, uninstrumented, reference protocol, --tmin 3,
#      6 launches in the order vector, scalar, scalar, vector, vector, scalar (3 per packer).
#   3. M2 (1T) / M3 (8T) (pack/compute split): the same config, --instr, warm-up protocol, 200 reps,
#      4 launches vector, scalar, scalar, vector.
# Every launch: --no-nc-overrides; 1T on core 4, 8T on cores 0-7 (user's #56 exception). Before each launch it
# waits while another user's job is busy on the board (unreadable counts as busy). After each launch the run
# folder is copied to --report-dir and a row is added to its runs.tsv.
#
# Usage (from the project root, detached):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue85.sh --board jbpi --threads 1 --report-dir <abs dir> \
#       > <log> 2>&1 < /dev/null &
# Flags: --board B, --threads 1|8, --report-dir DIR (required), --workspace W (board-side dir: a second board's
#        series needs its own workspace and its own checkout, shared NFS home), --after PID, --dry-run
# ============================================================
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD=""; THREADS=""; REPORT=""; WS=(); AFTER=""; DRY=false
die() { echo "Error: $*" >&2; exit 1; }
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --board) BOARD="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
        --report-dir) REPORT="$2"; shift ;;
        --workspace) WS=(--riscv-workspace "$2"); shift ;;
        --after) AFTER="$2"; shift ;;
        --dry-run) DRY=true ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[ -n "${BOARD}" ] || die "--board is required"
[ -n "${REPORT}" ] || die "--report-dir is required"
case "${THREADS}" in
    1) TFLAG=(--cpu 4); TAG=1t ;;
    8) TFLAG=(--threads 8); TAG=8t ;;
    *) die "--threads takes 1 or 8" ;;
esac

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"
busy_others() {  # other users' processes above 20% CPU on the board; unreadable counts as busy
    local out
    out=$(timeout 60 sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        "jlei@${REMOTE_HOST}" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 ${BOARD} 'ps -eo user=,pcpu=,comm='" 2>/dev/null) \
        || { echo "unreadable"; return; }
    echo "${out}" | awk '$1 != "jlei" && $1 != "root" && $2 > 20 {print $1 ":" $3 ":" $2}' | tr '\n' ' '
}

# launch STEP PACKER CONFIG FLAGS... : one compile_macro_bench.sh run; prints the new run folder's path.
launch() {
    local step="$1" packer="$2" cfg="$3"; shift 3
    local pflag=(); [ "${packer}" = scalar ] && pflag=(--scalar-pack)
    echo "[$(date +%H:%M)] ${step}: ${packer} packer, ${TAG}, $* on ${BOARD}" >&2
    [ "${DRY}" = true ] && return 0
    local b before after
    while b=$(busy_others); [ -n "${b}" ]; do echo "[$(date +%H:%M)] ${BOARD} busy (${b}); waiting 10 min" >&2; sleep 600; done
    before=$(ls -d "${HERE}"/logs/run_* 2>/dev/null | sort | tail -1)
    (cd "${HERE}/../.." && "${HERE}/compile_macro_bench.sh" --jump --riscv-ip "${BOARD}" "${WS[@]}" --no-nc-overrides \
        "${TFLAG[@]}" "${pflag[@]}" "$@" --config "${HERE}/${cfg}") >&2 || { echo "${step} (${packer}) failed" >&2; exit 1; }
    after=$(ls -d "${HERE}"/logs/run_* | sort | tail -1)
    [ "${after}" != "${before}" ] || { echo "${step}: no new run folder" >&2; exit 1; }
    mkdir -p "${REPORT}/runs" && cp -r "${after}" "${REPORT}/runs/"
    printf '%s\t%s\t%s\t%s\t%s\n' "${step}" "${TAG}" "${packer}" "${BOARD}" "$(basename "${after}")" >> "${REPORT}/runs.tsv"
    echo "[$(date +%H:%M)] ${step} done: $(basename "${after}")" >&2
    echo "${after}"
}

if [ -n "${AFTER}" ] && [ "${DRY}" != true ]; then
    echo "[$(date +%H:%M)] waiting for PID ${AFTER} to end"
    while kill -0 "${AFTER}" 2>/dev/null; do sleep 60; done
fi
mkdir -p "${REPORT}"

for packer in vector scalar; do
    dir=$(launch "correctness" "${packer}" config_issue85_correctness.json --tmin 0.2) || exit 1
    [ "${DRY}" = true ] && continue
    total=$(cat "${dir}"/all_raw_*.txt | grep -vc '^Project,')
    pass=$(cat "${dir}"/all_raw_*.txt | grep -v '^Project,' | awk -F, '$7 == "PASS"' | wc -l)
    echo "[$(date +%H:%M)] correctness ${packer} ${TAG}: ${pass} of ${total} rows PASS"
    [ "${total}" -gt 0 ] && [ "${pass}" -eq "${total}" ] || { echo "not all PASS; stopping before timing."; exit 1; }
done

n=0
for packer in vector scalar scalar vector vector scalar; do
    n=$((n + 1)); launch "M1-launch${n}" "${packer}" config_issue85_sweep.json --tmin 3 >/dev/null || exit 1
done
step=M2; [ "${THREADS}" = 8 ] && step=M3
n=0
for packer in vector scalar scalar vector; do
    n=$((n + 1)); launch "${step}-launch${n}" "${packer}" config_issue85_sweep.json --instr >/dev/null || exit 1
done
echo "[$(date +%H:%M)] issue #85 ${TAG} series on ${BOARD} finished."
