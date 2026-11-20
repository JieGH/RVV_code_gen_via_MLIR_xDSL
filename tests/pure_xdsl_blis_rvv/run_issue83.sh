#!/bin/bash
# ============================================================
# run_issue83.sh -- the scalar-packing rerun (issue #83) on one board: correctness first, then timing.
#
#   --part yolo  : C1 YOLO 1T (core 4), C2 YOLO 8T, both oracle on (config_yolo_correctness.json); then T2 = run_issue63.sh --scalar-pack
#                  (ours vs his gemm.x, A-B-B-A) and T1 = run_yolo_sweep.sh --threads 1 --scalar-pack (3 launches).
#   --part paper : C3 paper 1T, C4 paper 8T, oracle on (config_paper_correctness.json); then T3 = run_paper_sweep.sh --scalar-pack.
# The correctness configs keep to the scalar oracle's B2 size limit (docs/agents/execution.md, "Correctness pass").
# A correctness pass with any row not PASS stops everything before timing. Every step checks the board first
# and waits while another user's job is busy (the sweeps do this themselves; the correctness passes here).
#
# Usage (from the project root, detached):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue83.sh --part yolo --board jbpi --src <his tree> \
#       > tests/pure_xdsl_blis_rvv/logs/issue83_yolo.out 2>&1 < /dev/null &
# Flags: --part yolo|paper, --board B, --src DIR (yolo only), --workspace W (board-side dir, for a second
#        board's series; needs its own checkout, shared NFS home), --dry-run
# ============================================================
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/board_pin.sh"   # board access (issue #116)
PART=""; BOARD=""; SRC=""; WS=(); SWS=(); DRY=false
die() { echo "Error: $*" >&2; exit 1; }
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --part) PART="$2"; shift ;;
        --board) BOARD="$2"; shift ;;
        --src) SRC="$2"; shift ;;
        --workspace) WS=(--riscv-workspace "$2"); SWS=(--workspace "$2"); shift ;;
        --dry-run) DRY=true ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[ -n "${BOARD}" ] || die "--board is required"
case "${PART}" in
    yolo)  [ -n "${SRC}" ] || die "--part yolo needs --src (his GEMMLib_yolo tree at b292c92)"; CFG="config_yolo_correctness.json" ;;
    paper) CFG="config_paper_correctness.json" ;;
    *) die "--part takes yolo or paper" ;;
esac
[ "${#WS[@]}" -eq 0 ] || [ "${PART}" = paper ] || die "--workspace is for the paper part only (run_issue63.sh has no workspace flag)"

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"
correctness() {  # $1 label, rest: extra flags
    local label="$1"; shift
    echo "[$(date +%H:%M)] ${label}: ${CFG} on ${BOARD} $*, --scalar-pack, oracle on"
    [ "${DRY}" = true ] && return 0
    board_wait_idle "${BOARD}"
    local before after dir pass total
    before=$(ls -d "${HERE}"/logs/run_* 2>/dev/null | sort | tail -1)
    (cd "${HERE}/../.." && "${HERE}/compile_macro_bench.sh" --jump --riscv-ip "${BOARD}" "${WS[@]}" --no-nc-overrides \
        --tmin 0.2 --scalar-pack "$@" --config "${HERE}/${CFG}") || { echo "${label}: compile_macro_bench.sh failed"; exit 1; }
    after=$(ls -d "${HERE}"/logs/run_* | sort | tail -1)
    [ "${after}" != "${before}" ] || { echo "${label}: no new run folder"; exit 1; }
    dir="${after}"
    total=$(cat "${dir}"/all_raw_*.txt | grep -vc '^Project,')
    pass=$(cat "${dir}"/all_raw_*.txt | grep -v '^Project,' | awk -F, '$7 == "PASS"' | wc -l)
    echo "[$(date +%H:%M)] ${label}: ${pass} of ${total} rows PASS ($(basename "${dir}"))"
    [ "${total}" -gt 0 ] && [ "${pass}" -eq "${total}" ] || { echo "${label}: not all PASS; stopping before timing."; exit 1; }
}

if [ "${PART}" = yolo ]; then
    correctness C1 --cpu 4
    correctness C2 --threads 8
    echo "[$(date +%H:%M)] T2: run_issue63.sh --scalar-pack --issue 83"
    [ "${DRY}" = true ] || "${HERE}/run_issue63.sh" --src "${SRC}" --board "${BOARD}" --scalar-pack --issue 83 || exit 1
    echo "[$(date +%H:%M)] T1: run_yolo_sweep.sh --threads 1 --scalar-pack --issue 83"
    [ "${DRY}" = true ] || "${HERE}/run_yolo_sweep.sh" --boards "${BOARD}" --tmin 2 --scalar-pack --issue 83 || exit 1
else
    correctness C3 --cpu 4
    correctness C4 --threads 8
    echo "[$(date +%H:%M)] T3: run_paper_sweep.sh --scalar-pack --issue 83"
    [ "${DRY}" = true ] || "${HERE}/run_paper_sweep.sh" --board "${BOARD}" --scalar-pack --issue 83 "${SWS[@]}" || exit 1
fi
echo "[$(date +%H:%M)] issue #83 ${PART} part finished."
