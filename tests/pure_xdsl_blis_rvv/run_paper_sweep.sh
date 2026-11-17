#!/bin/bash

# ============================================================
# run_paper_sweep.sh -- the Tile Search on the paper's shapes (arXiv 2603.17800v1, Table 1: BERT B1-B5,
# Square S1-S5) at 1 thread (core 4) and at 8 threads (cores 0-7), alternating, on one board.
#
# Steps, one launch each: 1T L1, 8T L1, 8T L2, 1T L2, 1T L3, 8T L3 (A-B-B-A, house rule 2). Each step is
#   run_yolo_sweep.sh --config config_paper_search.json --threads <1|8> --launch <l>
# (19 Candidate Tiles, rows rotated per launch, idle-board wait, report after every launch). The first failed
# step stops the series. 8 threads use core 0: the user's exception for 8-thread runs (#63 §2).
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_paper_sweep.sh --after-pid <pid> \
#       > tests/pure_xdsl_blis_rvv/logs/paper_sweep.out 2>&1 < /dev/null &
#
# Flags:
#   --after-pid P   wait until process P (another series on the board) has exited, then start
#   --board B       the board for every step (default jbpi)
#   --launches N    launches per thread count (default 3)
#   --tmin S        --tmin of every launch (default 2)
#   --issue N       the sweep's --issue (report copies go to $HECTOR_YOLO_DIR/issue-<N>/)
#   --dry-run       print the order, then stop
# ============================================================

set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${PAPER_SWEEP:-${HERE}/run_yolo_sweep.sh}"      # PAPER_SWEEP lets the order be tried against a stub

AFTER=""; BOARD="jbpi"; LAUNCHES=3; TMIN=2; ISSUE=""; DRY_RUN=false
die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --after-pid) AFTER="$2"; shift ;;
        --board)     BOARD="$2"; shift ;;
        --launches)  LAUNCHES="$2"; shift ;;
        --tmin)      TMIN="$2"; shift ;;
        --issue)     ISSUE="$2"; shift ;;
        --dry-run)   DRY_RUN=true ;;
        -h|--help)   sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1"
[[ "${ISSUE}" =~ ^[0-9]+$ ]] || die "--issue N is required"
[ -z "${AFTER}" ] || [[ "${AFTER}" =~ ^[0-9]+$ ]] || die "--after-pid takes a process id"

# 1 8 8 1 1 8 ..., LAUNCHES of each
THREADS=()
for ((i = 0; i < 2 * LAUNCHES; i++)); do
    case $(( i % 4 )) in 0|3) THREADS+=(1) ;; *) THREADS+=(8) ;; esac
done
labels=(); n1=0; n8=0
for t in "${THREADS[@]}"; do
    if [ "${t}" = 1 ]; then n1=$((n1 + 1)); labels+=("1T L${n1}"); else n8=$((n8 + 1)); labels+=("8T L${n8}"); fi
done

echo "Paper shapes (B1-B5, S1-S5) Tile Search on ${BOARD}: 1T on core 4, 8T on cores 0-7, --tmin ${TMIN}, ${LAUNCHES} launches each"
echo "  Order: $(IFS=,; echo "${labels[*]}" | sed 's/,/, /g')"
[ "${DRY_RUN}" = true ] && { echo "  --dry-run: nothing was started."; exit 0; }

if [ -n "${AFTER}" ] && kill -0 "${AFTER}" 2>/dev/null; then
    echo "[$(date +%H:%M)] waiting for process ${AFTER} to end"
    while kill -0 "${AFTER}" 2>/dev/null; do sleep "${PAPER_POLL_S:-60}"; done
fi

n1=0; n8=0
for ((i = 0; i < ${#THREADS[@]}; i++)); do
    t="${THREADS[$i]}"
    if [ "${t}" = 1 ]; then n1=$((n1 + 1)); l=${n1}; else n8=$((n8 + 1)); l=${n8}; fi
    echo "[$(date +%H:%M)] step $((i + 1))/${#THREADS[@]}: ${labels[$i]}"
    "${SWEEP}" --config "${HERE}/config_paper_search.json" --threads "${t}" --launches "${LAUNCHES}" --launch "${l}" \
        --boards "${BOARD}" --tmin "${TMIN}" --issue "${ISSUE}" \
        || { echo "step $((i + 1)) (${labels[$i]}) failed; stopping the series."; exit 1; }
done
echo "Series finished: ${#THREADS[@]} of ${#THREADS[@]} steps ok."
