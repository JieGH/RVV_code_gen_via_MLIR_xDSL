#!/bin/bash

# ============================================================
# run_issue63.sh -- the 8-thread Tile Search against Héctor's 8-thread GEMMLib, alternating (issue #63)
#
# Steps, one launch each, on one board: ours L1, his, his, ours L2, ours L3, his (A-B-B-A, house rule 2).
#   ours: run_yolo_sweep.sh --threads 8 --launch <l> (19 Candidate Tiles, rows rotated per launch, cores 0-7)
#   his : run_hector_gemm.sh --threads 8 --launches 1 (his gemm.x at b292c92, cores 0-7); built once, by his
#         first launch, then reused (--no-build; jbpi and jbpi2 share one NFS home)
# Each runner reads the board before its launch and waits while another user's job is busy; each writes its own
# series folder and copies it to HectorYolo/issue-63/. The first failed step stops the series.
# Core 0 is used: the user's exception for 8-thread runs (#63 §2, decided 2026-11-16).
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue63.sh --src <his clone> \
#       > tests/pure_xdsl_blis_rvv/logs/issue63.out 2>&1 < /dev/null &
#
# Flags:
#   --src DIR       his GEMMLib_yolo tree at b292c92 (required)
#   --board B       the board for every step (default jbpi)
#   --launches N    launches per side (default 3)
#   --tmin S        our --tmin and his TIMIN (default 2)
#   --dry-run       print the order, then stop
# ============================================================

set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${ISSUE63_SWEEP:-${HERE}/run_yolo_sweep.sh}"      # ISSUE63_* let the order be tried against stubs
HECTOR="${ISSUE63_HECTOR:-${HERE}/run_hector_gemm.sh}"

SRC=""; BOARD="jbpi"; LAUNCHES=3; TMIN=2; DRY_RUN=false
die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --src)       SRC="$2"; shift ;;
        --board)     BOARD="$2"; shift ;;
        --launches)  LAUNCHES="$2"; shift ;;
        --tmin)      TMIN="$2"; shift ;;
        --dry-run)   DRY_RUN=true ;;
        -h|--help)   sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[ -n "${SRC}" ] || die "--src is required (his GEMMLib_yolo tree)"
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1"

# ours his his ours ours his ..., LAUNCHES of each
STEPS=()
for ((i = 0; i < 2 * LAUNCHES; i++)); do
    case $(( i % 4 )) in 0|3) STEPS+=(ours) ;; *) STEPS+=(his) ;; esac
done
labels=(); l=0
for s in "${STEPS[@]}"; do [ "${s}" = ours ] && { l=$((l + 1)); labels+=("ours L${l}"); } || labels+=(his); done
order="$(IFS=,; echo "${labels[*]}" | sed 's/,/, /g')"

echo "Issue #63: 8 threads on cores 0-7 of ${BOARD}, --tmin ${TMIN}, ${LAUNCHES} launches per side"
echo "  Order: ${order}"
[ "${DRY_RUN}" = true ] && { echo "  --dry-run: nothing was started."; exit 0; }

l=0; built=false
for ((i = 0; i < ${#STEPS[@]}; i++)); do
    side="${STEPS[$i]}"
    echo "[$(date +%H:%M)] step $((i + 1))/${#STEPS[@]}: ${labels[$i]}"
    if [ "${side}" = ours ]; then
        l=$((l + 1))
        "${SWEEP}" --threads 8 --launches "${LAUNCHES}" --launch "${l}" --boards "${BOARD}" --tmin "${TMIN}" --issue 63
    else
        build=(); [ "${built}" = true ] && build=(--no-build)
        "${HECTOR}" --src "${SRC}" --threads 8 --launches 1 --boards "${BOARD}" --timin "${TMIN}" --issue 63 "${build[@]}"
    fi
    rc=$?
    [ "${rc}" -eq 0 ] || { echo "step $((i + 1)) (${side}) failed (exit ${rc}); stopping the series."; exit 1; }
    [ "${side}" = his ] && built=true
done
echo "Series finished: ${#STEPS[@]} of ${#STEPS[@]} steps ok."
