#!/bin/bash
# ============================================================
# run_issue94.sh -- YOLO GEMM at 1 and 8 threads: our vector packer (V), our scalar packer (S), Héctor's gemm.bin (H)
# mixed in one series on one board (issue #94; the #56 setup with #83's scalar packer and #62's runs added).
#
# First C1 (1 thread, core 4) and C2 (8 threads, cores 0-7): config_issue94_correctness.json (24x8, scalar packer,
# oracle on, --tmin 0.2). A row not PASS stops everything before timing.
# Then 3 rounds; each round a 1T and an 8T block (thread order 1 8 8 1 1 8, as #56), the setup order rotated per
# round (V S H, S H V, H V S) so each setup takes each place once: 18 launches.
#   V: run_parallel_gemm.sh --threads T --launches 1                (tile 24x8, every row checked)
#   S: run_parallel_gemm.sh --threads T --launches 1 --scalar-pack
#   H: run_hector_gemm.sh   --threads T --launches 1 --omp-bind (his gemm.x at b292c92, unchanged; built by his first
#      launch; 8T with OMP_PROC_BIND=close OMP_PLACES=cores, the same run settings as ours, user 2026-11-18)
# Each runner reads the board before its launch and waits while another user's job is busy, writes its own series
# folder and copies it to .scratch/reports/issue-94/. The first failed step stops the series.
# Logs: logs/issue94_<stamp>/: plan.txt (plan and order), steps.tsv (step, setup, threads, series folder, exit).
#
# Usage (from anywhere; detached):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/experiments/issue94/run_issue94.sh --src <his tree> \
#       > tests/pure_xdsl_blis_rvv/logs/issue94.out 2>&1 < /dev/null &
# Flags: --src DIR (his GEMMLib_yolo tree at b292c92, required)  --board B (jbpi)  --tmin S (2)  --skip-correctness
#        --dry-run  --issue N (log folder issue-N; 94)
#        --config-1t FILE / --config-8t FILE (ours at 1T / 8T runs this config; default config_parallel_gemm.json; #109)
#        --check-1t FILE / --check-8t FILE (C1 / C2 run this correctness config; default config_issue94_correctness.json; #109)
# No tests or --dry-run of another wrapper meanwhile: they rewrite pj_gen/.
# ============================================================
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${HERE}/board_pin.sh"   # board access (issue #116)
OURS="${ISSUE94_OURS:-${HERE}/run_parallel_gemm.sh}"      # ISSUE94_* let the order be tried against stubs
HECTOR="${ISSUE94_HECTOR:-${HERE}/run_hector_gemm.sh}"
LOGS="${ISSUE94_LOGS:-${HERE}/logs}"
CHECK1="${HERE}/experiments/issue94/config_issue94_correctness.json"; CHECK8="${CHECK1}"
source "${HERE}/hector_yolo_dump.sh"
SRC=""; BOARD="jbpi"; TMIN=2; DRY=false; CHECK=true; ISSUE=94; CFG1=(); CFG8=()
die() { echo "Error: $*" >&2; exit 1; }
tile_of() { grep -oE '\[[0-9]+, [0-9]+\]' "$1" | head -1 | tr -d '[] ' | tr , x; }
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --src) SRC="$2"; shift ;;
        --board) BOARD="$2"; shift ;;
        --tmin) TMIN="$2"; shift ;;
        --skip-correctness) CHECK=false ;;
        --dry-run) DRY=true ;;
        --config-1t) CFG1=(--config "$(realpath "$2")"); shift ;;
        --config-8t) CFG8=(--config "$(realpath "$2")"); shift ;;
        --check-1t) CHECK1="$(realpath "$2")"; shift ;;
        --check-8t) CHECK8="$(realpath "$2")"; shift ;;
        --issue) ISSUE="$2"; shift ;;
        -h|--help) sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[ -n "${SRC}" ] || die "--src is required (his GEMMLib_yolo tree at b292c92)"
[ -n "${BOARD}" ] || die "--board takes a board name"

# 18 steps: "<setup> <threads>"
STEPS=(); rot=(V S H)
for r in 0 1 2; do
    order=("${rot[@]:r}" "${rot[@]:0:r}")
    if [ $((r % 2)) -eq 0 ]; then blocks=(1 8); else blocks=(8 1); fi
    for t in "${blocks[@]}"; do for s in "${order[@]}"; do STEPS+=("${s} ${t}"); done; done
done

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"
correctness() {  # $1 label, $2 config, rest: extra flags
    local label="$1" cfg="$2"; shift 2
    echo "[$(date +%H:%M)] ${label}: $(basename "${cfg}") on ${BOARD} $*, --scalar-pack, oracle on"
    [ "${DRY}" = true ] && return 0
    board_wait_idle "${BOARD}"
    local before after pass total
    before=$(ls -d "${LOGS}"/run_* 2>/dev/null | sort | tail -1)
    (cd "${HERE}/../.." && "${HERE}/compile_macro_bench.sh" --jump --riscv-ip "${BOARD}" --no-nc-overrides \
        --tmin 0.2 --scalar-pack "$@" --config "${cfg}") || { echo "${label}: compile_macro_bench.sh failed"; exit 1; }
    after=$(ls -d "${LOGS}"/run_* | sort | tail -1)
    [ "${after}" != "${before}" ] || { echo "${label}: no new run folder"; exit 1; }
    total=$(cat "${after}"/all_raw_*.txt | grep -vc '^Project,')
    pass=$(cat "${after}"/all_raw_*.txt | grep -v '^Project,' | awk -F, '$7 == "PASS"' | wc -l)
    echo "[$(date +%H:%M)] ${label}: ${pass} of ${total} rows PASS ($(basename "${after}"))" | tee -a "${SERIES}/plan.txt"
    [ "${total}" -gt 0 ] && [ "${pass}" -eq "${total}" ] || { echo "${label}: not all PASS; stopping before timing."; exit 1; }
}

print_plan() {
    echo "Issue #${ISSUE}: YOLO26, V = ours vector packer, S = ours scalar packer, H = Héctor's gemm.x (b292c92)"
    echo "  Ours 1T  : $(basename "${CFG1[1]:-config_parallel_gemm.json}") (tile $(tile_of "${CFG1[1]:-${HERE}/config_parallel_gemm.json}"))"
    echo "  Ours 8T  : $(basename "${CFG8[1]:-config_parallel_gemm.json}") (tile $(tile_of "${CFG8[1]:-${HERE}/config_parallel_gemm.json}"))"
    echo "  Board    : ${BOARD}; 1T on core 4, 8T on cores 0-7 (core 0: user's 8-thread exception)"
    echo "  Ours     : uninstrumented, reference protocol, --tmin ${TMIN}, --no-nc-overrides, every row checked"
    echo "  Héctor   : TIMIN=${TMIN}, TEST=T, his tiles; 8T OMP_PROC_BIND=close OMP_PLACES=cores (as ours); built by his first launch, then reused"
    echo "  Check    : $([ "${CHECK}" = true ] && echo "C1 (1T) $(basename "${CHECK1}"), C2 (8T) $(basename "${CHECK8}") first, --scalar-pack, oracle on" || echo "skipped (--skip-correctness)")"
    local i=0
    for st in "${STEPS[@]}"; do i=$((i + 1)); echo "  step ${i}: ${st% *} ${st#* }T"; done
}

if [ "${DRY}" = true ]; then SERIES=/dev/null; print_plan; [ "${CHECK}" = true ] && { correctness C1 "${CHECK1}" --cpu 4; correctness C2 "${CHECK8}" --threads 8; }; echo "  --dry-run: nothing was started."; exit 0; fi

SERIES="${LOGS}/issue94_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'step\tsetup\tthreads\tseries\texit\n' > "${SERIES}/steps.tsv"
if [ "${CHECK}" = true ]; then correctness C1 "${CHECK1}" --cpu 4; correctness C2 "${CHECK8}" --threads 8; fi

built=false; i=0
for st in "${STEPS[@]}"; do
    i=$((i + 1)); s="${st% *}"; t="${st#* }"
    echo "[$(date +%H:%M)] step ${i}/${#STEPS[@]}: ${s} ${t}T"
    start="$(date +%Y-%m-%d_%H%M%S)"
    cfg=("${CFG1[@]}"); [ "${t}" = 8 ] && cfg=("${CFG8[@]}")
    case "${s}" in
        V) "${OURS}" --threads "${t}" --launches 1 --boards "${BOARD}" --tmin "${TMIN}" --issue "${ISSUE}" "${cfg[@]}" ;;
        S) "${OURS}" --threads "${t}" --launches 1 --boards "${BOARD}" --tmin "${TMIN}" --issue "${ISSUE}" "${cfg[@]}" --scalar-pack ;;
        H) build=(); [ "${built}" = true ] && build=(--no-build)
           "${HECTOR}" --src "${SRC}" --threads "${t}" --launches 1 --boards "${BOARD}" --timin "${TMIN}" --issue "${ISSUE}" --omp-bind "${build[@]}" ;;
    esac
    rc=$?
    pat=parallel_gemm_; [ "${s}" = H ] && pat=hector_gemm_
    dir="$(ls -d "${LOGS}/${pat}"* 2>/dev/null | sort | tail -1)"
    [[ -n "${dir}" && ! "$(basename "${dir}")" < "${pat}${start}" ]] || dir="-"
    printf '%s\t%s\t%s\t%s\t%s\n' "${i}" "${s}" "${t}" "$(basename "${dir}")" "${rc}" >> "${SERIES}/steps.tsv"
    hector_yolo_dump "${ISSUE}" "${SERIES}"
    [ "${rc}" -eq 0 ] || { echo "step ${i} (${s} ${t}T) failed (exit ${rc}); stopping the series."; exit 1; }
    [ "${s}" = H ] && built=true
done
echo "[$(date +%H:%M)] Series finished: ${#STEPS[@]} of ${#STEPS[@]} steps ok. Logs: ${SERIES}"
