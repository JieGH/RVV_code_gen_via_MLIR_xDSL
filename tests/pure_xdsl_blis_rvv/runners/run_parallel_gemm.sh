#!/bin/bash

# ============================================================
# run_parallel_gemm.sh -- our GEMM at 1 and 8 threads on the YOLO26 shapes (issue #56)
#
# One LAUNCH is one compile_macro_bench.sh run of configs/benchmark/config_parallel_gemm.json (tile 24x8, datasets/yolo26_k1.dat,
# every row checked), uninstrumented, reference protocol, --no-nc-overrides, --tmin 2. 1 thread runs on core 4
# (--cpu 4); 8 threads run with --threads 8 (cores 0-7, OpenMP bound close to cores; core 0 by the user's
# exception, #56 §5). Order 1 8 8 1 1 8 (--launches of each), all on one board: the first of --boards with no
# other user's busy job, kept for every launch. Before every launch the board's processes are read and, while
# another user's job is busy there, the series WAITS (--wait-min) and reads again; a board that cannot be read
# counts as busy. Read again after each launch: a job seen then is a RED NOTE in load_warnings.txt. A launch
# that fails or misses a shape stops the series.
#
# Logs: logs/parallel_gemm_<stamp>/ : plan.txt, runs.tsv (launch, threads, cores, board, run folder, exit), one
# log per launch, load_*.txt. After every launch parallel_gemm_report.py writes parallel_gemm_report.md and
# parallel_gemm_fig.png there (beside Héctor's #62 series given by --hector), and the folder is copied to
# .scratch/reports/issue-<N>/ (--issue, default 56).
#
# Usage (from anywhere; detach it):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/runners/run_parallel_gemm.sh \
#       > tests/pure_xdsl_blis_rvv/logs/parallel_gemm.out 2>&1 < /dev/null &
# Flags: --boards a,b (default jbpi: the board of the #62 series compared against; give --hector to match another)  --launches N (3)  --tmin S (2)  --wait-min M (10)  --workspace NAME
#        --direct  --riscv-user U  --riscv-port P  --issue N (56)  --hector <#62 series folder>  --dry-run
#        --threads 1|8 (run only that thread count, --launches of it; issue #94)  --scalar-pack (scalar packer; #83, #94)
#        --config FILE (another config, e.g. configs/benchmark/config_yolo53_24x8.json; the shape count comes from its first dataset; #108, #109)
# No tests or --dry-run of another wrapper meanwhile: they rewrite pj_gen/.
# ============================================================

set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the project folder
LOGS="${PARALLEL_GEMM_LOGS:-${HERE}/logs}"
BENCH_RUN="${PARALLEL_GEMM_RUN:-${HERE}/compile_macro_bench.sh}"
CONFIG="${HERE}/configs/benchmark/config_parallel_gemm.json"
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"
source "${HERE}/runners/hector_yolo_dump.sh"
PY="${HERE}/../../.venv/bin/python"; [ -x "${PY}" ] || PY=python3

BOARDS="jbpi"; LAUNCHES=3; TMIN=2; WAIT_MIN=10; WORKSPACE=""; MODE_ARG="--jump"; RISCV_USER="jlei"
RISCV_PORT="22"; DRY_RUN=false; ISSUE=56; THREADS=""; PACK_ARGS=(); PACKER="vector packer"
HECTOR="${HERE}/../../.scratch/reports/issue-62/hector_gemm_2026-11-16_111457"   # his jbpi series

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --boards)      BOARDS="$2"; shift ;;
        --launches)    LAUNCHES="$2"; shift ;;
        --tmin)        TMIN="$2"; shift ;;
        --wait-min)    WAIT_MIN="$2"; shift ;;
        --workspace)   WORKSPACE="$2"; shift ;;
        --direct)      MODE_ARG="--direct" ;;
        --riscv-user)  RISCV_USER="$2"; shift ;;
        --riscv-port)  RISCV_PORT="$2"; shift ;;
        --issue)       ISSUE="$2"; shift ;;
        --hector)      HECTOR="$2"; shift ;;
        --threads)     THREADS="$2"; shift ;;
        --scalar-pack) PACK_ARGS=(--scalar-pack); PACKER="scalar packer" ;;
        --config)      CONFIG="$(realpath "$2")"; shift ;;
        --dry-run)     DRY_RUN=true ;;
        -h|--help)     sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1 (got '${LAUNCHES}')"
[[ -z "${THREADS}" || "${THREADS}" =~ ^[18]$ ]] || die "--threads takes 1 or 8 (got '${THREADS}')"
[[ "${WAIT_MIN}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--wait-min takes minutes (got '${WAIT_MIN}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ -n "${BOARD_LIST[0]:-}" ] || die "--boards takes a comma-separated list"
export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"
DATASET="$(grep -o '"[^"]*\.dat"' "${CONFIG}" | head -1 | tr -d '"')"
[ -f "${HERE}/datasets/${DATASET}" ] || die "no dataset named in ${CONFIG}"
SHAPES="$(grep -c '^[A-Za-z0-9_]' "${HERE}/datasets/${DATASET}")"

# The order 1 8 8 1 1 8 ..., LAUNCHES of each.
ORDER=()
for ((i = 0; i < 2 * LAUNCHES; i++)); do
    case $(( i % 4 )) in 0|3) ORDER+=(1) ;; *) ORDER+=(8) ;; esac
done
[ -n "${THREADS}" ] && { ORDER=(); for ((i = 0; i < LAUNCHES; i++)); do ORDER+=("${THREADS}"); done; }
args_of() { [ "$1" = 1 ] && echo "--cpu 4" || echo "--threads 8"; }
cores_of() { [ "$1" = 1 ] && echo 4 || echo 0-7; }

print_plan() {
    echo "Issue #${ISSUE}: our GEMM (tile $(grep -oE '\[[0-9]+, [0-9]+\]' "${CONFIG}" | head -1 | tr -d '[] ' | tr , x)) at 1 and 8 threads on datasets/${DATASET} (${SHAPES} shapes, every row checked)"
    echo "  Settings : uninstrumented, reference protocol, --tmin ${TMIN}, --no-nc-overrides, ${PACKER}"
    echo "  Cores    : 1 thread on core 4; 8 threads on cores 0-7, OMP_PROC_BIND=close OMP_PLACES=cores (core 0: user's exception, #56 §5)"
    echo "  Order    : ${ORDER[*]} (threads per launch), one board: the first idle one of ${BOARDS}"
    echo "  Load     : the board's processes are read before every launch; busy -> wait ${WAIT_MIN} min and read again"
}

if [ "${DRY_RUN}" = true ]; then print_plan; echo ""; echo "  --dry-run: nothing was started."; exit 0; fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/parallel_gemm_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'launch\tthreads\tcores\tboard\tresult_dir\texit\n' > "${SERIES}/runs.tsv"
wait_s() { awk -v m="${WAIT_MIN}" 'BEGIN { printf "%d", m * 60 }'; }

BOARD=""
while [ -z "${BOARD}" ]; do
    for b in "${BOARD_LIST[@]}"; do
        others="$(board_busy "${b}" "${SERIES}/load_${b}_pick.txt")"
        if [ -z "${others}" ]; then BOARD="${b}"; break; fi
        echo "[$(date +%H:%M)] ${b} is busy: ${others}"
    done
    [ -z "${BOARD}" ] && { echo "[$(date +%H:%M)] no idle board; reading again in ${WAIT_MIN} min"; sleep "$(wait_s)"; }
done
echo "[$(date +%H:%M)] board: ${BOARD}" | tee -a "${SERIES}/plan.txt"
CONN=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}" --riscv-ip "${BOARD}")
[ -n "${WORKSPACE}" ] && CONN+=(--riscv-workspace "${WORKSPACE}")

newest_since() {
    local d
    d="$(ls -d "${LOGS}"/run_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "run_$1" ]]; then return 1; fi
    echo "${d}"
}

report_and_dump() {
    "${PY}" "${HERE}/readers/parallel_gemm_report.py" "${SERIES}" --logs "${LOGS}" --hector "${HECTOR}" \
        -o "${SERIES}/parallel_gemm_report.md" --plot "${SERIES}/parallel_gemm_fig.png" \
        || echo "    report failed: ${SERIES}"
    hector_yolo_dump "${ISSUE}" "${SERIES}"
}

OK=0; TOTAL=${#ORDER[@]}
for ((l = 1; l <= TOTAL; l++)); do
    th="${ORDER[$((l - 1))]}"
    log="${SERIES}/L${l}_T${th}_${BOARD}.log"
    while others="$(board_busy "${BOARD}" "${SERIES}/load_${BOARD}_L${l}.txt")"; [ -n "${others}" ]; do
        echo "[$(date +%H:%M)] launch ${l}: ${BOARD} is busy (${others}); waiting ${WAIT_MIN} min"
        sleep "$(wait_s)"
    done
    echo "[$(date +%H:%M)] launch ${l}/${TOTAL}: ${th} thread(s) on cores $(cores_of "${th}") of ${BOARD}: ${log}"
    start="$(date +%Y-%m-%d_%H%M%S)"
    # shellcheck disable=SC2046  # args_of is two words on purpose
    (
        flock -w 21600 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
        cd "${HERE}/../.." && "${BENCH_RUN}" "${CONN[@]}" --config "${CONFIG}" --no-nc-overrides --tmin "${TMIN}" $(args_of "${th}") "${PACK_ARGS[@]}"
    ) 9>"${LOCK}" >> "${log}" 2>&1
    rc=$?
    dir="$(newest_since "${start}")"
    rows=0
    for f in ${dir:+"${dir}"/*/reps_*.txt*}; do
        [ -e "${f}" ] && rows="$(zcat -f "${f}" | tail -n +2 | cut -d, -f1 | sort -u | wc -l)"
    done
    fails=0
    [ -n "${dir}" ] && fails="$(cat "${dir}"/all_raw_*.txt 2>/dev/null | grep -c FAIL)"
    why=""
    [ "${rc}" -ne 0 ] && why="exit ${rc}"
    [ "${rows}" -ne "${SHAPES}" ] && why="${why:+${why}, }${rows} of ${SHAPES} shapes"
    [ "${fails}" -ne 0 ] && why="${why:+${why}, }${fails} FAIL rows"
    others="$(board_busy "${BOARD}" "${SERIES}/load_${BOARD}_L${l}_after.txt")"
    if [ -n "${others}" ]; then
        echo "$(date -Is) launch ${l} on ${BOARD}, read after: ${others}" >> "${SERIES}/load_warnings.txt"
        echo "    RED NOTE: another user's job was on ${BOARD} at the end of launch ${l}: ${others}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${l}" "${th}" "$(cores_of "${th}")" "${BOARD}" \
        "$([ -n "${dir}" ] && basename "${dir}" || echo -)" "$([ -n "${why}" ] && echo "FAIL(${why})" || echo 0)" >> "${SERIES}/runs.tsv"
    report_and_dump
    if [ -n "${why}" ]; then echo "    launch ${l} failed (${why}); see ${log}. Stopping the series."; exit 1; fi
    OK=$((OK + 1))
    echo "    ok: $(basename "${dir}")"
done
echo "Series finished: ${OK} of ${TOTAL} launches ok on ${BOARD}. Logs: ${SERIES}"
