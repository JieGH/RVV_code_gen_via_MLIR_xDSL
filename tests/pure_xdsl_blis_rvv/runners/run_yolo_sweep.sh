#!/bin/bash

# ============================================================
# run_yolo_sweep.sh -- the Tile Search on the YOLO26 shapes: one core (#61) or 8 threads (#63)
#
# One LAUNCH is one compile_macro_bench.sh run of configs/tile_search/config_yolo_search.json (19 pinned rows, one per
# Candidate Tile, on datasets/yolo26_k1.dat), on one board, core 4, uninstrumented, reference
# protocol, --no-nc-overrides. The tiles run in config order, so each launch gets its own copy of
# the config with the rows rotated (launch l starts at tile (l-1) x ceil(19 / launches)): no tile
# is always first or last (house rule 2). The copies are written into the series folder.
#
# Board: the first board of --boards with no other user's busy job is picked at the start and
# kept for every launch (launches on one board compare like for like). Before every launch the
# board's processes are read; while another user's job is busy there, the series WAITS and reads
# again every --wait-min minutes. It never starts a launch under another user's load. A board whose
# processes cannot be read counts as busy, so an unreachable board means waiting with no time limit:
# watch the log and stop the series by hand. The board is read again after each launch: a job that came
# up meanwhile is a RED NOTE in load_warnings.txt. A failed or incomplete launch (a tile missing, or a
# per-rep log without every shape) stops the series. Limit: the check reads ps %CPU, a lifetime average.
#
# Logs: logs/yolo_sweep_<stamp>/ : plan.txt, config_L<n>.json, runs.tsv (board, launch, first tile,
# run folder, exit), one log per launch, load_<board>_L<n>.txt. The run folders are the usual
# logs/run_*. Read them with: search_report.py <run folders> --compare 16x8 --compare n=4:16x4
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/runners/run_yolo_sweep.sh --tmin 2 \
#       > tests/pure_xdsl_blis_rvv/logs/yolo_sweep.out 2>&1 < /dev/null &
#
# Flags:
#   --boards a,b      boards to pick from, in order (default jbpi,jbpi2)
#   --launches N      launches (default 3, house rule 2)
#   --tmin S          seconds of reps per shape (default 10; #61 decided 2)
#   --cpu N           core (default 4)
#   --allow-core-0    allow --cpu 0 (house rule 1: never for a quoted number)
#   --threads N       1 (default, on --cpu) or 8 (cores 0-7) instead of --cpu (compile_macro_bench.sh --threads; issue #63)
#   --launch L        run launch L of --launches only (its rotation), so other launches can go between (#63)
#   --wait-min M      minutes between reads of a busy board (default 10)
#   --workspace NAME  the board-side directory (compile_macro_bench.sh --riscv-workspace)
#   --direct          reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --issue N         .scratch/reports/issue-<N>/ folder for the report and logs (default 61)
#   --config FILE     the search config (default configs/tile_search/config_yolo_search.json); every row runs the same one dataset
#   --scalar-pack     the scalar packer instead of the vector one (compile_macro_bench.sh --scalar-pack; issue #83)
#   --dry-run         print the plan and the launch configs, then stop
# After every finished launch: search_report.py over the launches so far (--compare 16x8 --compare n=4:16x4,
# heatmap) into the series folder as tile_search_report.md / tile_search_heatmap.png, then the series folder
# is copied to .scratch/reports/issue-<N>/<series>/ (user, 2026-11-16).
#
# Jump-host settings come from the environment: REMOTE_PASS (default empty), REMOTE_HOST (default
# fej), REMOTE_PORT (default 3322). Do not run tests, --dry-run of a wrapper or another benchmark
# on the host meanwhile: they rewrite pj_gen/ under a running series.
# ============================================================

set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the project folder
LOGS="${YOLO_SWEEP_LOGS:-${HERE}/logs}"                 # YOLO_SWEEP_* let the loop be tried against stubs
BENCH_RUN="${YOLO_SWEEP_RUN:-${HERE}/compile_macro_bench.sh}"
CONFIG="${HERE}/configs/tile_search/config_yolo_search.json"
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"
source "${HERE}/runners/hector_yolo_dump.sh"
PY="${HERE}/../../.venv/bin/python"; [ -x "${PY}" ] || PY=python3
REPORT_CMD=("${PY}" "${HERE}/readers/search_report.py"); [ -n "${YOLO_SWEEP_REPORT:-}" ] && REPORT_CMD=("${YOLO_SWEEP_REPORT}")

BOARDS="jbpi,jbpi2"; LAUNCHES=3; TMIN=10; CPU_ARGS=(--cpu 4); CPU_FLAG=false; CPU=""; WAIT_MIN=10
PACK_ARGS=(); PACKER="vector packer"; WORKSPACE=""; MODE_ARG="--jump"; RISCV_USER="jlei"; RISCV_PORT="22"; DRY_RUN=false; ISSUE=61; THREADS=1; ONLY=""
CORES="core 4"

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --boards)        BOARDS="$2"; shift ;;
        --launches)      LAUNCHES="$2"; shift ;;
        --tmin)          TMIN="$2"; shift ;;
        --cpu)           CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --allow-core-0)  ALLOW_CORE0=true ;;
        --threads)       THREADS="$2"; shift ;;
        --launch)        ONLY="$2"; shift ;;
        --wait-min)      WAIT_MIN="$2"; shift ;;
        --workspace)     WORKSPACE="$2"; shift ;;
        --direct)        MODE_ARG="--direct" ;;
        --riscv-user)    RISCV_USER="$2"; shift ;;
        --riscv-port)    RISCV_PORT="$2"; shift ;;
        --issue)         ISSUE="$2"; shift ;;
        --config)        CONFIG="$2"; shift ;;
        --scalar-pack)   PACK_ARGS=(--scalar-pack); PACKER="scalar packer" ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)       sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

pin_check "${CPU_FLAG}" false "${CPU}" "${ALLOW_CORE0:-false}" || exit 1
[ "${ALLOW_CORE0:-false}" = true ] && CPU_ARGS+=(--allow-core-0)
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1 (got '${LAUNCHES}')"
[[ "${THREADS}" =~ ^[18]$ ]] || die "--threads takes 1 or 8 (got '${THREADS}'): 2-7 would use core 0 outside the 8-thread exception"
if [ "${THREADS}" -gt 1 ]; then
    [ "${CPU_FLAG}" = true ] && die "--threads fixes the cores (0-$((THREADS - 1))): drop --cpu"
    CPU_ARGS=(--threads "${THREADS}"); CORES="cores 0-$((THREADS - 1)) (core 0: user's exception, #63 §2)"
fi
FIRST=1; LAST=${LAUNCHES}
if [ -n "${ONLY}" ]; then
    { [[ "${ONLY}" =~ ^[0-9]+$ ]] && [ "${ONLY}" -ge 1 ] && [ "${ONLY}" -le "${LAUNCHES}" ]; } \
        || die "--launch takes 1 to ${LAUNCHES} (got '${ONLY}')"
    FIRST=${ONLY}; LAST=${ONLY}
fi
[[ "${WAIT_MIN}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--wait-min takes minutes (got '${WAIT_MIN}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ -n "${BOARD_LIST[0]:-}" ] || die "--boards takes a comma-separated list"
export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

mapfile -t ROWS < <(grep '^ *\["' "${CONFIG}" | sed 's/,[[:space:]]*$//')
N=${#ROWS[@]}
[ "${N}" -gt 0 ] || die "no rows in ${CONFIG} (one [\"fp32\", ...] row per line is expected)"
STEP=$(( (N + LAUNCHES - 1) / LAUNCHES ))

tile_of() { echo "$1" | sed -E 's/.*"AXPY_FP32_Pack", \[([0-9]+), ([0-9]+)\].*/\1x\2/'; }
DATASET="$(echo "${ROWS[0]}" | grep -oE "\"[^\"]+\.dat\"" | tr -d "\"")"  # one dataset, shared by every row
for r in "${ROWS[@]}"; do [[ "${r}" == *"\"${DATASET}\""* ]] || die "every row of ${CONFIG} must run ${DATASET}: ${r}"; done
SHAPES="$(grep -c "^[A-Za-z0-9_]" "${HERE}/datasets/${DATASET}")"
for r in "${ROWS[@]}"; do [[ "$(tile_of "${r}")" =~ ^[0-9]+x[0-9]+$ ]] || die "cannot read a tile from row: ${r}"; done

# write_config <launch> <file> -- the search config with its rows rotated left by (launch-1) x STEP
write_config() {
    local l="$1" i out=() first
    for ((i = 0; i < N; i++)); do out+=("${ROWS[$(( (i + (l - 1) * STEP) % N ))]}"); done
    { echo "["
      echo "    // run_yolo_sweep.sh launch $l: $(basename "${CONFIG}") with the rows rotated (issue #61)"
      first=true
      for r in "${out[@]}"; do ${first} || printf ',\n'; printf '%s' "${r}"; first=false; done
      printf '\n]\n'; } > "$2"
}

print_plan() {
    echo "Issue #${ISSUE} Tile Search: ${LAUNCHES} launches on the first idle board of ${BOARDS}, ${N} tiles x datasets/${DATASET}"
    echo "  Threads   : ${THREADS} on ${CORES}"
    echo "  Settings  : ${CPU_ARGS[*]} --no-nc-overrides --tmin ${TMIN}, uninstrumented, reference protocol, ${PACKER}"
    echo "  Load      : the board's processes are read before every launch; another user's busy job means wait ${WAIT_MIN} min and read again"
    local l
    for ((l = FIRST; l <= LAST; l++)); do echo "  Launch ${l}  : starts at $(tile_of "${ROWS[$(( ((l - 1) * STEP) % N ))]}")"; done
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    tmp="$(mktemp -d)"
    for ((l = FIRST; l <= LAST; l++)); do write_config "${l}" "${tmp}/config_L${l}.json"; echo ""; echo "=== config_L${l}.json ==="; cat "${tmp}/config_L${l}.json"; done
    rm -rf "${tmp}"
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/yolo_sweep_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'board\tlaunch\tfirst_tile\tresult_dir\texit\n' > "${SERIES}/runs.tsv"

wait_s() { awk -v m="${WAIT_MIN}" 'BEGIN { printf "%d", m * 60 }'; }

# Pick the board: the first idle one, read again every --wait-min while none is.
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

# newest_since <stamp> -- the newest logs/run_<date>_<time> folder made since the stamp
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/run_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "run_$1" ]]; then return 1; fi
    echo "${d}"
}

# report_and_dump -- the report over the launches that finished so far, then the copy to .scratch/reports/
report_and_dump() {
    local dirs=()
    mapfile -t dirs < <(awk -F'\t' -v L="${LOGS}" 'NR > 1 && $5 == "0" { print L "/" $4 }' "${SERIES}/runs.tsv")
    if [ "${#dirs[@]}" -gt 0 ]; then
        "${REPORT_CMD[@]}" "${dirs[@]}" --compare 16x8 --compare n=4:16x4 --heatmap "${SERIES}/tile_search_heatmap.png" \
            > "${SERIES}/tile_search_report.md" 2>&1 || echo "    report failed: ${SERIES}/tile_search_report.md"
    fi
    hector_yolo_dump "${ISSUE}" "${SERIES}"
}

OK=0; FAILED=()
for ((l = FIRST; l <= LAST; l++)); do
    cfg="${SERIES}/config_L${l}.json"
    write_config "${l}" "${cfg}"
    first="$(tile_of "${ROWS[$(( ((l - 1) * STEP) % N ))]}")"
    log="${SERIES}/${BOARD}_L${l}.log"
    while others="$(board_busy "${BOARD}" "${SERIES}/load_${BOARD}_L${l}.txt")"; [ -n "${others}" ]; do
        echo "[$(date +%H:%M)] launch ${l}: ${BOARD} is busy (${others}); waiting ${WAIT_MIN} min"
        sleep "$(wait_s)"
    done
    echo "[$(date +%H:%M)] launch ${l}/${LAUNCHES} on ${BOARD}, starting at ${first}: log ${log}"
    start="$(date +%Y-%m-%d_%H%M%S)"
    (
        flock -w 21600 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
        cd "${HERE}/../.." && "${BENCH_RUN}" "${CONN[@]}" --config "${cfg}" --no-nc-overrides "${CPU_ARGS[@]}" "${PACK_ARGS[@]}" --tmin "${TMIN}"
    ) 9>"${LOCK}" >> "${log}" 2>&1
    rc=$?
    dir="$(newest_since "${start}")"
    # complete = one per-rep file per tile, each holding every shape (a tile that crashed partway leaves fewer)
    reps=0; short=0
    for f in ${dir:+"${dir}"/*/reps_*.txt*}; do
        [ -e "${f}" ] || continue
        reps=$((reps + 1))
        [ "$(zcat -f "${f}" | tail -n +2 | cut -d, -f1 | sort -u | wc -l)" -eq "${SHAPES}" ] || short=$((short + 1))
    done
    if [ "${rc}" -ne 0 ] || [ "${reps}" -ne "${N}" ] || [ "${short}" -ne 0 ]; then
        why="exit ${rc}, ${reps} of ${N} per-rep logs, ${short} missing shapes"
        FAILED+=("launch ${l} (${why})")
        echo "    FAILED or incomplete (${why}); see ${log}. Stopping the series."
        printf '%s\t%s\t%s\t%s\t%s\n' "${BOARD}" "${l}" "${first}" "$([ -n "${dir}" ] && basename "${dir}" || echo -)" "FAIL(${rc})" >> "${SERIES}/runs.tsv"
        break
    fi
    others="$(board_busy "${BOARD}" "${SERIES}/load_${BOARD}_L${l}_after.txt")"
    if [ -n "${others}" ]; then
        echo "$(date -Is) launch ${l} on ${BOARD}, read after: ${others}" >> "${SERIES}/load_warnings.txt"
        echo "    RED NOTE: another user's job was on ${BOARD} at the end of launch ${l}: ${others}"
    fi
    OK=$((OK + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' "${BOARD}" "${l}" "${first}" "$(basename "${dir}")" 0 >> "${SERIES}/runs.tsv"
    echo "    ok: $(basename "${dir}")"
    report_and_dump
done

echo ""; echo "Series finished: ${OK} of $((LAST - FIRST + 1)) launches ok on ${BOARD}. Run folders: ${SERIES}/runs.tsv"
report_and_dump
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"
[ -s "${SERIES}/load_warnings.txt" ] && { echo "RED NOTE: load seen after a launch (${SERIES}/load_warnings.txt):"; sed 's/^/  /' "${SERIES}/load_warnings.txt"; }
[ "${#FAILED[@]}" -eq 0 ]
