#!/bin/bash

# ============================================================
# run_scalar_pack_sweep.sh -- an unattended sweep with the scalar packer only (issue #26).
#
# The 19-20 Sep overnight sweep (16 runs, jbpi and jbpi2 taking turns, every layer in
# config.json) used the RVV vector packer. This repeats it with `--scalar-pack`, so every timed
# call packs with the scalar packer, to see how much of each layer's speed is the packer:
#
#   - same config (config.json), same flags as the default run command
#     (--no-nc-overrides --cpu 0 --tmin 10), same protocol (reference, uninstrumented);
#   - the boards take turns, one run every --interval minutes (default 30: about 7 min of
#     run, then idle), 8 runs per board by default, so 16 runs like the overnight one;
#   - one run at a time, under the same lock as cron_bench_run.sh, because jbpi, jbpi2 and fej
#     share one /nfs directory and a second run would delete the first one's build tree;
#   - a run that fails is logged and the sweep goes on; the exit code says how many failed.
#
# When the last run ends, the sweep is compared against the vector-pack runs made in a time
# window (--vs-from / --vs-to, default the 19-20 Sep overnight sweep) with
# compare_pack_swap.py, layer by layer. A difference smaller than the spread between runs is
# called "within noise".
#
# Logs: logs/run_<stamp>/ for every run (as any compile_macro_bench.sh run) and
# logs/sweep_scalar_<stamp>/ for the sweep itself: plan.txt, one log per run, runs.txt (board and
# run dir of each), comparison.txt.
#
# Usage (jump-host, from the project root; detach it so it survives the terminal):
#   nohup ./tests/pure_xdsl_blis_rvv/run_scalar_pack_sweep.sh --start-at 23:00 \
#       > tests/pure_xdsl_blis_rvv/logs/sweep_scalar.out 2>&1 &
#
# Flags:
#   --start-at HH:MM      wait until this time (today, or tomorrow if it has passed); default now
#   --boards a,b          boards that take turns (default jbpi,jbpi2)
#   --runs N              runs per board (default 8)
#   --interval MIN        minutes between run starts (default 30; a run that overruns delays
#                         the next one, nothing overlaps)
#   --tmin S | --cpu N | --no-pin   as for compile_macro_bench.sh (defaults 10 and core 0)
#   --shipped-nc          keep the nc overrides in config.json (default: --no-nc-overrides)
#   --config FILE         project config (default config.json)
#   --direct              reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --vs-from STAMP --vs-to STAMP    vector-pack runs to compare against, by run dir stamp
#                         (default 2026-09-19_230000 to 2026-09-20_070000, the overnight sweep)
#   --no-compare          skip the comparison
#   --dry-run             print the schedule and the first run's settings, then stop
#
# Jump-host settings come from the environment, as for cron_bench_run.sh:
# REMOTE_PASS (default empty), REMOTE_HOST (default fej), REMOTE_PORT (default 3322).
# ============================================================

# No `set -e`: a failed run must not end the sweep.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${SWEEP_LOGS:-${HERE}/logs}"                 # SWEEP_LOGS and SWEEP_BENCH exist so the
BENCH="${SWEEP_BENCH:-${HERE}/compile_macro_bench.sh}"   # loop can be tried against a stub
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"

BOARDS="jbpi,jbpi2"
RUNS=8
INTERVAL=30
START_AT=""
CPU_ARGS=(--cpu 0)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN=10
NC_ARGS=(--no-nc-overrides)
SHIPPED_NC=false
CONFIG="config.json"
MODE_ARG="--jump"
RISCV_USER="jlei"
RISCV_PORT="22"
VS_FROM="2026-09-19_230000"
VS_TO="2026-09-20_070000"
COMPARE=true
DRY_RUN=false

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start-at)      START_AT="$2"; shift ;;
        --boards)        BOARDS="$2"; shift ;;
        --runs)          RUNS="$2"; shift ;;
        --interval)      INTERVAL="$2"; shift ;;
        --tmin)          TMIN="$2"; shift ;;
        --cpu)           CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)        CPU_ARGS=(--no-pin); NO_PIN=true ;;
        --shipped-nc)    NC_ARGS=(); SHIPPED_NC=true ;;
        --config)        CONFIG="$2"; shift ;;
        --direct)        MODE_ARG="--direct" ;;
        --riscv-user)    RISCV_USER="$2"; shift ;;
        --riscv-port)    RISCV_PORT="$2"; shift ;;
        --vs-from)       VS_FROM="$2"; shift ;;
        --vs-to)         VS_TO="$2"; shift ;;
        --no-compare)    COMPARE=false ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)       sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" || exit 1
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]]     || die "--runs takes a whole number of at least 1 (got '${RUNS}')"
[[ "${INTERVAL}" =~ ^[0-9]+$ ]]      || die "--interval takes whole minutes (got '${INTERVAL}')"
[[ -z "${START_AT}" || "${START_AT}" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || die "--start-at takes HH:MM (got '${START_AT}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
NB=${#BOARD_LIST[@]}
[ "${NB}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"
TOTAL=$((NB * RUNS))

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

# The settings of one run, minus the board.
BENCH_ARGS=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}" --config "${CONFIG}"
            --scalar-pack "${NC_ARGS[@]}" "${CPU_ARGS[@]}" --tmin "${TMIN}")

NOW="$(date +%s)"
T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi
due_of() { echo $((T0 + $1 * INTERVAL * 60)); }
stamp_of() { date -d "@$1" '+%a %d %b %H:%M'; }

print_plan() {
    echo "Scalar-pack sweep: ${TOTAL} runs (${RUNS} per board on ${BOARDS}), one every ${INTERVAL} min"
    echo "  Packing   : scalar only (--scalar-pack); uninstrumented build, reference protocol"
    echo "  Settings  : ${BENCH_ARGS[*]}"
    echo "  Config    : ${CONFIG}"
    echo "  First run : $(stamp_of "${T0}")"
    echo "  Last start: $(stamp_of "$(due_of $((TOTAL - 1)))") (a run takes about 7 min on the full config.json)"
    if [ "${COMPARE}" = true ]; then
        echo "  Compare to: vector-pack runs with a stamp from ${VS_FROM} to ${VS_TO}"
    else
        echo "  Compare to: nothing (--no-compare)"
    fi
    echo "  Schedule  :"
    local k
    for ((k = 0; k < TOTAL; k++)); do
        printf "    %2d  %-8s %s\n" $((k + 1)) "${BOARD_LIST[$((k % NB))]}" "$(stamp_of "$(due_of ${k})")"
    done
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    echo ""; echo "=== the first run's own overview ==="
    "${BENCH}" "${BENCH_ARGS[@]}" --riscv-ip "${BOARD_LIST[0]}" --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SWEEP_DIR="${LOGS}/sweep_scalar_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"
print_plan | tee "${SWEEP_DIR}/plan.txt"
: > "${SWEEP_DIR}/runs.txt"

# newest_since <stamp> -- the newest logs/run_* dir made since the stamp (%Y-%m-%d_%H%M%S names
# sort in time order).
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/run_* 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "run_$1" ]]; then return 1; fi
    echo "${d}"
}

OK=0; FAILED=()
SCALAR_DIRS=()
for ((k = 0; k < TOTAL; k++)); do
    board="${BOARD_LIST[$((k % NB))]}"
    due="$(due_of ${k})"; now="$(date +%s)"
    if [ "${due}" -gt "${now}" ]; then
        echo ""; echo "[$(date +%H:%M)] run $((k + 1))/${TOTAL} (${board}) is due at $(date -d "@${due}" +%H:%M); waiting $(((due - now + 59) / 60)) min"
        sleep $((due - now))
    fi
    runlog="${SWEEP_DIR}/run_$(printf '%02d' $((k + 1)))_${board}.log"
    echo "[$(date +%H:%M)] run $((k + 1))/${TOTAL} on ${board}, log ${runlog}"
    start="$(date +%Y-%m-%d_%H%M%S)"
    (
        # One run at a time across boards and across this script and the cron wrapper.
        flock -w 10800 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
        echo "$(date -Is) got the lock for ${board}"
        cd "${HERE}/../.." && "${BENCH}" "${BENCH_ARGS[@]}" --riscv-ip "${board}"
    ) 9>"${LOCK}" >> "${runlog}" 2>&1
    rc=$?
    dir="$(newest_since "${start}")"
    if [ "${rc}" -eq 0 ] && [ -n "${dir}" ]; then
        OK=$((OK + 1)); SCALAR_DIRS+=("$(basename "${dir}")")
        printf '%s\t%s\n' "${board}" "$(basename "${dir}")" >> "${SWEEP_DIR}/runs.txt"
        echo "  ok: $(basename "${dir}")"
    else
        FAILED+=("run $((k + 1)) on ${board} (exit ${rc})")
        echo "  FAILED (exit ${rc}); see ${runlog}"
    fi
done

echo ""; echo "Sweep finished: ${OK} of ${TOTAL} runs ok."
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"

if [ "${COMPARE}" = true ] && [ "${OK}" -gt 0 ]; then
    GLOB="all_raw_*_noinstr_refproto_no_nc_override.txt"; CMP_ARGS=()
    if [ "${SHIPPED_NC}" = true ]; then GLOB="all_raw_*_noinstr_refproto.txt"; CMP_ARGS=(--shipped-nc); fi
    VECTOR_DIRS=()
    for d in $(ls -d "${LOGS}"/run_* 2>/dev/null | sort); do
        name="$(basename "${d}")"; stamp="${name#run_}"
        if [[ "${stamp}" < "${VS_FROM}" || "${stamp}" > "${VS_TO}" ]]; then continue; fi
        compgen -G "${d}/${GLOB}" > /dev/null && VECTOR_DIRS+=("${name}")
    done
    echo ""; echo "=== comparison: ${#SCALAR_DIRS[@]} scalar-pack runs against ${#VECTOR_DIRS[@]} vector-pack runs (${VS_FROM} to ${VS_TO}) ==="
    if [ "${#VECTOR_DIRS[@]}" -eq 0 ]; then
        echo "  no vector-pack runs (${GLOB}) in that window; nothing to compare. Compare later with:"
        echo "  python3 tests/pure_xdsl_blis_rvv/compare_pack_swap.py --vector <run dirs> --scalar ${SCALAR_DIRS[*]}"
    else
        python3 "${HERE}/compare_pack_swap.py" --vector "${VECTOR_DIRS[@]/#/${LOGS}/}" --scalar "${SCALAR_DIRS[@]/#/${LOGS}/}" "${CMP_ARGS[@]}" \
            | tee "${SWEEP_DIR}/comparison.txt"
    fi
fi

[ "${#FAILED[@]}" -eq 0 ]
