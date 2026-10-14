#!/bin/bash

# ============================================================
# run_issue50.sh -- prefetcher counters at the four tiles that flip the slow state of #48 (issue #50)
#
# One LAUNCH is one instrumented run on one board, back to back on core 4: config_issue50_sweep.json
# (16x6, 16x7, 16x8, the shipped 16x15, each on the three shapes of datasets/issue46_ref_b3.dat: B3,
# B3_K256, B2 -- cut from config_issue48_sweep.json to just these four rows), --instr (warm-up
# protocol, 1000 reps), vector packer, --no-nc-overrides (B3 at nc=384), --raw-events x60-l2 by
# default (#42: the X60's L2 and L1D-prefetch events, per rep, pack and compute apart).
#
# No code change and no setting to compare between launches: the repeats are to see the fast/slow
# split settle (house rule 2), not to rotate an order. The boards take turns launch by launch, one
# run at a time, under the lock the other sweeps use (they share the host's pj_gen/ and one /nfs
# directory).
#
# Around every run the board's processes are read (`ps`, other users only). Another user's job is
# NOT a reason to skip a run (2026-09-29 rule): the run goes ahead, the load is written to
# load_warnings.txt, and reps_report.py prints a RED NOTE for every run folder whose provenance.txt
# records it. Numbers that decide something come from a quiet window (midnight): use --start-at 00:00.
#
# Board time (one launch, one board): about 15-20 min (4 tiles x 3 shapes, instrumented).
#
# Logs: logs/issue50_<stamp>/ : plan.txt, runs.tsv (board, launch, run folder, exit), one log per run,
# load_<board>_L<n>_{before,after}.txt, load_warnings.txt. The run folders are the usual logs/run_*.
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue50.sh --start-at 00:00 \
#       > tests/pure_xdsl_blis_rvv/logs/issue50.out 2>&1 < /dev/null &
#
# Flags:
#   --boards a,b        boards that take turns (default jbpi,jbpi2)
#   --launches N        launches per board (default 3; house rule 2 needs at least 3)
#   --start-at HH:MM    wait until this time (today, or tomorrow if it has passed); default now
#   --cpu N             as for the other scripts (default core 4; core 0 needs --allow-core-0)
#   --raw-events E      give E (a name=code list, or x60-l2) instead of the default x60-l2
#   --workspace NAME    the workspace directory on the board (compile_macro_bench.sh --riscv-workspace). The boards
#                       share one NFS home, so two series running at once, one per board, each need their own
#                       (and their own checkout of the repo on this host: every run rewrites pj_gen/)
#   --direct            reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --dry-run           print the plan and the first run's settings, then stop
#
# Jump-host settings come from the environment: REMOTE_PASS (default empty), REMOTE_HOST (default fej),
# REMOTE_PORT (default 3322). Do not start other benchmarks, `--dry-run` of a wrapper or the test suite on the host
# meanwhile: they rewrite pj_gen/ under a running series.
# ============================================================

# No `set -e`: a failed run must not end the series.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${ISSUE50_LOGS:-${HERE}/logs}"                                 # ISSUE50_* let the loop be tried against stubs
BENCH_RUN="${ISSUE50_RUN:-${HERE}/compile_macro_bench.sh}"
LOCK="${LOGS}/.cron_bench.lock"
SWEEP_CONFIG="config_issue50_sweep.json"
source "${HERE}/board_pin.sh"

BOARDS="jbpi,jbpi2"
LAUNCHES=3
START_AT=""
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
RAW_EVENTS="x60-l2"
WORKSPACE=""
MODE_ARG="--jump"
RISCV_USER="jlei"
RISCV_PORT="22"
DRY_RUN=false

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --boards)        BOARDS="$2"; shift ;;
        --launches)      LAUNCHES="$2"; shift ;;
        --start-at)      START_AT="$2"; shift ;;
        --cpu)           CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)        CPU_ARGS=(--no-pin); NO_PIN=true ;;
        --allow-core-0)  ALLOW_CORE0=true ;;
        --raw-events)    RAW_EVENTS="$2"; shift ;;
        --workspace)     WORKSPACE="$2"; shift ;;
        --direct)        MODE_ARG="--direct" ;;
        --riscv-user)    RISCV_USER="$2"; shift ;;
        --riscv-port)    RISCV_PORT="$2"; shift ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)       sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" "${ALLOW_CORE0:-false}" || exit 1
[ "${ALLOW_CORE0:-false}" = true ] && CPU_ARGS+=(--allow-core-0)
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1 (got '${LAUNCHES}')"
[[ -z "${START_AT}" || "${START_AT}" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || die "--start-at takes HH:MM (got '${START_AT}')"
[ -n "${RAW_EVENTS}" ] || die "--raw-events needs a value (this series exists to read counters)"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ "${#BOARD_LIST[@]}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

SWEEP_ROWS="$(grep -c '^[[:space:]]*\["fp32"' "${HERE}/${SWEEP_CONFIG}")"

CONN=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}")

# build_cmd <board> -- sets CMD to the command line of one run (an array, so nothing is word-split)
build_cmd() {
    CMD=("${BENCH_RUN}" "${CONN[@]}" --riscv-ip "$1" --config "${SWEEP_CONFIG}" --no-nc-overrides "${CPU_ARGS[@]}"
         --instr --raw-events "${RAW_EVENTS}")
    [ -n "${WORKSPACE}" ] && CMD+=(--riscv-workspace "${WORKSPACE}")
    return 0
}

# board_ps <board> -- the board's processes by CPU (user, %cpu, core, age, command), whoever's they are.
board_ps() {
    if [ -n "${ISSUE50_PS_CMD:-}" ]; then ${ISSUE50_PS_CMD} "$1"; return; fi
    local cmd='ps -eo user:20,pcpu,psr,etime,comm --sort=-pcpu --no-headers | head -12'
    if [ "${MODE_ARG}" = "--direct" ]; then
        ssh -p "${RISCV_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${RISCV_USER}@$1" "${cmd}"
    else
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            "${RISCV_USER}@${REMOTE_HOST}" "ssh -p ${RISCV_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '${cmd}'"
    fi
}

# snapshot <board> <tag> <before|after> -- reads the board's processes into the series dir and notes any other
# user's heavy job in load_warnings.txt. Never stops the run.
snapshot() {
    local f="${SERIES}/load_$1_$2_$3.txt" heavy
    board_ps "$1" > "${f}" 2>&1 || echo "(could not read the board's processes)" >> "${f}"
    heavy="$(awk -v me="${RISCV_USER}" "${OTHER_USERS_AWK}" "${f}")"
    if [ -n "${heavy}" ]; then
        echo "$(date -Is) $1 $2 ($3): ${heavy}" >> "${SERIES}/load_warnings.txt"
        echo "  RED NOTE: $1 $2 ($3): another user is running: ${heavy}"
    fi
}

TOTAL=$(( ${#BOARD_LIST[@]} * LAUNCHES ))
NOW="$(date +%s)"; T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi

print_plan() {
    echo "Issue #50 series: ${LAUNCHES} launches on each of ${BOARDS}, boards taking turns (${TOTAL} runs)"
    echo "  Start     : $(date -d "@${T0}" '+%a %d %b %H:%M')"
    echo "  Settings  : ${CPU_ARGS[*]} --no-nc-overrides --instr --raw-events ${RAW_EVENTS}"
    echo "  Config    : ${SWEEP_CONFIG} (${SWEEP_ROWS} tiles x 3 shapes: B3, B3_K256, B2)"
    echo "  Load      : other users' processes are read before and after every run; a heavy one is flagged red, not a stop"
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    echo ""; echo "=== the first run's command (${BOARD_LIST[0]}) ==="
    build_cmd "${BOARD_LIST[0]}"
    echo "  ${CMD[*]}"
    echo ""; echo "=== the first run's own overview ==="
    "${CMD[@]}" --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/issue50_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'board\tlaunch\tresult_dir\texit\n' > "${SERIES}/runs.tsv"

if [ "${T0}" -gt "$(date +%s)" ]; then
    echo ""; echo "[$(date +%H:%M)] waiting $(( (T0 - $(date +%s) + 59) / 60 )) min for $(date -d "@${T0}" +%H:%M)"
    sleep $(( T0 - $(date +%s) ))
fi

# newest_since <stamp> -- the newest logs/run_<date>_<time> folder made since the stamp. Only stamped names count.
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/run_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "run_$1" ]]; then return 1; fi
    echo "${d}"
}

# complete <dir> -- a finished run holds one per-rep file for every tile of the config
complete() {
    [ "$(ls "$1"/*/reps_*.txt* 2>/dev/null | wc -l)" -eq "${SWEEP_ROWS}" ]
}

OK=0; FAILED=()
for ((l = 1; l <= LAUNCHES; l++)); do
  for board in "${BOARD_LIST[@]}"; do
    echo ""; echo "[$(date +%H:%M)] launch ${l}/${LAUNCHES} on ${board}"
    tag="L${l}"
    log="${SERIES}/${board}_${tag}.log"
    start="$(date +%Y-%m-%d_%H%M%S)"
    build_cmd "${board}"
    echo "[$(date +%H:%M)]   log ${log}"
    (
        flock -w 21600 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
        echo "$(date -Is) got the lock for ${board}"
        snapshot "${board}" "${tag}" before
        cd "${HERE}/../.." && "${CMD[@]}"
        rc=$?
        snapshot "${board}" "${tag}" after
        exit ${rc}
    ) 9>"${LOCK}" >> "${log}" 2>&1
    rc=$?
    dir="$(newest_since "${start}")"
    if [ "${rc}" -ne 0 ] || [ -z "${dir}" ] || ! complete "${dir}"; then
        FAILED+=("launch ${l} ${board} (exit ${rc})")
        echo "    FAILED or incomplete (exit ${rc}); see ${log}"
        printf '%s\t%s\t%s\t%s\n' "${board}" "${l}" "${dir:+$(basename "${dir}")}" "FAIL(${rc})" | sed 's/\t\t/\t-\t/' >> "${SERIES}/runs.tsv"
        continue
    fi
    OK=$((OK + 1))
    printf '%s\t%s\t%s\t%s\n' "${board}" "${l}" "$(basename "${dir}")" 0 >> "${SERIES}/runs.tsv"
    echo "    ok: $(basename "${dir}")"
  done
done

echo ""; echo "Series finished: ${OK} of ${TOTAL} runs ok. Run folders: ${SERIES}/runs.tsv"
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"
if [ -s "${SERIES}/load_warnings.txt" ]; then
    echo ""; echo "RED NOTE: another user's job was on a board during this series (${SERIES}/load_warnings.txt):"
    sed 's/^/  /' "${SERIES}/load_warnings.txt"
    echo "  These runs are not clean: repeat them in a quiet window (e.g. --start-at 00:00)."
fi
[ "${#FAILED[@]}" -eq 0 ]
