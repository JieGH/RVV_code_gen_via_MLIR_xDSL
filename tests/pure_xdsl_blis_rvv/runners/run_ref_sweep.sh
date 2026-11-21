#!/bin/bash

# ============================================================
# run_ref_sweep.sh -- an unattended sweep of the reference design alone (issue #29).
#
# The #26 sweeps ran OUR design 16 times (vector pack overnight, scalar pack in the afternoon).
# This runs the REFERENCE design (ext/RVV_ukernels_benchmark, nothing of ours in it) the same
# way, so its numbers have an error bar and sit next to ours under the same conditions:
#
#   - every layer of ref_baseline/cases.tsv (the ten layers of config.json), the same shapes;
#   - the reference configuration: RVV_256 kernels, the LDX set (lda/ldb are runtime
#     arguments), the `macc` variant (vfmacc.vf), loadAB; `--bracket`, because the reference
#     ships no odd-NR kernel (32x7, 16x15) and those layers are measured at the even
#     neighbours either side;
#   - core 4 (issue #39; --cpu 0 --allow-core-0 to match the 19-20 Sep core-0 runs), tmin 10;
#   - the boards take turns, one run every --interval minutes (default 30), 8 runs per board
#     by default, so 16 runs like the sweeps of ours;
#   - one run at a time, under the same lock as cron_bench_run.sh and run_scalar_pack_sweep.sh,
#     because jbpi, jbpi2 and fej share one /nfs directory;
#   - a run that fails, or that leaves a row out or writes a GFLOPS that is not a number
#     (run_ref_baseline.sh only warns when one of its builds fails and goes on), is logged as
#     failed and the sweep goes on; the exit code is 1 if any run failed, and the log names them.
#
# When the last run ends, compare_ref_sweep.py puts the reference beside our vector-pack runs of
# the overnight sweep and the runs of the newest scalar-pack sweep, layer by layer.
#
# Logs: logs/refbase_<stamp>/ for every run (as any run_ref_baseline.sh run) and
# logs/sweep_ref_<stamp>/ for the sweep itself: plan.txt, one log per run, runs.txt (board and
# result dir of each), comparison.txt.
#
# Usage (jump-host, from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/runners/run_ref_sweep.sh \
#       > tests/pure_xdsl_blis_rvv/logs/sweep_ref.out 2>&1 < /dev/null &
#
# Flags:
#   --start-at HH:MM      wait until this time (today, or tomorrow if it has passed); default now
#   --boards a,b          boards that take turns (default jbpi,jbpi2)
#   --runs N              runs per board (default 8)
#   --interval MIN        minutes between run starts (default 30; a run that overruns delays
#                         the next one, nothing overlaps)
#   --tmin S | --cpu N | --no-pin   as for run_ref_baseline.sh (defaults 10 and core 4; core 0 needs --allow-core-0)
#   --opt OPT|LDX|BASE    the reference kernel set (default LDX)
#   --variants 0|1|2      the kernel variant, 2 = macc (default), 0 = bcast, 1 = gather
#   --direct              reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --scalar-sweep DIR    scalar-pack sweep to compare against (default: the newest)
#   --vs-from STAMP --vs-to STAMP    vector-pack runs to compare against, by run dir stamp
#                         (default 2026-09-19_230000 to 2026-09-20_070000, the overnight sweep)
#   --no-compare          skip the comparison
#   --dry-run             print the schedule and the first run's settings, then stop
#
# Do not run `--dry-run` (or run_ref_baseline.sh) while a sweep is going: each one rewrites
# ref_baseline/generated/, which a running reference run stages from.
#
# Jump-host settings come from the environment, as for cron_bench_run.sh:
# REMOTE_PASS (default empty), REMOTE_HOST (default fej), REMOTE_PORT (default 3322).
# ============================================================

# No `set -e`: a failed run must not end the sweep.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the project folder
LOGS="${SWEEP_LOGS:-${HERE}/logs}"                                   # SWEEP_LOGS and SWEEP_REF exist so the
REF_RUN="${SWEEP_REF:-${HERE}/ref_baseline/run_ref_baseline.sh}"     # loop can be tried against a stub
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"

BOARDS="jbpi,jbpi2"
RUNS=8
INTERVAL=30
START_AT=""
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN=10
OPT="LDX"
VARIANTS="2"
MODE_ARG="--jump"
RISCV_USER="jlei"
RISCV_PORT="22"
SCALAR_SWEEP=""
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
        --allow-core-0)  ALLOW_CORE0=true ;;
        --opt)           OPT="$2"; shift ;;
        --variants)      VARIANTS="$2"; shift ;;
        --direct)        MODE_ARG="--direct" ;;
        --riscv-user)    RISCV_USER="$2"; shift ;;
        --riscv-port)    RISCV_PORT="$2"; shift ;;
        --scalar-sweep)  SCALAR_SWEEP="$2"; shift ;;
        --vs-from)       VS_FROM="$2"; shift ;;
        --vs-to)         VS_TO="$2"; shift ;;
        --no-compare)    COMPARE=false ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)       sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" "${ALLOW_CORE0:-false}" || exit 1
[ "${ALLOW_CORE0:-false}" = true ] && CPU_ARGS+=(--allow-core-0)
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]]     || die "--runs takes a whole number of at least 1 (got '${RUNS}')"
[[ "${INTERVAL}" =~ ^[0-9]+$ ]]      || die "--interval takes whole minutes (got '${INTERVAL}')"
[[ -z "${START_AT}" || "${START_AT}" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || die "--start-at takes HH:MM (got '${START_AT}')"
[[ "${OPT}" =~ ^(BASE|LDX|OPT|ALL)$ ]] || die "--opt takes BASE, LDX, OPT or ALL (got '${OPT}')"
[[ "${VARIANTS}" =~ ^[012]$ ]]       || die "--variants takes one of 0, 1, 2 (got '${VARIANTS}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
NB=${#BOARD_LIST[@]}
[ "${NB}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"
TOTAL=$((NB * RUNS))

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

# The settings of one run, minus the board.
REF_ARGS=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}"
          --opt "${OPT}" --variants "${VARIANTS}" --bracket "${CPU_ARGS[@]}" --tmin "${TMIN}" --no-compare)

NOW="$(date +%s)"
T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi
due_of() { echo $((T0 + $1 * INTERVAL * 60)); }
stamp_of() { date -d "@$1" '+%a %d %b %H:%M'; }

print_plan() {
    echo "Reference sweep: ${TOTAL} runs (${RUNS} per board on ${BOARDS}), one every ${INTERVAL} min"
    echo "  Design    : the reference alone (its packers, its ${OPT} kernels, variant ${VARIANTS}); nothing of ours"
    echo "  Settings  : ${REF_ARGS[*]}"
    echo "  First run : $(stamp_of "${T0}")"
    echo "  Last start: $(stamp_of "$(due_of $((TOTAL - 1)))")"
    if [ "${COMPARE}" = true ]; then
        echo "  Compare to: vector-pack runs with a stamp from ${VS_FROM} to ${VS_TO}, and the scalar-pack sweep ${SCALAR_SWEEP:-(the newest)}"
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
    "${REF_RUN}" "${REF_ARGS[@]}" --riscv-ip "${BOARD_LIST[0]}" --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SWEEP_DIR="${LOGS}/sweep_ref_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"
print_plan | tee "${SWEEP_DIR}/plan.txt"
: > "${SWEEP_DIR}/runs.txt"

# newest_since <stamp> -- the newest logs/refbase_* dir made since the stamp (%Y-%m-%d_%H%M%S names
# sort in time order).
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/refbase_* 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "refbase_$1" ]]; then return 1; fi
    echo "${d}"
}

# missing_rows <result dir> -- prints each row of the run plan that no result CSV holds with a
# numeric GFLOPS (`l;m;n;k;Gflops`, e.g. `1;1024;384;1024;7.30e+00`).
# run_ref_baseline.sh only warns when one of its builds fails, and exits 0.
missing_rows() {
    local d="$1" numid tag
    [ -s "${d}/run_plan.tsv" ] || { echo "no run_plan.tsv"; return; }
    while IFS=$'\t' read -r _ numid _ _ _ _ _ _ _ _ _ _ _ tag; do
        if ! cat "${d}"/result_"${tag}"_*.csv 2>/dev/null | grep -Eq "^${numid};[0-9]+;[0-9]+;[0-9]+;[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$"; then
            echo "${tag}/${numid}"
        fi
    done < <(grep -v '^#' "${d}/run_plan.tsv")
}

OK=0; FAILED=()
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
        # One run at a time across boards and across this script, the cron wrapper and the
        # other sweep.
        flock -w 10800 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
        echo "$(date -Is) got the lock for ${board}"
        cd "${HERE}/../.." && "${REF_RUN}" "${REF_ARGS[@]}" --riscv-ip "${board}"
    ) 9>"${LOCK}" >> "${runlog}" 2>&1
    rc=$?
    dir="$(newest_since "${start}")"
    if [ "${rc}" -ne 0 ] || [ -z "${dir}" ]; then
        FAILED+=("run $((k + 1)) on ${board} (exit ${rc})")
        echo "  FAILED (exit ${rc}); see ${runlog}"
        continue
    fi
    missing="$(missing_rows "${dir}" | tr '\n' ' ')"
    if [ -n "${missing}" ]; then
        FAILED+=("run $((k + 1)) on ${board}: no result for ${missing}")
        echo "  INCOMPLETE: $(basename "${dir}") has no result for ${missing}; see ${runlog}"
        continue
    fi
    OK=$((OK + 1))
    printf '%s\t%s\n' "${board}" "$(basename "${dir}")" >> "${SWEEP_DIR}/runs.txt"
    echo "  ok: $(basename "${dir}")"
done

echo ""; echo "Sweep finished: ${OK} of ${TOTAL} runs ok."
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"

if [ "${COMPARE}" = true ] && [ "${OK}" -gt 0 ]; then
    echo ""; echo "=== comparison: ${OK} reference runs against our vector-pack and scalar-pack sets ==="
    CMP_ARGS=(--sweep "${SWEEP_DIR}" --logs "${LOGS}" --vector-from "${VS_FROM}" --vector-to "${VS_TO}")
    [ -n "${SCALAR_SWEEP}" ] && CMP_ARGS+=(--scalar-sweep "${SCALAR_SWEEP}")
    python3 "${HERE}/ref_baseline/compare_ref_sweep.py" "${CMP_ARGS[@]}" | tee "${SWEEP_DIR}/comparison.txt"
fi

[ "${#FAILED[@]}" -eq 0 ]
