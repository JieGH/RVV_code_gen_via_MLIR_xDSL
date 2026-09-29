#!/bin/bash

# ============================================================
# run_issue46.sh -- does the reference design's B3 also have two speeds? (issue #46)
#
# One LAUNCH is three runs on one board, back to back on core 4:
#   O  ours          compile_macro_bench.sh --config config_issue46_ref_b3.json --no-nc-overrides
#                    (B3 at nc=384 with B2, B3_K256 and R24 as controls, MR16_NR15)
#   P  reference     run_ref_baseline.sh, the driver unchanged (the control)
#   R  reference     the same with --per-rep: a staged test_gemm.c stores the clock after every call
#                    (ref_per_rep.py), so reps_report.py can read its per-rep times
# P and R must agree on mean GOPs before R's per-rep times are believed (issue #46, plan step 2).
#
# The order changes from launch to launch so no run is always first or last (house rule 2):
#   launch 1: O P R    launch 2: R P O    launch 3: O R P    launch 4: P R O    (then again)
# and the boards take turns launch by launch, one run at a time, under the lock the other sweeps
# use (they share the host's ref_baseline/generated/ and one /nfs directory).
#
# Around every launch the board's processes are read (`ps`, other users only). Another user's job
# is NOT a reason to skip a run (2026-09-29 rule): the run goes ahead and the load is written to
# load_warnings.txt, and reps_report.py prints a RED NOTE for every run folder whose provenance.txt
# records it. A launch made under load is not a clean result: repeat it in a quiet window (midnight),
# e.g. with --start-at 00:00.
#
# Logs: logs/issue46_<stamp>/ : plan.txt, runs.tsv (board, launch, step, run folder, exit), one log per
# run, load_<board>_L<n>_{before,after}.txt, load_warnings.txt. The run folders are the usual
# logs/run_* and logs/refbase_*.
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue46.sh --start-at 00:00 \
#       > tests/pure_xdsl_blis_rvv/logs/issue46.out 2>&1 < /dev/null &
#
# Flags:
#   --boards a,b        boards that take turns (default jbpi,jbpi2)
#   --launches N        launches per board (default 4; house rule 2 needs at least 3)
#   --start-at HH:MM    wait until this time (today, or tomorrow if it has passed); default now
#   --tmin S | --cpu N  as for the other scripts (defaults 10 and core 4; core 0 needs --allow-core-0)
#   --opt OPT|LDX|BASE  the reference kernel set (default LDX, as the #29 sweep)
#   --direct            reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --dry-run           print the plan and the first launch's settings, then stop
#
# Jump-host settings come from the environment: REMOTE_PASS (default empty), REMOTE_HOST (default
# fej), REMOTE_PORT (default 3322). Do not start other benchmarks on the boards meanwhile.
# ============================================================

# No `set -e`: a failed run must not end the series.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${ISSUE46_LOGS:-${HERE}/logs}"                                 # ISSUE46_* let the loop be tried against stubs
OURS_RUN="${ISSUE46_OURS:-${HERE}/compile_macro_bench.sh}"
REF_RUN="${ISSUE46_REF:-${HERE}/ref_baseline/run_ref_baseline.sh}"
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"

BOARDS="jbpi,jbpi2"
LAUNCHES=4
START_AT=""
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN=10
OPT="LDX"
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
        --tmin)          TMIN="$2"; shift ;;
        --cpu)           CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)        CPU_ARGS=(--no-pin); NO_PIN=true ;;
        --allow-core-0)  ALLOW_CORE0=true ;;
        --opt)           OPT="$2"; shift ;;
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
[[ "${OPT}" =~ ^(BASE|LDX|OPT|ALL)$ ]] || die "--opt takes BASE, LDX, OPT or ALL (got '${OPT}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ "${#BOARD_LIST[@]}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

ORDERS=("O P R" "R P O" "O R P" "P R O")
order_of() { echo "${ORDERS[$(( ($1 - 1) % ${#ORDERS[@]} ))]}"; }
step_name() { case "$1" in O) echo "ours" ;; P) echo "ref" ;; R) echo "ref-per-rep" ;; esac; }

CONN=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}")
REF_ARGS=(--opt "${OPT}" --variants 2 --bracket "${CPU_ARGS[@]}" --tmin "${TMIN}" --no-compare
          --cases "${HERE}/ref_baseline/cases_issue46.tsv")
OURS_ARGS=(--config config_issue46_ref_b3.json --no-nc-overrides "${CPU_ARGS[@]}" --tmin "${TMIN}")

# build_cmd <step> <board> -- sets CMD to the command line of one run (an array, so nothing is word-split)
build_cmd() {
    case "$1" in
        O) CMD=("${OURS_RUN}" "${CONN[@]}" --riscv-ip "$2" "${OURS_ARGS[@]}") ;;
        P) CMD=("${REF_RUN}" "${CONN[@]}" --riscv-ip "$2" "${REF_ARGS[@]}") ;;
        R) CMD=("${REF_RUN}" "${CONN[@]}" --riscv-ip "$2" "${REF_ARGS[@]}" --per-rep) ;;
    esac
}

# board_ps <board> -- the board's processes by CPU (user, %cpu, core, age, command), whoever's they are.
board_ps() {
    if [ -n "${ISSUE46_PS_CMD:-}" ]; then ${ISSUE46_PS_CMD} "$1"; return; fi
    local cmd='ps -eo user:20,pcpu,psr,etime,comm --sort=-pcpu --no-headers | head -12'
    if [ "${MODE_ARG}" = "--direct" ]; then
        ssh -p "${RISCV_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${RISCV_USER}@$1" "${cmd}"
    else
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            "${RISCV_USER}@${REMOTE_HOST}" "ssh -p ${RISCV_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '${cmd}'"
    fi
}

# snapshot <board> <launch> <before|after> -- reads the board's processes into the series dir and
# notes any other user's heavy job in load_warnings.txt. Never stops the run.
snapshot() {
    local f="${SERIES}/load_$1_L$2_$3.txt" heavy
    board_ps "$1" > "${f}" 2>&1 || echo "(could not read the board's processes)" >> "${f}"
    heavy="$(awk -v me="${RISCV_USER}" "${OTHER_USERS_AWK}" "${f}")"
    if [ -n "${heavy}" ]; then
        echo "$(date -Is) $1 launch $2 ($3): ${heavy}" >> "${SERIES}/load_warnings.txt"
        echo "  RED NOTE: $1 launch $2 ($3): another user is running: ${heavy}"
    fi
}

TOTAL=$(( ${#BOARD_LIST[@]} * LAUNCHES ))
NOW="$(date +%s)"; T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi

print_plan() {
    echo "Issue #46 series: ${LAUNCHES} launches on each of ${BOARDS}, boards taking turns, ${TOTAL} launches of 3 runs"
    echo "  Start     : $(date -d "@${T0}" '+%a %d %b %H:%M')"
    echo "  Settings  : ${CPU_ARGS[*]} --tmin ${TMIN}, reference ${OPT} kernels (variant 2, macc, bracketed)"
    echo "  Steps     : O ours | P reference, driver unchanged (control) | R reference with the per-rep record"
    echo "  Load      : other users' processes are read before and after every launch; a heavy one is flagged red, not a stop"
    local l
    for ((l = 1; l <= LAUNCHES; l++)); do echo "  Launch ${l}   : $(order_of "${l}")"; done
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    echo ""; echo "=== the first launch's runs (${BOARD_LIST[0]}) ==="
    for s in $(order_of 1); do build_cmd "${s}" "${BOARD_LIST[0]}"; echo "  $(step_name "${s}"): ${CMD[*]}"; done
    echo ""; echo "=== the per-rep reference run's own overview ==="
    "${REF_RUN}" "${CONN[@]}" --riscv-ip "${BOARD_LIST[0]}" "${REF_ARGS[@]}" --per-rep --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/issue46_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'board\tlaunch\tposition\tstep\tresult_dir\texit\n' > "${SERIES}/runs.tsv"

if [ "${T0}" -gt "$(date +%s)" ]; then
    echo ""; echo "[$(date +%H:%M)] waiting $(( (T0 - $(date +%s) + 59) / 60 )) min for $(date -d "@${T0}" +%H:%M)"
    sleep $(( T0 - $(date +%s) ))
fi

# newest_since <run_ | refbase_> <stamp> -- the newest logs/<prefix><date>_<time> folder made since the
# stamp. Only stamped names count: refbase_issue33_2026-09-25 and the like sort after them.
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/$1[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "$1$2" ]]; then return 1; fi
    echo "${d}"
}

# complete <step> <dir> -- what a finished run has to hold: reps files for ours and R, result CSVs for P.
complete() {
    case "$1" in
        O) ls "$2"/*/reps_*.txt* >/dev/null 2>&1 ;;
        R) ls "$2"/reps_ref_*.txt.gz >/dev/null 2>&1 && ! grep -q "MISMATCH\|later reps may be missing\|convert reported a problem" "${3}" ;;
        P) ls "$2"/result_*.csv >/dev/null 2>&1 && ! grep -Eq "^[0-9]+;[0-9]+;[0-9]+;[0-9]+;(nan|-nan)" "$2"/result_*.csv ;;
    esac
}

OK=0; FAILED=(); k=0
for ((l = 1; l <= LAUNCHES; l++)); do
  for board in "${BOARD_LIST[@]}"; do
    echo ""; echo "[$(date +%H:%M)] launch ${l}/${LAUNCHES} on ${board}: $(order_of "${l}")"
    snapshot "${board}" "${l}" before
    pos=0
    for s in $(order_of "${l}"); do
        pos=$((pos + 1)); k=$((k + 1))
        log="${SERIES}/${board}_L${l}_${pos}_$(step_name "${s}").log"
        prefix="run_"; [ "${s}" != O ] && prefix="refbase_"
        start="$(date +%Y-%m-%d_%H%M%S)"
        build_cmd "${s}" "${board}"
        echo "[$(date +%H:%M)]   ${pos}/3 $(step_name "${s}"): log ${log}"
        (
            flock -w 10800 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
            echo "$(date -Is) got the lock for ${board}"
            cd "${HERE}/../.." && "${CMD[@]}"
        ) 9>"${LOCK}" >> "${log}" 2>&1
        rc=$?
        dir="$(newest_since "${prefix}" "${start}")"
        if [ "${rc}" -ne 0 ] || [ -z "${dir}" ] || ! complete "${s}" "${dir}" "${log}"; then
            FAILED+=("launch ${l} ${board} $(step_name "${s}") (exit ${rc})")
            echo "    FAILED or incomplete (exit ${rc}); see ${log}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "$(step_name "${s}")" "${dir:--}" "FAIL(${rc})" >> "${SERIES}/runs.tsv"
            continue
        fi
        OK=$((OK + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "$(step_name "${s}")" "$(basename "${dir}")" 0 >> "${SERIES}/runs.tsv"
        echo "    ok: $(basename "${dir}")"
    done
    snapshot "${board}" "${l}" after
  done
done

echo ""; echo "Series finished: ${OK} of $((TOTAL * 3)) runs ok. Run folders: ${SERIES}/runs.tsv"
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"
if [ -s "${SERIES}/load_warnings.txt" ]; then
    echo ""; echo "RED NOTE: another user's job was on a board during this series (${SERIES}/load_warnings.txt):"
    sed 's/^/  /' "${SERIES}/load_warnings.txt"
    echo "  These launches are not clean: repeat them in a quiet window (e.g. --start-at 00:00)."
fi
[ "${#FAILED[@]}" -eq 0 ]
