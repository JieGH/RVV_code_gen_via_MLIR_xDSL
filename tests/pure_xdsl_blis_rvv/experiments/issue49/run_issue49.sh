#!/bin/bash

# ============================================================
# run_issue49.sh -- the decisive test: does padding ldb alone move the B3_K256 flip? (issue #49)
#
# One LAUNCH is one run per PAD on one board, back to back on core 4: config_issue49_sweep.json
# (16x6 and 16x8, the two tiles that flip in opposite directions between K=384 and K=256, each on
# the four shapes of datasets/issue46_ref_b3.dat: B3, B3_K256, B2, R24), uninstrumented, reference
# protocol, vector packer, --no-nc-overrides (B3 at nc=384), one of four pads of ldb
# (--pad-ldb <elements>, k unchanged, data unchanged, the correctness check unchanged):
#   pad0    ldb = k                (today's behaviour)
#   pad8    ldb = k + 8            ("8 floats" of the issue)
#   pad16   ldb = k + 16           ("16 floats" of the issue)
#   pad128  ldb = k + 128          (K=384: ldb=512, a power of two; K=256: ldb=384, K=384's own step)
#
# The pads rotate from launch to launch so no pad is always first or last (house rule 2): with
# pads a b c d, launch 1 runs a b c d, launch 2 b c d a, launch 3 c d a b.
# The boards take turns launch by launch, one run at a time, under the lock the other sweeps use
# (they share the host's pj_gen/ and one /nfs directory).
#
# Around every run the board's processes are read (`ps`, other users only). Another user's job is
# NOT a reason to skip a run (2026-09-29 rule): the run goes ahead, the load is written to
# load_warnings.txt, and reps_report.py prints a RED NOTE for every run folder whose provenance.txt
# records it. Numbers that decide something come from a quiet window (midnight): use --start-at 00:00.
#
# Board time (one launch, one board): about 10-15 min (2 tiles x 4 shapes, uninstrumented, tmin 10).
#
# Logs: logs/issue49_<stamp>/ : plan.txt, runs.tsv (board, launch, position, pad, run folder, exit),
# one log per run, load_<board>_L<n>_<pad>_{before,after}.txt, load_warnings.txt. The run folders
# are the usual logs/run_*.
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/experiments/issue49/run_issue49.sh --start-at 00:00 \
#       > tests/pure_xdsl_blis_rvv/logs/issue49.out 2>&1 < /dev/null &
#
# Flags:
#   --pads a,b           ldb pads in elements, in this order on launch 1 (default 0,8,16,128)
#   --boards a,b          boards that take turns (default jbpi,jbpi2)
#   --launches N          launches per board (default 3; house rule 2 needs at least 3)
#   --start-at HH:MM      wait until this time (today, or tomorrow if it has passed); default now
#   --tmin S | --cpu N    as for the other scripts (defaults 10 and core 4; core 0 needs --allow-core-0)
#   --workspace NAME      the workspace directory on the board (compile_macro_bench.sh --riscv-workspace). The boards
#                         share one NFS home, so two series running at once, one per board, each need their own
#                         (and their own checkout of the repo on this host: every run rewrites pj_gen/)
#   --direct              reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --dry-run             print the plan and the first launch's settings, then stop
#
# Jump-host settings come from the environment: REMOTE_PASS (default empty), REMOTE_HOST (default fej),
# REMOTE_PORT (default 3322). Do not start other benchmarks, `--dry-run` of a wrapper or the test suite on the host
# meanwhile: they rewrite pj_gen/ under a running series.
# ============================================================

# No `set -e`: a failed run must not end the series.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS="${ISSUE49_LOGS:-${HERE}/logs}"                                 # ISSUE49_* let the loop be tried against stubs
BENCH_RUN="${ISSUE49_RUN:-${HERE}/compile_macro_bench.sh}"
LOCK="${LOGS}/.cron_bench.lock"
SWEEP_CONFIG="experiments/issue49/config_issue49_sweep.json"
source "${HERE}/board_pin.sh"

PADS="0,8,16,128"
BOARDS="jbpi,jbpi2"
LAUNCHES=3
START_AT=""
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN=10
WORKSPACE=""
MODE_ARG="--jump"
RISCV_USER="jlei"
RISCV_PORT="22"
DRY_RUN=false

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pads)          PADS="$2"; shift ;;
        --boards)        BOARDS="$2"; shift ;;
        --launches)      LAUNCHES="$2"; shift ;;
        --start-at)      START_AT="$2"; shift ;;
        --tmin)          TMIN="$2"; shift ;;
        --cpu)           CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)        CPU_ARGS=(--no-pin); NO_PIN=true ;;
        --allow-core-0)  ALLOW_CORE0=true ;;
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
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ "${#BOARD_LIST[@]}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"
IFS=',' read -ra PAD_LIST <<< "${PADS}"
[ "${#PAD_LIST[@]}" -ge 1 ] && [ -n "${PAD_LIST[0]}" ] || die "--pads takes a comma-separated list"
for p in "${PAD_LIST[@]}"; do
    [[ "${p}" =~ ^[0-9]+$ ]] || die "--pads takes non-negative integers (got '${p}')"
done

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

SWEEP_ROWS="$(grep -c '^[[:space:]]*\["fp32"' "${HERE}/${SWEEP_CONFIG}")"

# pad_tag <pad> -- the step's name in logs and runs.tsv
pad_tag() { echo "pad$1"; }

# order_of <launch> -- the pads of a launch, the list rotated left by launch-1
order_of() {
    local l="$1" n="${#PAD_LIST[@]}" i out=()
    for ((i = 0; i < n; i++)); do out+=("${PAD_LIST[$(( (i + l - 1) % n ))]}"); done
    echo "${out[*]}"
}

CONN=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}")

# build_cmd <pad> <board> -- sets CMD to the command line of one run (an array, so nothing is word-split)
build_cmd() {
    local pad="$1" board="$2"
    CMD=("${BENCH_RUN}" "${CONN[@]}" --riscv-ip "${board}" --config "${SWEEP_CONFIG}" --no-nc-overrides "${CPU_ARGS[@]}" --tmin "${TMIN}")
    [ "${pad}" != 0 ] && CMD+=(--pad-ldb "${pad}")
    [ -n "${WORKSPACE}" ] && CMD+=(--riscv-workspace "${WORKSPACE}")
    return 0
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

TOTAL=$(( ${#BOARD_LIST[@]} * LAUNCHES * ${#PAD_LIST[@]} ))
NOW="$(date +%s)"; T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi

print_plan() {
    echo "Issue #49 series: ${LAUNCHES} launches on each of ${BOARDS}, boards taking turns, pads ${PADS} (${TOTAL} runs)"
    echo "  Start     : $(date -d "@${T0}" '+%a %d %b %H:%M')"
    echo "  Settings  : ${CPU_ARGS[*]} --no-nc-overrides --tmin ${TMIN}, vector packer"
    echo "  Config    : ${SWEEP_CONFIG} (${SWEEP_ROWS} tiles x 4 shapes: B3, B3_K256, B2, R24)"
    echo "  Load      : other users' processes are read before and after every run; a heavy one is flagged red, not a stop"
    local l
    for ((l = 1; l <= LAUNCHES; l++)); do echo "  Launch ${l}   : $(order_of "${l}")"; done
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    echo ""; echo "=== the first launch's runs (${BOARD_LIST[0]}) ==="
    for p in $(order_of 1); do build_cmd "${p}" "${BOARD_LIST[0]}"; echo "  $(pad_tag "${p}"): ${CMD[*]}"; done
    echo ""; echo "=== the first run's own overview ==="
    build_cmd "$(order_of 1 | cut -d' ' -f1)" "${BOARD_LIST[0]}"
    "${CMD[@]}" --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/issue49_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'board\tlaunch\tposition\tpad\tresult_dir\texit\n' > "${SERIES}/runs.tsv"

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
    echo ""; echo "[$(date +%H:%M)] launch ${l}/${LAUNCHES} on ${board}: $(order_of "${l}")"
    pos=0
    for p in $(order_of "${l}"); do
        pos=$((pos + 1))
        tag="L${l}_${pos}_$(pad_tag "${p}")"
        log="${SERIES}/${board}_${tag}.log"
        start="$(date +%Y-%m-%d_%H%M%S)"
        build_cmd "${p}" "${board}"
        echo "[$(date +%H:%M)]   ${pos}/${#PAD_LIST[@]} $(pad_tag "${p}"): log ${log}"
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
            FAILED+=("launch ${l} ${board} $(pad_tag "${p}") (exit ${rc})")
            echo "    FAILED or incomplete (exit ${rc}); see ${log}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "$(pad_tag "${p}")" "${dir:+$(basename "${dir}")}" "FAIL(${rc})" | sed 's/\t\t/\t-\t/' >> "${SERIES}/runs.tsv"
            continue
        fi
        OK=$((OK + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "$(pad_tag "${p}")" "$(basename "${dir}")" 0 >> "${SERIES}/runs.tsv"
        echo "    ok: $(basename "${dir}")"
    done
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
