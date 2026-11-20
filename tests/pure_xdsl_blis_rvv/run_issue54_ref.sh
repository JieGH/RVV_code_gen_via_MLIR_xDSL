#!/bin/bash
# ============================================================
# run_issue54_ref.sh -- issue #54: the reference design on B3, rep by rep, as it is and with our
# one-column B packer (run_ref_baseline.sh --pack-b-fix).
#
# 1. A correctness run: the reference with the fix, --check (the driver compares C with its plain gemm
#    after the timed loop), --tmin 0.2. The series stops unless every row says PASS.
# 2. --launches (default 3) launches of two arms on one board, the order alternating (A B, B A, A B):
#      ref     the reference as it is (its own scalar pack_CB)
#      ref-fix the reference with pack_CB reading one column at a time, 32 numbers per load
#    Each run: run_ref_baseline.sh --opt LDX --variants 2 --bracket --per-rep --cases
#    ref_baseline/cases_issue54_b3.tsv --no-compare --tmin <tmin> --cpu <cpu> (the #46 setting, B3 only:
#    the reference builds 16x14 and 16x16 around our 16x15).
# Before every run the board must be idle: two `top` readings --idle-gap s apart, overall idle at least
# --idle-min %, and no process of another user (not root, not us) at --idle-proc-max % of a core or more.
# If not, it waits --idle-poll s and tries again, up to --idle-max-wait s, then stops (exit 2). A reading
# is also taken after every run (recorded, never a stop). Never kills or waits out by force anyone's job.
#
# Output: logs/issue54_ref_<stamp>/ with runs.tsv (launch, arm, result dir, status), one console log per
# run, idle_readings.txt, idle_waits.txt, load_warnings.txt. The per-rep files are in each refbase_* dir
# (reps_ref_16x14_macc.txt.gz, reps_ref_16x16_macc.txt.gz).
#
# Usage (from the repo root; jump host by default, as the other series):
#   REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./tests/pure_xdsl_blis_rvv/run_issue54_ref.sh
# Flags: --board B (default jbpi2), --launches N, --tmin S (default 10), --cpu N (default 4),
#        --idle-gap/--idle-poll/--idle-max-wait/--idle-min/--idle-proc-max, --direct, --dry-run
# Do not run run_ref_baseline.sh, a --dry-run of it, or the test suite while this goes: they rewrite
# ref_baseline/generated/, which every run stages from.
# ============================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/board_pin.sh"   # board access (issue #116)
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
BOARD="jbpi2"; LAUNCHES=3; TMIN=10; CPU=4
IDLE_GAP=60; IDLE_POLL=300; IDLE_MAX_WAIT=10800; IDLE_MIN=85; IDLE_PROC_MAX=20
MODE_ARG="--jump"; DRY_RUN=false
RISCV_USER="jlei"; RISCV_PORT=22
REMOTE_HOST="${REMOTE_HOST:-fej}"; REMOTE_PORT="${REMOTE_PORT:-3322}"; REMOTE_PASS="${REMOTE_PASS:-}"
export REMOTE_HOST REMOTE_PORT REMOTE_PASS

die() { echo "ERROR: $*" >&2; exit 1; }
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --board)          BOARD="$2"; shift ;;
        --launches)       LAUNCHES="$2"; shift ;;
        --tmin)           TMIN="$2"; shift ;;
        --cpu)            CPU="$2"; shift ;;
        --idle-gap)       IDLE_GAP="$2"; shift ;;
        --idle-poll)      IDLE_POLL="$2"; shift ;;
        --idle-max-wait)  IDLE_MAX_WAIT="$2"; shift ;;
        --idle-min)       IDLE_MIN="$2"; shift ;;
        --idle-proc-max)  IDLE_PROC_MAX="$2"; shift ;;
        --direct)         MODE_ARG="--direct" ;;
        --dry-run)        DRY_RUN=true ;;
        -h|--help)        sed -n '3,31p' "$0"; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done
[ "${CPU}" != "0" ] || die "core 0 takes every device interrupt; use another core (house rule 1)"

STAMP="$(date +%Y-%m-%d_%H%M%S)"
SERIES="${HERE}/logs/issue54_ref_${STAMP}"
CASES="${HERE}/ref_baseline/cases_issue54_b3.tsv"
BASE=(--riscv-ip "${BOARD}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}" --opt LDX --variants 2 --bracket
      --cases "${CASES}" --no-compare --cpu "${CPU}")

# The order of the two arms in each launch: A B, B A, A B, ...
order_of() { if [ $(( $1 % 2 )) -eq 1 ]; then echo "ref ref-fix"; else echo "ref-fix ref"; fi; }

if [ "${DRY_RUN}" = true ]; then
    echo "Series folder : ${SERIES} (not created)"
    echo "Board ${BOARD}, core ${CPU}, tmin ${TMIN}, ${LAUNCHES} launches; idle gate ${IDLE_MIN}% idle, other users < ${IDLE_PROC_MAX}%"
    echo "0 check    : run_ref_baseline.sh ${MODE_ARG} ${BASE[*]} --pack-b-fix --check --tmin 0.2"
    for ((L = 1; L <= LAUNCHES; L++)); do
        for arm in $(order_of "${L}"); do
            extra=""; [ "${arm}" = "ref-fix" ] && extra=" --pack-b-fix"
            echo "${L} ${arm}$( [ "${arm}" = ref ] && echo '    ' || echo ' ') : run_ref_baseline.sh ${MODE_ARG} ${BASE[*]} --per-rep${extra} --tmin ${TMIN}"
        done
    done
    exit 0
fi
mkdir -p "${SERIES}"
printf 'board\tlaunch\tposition\tarm\tresult_dir\texit\tstatus\n' > "${SERIES}/runs.tsv"

# run_one <launch> <position> <arm> <extra args...>
run_one() {
    local L="$1" P="$2" arm="$3"; shift 3
    local tag="L${L}_${P}_${arm}" log rc dir status
    board_idle_gate "${BOARD}" "${tag}" || { echo "  board busy for too long before ${tag}: stopping"; exit 2; }
    log="${SERIES}/${BOARD}_${tag}.log"
    echo "$(date -Is) ${tag}: start"
    ( cd "${REPO_ROOT}" && ./tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh "${MODE_ARG}" "${BASE[@]}" "$@" ) \
        > "${log}" 2>&1
    rc=$?
    board_one_reading "${BOARD}" "${tag}" post || true
    dir="$(grep -o 'refbase_[0-9_-]*' "${log}" | tail -1)"
    status=ok
    [ "${rc}" -ne 0 ] && status=failed
    grep -q '\[WARN\]' "${log}" && status=failed
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${BOARD}" "${L}" "${P}" "${arm}" "${dir:-none}" "${rc}" "${status}" >> "${SERIES}/runs.tsv"
    echo "$(date -Is) ${tag}: ${status} (${dir:-no result dir})"
    LAST_LOG="${log}"
}

echo "Series ${SERIES}: board ${BOARD}, core ${CPU}, tmin ${TMIN}, ${LAUNCHES} launches"

# 1. correctness of the fixed reference
run_one 0 1 ref-fix-check --pack-b-fix --check --tmin 0.2
# The driver prints OK or ERROR in the last column of each result row (the header says "PASS", so do not count that word).
npass=$(sed 's/\x1b\[[0-9;]*m//g' "${LAST_LOG}" | grep -c -E '\|[[:space:]]+OK[[:space:]]+\|' || true)
nerr=$(sed 's/\x1b\[[0-9;]*m//g' "${LAST_LOG}" | grep -c -E '\|[[:space:]]+ERROR[[:space:]]+\|' || true)
echo "check: OK rows ${npass}, ERROR rows ${nerr}" | tee "${SERIES}/check.txt"
if [ "${nerr}" -ne 0 ] || [ "${npass}" -lt 2 ]; then
    echo "  the fixed reference did not pass its result check: stopping before the timed runs"; exit 3
fi

# 2. the timed launches
for ((L = 1; L <= LAUNCHES; L++)); do
    P=0
    for arm in $(order_of "${L}"); do
        P=$((P + 1))
        if [ "${arm}" = "ref-fix" ]; then run_one "${L}" "${P}" "${arm}" --per-rep --pack-b-fix --tmin "${TMIN}"
        else run_one "${L}" "${P}" "${arm}" --per-rep --tmin "${TMIN}"; fi
    done
done
echo "$(date -Is) series done: ${SERIES}"
