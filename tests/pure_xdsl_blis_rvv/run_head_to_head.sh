#!/bin/bash
set -e

# ============================================================
# run_head_to_head.sh -- our side and the reference design, back to back (issue #21).
#
# A head-to-head percentage is only as good as the conditions the two numbers were taken
# under. The board is shared and its thermal state drifts, so a pair taken a day apart
# carries an unknown-sized variable. This runs both sides in one go, on the same core,
# and compares them:
#
#   1. compile_macro_bench.sh   --no-nc-overrides   (uninstrumented, reference protocol)
#   2. ref_baseline/run_ref_baseline.sh --bracket --no-compare
#   3. compare_ref_baseline.py, on exactly the run and the baseline just made
#
# Ours goes first only because the comparison wants its run dir; one pair cannot cancel
# a drift between the sides, only bound it (the comparison prints the gap).
#
# Usage (jump-host; mirrors the reference command in docs/agents/execution.md):
#   REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./tests/pure_xdsl_blis_rvv/run_head_to_head.sh --jump \
#       --riscv-user jlei --riscv-ip jbpi --riscv-port 22
#
# Flags: --direct | --jump, --riscv-user/-ip/-port, --cpu N | --no-pin, --tmin <s>,
# --opt OPT|LDX (reference side only), --dry-run (each side prints its plan and stops).
# Anything else about either side: run that script on its own.
#
# Run from the project root (RVV_code_gen_via_MLIR_xDSL/).
# ============================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/board_pin.sh"

LOGS="${HERE}/logs"
SHARED=()      # goes to both scripts
REF_ONLY=()
DRY_RUN=false
CPU_FLAG=false
NO_PIN=false
CPU=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct|--jump)                        SHARED+=("$1")             ;;
        --riscv-user|--riscv-ip|--riscv-port)   SHARED+=("$1" "$2"); shift ;;
        --tmin)                                 SHARED+=("$1" "$2"); shift ;;
        --cpu)                                  SHARED+=("$1" "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)                               SHARED+=("$1"); NO_PIN=true ;;
        --opt)                                  REF_ONLY+=("$1" "$2"); shift ;;
        --dry-run)                              DRY_RUN=true; SHARED+=("$1") ;;
        -h|--help)                              sed -n '3,29p' "$0"; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Refuse a bad pin now: the reference side would only notice after ours had run for an hour.
pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" || exit 1

# newest_since <prefix> -- the newest logs/<prefix>_<stamp> dir, if made since START.
# Stamps are %Y-%m-%d_%H%M%S, so dir names sort in time order.
newest_since() {
    local d
    d="$(ls -d "${LOGS}/$1"_* 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "$1_${START}" ]]; then return 1; fi
    echo "${d}"
}

START="$(date +%Y-%m-%d_%H%M%S)"

echo ""; echo "=== [1/3] our side ==="
"${HERE}/compile_macro_bench.sh" --no-nc-overrides "${SHARED[@]}"

if [ "${DRY_RUN}" = true ]; then
    echo ""; echo "=== [2/3] the reference side ==="
    "${HERE}/ref_baseline/run_ref_baseline.sh" --bracket --no-compare "${SHARED[@]}" "${REF_ONLY[@]}"
    echo ""; echo "  --dry-run: stopped before the comparison."
    exit 0
fi

OURS="$(newest_since run)" || { echo "  [FAIL] our side left no new run dir in ${LOGS}"; exit 1; }

echo ""; echo "=== [2/3] the reference side ==="
"${HERE}/ref_baseline/run_ref_baseline.sh" --bracket --no-compare "${SHARED[@]}" "${REF_ONLY[@]}"

REF="$(newest_since refbase)" || { echo "  [FAIL] the reference side left no new baseline dir in ${LOGS}"; exit 1; }
RAW_FILE="$(ls "${OURS}"/all_raw_*_noinstr_refproto_no_nc_override.txt 2>/dev/null | head -1)"
[ -n "${RAW_FILE}" ] || { echo "  [FAIL] ${OURS} has no all_raw_*_noinstr_refproto_no_nc_override.txt to compare"; exit 1; }
RAW="$(basename "${RAW_FILE}")"

echo ""; echo "=== [3/3] comparison ==="
python3 "${HERE}/ref_baseline/compare_ref_baseline.py" \
    --result-dir "${REF}" --runs "$(basename "${OURS}")" --raw-name "${RAW}" \
    | tee "${REF}/comparison.txt"
