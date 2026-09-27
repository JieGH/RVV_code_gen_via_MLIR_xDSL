#!/bin/bash
set -e
set -o pipefail

# ============================================================
# run_pack_swap.sh -- vector pack against scalar pack, end to end (issue #26).
#
# Runs the same build and protocol four times on one pinned core, in the order
#
#   vector-live, scalar-live, scalar-live, vector-live
#
# (scalar-live = compile_macro_bench.sh --scalar-pack: every timed call packs with the scalar
# packer). The bracket puts each side at both ends of the session, so a drift in the board's
# thermal state shows up as a gap between the two runs of the same side instead of hiding
# inside the difference. Then compare_pack_swap.py prints the table.
#
# Cases: B3 (MR16_NR15) first, then S1 (MR32_NR7), from config_pack_swap.json; B2, B3 and B4
# from config_pack_swap_b2b3b4.json (--config).
# Defaults: --no-nc-overrides --cpu 4 --tmin 10 (core 4 since issue #39: never core 0).
#
# Usage (jump-host):
#   REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./tests/pure_xdsl_blis_rvv/run_pack_swap.sh --jump \
#       --riscv-user jlei --riscv-ip jbpi --riscv-port 22
#
# Flags: --direct | --jump, --riscv-user/-ip/-port, --cpu N | --no-pin, --tmin <s>,
# --config <file> (default config_pack_swap.json), --shipped-nc (keep the config's nc
# overrides instead of --no-nc-overrides), --dry-run (print each side's plan and stop).
#
# One four-run session takes about 4 x (B3 + S1 at tmin each + build), so 5-10 min per run.
# Do not start it while another benchmark is using the same board or the shared
# /nfs/gap directory (see cron_bench_run.sh).
#
# Run from the project root (RVV_code_gen_via_MLIR_xDSL/).
# ============================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/board_pin.sh"
LOGS="${HERE}/logs"
SHARED=()
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN_ARGS=(--tmin 10)
NC_ARGS=(--no-nc-overrides)
COMPARE_ARGS=()
CONFIG="config_pack_swap.json"
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct|--jump)                        SHARED+=("$1")             ;;
        --riscv-user|--riscv-ip|--riscv-port)   SHARED+=("$1" "$2"); shift ;;
        --cpu)                                  CPU_ARGS=("$1" "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)                               CPU_ARGS=("$1"); NO_PIN=true ;;
        --tmin)                                 TMIN_ARGS=("$1" "$2"); shift ;;
        --config)                               CONFIG="$2"; shift         ;;
        --shipped-nc)                           NC_ARGS=(); COMPARE_ARGS=(--shipped-nc) ;;
        --dry-run)                              DRY_RUN=true               ;;
        -h|--help)                              sed -n '5,34p' "$0"; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Refuse a bad pin now: the first run would only notice it after a build on the board.
pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" || exit 1

BENCH=("${HERE}/compile_macro_bench.sh" --config "${CONFIG}" "${SHARED[@]}" "${CPU_ARGS[@]}" "${TMIN_ARGS[@]}" "${NC_ARGS[@]}")

if [ "${DRY_RUN}" = true ]; then
    echo "=== plan: vector, scalar, scalar, vector (each side's overview shown once) ==="
    echo ""; echo "=== vector-live ==="
    "${BENCH[@]}" --dry-run
    echo ""; echo "=== scalar-live ==="
    "${BENCH[@]}" --scalar-pack --dry-run
    echo ""; echo "  --dry-run: stopped before touching the board."
    exit 0
fi

# newest_since <stamp> -- the newest logs/run_* dir made since the stamp (%Y-%m-%d_%H%M%S names
# sort in time order).
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/run_* 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "run_$1" ]]; then return 1; fi
    echo "${d}"
}

VECTOR=()
SCALAR=()
for side in vector scalar scalar vector; do
    echo ""; echo "=== ${side}-live run $((${#VECTOR[@]} + ${#SCALAR[@]} + 1)) of 4 ==="
    START="$(date +%Y-%m-%d_%H%M%S)"
    if [ "${side}" = scalar ]; then "${BENCH[@]}" --scalar-pack; else "${BENCH[@]}"; fi
    DIR="$(newest_since "${START}")" || { echo "  [FAIL] the ${side}-live run left no new run dir in ${LOGS}"; exit 1; }
    if [ "${side}" = scalar ]; then SCALAR+=("$(basename "${DIR}")"); else VECTOR+=("$(basename "${DIR}")"); fi
done

echo ""; echo "=== comparison ==="
python3 "${HERE}/compare_pack_swap.py" --vector "${VECTOR[@]}" --scalar "${SCALAR[@]}" "${COMPARE_ARGS[@]}" \
    | tee "${LOGS}/pack_swap_${VECTOR[0]#run_}.txt"
