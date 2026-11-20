#!/bin/bash
set -eu

# ============================================================
# run_direct_check.sh — issue #30: run ext/RVV_ukernels_benchmark's own
# make + run_gemm.sh, unwrapped, for B1 and B5 (the only two BERT-large
# shapes the reference ships a real, undamaged kernel for — see
# .scratch/issues/0015-direct-untouched-reference-check.md §6). No staging
# pipeline, no dataset generator, no comparison script — this prints the
# driver's own line, unprocessed.
#
# The only three things allowed to differ from a pristine
# ext/RVV_ukernels_benchmark checkout: the matrix input (B1/B5, hand-written
# below in the reference's own #layer/m/n/k/Flops format), the MR/NR (32x6 —
# a shape the reference already ships a kernel for, so nothing needs
# generating), and this jump-host connection (these boards aren't reachable
# any other way). Everything else — Makefile, run_gemm.sh, test_gemm.c,
# gemm_blis.c, the kernel source itself — runs untouched from a copy of
# ext/RVV_ukernels_benchmark. The BLIS header stand-in (blis_shim/blis.h,
# reused as-is) is the one thing required just to compile at all under
# RUN_MODE=FAMILY_EXO; see its own header comment for why that's safe.
#
# Connection: the "fej" jump host is a plain ~/.ssh/config alias with its
# own working key — no REMOTE_PASS/sshpass machinery needed here. fej's home
# (/nfs/gap/<user>) is the same NFS mount the boards see, so one rsync to
# fej is enough; no second-hop rsync onto the board itself.
#
# Usage:
#   ./tests/pure_xdsl_blis_rvv/ref_baseline/run_direct_check.sh [--dry-run] [--cpu N]
#
# Run from the project root.
# ============================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
REF_SRC="$(cd "${REPO_ROOT}/../ext/RVV_ukernels_benchmark" && pwd)"

source "${HERE}/../board_pin.sh"

RISCV_USER="jlei"
BOARDS="jbpi jbpi2"
PIN_CPU="${BENCH_CPU_DEFAULT}"
CPU_FLAG=false
ALLOW_CORE0=false
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --cpu)     PIN_CPU="$2"; CPU_FLAG=true; shift ;;
        --allow-core-0) ALLOW_CORE0=true ;;
        -h|--help) sed -n '4,32p' "$0"; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Issue #21's rationale, reused here: refuse a bad core now, not after staging both boards.
pin_check "${CPU_FLAG}" false "${PIN_CPU}" "${ALLOW_CORE0}" || exit 1

STAMP="$(date +%Y-%m-%d_%H%M%S)"
LOG="${REPO_ROOT}/tests/pure_xdsl_blis_rvv/logs/issue30_direct_check_${STAMP}.txt"
mkdir -p "$(dirname "${LOG}")"

export REMOTE_HOST=fej REMOTE_PORT=3322 REMOTE_PASS=""; MODE_ARG=--jump; RISCV_PORT=22

# Committed dataset, not regenerated here: tests/pure_xdsl_blis_rvv/datasets/*.dat are all
# static, checked-in files in this project, and a run silently overwriting a hand-edited
# one would be a surprise. If it's missing, that's a setup problem to fix, not paper over.
DATASET="${HERE}/bert_b1_b5.dat"
if [ ! -f "${DATASET}" ]; then
    echo "  [FAIL] missing ${DATASET}"; exit 1
fi

{
echo "Issue #30 — reference, direct and unwrapped. B1 (layer 1, 1024x384x1024) + B5 (layer 5,"
echo "1024x384x4096), both MR32 NR6 OPT=LDX GATHER=2(macc) SWAP=0, RUN_MODE=FAMILY_EXO."
echo ""
echo "Dataset (reference's own #layer m n k Flops format, Flops column unused by the parser):"
cat "${DATASET}"
echo ""
} | tee "${LOG}"

if [ "${DRY_RUN}" = true ]; then
    echo "--dry-run: stopping before touching a board." | tee -a "${LOG}"
    exit 0
fi

for BOARD in ${BOARDS}; do
    DEST="/nfs/gap/${RISCV_USER}/issue30_ref_direct_check_${BOARD}"

    {
    echo "============================================================"
    echo " Board: ${BOARD}  (staged at ${DEST})"
    echo "============================================================"
    } | tee -a "${LOG}"

    ssh -o StrictHostKeyChecking=no fej "rm -rf ${DEST} && mkdir -p ${DEST}/kernels/RVV_256_LDX/fp32"
    board_rsync "${BOARD}" "${REF_SRC}/gemm_blis_family/" "@T@:${DEST}/gemm_blis_family/" \
        --exclude 'build/' --exclude 'out_models/' --exclude 'output/'
    board_rsync "${BOARD}" "${REF_SRC}/kernels/RVV_256_LDX/fp32/32x6/" "@T@:${DEST}/kernels/RVV_256_LDX/fp32/32x6/"
    board_rsync "${BOARD}" "${HERE}/blis_shim/" "@T@:${DEST}/blis_shim/"
    board_rsync "${BOARD}" "${DATASET}" "@T@:${DEST}/gemm_blis_family/bert_b1_b5.dat"

    echo "" | tee -a "${LOG}"
    echo "--- build: make MR=32 NR=6 OPT=LDX GATHER=2 SWAP=0 SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO INCLUDE=-I../blis_shim LIBS=-lm ---" | tee -a "${LOG}"
    board_ssh "${BOARD}" \
        "cd ${DEST}/gemm_blis_family && make clean >/dev/null 2>&1; make MR=32 NR=6 OPT=LDX GATHER=2 SWAP=0 SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO INCLUDE=-I../blis_shim LIBS=-lm" \
        2>&1 | tee -a "${LOG}"

    PIN_PREFIX="$(pin_prefix "${PIN_CPU}")"
    echo "" | tee -a "${LOG}"
    echo "--- run: ${PIN_PREFIX}./run_gemm.sh bert_b1_b5.dat result.csv ---" | tee -a "${LOG}"
    board_ssh "${BOARD}" \
        "cd ${DEST}/gemm_blis_family && ${PIN_PREFIX}./run_gemm.sh bert_b1_b5.dat result.csv && cat result.csv" \
        2>&1 | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"
done

echo "Done. Full transcript: ${LOG}"
