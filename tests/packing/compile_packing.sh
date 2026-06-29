#!/bin/bash
set -e

# ============================================================
# compile_packing.sh — Standalone build & verify script for
#                      vectorized FP32 packing (pack_CB_v + pack_RB_v).
#
# Generates both branch kernels for CB and RB, then regenerates
# pack_dispatch.h to keep function names in sync with block sizes.
#
# Usage (jump-host / same pattern as compile.sh):
#   REMOTE_PASS="<pw>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./tests/packing/compile_packing.sh \
#       --jump \
#       --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22
#
# Usage (direct SSH to board):
#   ./tests/packing/compile_packing.sh \
#       --direct \
#       --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22
#
# All paths are relative to the project root (RVV_code_gen_via_MLIR_xDSL/).
# Run this script from the project root, not from tests/packing/.
# ============================================================


# ============================================================
# Defaults
# ============================================================
USE_DIRECT=true

RISCV_USER="jlei"
RISCV_REMOTE_IP="jbpi2"
RISCV_SSH_PORT="22"
RISCV_WORKSPACE="xdsl_rvv_microkernel"

MLIR_BUILD_DIR="../llvm-project"

# Jump-host vars (read from environment, same as compile.sh):
# REMOTE_PASS, REMOTE_HOST, REMOTE_PORT

SEP="------------------------------------------------------------"

# ============================================================
# Argument parsing
# ============================================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct)          USE_DIRECT=true                       ;;
        --jump)            USE_DIRECT=false                      ;;
        --riscv-user)      RISCV_USER="$2";          shift      ;;
        --riscv-ip)        RISCV_REMOTE_IP="$2";     shift      ;;
        --riscv-port)      RISCV_SSH_PORT="$2";      shift      ;;
        --riscv-workspace) RISCV_WORKSPACE="$2";     shift      ;;
        --mlir-build-dir)  MLIR_BUILD_DIR="$2";      shift      ;;
        --mlir-translate)  MLIR_TRANSLATE_PATH="$2"; shift      ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Default mlir-translate path
if [ -z "${MLIR_TRANSLATE_PATH+x}" ]; then
    MLIR_TRANSLATE_PATH="${MLIR_BUILD_DIR}/build/bin/mlir-translate"
fi

# ============================================================
# SSH / rsync helpers  (mirrors compile.sh exactly)
# ============================================================
if [ "$USE_DIRECT" = true ]; then
    echo ""; echo "  Mode: Direct"; echo ""
    ssh_board()        { ssh -p "${RISCV_SSH_PORT}" -o StrictHostKeyChecking=no \
                             "${RISCV_USER}@${RISCV_REMOTE_IP}" "$@"; }
    ssh_board_no_tty() { ssh_board "$@"; }
    rsync_board()      { rsync -av -e "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no" "$@"; }
    RSYNC_TARGET="${RISCV_USER}@${RISCV_REMOTE_IP}"
else
    echo ""; echo "  Mode: Jump-host (x86 → ${REMOTE_HOST} → ${RISCV_REMOTE_IP})"; echo ""
    ssh_board() {
        local escaped_args
        escaped_args=$(printf '%q ' "$@")
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no \
            -t "${RISCV_USER}@${REMOTE_HOST}" \
            "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no -t ${RISCV_REMOTE_IP} ${escaped_args}"
    }
    ssh_board_no_tty() {
        local escaped_args
        escaped_args=$(printf '%q ' "$@")
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no \
            "${RISCV_USER}@${REMOTE_HOST}" \
            "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no ${RISCV_REMOTE_IP} ${escaped_args}"
    }
    rsync_board() {
        sshpass -p "${REMOTE_PASS}" rsync -av \
            -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "$@"
    }
    RSYNC_TARGET="${RISCV_USER}@${REMOTE_HOST}"
fi

# ============================================================
# Dependency check
# ============================================================
echo ""; echo "$SEP"; echo " Dependency Check"; echo "$SEP"

for cmd in python3 rsync ssh; do
    if command -v "$cmd" > /dev/null 2>&1; then
        echo "  [OK] $cmd: $(command -v "$cmd")"
    else
        echo "  [FAIL] $cmd not found in PATH"
        exit 1
    fi
done

if [ "$USE_DIRECT" = false ]; then
    if command -v sshpass > /dev/null 2>&1; then
        echo "  [OK] sshpass: $(command -v sshpass)"
    else
        echo "  [FAIL] sshpass not found (required for jump-host mode)"
        exit 1
    fi
fi

if [ -f "$MLIR_TRANSLATE_PATH" ]; then
    echo "  [OK] mlir-translate: $(realpath "$MLIR_TRANSLATE_PATH")"
else
    echo "  [FAIL] mlir-translate not found at: $MLIR_TRANSLATE_PATH"
    echo "         Pass --mlir-build-dir or --mlir-translate to fix this."
    exit 1
fi

if ! source .venv/bin/activate 2>/dev/null; then
    echo "  [FAIL] Could not activate .venv — run from project root"
    exit 1
fi
echo "  [OK] .venv activated"

# ============================================================
# [1/4] Generate C code locally
# ============================================================
echo ""; echo "$SEP"; echo " [1/4] Generate C Code Locally"; echo "$SEP"

# Configurable block sizes & dtype
NR=8          # CB block width  (B-matrix, cols)
MR=4          # RB block height (A-matrix, rows)
DTYPE=fp32    # dtype suffix for generated functions

PYTHONPATH=src:tests/packing python tests/packing/generate_packing.py \
    --mlir-translate "$MLIR_TRANSLATE_PATH" \
    --output tests/packing/gen/pack_v_fp32.c \
    --NR ${NR} --MR ${MR} --elem-bytes 4 --suffix ${DTYPE}
echo "  [OK] Generated: tests/packing/gen/pack_v_fp32.c  (NR=${NR}, MR=${MR}, dtype=${DTYPE})"

# Cleanup obsolete files
rm -f tests/packing/test_pack_CB_v_fp32.py tests/packing/test_pack_RB_v_fp32.py
rm -f tests/packing/gen/pack_CB_v_fp32.c* tests/packing/gen/pack_RB_v_fp32.c*

# Re-generate pack_dispatch.h to match compiled function names
PYTHONPATH=src:tests/packing python tests/packing/gen_dispatch.py \
    --NR ${NR} --MR ${MR} --suffix ${DTYPE} \
    --output tests/packing/pack_dispatch.h
echo "  [OK] Generated: tests/packing/pack_dispatch.h"

# ============================================================
# [2/4] rsync tests/packing/ to board
# ============================================================
echo ""; echo "$SEP"; echo " [2/4] Sync to Board"; echo "$SEP"

REMOTE_PACKING_DIR="${RISCV_WORKSPACE}/tests/packing"
ssh_board "mkdir -p ${REMOTE_PACKING_DIR}/gen"
rsync_board tests/packing/ "${RSYNC_TARGET}:${REMOTE_PACKING_DIR}/"
echo "  [OK] Synced tests/packing/ → ${RSYNC_TARGET}:${REMOTE_PACKING_DIR}/"

# ============================================================
# [3/4] Compile on board
# ============================================================
echo ""; echo "$SEP"; echo " [3/4] Compile on Board"; echo "$SEP"

ssh_board "cd ${REMOTE_PACKING_DIR} && make clean && make"
echo "  [OK] Compiled verify_packing on board"

# ============================================================
# [4/4] Run verification on board
# ============================================================
echo ""; echo "$SEP"; echo " [4/4] Run Verification"; echo "$SEP"

ssh_board "cd ${REMOTE_PACKING_DIR} && ./verify_packing"
echo ""
echo "$SEP"
echo " Done."
echo "$SEP"
