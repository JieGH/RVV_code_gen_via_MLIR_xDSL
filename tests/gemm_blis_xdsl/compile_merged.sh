#!/bin/bash
set -e

# ============================================================
# compile_merged.sh — Standalone build & verify script for
#                     xDSL Macro-Kernel loop structure.
# ============================================================

# ============================================================
# Datasets to evaluate
# Edit this list to enable/disable models for benchmarking.
# ============================================================
DATASETS=(
    # "cnn_models/trail.dat"
    # "cnn_models/tiny.dat"
    # "cnn_models/gnn.dat"
    "cnn_models/bert_large.dat"
    # "cnn_models/special.dat"
    # "cnn_models/gpt2_large.dat"
    # "cnn_models/square.dat"
    "cnn_models/square_small.dat"
)
# ============================================================

USE_DIRECT=true
RISCV_USER="jlei"
RISCV_REMOTE_IP="jbpi2"
RISCV_SSH_PORT="22"
RISCV_WORKSPACE="xdsl_rvv_microkernel"
MLIR_BUILD_DIR="../llvm-project"

SEP="------------------------------------------------------------"

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
        --model)           MODEL_NAME="$2";          shift      ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Export variables for sub-scripts (like check_ram.sh)
export USE_DIRECT RISCV_USER RISCV_REMOTE_IP RISCV_SSH_PORT REMOTE_PASS REMOTE_PORT REMOTE_HOST

# Override DATASETS if --model is provided
if [ -n "$MODEL_NAME" ]; then
    DATASETS=("cnn_models/${MODEL_NAME}.dat")
fi

# Run RAM check / env setup as in compile.sh to populate jump-host vars
if [ -f "./check_ram.sh" ]; then
    bash ./check_ram.sh
fi

if [ -z "${MLIR_TRANSLATE_PATH+x}" ]; then
    MLIR_TRANSLATE_PATH="${MLIR_BUILD_DIR}/build/bin/mlir-translate"
fi

if [ "$USE_DIRECT" = true ]; then
    echo ""; echo "  Mode: Direct"; echo ""
    ssh_board()        { ssh -p "${RISCV_SSH_PORT}" -o StrictHostKeyChecking=no "${RISCV_USER}@${RISCV_REMOTE_IP}" "$@"; }
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
    rsync_board() {
        sshpass -p "${REMOTE_PASS}" rsync -av -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "$@"
    }
    RSYNC_TARGET="${RISCV_USER}@${REMOTE_HOST}"
fi

if ! source .venv/bin/activate 2>/dev/null; then
    echo "  [FAIL] Could not activate .venv — run from project root"
    exit 1
fi
echo "  [OK] .venv activated"

# ============================================================
# Start Batch Evaluation
# ============================================================

for DATASET_REL_PATH in "${DATASETS[@]}"; do
    MODEL_PATH="tests/gemm_blis_xdsl/${DATASET_REL_PATH}"
    MODEL_NAME=$(basename "$DATASET_REL_PATH" .dat)

    echo ""; echo "############################################################"
    echo "## EVALUATING MODEL: ${MODEL_NAME}"
    echo "############################################################"

    # [1/4] Local Preparation
    # ============================================================
    echo ""; echo "$SEP"; echo " [1/4] Local Preparation"; echo "$SEP"

# Generate C macro locally
echo ">> Generating xDSL packing and loop nest for model: ${MODEL_PATH}"
bash tests/gemm_blis_xdsl/compile_macro.sh "${MODEL_PATH}"

# Assemble kernels in tests/gemm_blis_xdsl/kernels/
echo ">> Assembling xDSL kernels into tests/gemm_blis_xdsl/kernels/..."
mkdir -p tests/gemm_blis_xdsl/kernels
SRC_DIR="tests/api_gen_cpp/merged"
DEST_DIR="tests/gemm_blis_xdsl/kernels"

cp ${SRC_DIR}/kernels_RVV_8x8_fp32.c ${DEST_DIR}/
cp ${SRC_DIR}/kernels_RVV_8x8_fp32.h ${DEST_DIR}/
cp ${SRC_DIR}/exo_matrix_RVV_fp32.c    ${DEST_DIR}/
cp ${SRC_DIR}/exo_matrix_RVV_fp32.h    ${DEST_DIR}/

# ============================================================
# [2/4] Sync to Board
# ============================================================
echo ""; echo "$SEP"; echo " [2/4] Sync to Board"; echo "$SEP"

BASE_REMOTE_DIR="${RISCV_WORKSPACE}/pre_rvv_integration"
ssh_board "mkdir -p ${BASE_REMOTE_DIR}/gemm_blis_xdsl"

# Sync driver application (includes kernels subfolder)
rsync_board tests/gemm_blis_xdsl/ "${RSYNC_TARGET}:${BASE_REMOTE_DIR}/gemm_blis_xdsl/"

# ============================================================
# [3/4] Compile on Board
# ============================================================
echo ""; echo "$SEP"; echo " [3/4] Compile on Board"; echo "$SEP"

DRIVER_REMOTE_DIR="${BASE_REMOTE_DIR}/gemm_blis_xdsl"
ssh_board "cd ${DRIVER_REMOTE_DIR} && make clean && make"

# ============================================================
# [3.5] Packing Performance Analysis
# ============================================================
echo ""; echo "$SEP"; echo " [3.5] Packing Performance Analysis"; echo "$SEP"

echo ">> Running RVV vs Scalar Packing Benchmark..."
ssh_board "cd ${DRIVER_REMOTE_DIR} && ./benchmark_packing ${DATASET_REL_PATH}"

# ============================================================
# [4/4] Run Verification on Board
# ============================================================
echo ""; echo "$SEP"; echo " [4/4] Run Verification"; echo "$SEP"

echo ">> Running MOCK Verification..."
ssh_board "cd ${DRIVER_REMOTE_DIR} && ./test_macro"

echo ""
echo ">> Running RVV HARDWARE Verification..."
    ssh_board "cd ${DRIVER_REMOTE_DIR} && ./test_macro_hw ${DATASET_REL_PATH}"

    echo ""; echo "$SEP"
    echo " Done: ${MODEL_NAME}"
    echo "$SEP"
done

echo ""
echo "$SEP"
echo " Done."
echo "$SEP"
