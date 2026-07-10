#!/bin/bash
set -e

# ============================================================
# compile_macro_bench.sh — Standalone build & benchmark script 
# for the generated pure xDSL BLIS project.
#
# Usage (jump-host):
#   REMOTE_PASS="<your_password>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh \
#       --config fp32_axpy_config.json \
#       --jump \
#       --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 
#
# Run this script from the project root!
# ============================================================

# Defaults
USE_DIRECT=true
RISCV_USER="jlei"
RISCV_REMOTE_IP="jbpi2"
RISCV_SSH_PORT="22"
RISCV_WORKSPACE="xdsl_rvv_microkernel"
CONFIG="config.json"

SEP="------------------------------------------------------------"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct)          USE_DIRECT=true                       ;;
        --jump)            USE_DIRECT=false                      ;;
        --riscv-user)      RISCV_USER="$2";          shift      ;;
        --riscv-ip)        RISCV_REMOTE_IP="$2";     shift      ;;
        --riscv-port)      RISCV_SSH_PORT="$2";      shift      ;;
        --riscv-workspace) RISCV_WORKSPACE="$2";     shift      ;;
        --config)          CONFIG="$2";              shift      ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Output configuration parsing
# (Remove .json from config name to get the project name)
PJ_NAME=$(basename "$CONFIG" .json)
# If config is "fp32_axpy_config", PJ_NAME is "fp32_axpy_config".
# Actually generate_pure_blis_project.py determines out_dir using the project_name from config.
# We will just sync the entire `pj_gen` directory.

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

echo ""; echo "$SEP"; echo " Dependency Check"; echo "$SEP"

for cmd in python3 rsync ssh; do
    if command -v "$cmd" > /dev/null 2>&1; then
        echo "  [OK] $cmd: $(command -v "$cmd")"
    else
        echo "  [FAIL] $cmd not found in PATH"
        exit 1
    fi
done

if ! source .venv/bin/activate 2>/dev/null; then
    echo "  [FAIL] Could not activate .venv — run from project root"
    exit 1
fi
echo "  [OK] .venv activated"

# ============================================================
# [1/3] Generate C Code Locally
# ============================================================
echo ""; echo "$SEP"; echo " [1/3] Generate C Code Locally"; echo "$SEP"

# Purge local generation directory to avoid stale files
rm -rf tests/pure_xdsl_blis_rvv/pj_gen

cd tests/pure_xdsl_blis_rvv
python3 generate_pure_blis_project.py --config "$CONFIG"
cd ../..
echo "  [OK] Generated project locally."

# ============================================================
# [2/3] rsync to board
# ============================================================
echo ""; echo "$SEP"; echo " [2/3] Sync to Board"; echo "$SEP"

REMOTE_DIR="${RISCV_WORKSPACE}/tests/pure_xdsl_blis_rvv"
ssh_board "mkdir -p ${REMOTE_DIR}"

# Purge remote generation directory before syncing to avoid stale files
ssh_board "rm -rf ${REMOTE_DIR}/pj_gen"

# Rsync the entire pure_xdsl_blis_rvv directory (or just pj_gen and datasets)
rsync_board --delete tests/pure_xdsl_blis_rvv/pj_gen/ "${RSYNC_TARGET}:${REMOTE_DIR}/pj_gen/"
rsync_board tests/pure_xdsl_blis_rvv/datasets/ "${RSYNC_TARGET}:${REMOTE_DIR}/datasets/"
echo "  [OK] Synced pj_gen/ and datasets/ → ${RSYNC_TARGET}:${REMOTE_DIR}/"

# ============================================================
# [3/3] Compile and benchmark on board
# ============================================================
echo ""; echo "$SEP"; echo " [3/3] Compile and Benchmark on Board"; echo "$SEP"

# We don't know the exact folder name inside pj_gen without parsing the json,
# but we know it generated into pj_gen/* . Let's just find the directory inside pj_gen and run make.
ssh_board "cd ${REMOTE_DIR}/pj_gen && for d in */; do (cd \"\$d\" && make clean && make all && make run); done"

echo ""
echo "$SEP"
echo " Done."
echo "$SEP"
