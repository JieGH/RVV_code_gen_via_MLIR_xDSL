#!/bin/bash
set -e

# ============================================================
# run_ldx_macro_bench.sh — Standalone build & benchmark script 
# for the LDX reference testing framework.
#
# Usage (jump-host):
#   REMOTE_PASS="<your_password>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./run_ldx_macro_bench.sh \
#       --jump \
#       --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 
#
# Run this script from anywhere; it will auto-detect its directory.
# ============================================================

# Navigate to the directory containing this script
cd "$(dirname "$0")"

# Defaults
USE_DIRECT=true
RISCV_USER="jlei"
RISCV_REMOTE_IP="jbpi2"
RISCV_SSH_PORT="22"
RISCV_WORKSPACE="xdsl_rvv_microkernel"

SEP="------------------------------------------------------------"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct)          USE_DIRECT=true                       ;;
        --jump)            USE_DIRECT=false                      ;;
        --riscv-user)      RISCV_USER="$2";          shift      ;;
        --riscv-ip)        RISCV_REMOTE_IP="$2";     shift      ;;
        --riscv-port)      RISCV_SSH_PORT="$2";      shift      ;;
        --riscv-workspace) RISCV_WORKSPACE="$2";     shift      ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$USE_DIRECT" = true ]; then
    echo ""; echo "  Mode: Direct"; echo ""
    ssh_board()        { ssh -p "${RISCV_SSH_PORT}" -o StrictHostKeyChecking=no \
                             "${RISCV_USER}@${RISCV_REMOTE_IP}" "$@"; }
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
        sshpass -p "${REMOTE_PASS}" rsync -av \
            -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "$@"
    }
    RSYNC_TARGET="${RISCV_USER}@${REMOTE_HOST}"
fi

echo ""; echo "$SEP"; echo " Dependency Check"; echo "$SEP"
for cmd in rsync ssh; do
    if command -v "$cmd" > /dev/null 2>&1; then
        echo "  [OK] $cmd: $(command -v "$cmd")"
    else
        echo "  [FAIL] $cmd not found in PATH"
        exit 1
    fi
done

# ============================================================
# [1/3] rsync to board
# ============================================================
echo ""; echo "$SEP"; echo " [1/3] Sync to Board"; echo "$SEP"

REMOTE_DIR="${RISCV_WORKSPACE}/test_ext_ldx_macc"
ssh_board "mkdir -p ${REMOTE_DIR}"

rsync_board --delete ./ "${RSYNC_TARGET}:${REMOTE_DIR}/"
echo "  [OK] Synced current directory (.) → ${RSYNC_TARGET}:${REMOTE_DIR}/"

# ============================================================
# [2/3] Compile on board
# ============================================================
echo ""; echo "$SEP"; echo " [2/3] Compile on Board"; echo "$SEP"

ssh_board "cd ${REMOTE_DIR} && make clean && make"

# ============================================================
# [3/3] Benchmark on board
# ============================================================
echo ""; echo "$SEP"; echo " [3/3] Benchmark on Board"; echo "$SEP"

ssh_board "cd ${REMOTE_DIR} && ./test_gemm"

echo ""
echo "$SEP"
echo " Done."
echo "$SEP"
