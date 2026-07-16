#!/bin/bash
set -e

# Usage:
#   REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" ./run_on_board.sh

RISCV_USER="jlei"
RISCV_REMOTE_IP="jbpi2"
RISCV_SSH_PORT="22"
RISCV_WORKSPACE="xdsl_rvv_microkernel"
REMOTE_DIR="${RISCV_WORKSPACE}/tests/pure_xdsl_blis_rvv/studyfp32dot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "------------------------------------------------------------"
echo "  Deploying and Testing RVV FP32 DOT Study"
echo "------------------------------------------------------------"

if [ -z "$REMOTE_HOST" ]; then
    echo "Direct Mode to ${RISCV_REMOTE_IP}"
    ssh_board() { ssh -p "${RISCV_SSH_PORT}" -o StrictHostKeyChecking=no "${RISCV_USER}@${RISCV_REMOTE_IP}" "$@"; }
    rsync_board() { rsync -av -e "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no" "$@"; }
    RSYNC_TARGET="${RISCV_USER}@${RISCV_REMOTE_IP}"
else
    echo "Jump-host Mode (x86 → ${REMOTE_HOST} → ${RISCV_REMOTE_IP})"
    ssh_board() {
        local escaped_args
        escaped_args=$(printf '%q ' "$@")
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no \
            -t "${RISCV_USER}@${REMOTE_HOST}" \
            "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no -t ${RISCV_REMOTE_IP} ${escaped_args}"
    }
    rsync_board() {
        # Sync to jump host first, then from jump host to board (simplified approach)
        # Note: The main script does a direct rsync to jump host. We will just use the jump host as an intermediary.
        echo "Syncing to jump host..."
        sshpass -p "${REMOTE_PASS}" rsync -av -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "$@"
    }
    RSYNC_TARGET="${RISCV_USER}@${REMOTE_HOST}"
fi

echo "1. Creating remote directory..."
ssh_board "mkdir -p ${REMOTE_DIR}"

echo "2. Syncing files..."
if [ -n "$REMOTE_HOST" ]; then
    # Two-step sync for jump host
    rsync_board "${SCRIPT_DIR}/" "${RSYNC_TARGET}:${REMOTE_DIR}/"
    sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no "${RISCV_USER}@${REMOTE_HOST}" \
        "rsync -av ${REMOTE_DIR}/ ${RISCV_USER}@${RISCV_REMOTE_IP}:${REMOTE_DIR}/"
else
    rsync_board "${SCRIPT_DIR}/" "${RSYNC_TARGET}:${REMOTE_DIR}/"
fi

echo "3. Compiling and running on board..."
ssh_board "cd ${REMOTE_DIR} && make clean && make && make run"

echo "------------------------------------------------------------"
echo "  Done!"
echo "------------------------------------------------------------"
