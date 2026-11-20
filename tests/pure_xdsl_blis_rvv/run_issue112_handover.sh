#!/bin/bash
# run_issue112_handover.sh -- issue #112: build and run the hand-over microkernel check on the board.
#
# Copies the bundle (--bundle, made by gen_handover_ukernels.py) to the board as ~/<--dir> via the jump host
# (the boards share one NFS home with it), builds it there with the bundle's Makefile, runs
# test_xdsl_ukernels.x on core 4, and writes the board's processes, gcc version and the test output to
# <bundle>/logs/board_<board>_<stamp>.log. Correctness only: no timing.
#
# Usage: REMOTE_HOST=fej REMOTE_PORT=3322 REMOTE_PASS="" ./run_issue112_handover.sh --bundle DIR [--board jbpi] [--cpu 4]
#        [--allow-core-0] [--dir NAME]
set -euo pipefail
source "$(dirname "$0")/board_pin.sh"

BUNDLE="" BOARD="jbpi" CPU="$BENCH_CPU_DEFAULT" CPU_GIVEN=false ALLOW0=false DIR="issue112_handover" USER_="${RISCV_USER:-jlei}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle) BUNDLE="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --cpu) CPU="$2"; CPU_GIVEN=true; shift 2 ;;
        --allow-core-0) ALLOW0=true; shift ;;
        --dir) DIR="$2"; shift 2 ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done
[[ -n "$BUNDLE" && -f "$BUNDLE/test_xdsl_ukernels.c" ]] || { echo "--bundle DIR with test_xdsl_ukernels.c required" >&2; exit 2; }
pin_check "$CPU_GIVEN" false "$CPU" "$ALLOW0" || exit 2
: "${REMOTE_HOST:?set REMOTE_HOST}" "${REMOTE_PORT:?set REMOTE_PORT}"

REMOTE_PASS="${REMOTE_PASS:-}"; MODE_ARG=--jump; RISCV_USER="$USER_"; RISCV_PORT=22
board() { board_ssh "$BOARD" "$*"; }

board_rsync "$BOARD" --delete --exclude logs "$BUNDLE/" "@T@:$DIR/"

mkdir -p "$BUNDLE/logs"
LOG="$BUNDLE/logs/board_${BOARD}_$(date +%Y-%m-%d_%H%M%S).log"
{
    echo "# issue #112 hand-over check on $BOARD, core $CPU, $(date -Is). Correctness only, no timing; build: the bundle's Makefile (gcc-14)."
    echo "## other users / load"
    board "who; uptime; ps -eo user,pcpu,comm --sort=-pcpu | head -6"
    echo "## build"
    board "cd ~/$DIR && gcc-14 --version | head -1 && make clean >/dev/null && make 2>&1"
    echo "## run"
    board "cd ~/$DIR && taskset -c $CPU ./test_xdsl_ukernels.x 2>&1; echo exit=\$?"
} 2>&1 | tee "$LOG"
grep -q '^exit=0$' "$LOG"
