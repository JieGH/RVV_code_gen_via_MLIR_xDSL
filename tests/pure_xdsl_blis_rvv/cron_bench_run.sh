#!/bin/bash
# Wrapper for unattended (cron) invocation of compile_macro_bench.sh.
# Usage: cron_bench_run.sh <riscv-ip>   e.g. cron_bench_run.sh jbpi
#
# Only one board runs at a time, whichever cron entry fires first. jbpi, jbpi2 and fej
# all see the same directory (/nfs/gap/jlei/xdsl_rvv_microkernel, exported by fej), and
# compile_macro_bench.sh does `rm -rf pj_gen` there before it syncs. Two boards running
# together delete each other's build tree, and the first one fails with
# "Fatal error: can't create gemm_b1_xdsl.o: Stale file handle". The local pj_gen is
# shared the same way (the generator wipes it). So a second run waits for the first.
#
# Flags are the default run command (--no-nc-overrides --cpu 0 --tmin 10). One run takes
# about 10 min; the schedule alternates boards every 30 min (10 run + 20 idle).

set -e

BOARD="$1"
if [ -z "$BOARD" ]; then
    echo "Usage: $0 <riscv-ip>"
    exit 1
fi

cd /home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL
source .venv/bin/activate

mkdir -p tests/pure_xdsl_blis_rvv/logs
LOG="tests/pure_xdsl_blis_rvv/logs/cron_${BOARD}_$(date +%Y%m%d_%H%M%S).log"
LOCK="tests/pure_xdsl_blis_rvv/logs/.cron_bench.lock"

# Wait up to 3 h for the other board to finish; a run takes about 40 min.
exec 9>"${LOCK}"
if ! flock -w 10800 9; then
    echo "$(date -Is) gave up waiting for the lock, another run is still going" >> "${LOG}"
    exit 1
fi
echo "$(date -Is) got the lock for ${BOARD}" >> "${LOG}"

REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip "${BOARD}" --riscv-port 22 \
  --no-nc-overrides --cpu 0 --tmin 10 >> "${LOG}" 2>&1
