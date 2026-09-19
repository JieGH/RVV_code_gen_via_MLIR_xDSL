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
# Benchmarking defaults to the UNINSTRUMENTED build (issue #14): its numbers are the
# ones comparable to the reference design, which carries no per-tile timers of its own.
# It reports no pack/compute breakdown and logs to *_noinstr.txt.
#
# Pass --instr for the instrumented build instead, when you are debugging where our own
# time goes (pack vs compute, IPC, cache misses). That build's timers cost 11-17% of
# measured time, so do NOT quote its numbers against the reference.
#
# The uninstrumented build times the way the reference design does (issue #27): no
# warm-up, reps run until --tmin seconds (default 3.0) have passed, correctness checked
# after timing. That is the default; no flag needed. Its logs are named
# *_noinstr_refproto.txt. Pass --no-ref-protocol for the older protocol instead (5 warm-up
# calls and a scalar-pack pass, then the fixed `iterations` from config.json), which gives
# steadier numbers on short shapes such as B3 but is not the reference's protocol.
# The instrumented build (--instr) cannot run the reference protocol, so it defaults to
# the older one; asking it for --ref-protocol is refused.
#
# --no-nc-overrides (issue #16) ignores the per-shape nc overrides in config.json, which the
# reference design has no counterpart for (B3 ships at nc=128, the reference runs nc=384).
# Use it for a head-to-head number; a run without it is the shipped configuration. The two
# never share a log: this one's are named *_no_nc_override.txt.
#
# Every run starts by printing an overview of the settings in effect, defaults included.
# --dry-run prints that overview and stops, without touching the board.
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
TEST_ARGS=""
FORCE_NC=""
# Issue #14: 0 (default) compiles the per-tile timers and counters out, so RVV_Time_ms
# carries no timer tax and is comparable to the reference. 1 keeps them, for debugging
# our own pack/compute split. The two write to separate logs.
INSTRUMENTED="0"
BUILD_SOURCE="default"
# Issue #27: PROTOCOL_FLAG is only what the user typed (--ref-protocol or
# --no-ref-protocol); it is passed to the driver as given, so the driver's own default is
# the one source of truth. REF_PROTOCOL is what this script resolves it to, for the
# banner and the log names. TMIN stays empty unless given (driver default: 3.0 s,
# matching run_ref_baseline.sh).
PROTOCOL_FLAG=""
REF_PROTOCOL=""
TMIN=""
DRY_RUN=false

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
        --scalar-pack)     TEST_ARGS="${TEST_ARGS} --scalar-pack" ;;
        --ref-protocol)    PROTOCOL_FLAG="--ref-protocol"       ;;
        --no-ref-protocol) PROTOCOL_FLAG="--no-ref-protocol"    ;;
        --dry-run)         DRY_RUN=true                         ;;
        --tmin)            TMIN="$2";                shift      ;;
        --force-nc)        FORCE_NC="$2";            shift      ;;
        --no-nc-overrides) TEST_ARGS="${TEST_ARGS} --no-nc-overrides" ;;
        --no-instr)        INSTRUMENTED="0"; BUILD_SOURCE="--no-instr" ;;
        --instr)           INSTRUMENTED="1"; BUILD_SOURCE="--instr"    ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Issue #27: resolve the protocol and refuse bad combinations now, during argument
# parsing, not after generating and syncing.
if [ "${PROTOCOL_FLAG}" = "--ref-protocol" ] && [ "${INSTRUMENTED}" != "0" ]; then
    echo "Error: --ref-protocol needs the uninstrumented build (drop --instr)."
    echo "       The instrumented build's timers cost 11-17% of measured time, so its numbers"
    echo "       are not comparable to the reference design under any protocol (issue #14)."
    exit 1
fi
if [ -n "${PROTOCOL_FLAG}" ]; then
    PROTOCOL_SOURCE="${PROTOCOL_FLAG}"
    if [ "${PROTOCOL_FLAG}" = "--ref-protocol" ]; then REF_PROTOCOL="1"; else REF_PROTOCOL="0"; fi
elif [ "${INSTRUMENTED}" = "0" ]; then
    PROTOCOL_SOURCE="default"; REF_PROTOCOL="1"
else
    PROTOCOL_SOURCE="default for this build: the reference protocol needs the uninstrumented build"
    REF_PROTOCOL="0"
fi
if [ -n "${TMIN}" ] && [ "${REF_PROTOCOL}" != "1" ]; then
    echo "Error: --tmin only applies to the reference protocol (this run uses the warm-up protocol)."
    exit 1
fi
# Issue #16: RVV_FORCE_NC beats every other nc source, so this run would not be at the
# heuristic's nc yet would carry the head-to-head log name.
if [ -n "${FORCE_NC}" ] && [[ "${TEST_ARGS}" == *--no-nc-overrides* ]]; then
    echo "Error: --force-nc cannot be combined with --no-nc-overrides (--force-nc sets nc for every shape)."
    exit 1
fi
TEST_ARGS="${TEST_ARGS}${PROTOCOL_FLAG:+ ${PROTOCOL_FLAG}}${TMIN:+ --tmin ${TMIN}}"

# Issue #14: say which build this is, loudly, at both ends of the run. Quoting an
# instrumented number against the reference understates us by 11-17%, and the only
# thing that distinguishes the two logs is a filename suffix.
if [ "${INSTRUMENTED}" = "0" ]; then
    BUILD_BANNER="UNINSTRUMENTED (INSTRUMENTED=0) -- comparable to the reference design"
    BUILD_NOTE="pack/compute breakdown and perf counters are nan in this build"
    BUILD_LOGS="raw_*_noinstr.txt"
    if [ "${REF_PROTOCOL}" = "1" ]; then BUILD_LOGS="raw_*_noinstr_refproto.txt"; fi
else
    BUILD_BANNER="INSTRUMENTED (INSTRUMENTED=1) -- DEBUG BUILD"
    BUILD_NOTE="timers cost 11-17% of measured time: do NOT quote these against the reference"
    BUILD_LOGS="raw_*.txt"
fi
case "${TEST_ARGS}" in *--no-nc-overrides*) NO_NC_OVERRIDES=true; BUILD_LOGS="${BUILD_LOGS%.txt}_no_nc_override.txt" ;; *) NO_NC_OVERRIDES=false ;; esac
# Issue #27: every number names its protocol as well as its build.
if [ "${REF_PROTOCOL}" = "1" ]; then
    PROTOCOL_BANNER="REFERENCE (tmin=${TMIN:-3.0} s) -- no warm-up, reps until tmin"
else
    PROTOCOL_BANNER="WARM-UP -- warm-up first, fixed rep count from config.json"
fi

# What this run will do: every setting in effect, defaults included, each marked with
# where it came from. (The driver prints its own, fuller overview per project on the board.)
ov() {  # key value [source]
    printf "  %-14s : %s" "$1" "$2"
    if [ -n "$3" ]; then printf "   [%s]" "$3"; fi
    printf "\n"
}
if [ "${USE_DIRECT}" = true ]; then TARGET_MODE="direct"; else TARGET_MODE="jump-host ${REMOTE_HOST:-?} -> board"; fi
echo ""; echo "$SEP"
echo " Run overview   ([default] = not set by you)"
echo "$SEP"
ov "Build" "${BUILD_BANNER}" "${BUILD_SOURCE}"
if [ "${REF_PROTOCOL}" = "1" ]; then
    ov "Protocol" "REFERENCE (matches the reference design's test_gemm.c)" "${PROTOCOL_SOURCE}"
    ov "Warm-up" "none: the cold first rep is inside the mean"
    ov "Reps" "until tmin = ${TMIN:-3.0} s has passed" "$([ -n "${TMIN}" ] && echo "--tmin" || echo default)"
    ov "Correctness" "checked AFTER timing (unless skip_ref_run in ${CONFIG})"
else
    ov "Protocol" "WARM-UP (steadier on short shapes; not the reference's protocol)" "${PROTOCOL_SOURCE}"
    ov "Warm-up" "5 kernel calls, then a scalar-pack pass"
    ov "Reps" "fixed, 'iterations' from ${CONFIG}"
    ov "Correctness" "checked BEFORE timing (unless skip_ref_run in ${CONFIG})"
fi
ov "Packing" "$(case "${TEST_ARGS}" in *--scalar-pack*) echo "scalar packing forced";; *) echo "RVV vector packing";; esac)" "$(case "${TEST_ARGS}" in *--scalar-pack*) echo "--scalar-pack";; *) echo default;; esac)"
ov "Cache blocking" "analytical model" "default"
if [ "${NO_NC_OVERRIDES}" = true ]; then
    ov "" "nc overrides: OFF (every shape uses the heuristic's nc; the nc table in ${CONFIG} is ignored)" "--no-nc-overrides"
else
    ov "" "nc overrides: as listed in ${CONFIG}, applied to the row with exactly that label" "default"
fi
ov "" "$([ -n "${FORCE_NC}" ] && echo "RVV_FORCE_NC=${FORCE_NC} overrides nc for every shape" || echo "RVV_FORCE_NC: not set")" "$([ -n "${FORCE_NC}" ] && echo "--force-nc" || echo default)"
ov "Config" "${CONFIG}" "$([ "${CONFIG}" = "config.json" ] && echo default || echo "--config")"
ov "Board" "${RISCV_USER}@${RISCV_REMOTE_IP}:${RISCV_SSH_PORT}, ${TARGET_MODE}, workspace ${RISCV_WORKSPACE}"
ov "Logs" "${BUILD_LOGS}"
echo "$SEP"
if [ "${DRY_RUN}" = true ]; then
    echo "  --dry-run: stopping here. Nothing was generated, synced or run."
    exit 0
fi

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
rsync_board tests/pure_xdsl_blis_rvv/modelLevel/ "${RSYNC_TARGET}:${REMOTE_DIR}/modelLevel/"
echo "  [OK] Synced pj_gen/, datasets/, and modelLevel/ → ${RSYNC_TARGET}:${REMOTE_DIR}/"

# ============================================================
# [3/4] Compile and benchmark on board
# ============================================================
echo ""; echo "$SEP"; echo " [3/4] Compile and Benchmark on Board"; echo "$SEP"

# Read compile_order.txt to run exactly in the JSON config order,
# avoiding bash default alphabetical expansion on 'for d in */'
FORCE_NC_ENV=""
if [ -n "${FORCE_NC}" ]; then
    FORCE_NC_ENV="RVV_FORCE_NC=${FORCE_NC} "
fi
ssh_board "cd ${REMOTE_DIR}/pj_gen && while read d; do (cd \"\$d\" && make clean && make all INSTRUMENTED=${INSTRUMENTED} && ${FORCE_NC_ENV}make run INSTRUMENTED=${INSTRUMENTED} TEST_ARGS=\"${TEST_ARGS}\"); done < compile_order.txt"

# ============================================================
# [4/4] Fetch logs
# ============================================================
echo ""; echo "$SEP"; echo " [4/4] Fetch Logs"; echo "$SEP"

TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
RUN_DIR="tests/pure_xdsl_blis_rvv/logs/run_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Rsync the logs folder while preserving the pj_gen project subdirectories locally
rsync_board -am --include='*/' --include='logs/***' --exclude='*' "${RSYNC_TARGET}:${REMOTE_DIR}/pj_gen/" "${RUN_DIR}/"

# One run = one timestamped directory. Flatten each project's nested logs/ into
# <RUN_DIR>/<project>/ (raw/summary/perf per project, unchanged), and additionally
# build one merged CSV per log kind (all_raw_*.txt, all_perf_*.txt, all_summary_*.txt)
# across every project in this run, tagged by a leading Project column, so cross-shape
# comparisons don't require opening N separate files.
for proj_dir in "${RUN_DIR}"/*/; do
    [ -d "$proj_dir" ] || continue
    proj_name=$(basename "$proj_dir")
    if [ -d "${proj_dir}logs" ]; then
        mv "${proj_dir}logs/"* "$proj_dir" 2>/dev/null || true
        rmdir "${proj_dir}logs" 2>/dev/null || true
    fi
    for f in "$proj_dir"*.txt; do
        [ -f "$f" ] || continue
        kind=$(basename "$f")  # e.g. raw_fp32_AXPY_FP32_Pack.txt
        merged="${RUN_DIR}/all_${kind}"
        if [[ "$kind" == summary_* ]]; then
            { echo "=== ${proj_name} ==="; cat "$f"; echo ""; } >> "$merged"
        else
            if [ ! -f "$merged" ]; then
                head -1 "$f" | sed 's/^/Project,/' > "$merged"
            fi
            tail -n +2 "$f" | sed "s/^/${proj_name},/" >> "$merged"
        fi
    done
    echo "  [OK] Fetched logs to ${proj_dir}"
done
echo "  [OK] Merged cross-project results into ${RUN_DIR}/all_*.txt"

# Per-case comparison table (pack/compute ms, pack % of total, KerGOPs, Ref GOPs,
# speedup vs ref, Vector Pack Gain %) built from all_raw_*.txt. See docs/metrics.md.
# A bug here must not fail the whole script: the board run and log fetch above
# already succeeded and are safely on disk regardless of what happens next.
if python3 tests/pure_xdsl_blis_rvv/summarize_run.py "${RUN_DIR}"; then
    echo "  [OK] Wrote comparison table to ${RUN_DIR}/comparison.md"
else
    echo "  [WARN] comparison.md generation failed (see above); raw logs are still in ${RUN_DIR}"
fi

echo ""
echo "$SEP"
echo " Done."
echo ""
echo "  Build    : ${BUILD_BANNER}"
echo "  Protocol : ${PROTOCOL_BANNER}"
echo "  Note     : ${BUILD_NOTE}"
echo "  Logs     : ${RUN_DIR}/ (${BUILD_LOGS})"
echo "$SEP"
