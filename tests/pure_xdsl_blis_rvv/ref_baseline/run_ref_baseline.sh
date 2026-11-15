#!/bin/bash
set -e

# ============================================================
# run_ref_baseline.sh — produce a FRESH reference-design baseline
# on the board, at our exact shapes and our exact MR/NR pinning.
#
# The "reference design" here is the real one:
#   ext/RVV_ukernels_benchmark/gemm_blis_family
# NOT the archived ext/*_comparison.txt tables (those were produced by an
# older xDSL sweep driver, and their provenance is what we are replacing).
#
# Usage (jump-host, mirrors compile_macro_bench.sh):
#   REMOTE_PASS="<pw>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh \
#       --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --bracket
#
# The timed runs are pinned to one core (--cpu N, --no-pin to disable; issue #21), the same
# default core as compile_macro_bench.sh, and the conditions they ran under are appended to
# provenance.txt. For a head-to-head against our side, run_head_to_head.sh runs both back
# to back; --no-compare skips step 5 for it.
#
# --per-rep (issue #46) adds a per-rep record to the reference: a STAGED copy of test_gemm.c
# (ext/ is never touched) stores the clock after every call and writes it out after the timed
# loop, and the results come back as reps_ref_<tile>_<variant>.txt.gz, the layout reps_report.py
# reads. The timed loop is unchanged. --cases FILE runs another case list than cases.tsv (same
# columns), e.g. cases_issue46.tsv; use --no-compare with it.
#
# --pack-b-fix (issue #54) packs B with our one-column packer instead of the reference's own: a STAGED
# copy of gemm_blis.c (ext/ is never touched) has pack_CB renamed to pack_CB_orig and a new pack_CB
# that reads a column-major fp32 B one column at a time, 32 numbers per load (ref_pack_fix.py).
# --check sets TEST=T in the staged run_gemm.sh, so the driver compares C with its plain gemm after the
# timed loop and prints PASS or ERROR (it does not change the timing; off by default, as upstream).
#
# Run from the project root (RVV_code_gen_via_MLIR_xDSL/).
# ============================================================

USE_DIRECT=true
RISCV_USER="jlei"
RISCV_REMOTE_IP="jbpi2"
RISCV_SSH_PORT="22"
RISCV_WORKSPACE="xdsl_rvv_refbase"
OPT_MODE="OPT"
SWAP=0
GATHER="2"
BRACKET=""
TIMIN="3.0"
DRY_RUN=false
PREFLIGHT=true
CROSS_CC="/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc"

SEP="------------------------------------------------------------"

source "${BASH_SOURCE[0]%/*}/../board_pin.sh"
PIN_CPU="${BENCH_CPU_DEFAULT}"
CPU_FLAG=false
NO_PIN=false
ALLOW_CORE0=false
COMPARE=true
PER_REP=false
PACK_FIX=false
CHECK=false
CASES_FILE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct)          USE_DIRECT=true                       ;;
        --jump)            USE_DIRECT=false                      ;;
        --riscv-user)      RISCV_USER="$2";          shift        ;;
        --riscv-ip)        RISCV_REMOTE_IP="$2";     shift        ;;
        --riscv-port)      RISCV_SSH_PORT="$2";      shift        ;;
        --riscv-workspace) RISCV_WORKSPACE="$2";     shift        ;;
        --opt)             OPT_MODE="$2";            shift        ;;
        --swap)            SWAP="$2";                shift        ;;
        --gather)          GATHER="$2";              shift        ;;
        --variants)        GATHER="$2";              shift        ;;
        --tmin)            TIMIN="$2";               shift        ;;
        --bracket)         BRACKET="--bracket"                    ;;
        --dry-run)         DRY_RUN=true                           ;;
        --no-preflight)    PREFLIGHT=false                        ;;
        --cross-cc)        CROSS_CC="$2";            shift        ;;
        --cpu)             PIN_CPU="$2"; CPU_FLAG=true; shift     ;;
        --no-pin)          NO_PIN=true                            ;;
        --allow-core-0)    ALLOW_CORE0=true                       ;;
        --no-compare)      COMPARE=false                          ;;
        --per-rep)         PER_REP=true                           ;;
        --pack-b-fix)      PACK_FIX=true                          ;;
        --check)           CHECK=true                             ;;
        --cases)           CASES_FILE="$2";          shift        ;;
        -h|--help)         sed -n '3,30p' "$0"; exit 0            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Issue #21: refuse a bad core, and --cpu with --no-pin, before anything runs.
pin_check "${CPU_FLAG}" "${NO_PIN}" "${PIN_CPU}" "${ALLOW_CORE0}" || exit 1
if [ "${NO_PIN}" = true ]; then PIN_CPU=""; fi
PIN_PREFIX="$(pin_prefix "${PIN_CPU}")"

vname() { case "$1" in 0) echo bcast ;; 1) echo gather ;; 2) echo macc ;; *) echo "unknown" ;; esac; }
GATHER_LIST="${GATHER//,/ }"
GATHER_FIRST="${GATHER_LIST%% *}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"                       # RVV_code_gen_via_MLIR_xDSL
REF_SRC="$(cd "${REPO_ROOT}/../ext/RVV_ukernels_benchmark" && pwd)"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
RESULT_DIR="${REPO_ROOT}/tests/pure_xdsl_blis_rvv/logs/refbase_${STAMP}"

if [ "$USE_DIRECT" = true ]; then
    echo ""; echo "  Mode: Direct"; echo ""
    ssh_board()   { ssh -p "${RISCV_SSH_PORT}" -o StrictHostKeyChecking=no \
                        "${RISCV_USER}@${RISCV_REMOTE_IP}" "$@"; }
    rsync_board() { rsync -a -e "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no" "$@"; }
    RSYNC_TARGET="${RISCV_USER}@${RISCV_REMOTE_IP}"
else
    echo ""; echo "  Mode: Jump-host (x86 -> ${REMOTE_HOST} -> ${RISCV_REMOTE_IP})"; echo ""
    ssh_board() {
        local escaped_args
        escaped_args=$(printf '%q ' "$@")
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no \
            "${RISCV_USER}@${REMOTE_HOST}" \
            "ssh -p ${RISCV_SSH_PORT} -o StrictHostKeyChecking=no ${RISCV_REMOTE_IP} ${escaped_args}"
    }
    rsync_board() {
        sshpass -p "${REMOTE_PASS}" rsync -a \
            -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "$@"
    }
    RSYNC_TARGET="${RISCV_USER}@${REMOTE_HOST}"
fi

echo "$SEP"; echo " Reference-design baseline"; echo "$SEP"
echo "  Reference tree : ${REF_SRC}"
VNAMES=""; for gg in ${GATHER_LIST}; do VNAMES="${VNAMES} $(vname ${gg})"; done
echo "  Variants       : OPT=${OPT_MODE} SWAP=${SWAP} kernel:${VNAMES}"
if [ -n "${PIN_CPU}" ]; then
    echo "  Pinning        : ${PIN_PREFIX}./run_gemm.sh (the timed run only), cpu ${PIN_CPU}"
else
    echo "  Pinning        : NOT pinned (--no-pin): the scheduler may move the run between cores"
fi
if [ "${PER_REP}" = true ]; then
    echo "  Per-rep record : ON (staged test_gemm.c patched, ref_per_rep.py); one row per call in reps_ref_*.txt.gz"
fi
if [ "${PACK_FIX}" = true ]; then
    echo "  B packer       : one column at a time, 32 numbers per load (staged gemm_blis.c patched, ref_pack_fix.py)"
fi
if [ "${CHECK}" = true ]; then
    echo "  Result check   : ON (TEST=T: the driver compares C with its plain gemm after the timed loop)"
fi
if [ -n "${CASES_FILE}" ]; then
    echo "  Cases          : ${CASES_FILE}"
fi
echo "  Results        : ${RESULT_DIR}"
echo ""

# ------------------------------------------------------------
# 1. Generate the per-(MR,NR) datasets + run plan
# ------------------------------------------------------------
echo "$SEP"; echo " [1/5] Generating datasets for our case set"; echo "$SEP"
CASES_ARGS=()
if [ -n "${CASES_FILE}" ]; then CASES_ARGS=(--cases "${CASES_FILE}"); fi
python3 "${HERE}/gen_ref_datasets.py" "${CASES_ARGS[@]}" \
    --opt "${OPT_MODE}" --swap "${SWAP}" --gather "${GATHER_FIRST}" --tmin "${TIMIN}" ${BRACKET}

PLAN="${HERE}/generated/run_plan.tsv"
if [ ! -s "${PLAN}" ]; then
    echo "  [FAIL] no run plan produced"; exit 1
fi
TAGS=$(awk -F'\t' '!/^#/ {print $14}' "${PLAN}" | sort -u)
echo ""
echo "  Builds required: $(echo ${TAGS} | tr '\n' ' ')"

# ------------------------------------------------------------
# 1b. Pre-flight: cross-compile every required (MR,NR) locally, so a build
#     error is caught here instead of after staging onto the board.
# ------------------------------------------------------------
if [ "$PREFLIGHT" = true ] && [ -x "${CROSS_CC}" ]; then
    echo ""; echo "$SEP"; echo " [1b] Pre-flight cross-build (${CROSS_CC##*/})"; echo "$SEP"
    PF_DIR="$(mktemp -d)"
    trap 'rm -rf "${PF_DIR}"' EXIT
    cp -r "${REF_SRC}/gemm_blis_family" "${PF_DIR}/"
    mkdir -p "${PF_DIR}/kernels"
    cp -r "${REF_SRC}/kernels/RVV_256_${OPT_MODE}" "${PF_DIR}/kernels/"
    cp -r "${HERE}/blis_shim" "${PF_DIR}/"
    python3 "${HERE}/repair_ref_kernels.py" "${PF_DIR}/kernels" || true
    if [ "${PER_REP}" = true ]; then
        python3 "${HERE}/ref_per_rep.py" patch "${PF_DIR}/gemm_blis_family/src/test_gemm.c" || exit 1
    fi
    if [ "${PACK_FIX}" = true ]; then
        python3 "${HERE}/ref_pack_fix.py" patch "${PF_DIR}/gemm_blis_family/src/gemm_blis.c" || exit 1
    fi
    PF_FAIL=0
    for TAG in ${TAGS}; do
      for GG in ${GATHER_LIST}; do
        MR="${TAG%x*}"; NR="${TAG#*x}"
        rm -rf "${PF_DIR}/gemm_blis_family/build"
        if ( cd "${PF_DIR}/gemm_blis_family" && \
             make CC="${CROSS_CC}" CLINKER="${CROSS_CC}" \
                  MR="${MR}" NR="${NR}" OPT="${OPT_MODE}" \
                  GATHER="${GG}" SWAP="${SWAP}" \
                  SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO \
                  INCLUDE='-I../blis_shim' LIBS='-lm' >/dev/null 2>&1 \
             && [ -f build/test_gemm.x ] ); then
            echo "  [OK]   ${TAG} $(vname ${GG})"
        else
            echo "  [FAIL] ${TAG} $(vname ${GG}) — does the reference ship this kernel?"
            PF_FAIL=1
        fi
      done
    done
    rm -rf "${PF_DIR}"; trap - EXIT
    if [ "${PF_FAIL}" -ne 0 ]; then
        echo ""
        echo "  Pre-flight failed. Fix the build before spending board time,"
        echo "  or pass --no-preflight to run anyway."
        exit 1
    fi
elif [ "$PREFLIGHT" = true ]; then
    echo ""
    echo "  [skip] pre-flight: no cross compiler at ${CROSS_CC}"
    echo "         (pass --cross-cc <path>, or --no-preflight to silence this)"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""; echo "  --dry-run: stopping before touching the board."; exit 0
fi

# Issue #21: find out now, not after the build, that the board cannot pin to this core.
if [ -n "${PIN_CPU}" ]; then
    if ! ssh_board "${PIN_PREFIX}true" > /dev/null 2>&1; then
        echo "  [FAIL] the board cannot run '${PIN_PREFIX}true' (no taskset, or no core ${PIN_CPU})."
        echo "         Pass --cpu <core>, or --no-pin to run unpinned."
        exit 1
    fi
    echo "  [OK] board can pin to core ${PIN_CPU}"
fi

# ------------------------------------------------------------
# 2. Stage the reference tree on the board
# ------------------------------------------------------------
echo ""; echo "$SEP"; echo " [2/5] Staging reference tree on ${RISCV_REMOTE_IP}"; echo "$SEP"
REMOTE_DIR="${RISCV_WORKSPACE}"
ssh_board "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR}/kernels/RVV_256_${OPT_MODE}/fp32"
# The Exo kernel tree is large; ship only the fp32 256-bit variant in use.
GEMM_SRC="${REF_SRC}/gemm_blis_family"
if [ "${PER_REP}" = true ] || [ "${PACK_FIX}" = true ]; then
    # Issues #46/#54: patch a local staged copy, never the one in ext/.
    STAGE_G="$(mktemp -d)"
    cp -r "${REF_SRC}/gemm_blis_family" "${STAGE_G}/"
    if [ "${PER_REP}" = true ]; then
        python3 "${HERE}/ref_per_rep.py" patch "${STAGE_G}/gemm_blis_family/src/test_gemm.c" || exit 1
    fi
    if [ "${PACK_FIX}" = true ]; then
        python3 "${HERE}/ref_pack_fix.py" patch "${STAGE_G}/gemm_blis_family/src/gemm_blis.c" || exit 1
    fi
    GEMM_SRC="${STAGE_G}/gemm_blis_family"
fi
rsync_board --delete \
    --exclude 'build/' --exclude 'out_models/' --exclude 'output/' \
    "${GEMM_SRC}/" "${RSYNC_TARGET}:${REMOTE_DIR}/gemm_blis_family/"
if [ -n "${STAGE_G:-}" ]; then rm -rf "${STAGE_G}"; fi
# Six vendored kernel files (16x14/*/macc) are truncated upstream. Repair a local
# staging copy — never ext/ itself — so the board gets buildable sources.
STAGE_K="$(mktemp -d)"
cp -r "${REF_SRC}/kernels/RVV_256_${OPT_MODE}/fp32" "${STAGE_K}/"
python3 "${HERE}/repair_ref_kernels.py" "${STAGE_K}" || true
rsync_board --delete \
    "${STAGE_K}/fp32/" \
    "${RSYNC_TARGET}:${REMOTE_DIR}/kernels/RVV_256_${OPT_MODE}/fp32/"
rm -rf "${STAGE_K}"
rsync_board "${HERE}/blis_shim/" "${RSYNC_TARGET}:${REMOTE_DIR}/blis_shim/"
rsync_board "${HERE}/generated/" "${RSYNC_TARGET}:${REMOTE_DIR}/gemm_blis_family/xdsl_cases/"
echo "  [OK] staged -> ${RSYNC_TARGET}:${REMOTE_DIR}/"

TEST_FLAG="F"; if [ "${CHECK}" = true ]; then TEST_FLAG="T"; fi

# ------------------------------------------------------------
# 3. Build + run one binary per (MR,NR)
# ------------------------------------------------------------
echo ""; echo "$SEP"; echo " [3/5] Build + run per (MR,NR)"; echo "$SEP"
# Issue #21: the conditions of this run, taken on the board just before it. If the probe
# fails, what lands in provenance.txt is the ssh error, and compare_ref_baseline.py then
# reports the pinning as unknown rather than assume it.
PROVENANCE="$(board_provenance "${PIN_CPU}" | ssh_board "sh -s" 2>&1)" || true
echo "  NOTE: building with blis_shim/ (RUN_MODE=FAMILY_EXO only — the BLIS"
echo "        ukernel is never invoked on that path; see blis_shim/blis.h)."
echo ""

for TAG in ${TAGS}; do
  for GG in ${GATHER_LIST}; do
    MR="${TAG%x*}"
    NR="${TAG#*x}"
    VN="$(vname ${GG})"
    echo "  --- ${TAG} / ${VN} ---"
    ssh_board "cd ${REMOTE_DIR}/gemm_blis_family && \
        sed -i 's|^TIMIN=.*|TIMIN=${TIMIN}|' run_gemm.sh && \
        sed -i 's|^TEST=.*|TEST=${TEST_FLAG}|' run_gemm.sh && \
        make clean >/dev/null 2>&1 ; \
        make MR=${MR} NR=${NR} OPT=${OPT_MODE} GATHER=${GG} SWAP=${SWAP} \
             SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO \
             INCLUDE='-I../blis_shim' LIBS='-lm' 2>&1 | tail -5 && \
        ${PIN_PREFIX}./run_gemm.sh xdsl_cases/xdsl_cases_${TAG}.dat \
                      xdsl_cases/result_${TAG}_${VN}.csv 2>&1 | tail -25" \
        || { echo "  [WARN] ${TAG}/${VN} failed — continuing"; continue; }
    echo ""
  done
done

# ------------------------------------------------------------
# 4. Pull results back
# ------------------------------------------------------------
echo "$SEP"; echo " [4/5] Retrieving results"; echo "$SEP"
mkdir -p "${RESULT_DIR}"
rsync_board "${RSYNC_TARGET}:${REMOTE_DIR}/gemm_blis_family/xdsl_cases/" "${RESULT_DIR}/"
cp "${PLAN}" "${RESULT_DIR}/run_plan.tsv"
if [ -n "${CASES_FILE}" ]; then cp "${CASES_FILE}" "${RESULT_DIR}/cases.tsv"; fi
if [ "${PER_REP}" = true ]; then
    python3 "${HERE}/ref_per_rep.py" convert "${RESULT_DIR}" --cases "${CASES_FILE:-${HERE}/cases.tsv}" \
        || echo "  [WARN] ref_per_rep.py convert reported a problem: read the lines above before using the per-rep files"
fi
cat > "${RESULT_DIR}/provenance.txt" <<EOF
Reference-design baseline, generated $(date -Is)
  board          : ${RISCV_REMOTE_IP} (user ${RISCV_USER})
  reference tree : ${REF_SRC}/gemm_blis_family
  variants       : OPT=${OPT_MODE} SWAP=${SWAP} kernel:${VNAMES}
  build          : SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO, blis_shim for blis.h
  tmin           : ${TIMIN} s
  protocol       : REFERENCE driver (test_gemm.c): no warm-up, reps until tmin, one clock read per call
  B packer       : $([ "${PACK_FIX}" = true ] && echo "one column at a time, 32 numbers per load (staged gemm_blis.c patched by ref_pack_fix.py, issue #54)" || echo "the reference's own pack_CB (unchanged)")
  result check   : $([ "${CHECK}" = true ] && echo "on (TEST=T, after the timed loop)" || echo "off (TEST=F, as upstream)")
  per-rep patch  : $([ "${PER_REP}" = true ] && echo "yes, v3 (staged test_gemm.c stores the clock after every call in a static array and writes it with write(2), no heap, issue #46)" || echo "no (unchanged driver)")
  metric         : GFLOPS = 2*m*n*k / wall time of gemm_blis_B3A2C0()
                   -> PACK TIME IS INCLUDED (src/test_gemm.c:399-400)
  cases          : see run_plan.tsv (pinned MR/NR vs actually-built MR/NR)
${PROVENANCE}
EOF
echo "  [OK] ${RESULT_DIR}"

# ------------------------------------------------------------
# 5. Compare against our latest run
# ------------------------------------------------------------
if [ "${COMPARE}" = false ]; then
    echo ""; echo "  --no-compare: skipping the comparison. Results in ${RESULT_DIR}"
    exit 0
fi
echo ""; echo "$SEP"; echo " [5/5] Comparison"; echo "$SEP"
python3 "${HERE}/compare_ref_baseline.py" --result-dir "${RESULT_DIR}" \
    | tee "${RESULT_DIR}/comparison.txt"
echo ""
echo "  Done. Baseline + comparison in ${RESULT_DIR}"
