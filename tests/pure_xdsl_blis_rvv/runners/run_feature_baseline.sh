#!/bin/bash
set -o pipefail

# ============================================================
# run_feature_baseline.sh -- every user-facing option, one correctness run each (issue #137).
#
# The baseline of the design's features, not of past issues: each case is one
# compile_macro_bench.sh run against the scalar oracle (skip_ref_run off, datasets/correctness_small.dat:
# full tiles, edge tails, several K blocks), and passes when every row of its all_raw_*.txt says PASS.
#
#   data types      fp32 AXPY, fp32 DOT, f16f16f32 (scalar pack only), int8
#   packing         RVV (default), --scalar-pack, every --pack-b variant, --zero-pack-lines, --pad-ldb
#   microkernel     pinned 16x15, 32x6, 32x7 (and 24x8 in f16f16f32); search mode sweeps every tile
#   threads         --threads 8
#
# Usage (jump-host), from the repo root:
#   REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
#   ./tests/pure_xdsl_blis_rvv/runners/run_feature_baseline.sh --jump \
#       --riscv-user jlei --riscv-ip jbpi --riscv-port 22
#
# Flags: --direct | --jump, --riscv-user/-ip/-port, --only NAME[,NAME] (run a subset),
# --list (print the case names), --dry-run (each case's compile_macro_bench.sh --dry-run: checks
# every option is accepted, no board).
# Writes logs/feature_baseline_<stamp>.tsv: case, run folder, rows passed / rows, verdict.
# A few minutes per case, about an hour in all. Correctness only: no number from it is quoted.
# ============================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the project folder
LOGS="${HERE}/logs"
C=configs/correctness
# name | config | extra flags
CASES=(
    "fp32-axpy-rvv-pack|${C}/config_correctness.json|"
    "fp32-dot|${C}/config_feature_dot.json|"
    "f16f16f32-scalar-pack|${C}/config_issue112_mixed_correctness.json|--scalar-pack"
    "int8|${C}/config_feature_int8.json|"
    "fp32-scalar-pack|${C}/config_correctness.json|--scalar-pack"
    "pack-b-row-intr|${C}/config_correctness.json|--pack-b row-intr"
    "pack-b-col-g1|${C}/config_correctness.json|--pack-b col-g1"
    "pack-b-col-g2|${C}/config_correctness.json|--pack-b col-g2"
    "pack-b-col-g4|${C}/config_correctness.json|--pack-b col-g4"
    "pack-b-col-gall|${C}/config_correctness.json|--pack-b col-gall"
    "pack-b-col-g1-m4|${C}/config_correctness.json|--pack-b col-g1-m4"
    "zero-pack-lines|${C}/config_correctness.json|--zero-pack-lines"
    "pad-ldb-16|${C}/config_correctness.json|--pad-ldb 16"
    "search-mode|${C}/config_feature_search.json|"
    "threads-8|${C}/config_correctness.json|--threads 8"
)
SHARED=()
ONLY=""
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --direct|--jump)                        SHARED+=("$1")             ;;
        --riscv-user|--riscv-ip|--riscv-port)   SHARED+=("$1" "$2"); shift ;;
        --only)                                 ONLY=",$2,"; shift         ;;
        --list)                                 for c in "${CASES[@]}"; do echo "${c%%|*}"; done; exit 0 ;;
        --dry-run)                              DRY_RUN=true               ;;
        -h|--help)                              sed -n '5,27p' "$0"; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

cd "${HERE}/../.." || exit 1   # compile_macro_bench.sh runs from the repo root
STAMP="$(date +%Y-%m-%d_%H%M%S)"
SUMMARY="${LOGS}/feature_baseline_${STAMP}.tsv"
[ "${DRY_RUN}" = true ] || printf 'case\trun\tpassed\trows\tverdict\n' > "${SUMMARY}"
FAILED=0

for c in "${CASES[@]}"; do
    IFS='|' read -r name config flags <<< "${c}"
    [ -z "${ONLY}" ] || [[ "${ONLY}" == *",${name},"* ]] || continue
    # shellcheck disable=SC2206  # flags split on spaces on purpose
    BENCH=("${HERE}/compile_macro_bench.sh" "${SHARED[@]}" --config "${config}" --no-nc-overrides --tmin 0.2 ${flags})
    [[ " ${flags} " == *" --threads "* ]] || BENCH+=(--cpu 4)   # --threads fixes its own cores
    echo ""; echo "=== ${name}: ${config##*/} ${flags} ==="
    if [ "${DRY_RUN}" = true ]; then
        if ! out="$("${BENCH[@]}" --dry-run 2>&1)"; then echo "${out}" | tail -5; echo "  [REFUSED] ${name}"; FAILED=$((FAILED + 1)); else echo "  accepted"; fi
        continue
    fi
    before="$(ls -d "${LOGS}"/run_* 2>/dev/null | sort | tail -1)"
    "${BENCH[@]}"
    run="$(ls -d "${LOGS}"/run_* 2>/dev/null | sort | tail -1)"
    if [ "${run}" = "${before}" ]; then
        printf '%s\t-\t0\t0\tNO RUN\n' "${name}" >> "${SUMMARY}"; FAILED=$((FAILED + 1)); continue
    fi
    # Status is the 6th column of every all_raw_*.txt row; every row must say PASS.
    read -r passed rows <<< "$(cat "${run}"/all_raw_*.txt 2>/dev/null | awk -F, '$1 != "Layer" && NF > 5 { n++; if ($6 == "PASS") p++ } END { print p + 0, n + 0 }')"
    verdict=PASS; { [ "${rows}" -gt 0 ] && [ "${passed}" -eq "${rows}" ]; } || { verdict=FAIL; FAILED=$((FAILED + 1)); }
    printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "$(basename "${run}")" "${passed}" "${rows}" "${verdict}" | tee -a "${SUMMARY}"
done

echo ""
if [ "${DRY_RUN}" = true ]; then echo "  --dry-run: ${FAILED} case(s) refused; nothing was started."; exit $(( FAILED > 0 )); fi
column -t -s $'\t' "${SUMMARY}"
echo "  ${FAILED} case(s) failed. Summary: ${SUMMARY#"${HERE}"/}"
exit $(( FAILED > 0 ))
