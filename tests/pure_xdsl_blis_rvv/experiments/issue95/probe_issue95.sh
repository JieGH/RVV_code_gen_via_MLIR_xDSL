#!/usr/bin/env bash
# Issue #95 test loop: one GEMM shape, ours then Héctor's gemm.x, back to back, 1 thread on core 4 of one board.
#
#   probe_issue95.sh M N K [--tile 24x8] [--tmin 1] [--board jbpi] [--label NAME] [-- <extra compile_macro_bench args>]
#
# Ours: compile_macro_bench.sh, uninstrumented build, reference protocol, --no-nc-overrides, vector packer
# (as #94), one-row dataset datasets/experiments/issue95_probe.dat. Héctor's: his built gemm.x in ~/hector_gemmlib_b292c92
# (built by run_hector_gemm.sh in #94; his code unchanged), TIMIN = --tmin, TEST=T, one-row dataset in ~/issue95/.
# Prints "ours <GOPs> his <GOPs> ratio <ours/his> RED|GREEN" (RED when ratio < 0.8) and keeps everything in
# logs/issue95_probe_<stamp>_<label>/. Refuses (exit 3) when another user's job is busy on the board.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
source "${HERE}/board_pin.sh"
M="$1"; N="$2"; K="$3"; shift 3
TILE="24x8"; TMIN=1; BOARD="jbpi"; LABEL="p"; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tile)  TILE="$2"; shift ;;
        --tmin)  TMIN="$2"; shift ;;
        --board) BOARD="$2"; shift ;;
        --label) LABEL="$2"; shift ;;
        --)      shift; EXTRA=("$@"); break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
MR="${TILE%x*}"; NR="${TILE#*x}"
export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"  # as run_parallel_gemm.sh
MODE_ARG=--jump; RISCV_USER=jlei; RISCV_PORT=22
on_board() { board_ssh "${BOARD}" "$1"; }

OUT="${HERE}/logs/issue95_probe_$(date +%Y-%m-%d_%H%M%S)_${LABEL}"; mkdir -p "${OUT}"
busy="$(on_board 'ps -eo user=,pcpu=,comm=' | awk '$1 != "jlei" && $1 != "root" && $2 > 20 {print $1 ":" $3 ":" $2}' | tr '\n' ' ')" \
    || busy="unreadable"
[ -n "${busy}" ] && { echo "board ${BOARD} busy: ${busy}" | tee "${OUT}/busy.txt"; exit 3; }

# ours
printf '# issue #95 probe (one shape)\nP0 %s %s %s\n' "${M}" "${N}" "${K}" > "${HERE}/datasets/experiments/issue95_probe.dat"
printf '[\n    ["fp32", "AXPY_FP32_Pack", [%s, %s], 256, "jbpi2", ["experiments/issue95_probe.dat"], false, 1000]\n]\n' "${MR}" "${NR}" > "${OUT}/config.json"
TMIN_ARGS=(--tmin "${TMIN}"); [[ " ${EXTRA[*]} " == *" --instr "* ]] && TMIN_ARGS=()  # --instr runs the warm-up protocol
before="$(ls -d "${HERE}"/logs/run_* | sort | tail -1)"
( cd "${HERE}/../.." && "${HERE}/compile_macro_bench.sh" --jump --riscv-user jlei --riscv-port 22 --riscv-ip "${BOARD}" \
    --config "${OUT}/config.json" --no-nc-overrides "${TMIN_ARGS[@]}" --cpu 4 "${EXTRA[@]}" ) > "${OUT}/ours.log" 2>&1
run="$(ls -d "${HERE}"/logs/run_* | sort | tail -1)"
[ "${run}" != "${before}" ] || { echo "no new run folder; see ${OUT}/ours.log"; exit 1; }
echo "$(basename "${run}")" > "${OUT}/ours_run.txt"
ours="$(cd "${HERE}/readers" && python3 -c "
import sys; from pathlib import Path
from reps_report import case_stats, read_reps, reps_files
for f in reps_files(Path(sys.argv[1])):
    for (_l, _nc), c in read_reps(f).items(): print(f'{case_stats(c.ops, c.seconds).gops_mean_time:.3f}')
" "${run}")"
grep -q ",PASS," "${run}"/all_perf_*.txt || echo "WARNING: ours not PASS (see ${run})"

# Héctor's
on_board "mkdir -p ~/issue95 && printf '0\t${M}\t${N}\t${K}\n' > ~/issue95/probe.dat && cd ~/hector_gemmlib_b292c92 && \
export OMP_NESTEED=FALSE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && LD_LIBRARY_PATH=\$(pwd)/install/lib \
LD_PRELOAD=\$(pwd)/install/lib/libGEMM.so taskset -c 4 ./build/gemm.x '' C C C N N 1.0 0.0 ${TMIN} T ~/issue95/probe.dat \
~/issue95/probe.csv 1 1 0 0 0 0 0 0 && echo ===CSV=== && cat ~/issue95/probe.csv" > "${OUT}/his.raw" 2>&1
sed '/^===CSV===$/,$d' "${OUT}/his.raw" > "${OUT}/his.log"; sed '1,/^===CSV===$/d' "${OUT}/his.raw" > "${OUT}/his.csv"; rm -f "${OUT}/his.raw"
his="$(awk -F';' 'NR > 1 && $2 != "" {printf "%.3f", 2 * $2 * $3 * $4 / $6 / 1e9}' "${OUT}/his.csv")"
grep -q OK "${OUT}/his.log" || echo "WARNING: his check not OK (see ${OUT}/his.log)"

line="$(python3 -c "
o, h = float('${ours:-0}'), float('${his:-0}')
r = o / h if h else 0
print(f'${M}x${N}x${K} tile ${TILE} ours {o:.2f} his {h:.2f} ratio {r:.2f} ' + ('RED' if r < 0.8 else 'GREEN'))")"
echo "${line}" | tee "${OUT}/result.txt"
