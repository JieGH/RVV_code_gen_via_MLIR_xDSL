#!/bin/bash

# ============================================================
# run_hector_gemm.sh -- Héctor's GEMMLib on the board, as he runs it (issue #62)
#
# Copies his source tree (--src, a clone at commit b292c92, NOT edited) to the board as ~/<--dir>
# (jbpi and jbpi2 share one NFS home, so one build serves both), builds it there with his Makefile and
# Makefile.inc as they are (gcc-14 -O3 -fopenmp, NQ_FP32, K1; `make clean && make && make install`, serial as in
# his README), then runs his gemm.x with the gemm.bin command line (TIMIN, TEST=T, gemm_study/yolo.dat). Two changes:
# LD_LIBRARY_PATH=install/lib is added (gemm.x needs libGEMM.so by name; his LD_PRELOAD names the same file), and
# the core list is ours (no --cpu flag: the cores are fixed by the #62 decisions, so pin_check is not used):
#   1 thread  : taskset -c 4    (his gemm.bin uses core 0; house rule 1)
#   8 threads : taskset -c 0-7  (core 0 allowed for the 8-thread run only: user, 2026-11-16)
# Launch order 1T 8T 8T 1T 1T 8T ... (A-B-B-A, house rule 2), --launches of each.
#
# Board, per launch (user, 2026-11-16): both boards' processes are read before every launch; the launch
# runs on an idle one, the last launch's board when both are idle. A busy board is written to
# side_notes.txt. With no idle board the series waits --wait-min minutes and reads again; it never
# starts under another user's load. The board used is read again after the launch (RED NOTE if busy).
# A launch that exits non-zero, prints ERROR (his correctness check) or leaves no CSV stops the series.
#
# Logs: logs/hector_gemm_<stamp>/: plan.txt, build.log, runs.tsv, L<n>_T<th>_<board>.{log,csv},
# load_<board>_L<n>.txt, side_notes.txt. His driver reports the mean time per call only, so his GOPs
# are "by mean time"; there is no per-rep time.
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_hector_gemm.sh --src <clone> \
#       > tests/pure_xdsl_blis_rvv/logs/hector_gemm.out 2>&1 < /dev/null &
#
# Flags:
#   --src DIR        his source tree (required; must be at commit b292c92)
#   --boards a,b     boards to pick from (default jbpi,jbpi2)
#   --launches N     launches per thread count (default 3)
#   --threads 1|8    run only that thread count, --launches of it (issue #63); default both, in the order above
#   --no-build       skip the copy and build: reuse ~/<--dir> from an earlier series (shared NFS home; #63)
#   --timin S        his TIMIN (default 2.0, as in gemm.bin)
#   --wait-min M     minutes between reads when no board is idle (default 10)
#   --dir NAME       board-side directory under ~ (default hector_gemmlib_b292c92)
#   --issue N        .scratch/reports/issue-<N>/ folder for the report and logs (default 62)
#   --dry-run        print the plan, then stop
# After every launch (and when the series stops): hector_gemm_report.py writes hector_gemm_report.md and
# hector_gemm_fig.png into the series folder, which is then copied to .scratch/reports/issue-<N>/<series>/
# (user, 2026-11-16).
# Jump host from the environment: REMOTE_HOST (default fej), REMOTE_PORT (default 3322), user jlei.
# ============================================================

set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${HECTOR_GEMM_LOGS:-${HERE}/logs}"          # HECTOR_GEMM_* let the loop be tried against a stub
WANT_COMMIT="b292c92"
source "${HERE}/board_pin.sh"                    # OTHER_USERS_AWK
source "${HERE}/hector_yolo_dump.sh"
PY="${HERE}/../../.venv/bin/python"; [ -x "${PY}" ] || PY=python3

SRC=""; BOARDS="jbpi,jbpi2"; LAUNCHES=3; TIMIN="2.0"; WAIT_MIN=10; DIR="hector_gemmlib_b292c92"; DRY_RUN=false; ISSUE=62; THREADS=""; BUILT=false
USER_ON_BOARD="jlei"

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --src)       SRC="$2"; shift ;;
        --boards)    BOARDS="$2"; shift ;;
        --launches)  LAUNCHES="$2"; shift ;;
        --timin)     TIMIN="$2"; shift ;;
        --wait-min)  WAIT_MIN="$2"; shift ;;
        --dir)       DIR="$2"; shift ;;
        --issue)     ISSUE="$2"; shift ;;
        --threads)   THREADS="$2"; shift ;;
        --no-build)  BUILT=true ;;
        --dry-run)   DRY_RUN=true ;;
        -h|--help)   sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

[ -d "${SRC}/gemm_study" ] || die "--src must be his GEMMLib_yolo tree (no gemm_study/ in '${SRC}')"
COMMIT="${HECTOR_GEMM_COMMIT:-$(git -C "${SRC}" rev-parse --short=7 HEAD 2>/dev/null)}"
[ "${COMMIT}" = "${WANT_COMMIT}" ] || die "--src is at '${COMMIT}', not ${WANT_COMMIT}"
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1"
[[ "${WAIT_MIN}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--wait-min takes minutes"
[[ -z "${THREADS}" || "${THREADS}" =~ ^[18]$ ]] || die "--threads takes 1 or 8 (got '${THREADS}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
REMOTE_HOST="${REMOTE_HOST:-fej}"; REMOTE_PORT="${REMOTE_PORT:-3322}"

# on_board <board> <command> -- runs the command in a login shell on the board, stdin passed through
on_board() {
    if [ -n "${HECTOR_GEMM_ON_BOARD:-}" ]; then "${HECTOR_GEMM_ON_BOARD}" "$1" "$2"; return; fi
    ssh -p "${REMOTE_PORT}" -o ConnectTimeout=15 "${USER_ON_BOARD}@${REMOTE_HOST}" \
        "ssh -p 22 -o ConnectTimeout=15 $1 $(printf '%q' "$2")"
}

# busy <board> <file> -- reads the board's processes into the file; prints another user's busy job, if any.
# A board that cannot be read counts as busy.
busy() {
    on_board "$1" 'ps -eo user:20,pcpu,psr,etime,comm --sort=-pcpu --no-headers | head -12' < /dev/null > "$2" 2>&1 \
        || { echo "cannot read the board's processes"; return; }
    awk -v me="${USER_ON_BOARD}" "${OTHER_USERS_AWK}" "$2"
}

# The order 1 8 8 1 1 8 8 1 ..., LAUNCHES of each.
ORDER=()
for ((i = 0; i < 2 * LAUNCHES; i++)); do
    case $(( i % 4 )) in 0|3) ORDER+=(1) ;; *) ORDER+=(8) ;; esac
done
if [ -n "${THREADS}" ]; then ORDER=(); for ((i = 0; i < LAUNCHES; i++)); do ORDER+=("${THREADS}"); done; fi
cpus_of() { [ "$1" = 1 ] && echo 4 || echo 0-7; }

print_plan() {
    echo "Issue #${ISSUE}: Héctor's GEMMLib @ ${WANT_COMMIT}, unedited, built and run on an idle board of ${BOARDS}"
    echo "  Build    : his Makefile + Makefile.inc (gcc-14 -O3 -fopenmp, NQ_FP32, K1), make clean && make && make install (serial, his README), in ~/${DIR}$([ "${BUILT}" = true ] && echo " -- skipped (--no-build), the existing build is used")"
    echo "  Run      : his gemm.bin command line, TIMIN=${TIMIN}, TEST=T, gemm_study/yolo.dat; no OMP_PROC_BIND (as his)"
    echo "  Cores    : 1 thread on core 4; 8 threads on cores 0-7 (core 0: user's exception, 8-thread run only)"
    echo "  Order    : ${ORDER[*]} (threads per launch)"
    echo "  Load     : both boards read before every launch; busy -> side note; none idle -> wait ${WAIT_MIN} min"
}

if [ "${DRY_RUN}" = true ]; then print_plan; echo ""; echo "  --dry-run: nothing was started."; exit 0; fi

STAMP="$(date +%Y-%m-%d_%H%M%S)"
SERIES="${LOGS}/hector_gemm_${STAMP}"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'launch\tthreads\tcpus\tboard\texit\n' > "${SERIES}/runs.tsv"
NOTES="${SERIES}/side_notes.txt"; : > "${NOTES}"

# pick_board <launch> -- prints the board for this launch; waits while none is idle. Messages go to fd 3
# (the series' stdout), since stdout is the answer.
exec 3>&1
LAST=""
pick_board() {
    local l="$1" b who pick
    while true; do
        pick=""
        for b in "${BOARD_LIST[@]}"; do
            who="$(busy "${b}" "${SERIES}/load_${b}_L${l}.txt")"
            if [ -n "${who}" ]; then
                echo "[$(date +%H:%M)] launch ${l}: ${b} busy: ${who}" >> "${NOTES}"
                echo "[$(date +%H:%M)] launch ${l}: ${b} busy: ${who}" >&3
            elif [ -z "${pick}" ] || [ "${b}" = "${LAST}" ]; then
                pick="${b}"
            fi
        done
        [ -n "${pick}" ] && { echo "${pick}"; return; }
        echo "[$(date +%H:%M)] launch ${l}: no idle board; reading again in ${WAIT_MIN} min" >&3
        sleep "$(awk -v m="${WAIT_MIN}" 'BEGIN { print m * 60 }')"
    done
}

# report_and_dump -- the report and figure over the launches so far, then the copy to .scratch/reports/
report_and_dump() {
    "${PY}" "${HERE}/hector_gemm_report.py" "${SERIES}" -o "${SERIES}/hector_gemm_report.md" \
        --plot "${SERIES}/hector_gemm_fig.png" || echo "report failed: ${SERIES}"
    hector_yolo_dump "${ISSUE}" "${SERIES}"
}

OK=0; TOTAL=${#ORDER[@]}
for ((l = 1; l <= TOTAL; l++)); do
    th="${ORDER[$((l - 1))]}"; cpus="$(cpus_of "${th}")"
    board="$(pick_board "${l}")"
    LAST="${board}"

    if [ "${BUILT}" = false ]; then
        echo "[$(date +%H:%M)] copying ${SRC} to ${board}:~/${DIR} and building (his Makefile)"
        tar -C "${SRC}" --exclude=.git -cf - . \
            | on_board "${board}" "mkdir -p ~/${DIR} && tar -C ~/${DIR} -xf -" > "${SERIES}/build.log" 2>&1 \
            || die "copy to ${board} failed: ${SERIES}/build.log"
        on_board "${board}" "cd ~/${DIR} && gcc-14 --version | head -1 && make clean && make && make install" \
            < /dev/null >> "${SERIES}/build.log" 2>&1 || die "build on ${board} failed: ${SERIES}/build.log"
        BUILT=true
    fi

    name="L${l}_T${th}_${board}"
    echo "[$(date +%H:%M)] launch ${l}/${TOTAL}: ${th} thread(s) on cores ${cpus} of ${board}: ${SERIES}/${name}.log"
    # OMP_NESTEED (sic) is his gemm.bin's spelling, kept as he runs it
    cmd="cd ~/${DIR} && mkdir -p output && export OMP_NESTEED=FALSE OPENBLAS_NUM_THREADS=${th} OMP_NUM_THREADS=${th}"
    # LD_LIBRARY_PATH: gemm.x NEEDs libGEMM.so by name, which his LD_PRELOAD alone does not satisfy (same file)
    cmd+=" && LD_LIBRARY_PATH=\$(pwd)/install/lib LD_PRELOAD=\$(pwd)/install/lib/libGEMM.so taskset -c ${cpus} ./build/gemm.x \"\" C C C N N 1.0 0.0"
    cmd+=" ${TIMIN} T gemm_study/yolo.dat output/${name}.csv ${th} 1 0 0 0 0 0 0"
    cmd+=" && echo ===CSV=== && cat output/${name}.csv"
    on_board "${board}" "${cmd}" < /dev/null > "${SERIES}/${name}.raw" 2>&1
    rc=$?
    sed '/^===CSV===$/,$d' "${SERIES}/${name}.raw" > "${SERIES}/${name}.log"
    sed '1,/^===CSV===$/d' "${SERIES}/${name}.raw" > "${SERIES}/${name}.csv"
    rm -f "${SERIES}/${name}.raw"

    after="$(busy "${board}" "${SERIES}/load_${board}_L${l}_after.txt")"
    [ -n "${after}" ] && echo "RED NOTE: another user's job was on ${board} at the end of launch ${l}: ${after}" | tee -a "${NOTES}"

    why=""
    [ "${rc}" -ne 0 ] && why="exit ${rc}"
    grep -q 'ERROR' "${SERIES}/${name}.log" && why="${why:+${why}, }his check printed ERROR"
    [ -s "${SERIES}/${name}.csv" ] || why="${why:+${why}, }no CSV"
    printf '%s\t%s\t%s\t%s\t%s\n' "${l}" "${th}" "${cpus}" "${board}" "$([ -n "${why}" ] && echo "FAIL(${why})" || echo 0)" \
        >> "${SERIES}/runs.tsv"
    report_and_dump
    if [ -n "${why}" ]; then
        echo "launch ${l} (${why}): ${SERIES}/${name}.log"
        echo "Series finished: ${OK} of ${TOTAL} launches ok. Stopping the series."
        exit 1
    fi
    OK=$((OK + 1))
done
echo "Series finished: ${OK} of ${TOTAL} launches ok. Logs: ${SERIES}"
