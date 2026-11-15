#!/bin/bash

# ============================================================
# run_issue51.sh -- how small does M have to be? Sweep M at fixed tiles (issue #51)
#
# One LAUNCH is one run per TILE on one board, back to back on core 4: config_issue51_<tile>.json
# (one pinned row each, all five over datasets/issue51_m_sweep.dat: M in {64, 128, 192, 256, 384,
# 512, 1000} x K in {384, 256}, plus B2 as the steady control), uninstrumented, reference
# protocol, vector packer, --no-nc-overrides. Five tiles: 16x6 and 32x6 (steady on B3), 16x8 and
# 16x15 (two-speed on B3, 16x15 the shipped one), 32x7 (two-speed on B3, steady on the large
# shapes, #47).
#
# The tiles rotate from launch to launch so no tile is always first or last (house rule 2): with
# tiles a b c d e, launch 1 runs a b c d e, launch 2 b c d e a, launch 3 c d e a b, launch 4
# d e a b c, launch 5 e a b c d -- a full cycle needs as many launches as there are tiles, so the
# default --launches is 5 (the default tile count), not house rule 2's bare minimum of 3: with
# only 3 launches and 5 tiles, two tiles (the two this issue most wants isolated, 16x15 and 32x7)
# would never run first, and a different two never last. Each tile is its own config (one
# project each), not bundled into one multi-row sweep, specifically so this rotation is real --
# #50's review found that a bundled multi-row config cannot have its row order rotated between
# launches without rebuilding the whole sweep machinery; this script avoids that gap from the
# start. A custom --tiles with a different count should get a matching --launches.
#
# The boards take turns launch by launch, one run at a time, under the lock the other sweeps use
# (they share the host's pj_gen/ and one /nfs directory).
#
# Around every run the board's processes are read (`ps`, other users only). Another user's job is
# NOT a reason to skip a run (2026-09-29 rule): the run goes ahead, the load is written to
# load_warnings.txt, and reps_report.py prints a RED NOTE for every run folder whose provenance.txt
# records it. Numbers that decide something come from a quiet window (midnight): use --start-at 00:00.
#
# Board time (one launch, one board): about 10 min per tile (15 shapes, uninstrumented, tmin 10) --
# about 50 min for all five tiles, so about 4h10 for the default 5 launches on one board.
#
# Logs: logs/issue51_<stamp>/ : plan.txt, runs.tsv (board, launch, position, tile, run folder,
# exit), one log per run, load_<board>_L<n>_<tile>_{before,after}.txt, load_warnings.txt. The run
# folders are the usual logs/run_*.
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/run_issue51.sh --start-at 00:00 \
#       > tests/pure_xdsl_blis_rvv/logs/issue51.out 2>&1 < /dev/null &
#
# Flags:
#   --tiles a,b           tiles (MRxNR, matching a config_issue51_<tile>.json each), in this
#                          order on launch 1 (default 16x6,32x6,16x8,16x15,32x7)
#   --boards a,b           boards that take turns (default jbpi,jbpi2)
#   --launches N           launches per board (default 5, matching the default tile count so the
#                          rotation's full cycle completes; house rule 2 needs at least 3)
#   --start-at HH:MM       wait until this time (today, or tomorrow if it has passed); default now
#   --tmin S | --cpu N     as for the other scripts (defaults 10 and core 4; core 0 needs --allow-core-0)
#   --workspace NAME       the workspace directory on the board (compile_macro_bench.sh --riscv-workspace). The boards
#                          share one NFS home, so two series running at once, one per board, each need their own
#                          (and their own checkout of the repo on this host: every run rewrites pj_gen/)
#   --direct               reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --dry-run              print the plan and the first launch's settings, then stop
#
# Jump-host settings come from the environment: REMOTE_PASS (default empty), REMOTE_HOST (default
# fej), REMOTE_PORT (default 3322). Do not start other benchmarks, `--dry-run` of a wrapper or the
# test suite on the host meanwhile: they rewrite pj_gen/ under a running series.
# ============================================================

# No `set -e`: a failed run must not end the series.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${ISSUE51_LOGS:-${HERE}/logs}"                                 # ISSUE51_* let the loop be tried against stubs
BENCH_RUN="${ISSUE51_RUN:-${HERE}/compile_macro_bench.sh}"
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"

TILES="16x6,32x6,16x8,16x15,32x7"
BOARDS="jbpi,jbpi2"
LAUNCHES=5
START_AT=""
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN=10
WORKSPACE=""
MODE_ARG="--jump"
RISCV_USER="jlei"
RISCV_PORT="22"
DRY_RUN=false

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tiles)         TILES="$2"; shift ;;
        --boards)        BOARDS="$2"; shift ;;
        --launches)      LAUNCHES="$2"; shift ;;
        --start-at)      START_AT="$2"; shift ;;
        --tmin)          TMIN="$2"; shift ;;
        --cpu)           CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)        CPU_ARGS=(--no-pin); NO_PIN=true ;;
        --allow-core-0)  ALLOW_CORE0=true ;;
        --workspace)     WORKSPACE="$2"; shift ;;
        --direct)        MODE_ARG="--direct" ;;
        --riscv-user)    RISCV_USER="$2"; shift ;;
        --riscv-port)    RISCV_PORT="$2"; shift ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)       sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" "${ALLOW_CORE0:-false}" || exit 1
[ "${ALLOW_CORE0:-false}" = true ] && CPU_ARGS+=(--allow-core-0)
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1 (got '${LAUNCHES}')"
[[ -z "${START_AT}" || "${START_AT}" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || die "--start-at takes HH:MM (got '${START_AT}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ "${#BOARD_LIST[@]}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"
IFS=',' read -ra TILE_LIST <<< "${TILES}"
[ "${#TILE_LIST[@]}" -ge 1 ] && [ -n "${TILE_LIST[0]}" ] || die "--tiles takes a comma-separated list"
for t in "${TILE_LIST[@]}"; do
    [[ "${t}" =~ ^[0-9]+x[0-9]+$ ]] || die "--tiles takes MRxNR entries (got '${t}')"
    [ -f "${HERE}/config_issue51_${t}.json" ] || die "no config_issue51_${t}.json for tile '${t}'"
done

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

# tile_rows <tile> -- the dataset's row count, the same for every tile (one shared dataset)
tile_rows() { grep -c '^[A-Za-z0-9_]' "${HERE}/datasets/issue51_m_sweep.dat"; }

# order_of <launch> -- the tiles of a launch, the list rotated left by launch-1
order_of() {
    local l="$1" n="${#TILE_LIST[@]}" i out=()
    for ((i = 0; i < n; i++)); do out+=("${TILE_LIST[$(( (i + l - 1) % n ))]}"); done
    echo "${out[*]}"
}

CONN=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}")

# build_cmd <tile> <board> -- sets CMD to the command line of one run (an array, so nothing is word-split)
build_cmd() {
    local tile="$1" board="$2"
    CMD=("${BENCH_RUN}" "${CONN[@]}" --riscv-ip "${board}" --config "config_issue51_${tile}.json" --no-nc-overrides "${CPU_ARGS[@]}" --tmin "${TMIN}")
    [ -n "${WORKSPACE}" ] && CMD+=(--riscv-workspace "${WORKSPACE}")
    return 0
}

# board_ps <board> -- the board's processes by CPU (user, %cpu, core, age, command), whoever's they are.
board_ps() {
    if [ -n "${ISSUE51_PS_CMD:-}" ]; then ${ISSUE51_PS_CMD} "$1"; return; fi
    local cmd='ps -eo user:20,pcpu,psr,etime,comm --sort=-pcpu --no-headers | head -12'
    if [ "${MODE_ARG}" = "--direct" ]; then
        ssh -p "${RISCV_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${RISCV_USER}@$1" "${cmd}"
    else
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            "${RISCV_USER}@${REMOTE_HOST}" "ssh -p ${RISCV_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '${cmd}'"
    fi
}

# snapshot <board> <tag> <before|after> -- reads the board's processes into the series dir and notes any other
# user's heavy job in load_warnings.txt. Never stops the run.
snapshot() {
    local f="${SERIES}/load_$1_$2_$3.txt" heavy
    board_ps "$1" > "${f}" 2>&1 || echo "(could not read the board's processes)" >> "${f}"
    heavy="$(awk -v me="${RISCV_USER}" "${OTHER_USERS_AWK}" "${f}")"
    if [ -n "${heavy}" ]; then
        echo "$(date -Is) $1 $2 ($3): ${heavy}" >> "${SERIES}/load_warnings.txt"
        echo "  RED NOTE: $1 $2 ($3): another user is running: ${heavy}"
    fi
}

TOTAL=$(( ${#BOARD_LIST[@]} * LAUNCHES * ${#TILE_LIST[@]} ))
NOW="$(date +%s)"; T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi

print_plan() {
    echo "Issue #51 series: ${LAUNCHES} launches on each of ${BOARDS}, boards taking turns, tiles ${TILES} (${TOTAL} runs)"
    echo "  Start     : $(date -d "@${T0}" '+%a %d %b %H:%M')"
    echo "  Settings  : ${CPU_ARGS[*]} --no-nc-overrides --tmin ${TMIN}, vector packer"
    echo "  Dataset   : datasets/issue51_m_sweep.dat ($(tile_rows) shapes: M in {64,128,192,256,384,512,1000} x K in {384,256}, plus B2)"
    echo "  Load      : other users' processes are read before and after every run; a heavy one is flagged red, not a stop"
    local l
    for ((l = 1; l <= LAUNCHES; l++)); do echo "  Launch ${l}   : $(order_of "${l}")"; done
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    echo ""; echo "=== the first launch's runs (${BOARD_LIST[0]}) ==="
    for t in $(order_of 1); do build_cmd "${t}" "${BOARD_LIST[0]}"; echo "  ${t}: ${CMD[*]}"; done
    echo ""; echo "=== the first run's own overview ==="
    build_cmd "$(order_of 1 | cut -d' ' -f1)" "${BOARD_LIST[0]}"
    "${CMD[@]}" --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/issue51_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'board\tlaunch\tposition\ttile\tresult_dir\texit\n' > "${SERIES}/runs.tsv"

if [ "${T0}" -gt "$(date +%s)" ]; then
    echo ""; echo "[$(date +%H:%M)] waiting $(( (T0 - $(date +%s) + 59) / 60 )) min for $(date -d "@${T0}" +%H:%M)"
    sleep $(( T0 - $(date +%s) ))
fi

# newest_since <stamp> -- the newest logs/run_<date>_<time> folder made since the stamp. Only stamped names count.
newest_since() {
    local d
    d="$(ls -d "${LOGS}"/run_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1)"
    if [ -z "${d}" ] || [[ "$(basename "${d}")" < "run_$1" ]]; then return 1; fi
    echo "${d}"
}

# complete <dir> -- a finished run holds one per-rep file for the (single-row) config's project
complete() {
    [ "$(ls "$1"/*/reps_*.txt* 2>/dev/null | wc -l)" -eq 1 ]
}

OK=0; FAILED=()
for ((l = 1; l <= LAUNCHES; l++)); do
  for board in "${BOARD_LIST[@]}"; do
    echo ""; echo "[$(date +%H:%M)] launch ${l}/${LAUNCHES} on ${board}: $(order_of "${l}")"
    pos=0
    for t in $(order_of "${l}"); do
        pos=$((pos + 1))
        tag="L${l}_${pos}_${t}"
        log="${SERIES}/${board}_${tag}.log"
        start="$(date +%Y-%m-%d_%H%M%S)"
        build_cmd "${t}" "${board}"
        echo "[$(date +%H:%M)]   ${pos}/${#TILE_LIST[@]} ${t}: log ${log}"
        (
            flock -w 21600 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
            echo "$(date -Is) got the lock for ${board}"
            snapshot "${board}" "${tag}" before
            cd "${HERE}/../.." && "${CMD[@]}"
            rc=$?
            snapshot "${board}" "${tag}" after
            exit ${rc}
        ) 9>"${LOCK}" >> "${log}" 2>&1
        rc=$?
        dir="$(newest_since "${start}")"
        if [ "${rc}" -ne 0 ] || [ -z "${dir}" ] || ! complete "${dir}"; then
            FAILED+=("launch ${l} ${board} ${t} (exit ${rc})")
            echo "    FAILED or incomplete (exit ${rc}); see ${log}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "${t}" "${dir:+$(basename "${dir}")}" "FAIL(${rc})" | sed 's/\t\t/\t-\t/' >> "${SERIES}/runs.tsv"
            continue
        fi
        OK=$((OK + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "${t}" "$(basename "${dir}")" 0 >> "${SERIES}/runs.tsv"
        echo "    ok: $(basename "${dir}")"
    done
  done
done

echo ""; echo "Series finished: ${OK} of ${TOTAL} runs ok. Run folders: ${SERIES}/runs.tsv"
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"
if [ -s "${SERIES}/load_warnings.txt" ]; then
    echo ""; echo "RED NOTE: another user's job was on a board during this series (${SERIES}/load_warnings.txt):"
    sed 's/^/  /' "${SERIES}/load_warnings.txt"
    echo "  These runs are not clean: repeat them in a quiet window (e.g. --start-at 00:00)."
fi
[ "${#FAILED[@]}" -eq 0 ]
