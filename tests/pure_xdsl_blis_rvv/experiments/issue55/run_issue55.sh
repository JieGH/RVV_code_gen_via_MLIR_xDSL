#!/bin/bash

# ============================================================
# run_issue55.sh -- the B packer read-order variants (--pack-b) on the three tiles of #54, issue #55
#
# One LAUNCH is one run of every selected ARM on one board, back to back on core 4. An arm is one
# compile_macro_bench.sh call over config_issue55_tiles.json (pinned 16x15, 16x8, 16x6, each on the four shapes of
# datasets/b3_controls.dat: B3, B3_K256, B2, R24), always uninstrumented, reference protocol, --no-nc-overrides
# --cpu 4 --tmin 10, plus the arm's packer:
#   row        the shipped xDSL vector packer (no --pack-b)
#   row-intr   --pack-b row-intr      today's order in C intrinsics (the control)
#   scalar     --scalar-pack          the scalar packer
#   col-g1 col-g2 col-g4 col-gall col-g1-m4    --pack-b <name>
# The arms rotate left from launch to launch (a b c / b c a / c a b), as run_issue48.sh does.
# The boards take turns launch by launch, one run at a time, under the lock the other sweeps use (they share the host's
# pj_gen/ and one /nfs directory).
#
# --pack-only adds, after the GEMM arms of every launch, the pack-only loop (pack_only/pack_b_only.cpp) for each chosen
# variant except `row` (not linked there): B3 (64 384 384) and B3_K256 (64 384 256) at 16x8 and 16x15, 1000 reps, core 4.
# pack_only/, templates/pack_b_variants.h and modelLevel/ are copied to the board workspace (rsync, the same connection
# helpers as compile_macro_bench.sh) and built there with the board's g++ at the start of each launch's pack-only part.
# scalar_pack.inc (the scalar twin) is generated on this host from render_scalar_pack_functions before the copy: the
# board needs no python or xdsl.
#
# IDLE GATE (2026-10-22 rule: never time under load). Before every run, on that board, two readings of
# `top -bn2 -d1` (the SECOND sample: an instant per-process %CPU, not `ps`'s lifetime average and not the load average,
# which sits at 2.00 on both boards from two kernel threads in D state) taken --idle-gap seconds apart. The board is idle
# when no process of a user other than the runner and root is at --idle-proc-max % of a core or more, and the overall CPU
# idle is at least --idle-min %. Both readings must pass. If not: the reason is written to idle_waits.txt, the script waits
# --idle-poll seconds and tries again, for up to --idle-max-wait seconds per run; then it STOPS the series (exit 2) rather
# than run under load. After every run one more reading is taken: if it fails the run is marked discarded-load in runs.tsv
# and queued once more at the end of that launch (a second discard stays discarded). Root's processes (bluetoothd) are
# recorded, never block. Other users' processes are never touched. The gate runs inside the lock, so a waiting series holds
# the lock; other series on this host wait behind it.
#
# Board time (estimates, not measured here): about 2.4 min per tile per arm (3 tiles x 4 shapes) for a GEMM arm; the pack-only
# loop about 1 min per variant per case (4 cases); the gate adds about (idle-gap + 10 s) per run. --dry-run prints the sum.
#
# Logs: logs/issue55_<stamp>/ : plan.txt, runs.tsv (board, launch, position, arm, run folder, exit, status),
# packonly.tsv (the pack-only runs), one log per run, idle_readings.txt (every reading), idle_<board>_<tag>_<n>.txt
# (each reading's top sample), idle_waits.txt, load_warnings.txt, packonly_*.txt (the loop's output).
# status is ok / failed / discarded-load. Read the series with issue55_report.py.
#
# Usage (from the project root; detach it so it survives the terminal):
#   setsid nohup ./tests/pure_xdsl_blis_rvv/experiments/issue55/run_issue55.sh --start-at 00:00 --variants row,row-intr,scalar,col-gall \
#       --pack-only > tests/pure_xdsl_blis_rvv/logs/issue55.out 2>&1 < /dev/null &
#
# Flags:
#   --variants a,b,c    arms to run, in this order on launch 1 (default row,row-intr)
#                       names: row row-intr scalar col-g1 col-g2 col-g4 col-gall col-g1-m4
#   --pack-only         also run the pack-only loop (see above) for every chosen variant but `row`
#   --po-reps N         reps of the pack-only loop (default 1000)
#   --boards a,b        boards that take turns (default jbpi,jbpi2)
#   --launches N        launches per board (default 3; house rule 2 needs at least 3)
#   --start-at HH:MM    wait until this time (today, or tomorrow if it has passed); default now
#   --tmin S | --cpu N  as for the other scripts (defaults 10 and core 4; core 0 needs --allow-core-0)
#   --idle-gap S        seconds between the two readings of the gate (default 60)
#   --idle-poll S       seconds between gate attempts while the board is busy (default 300)
#   --idle-max-wait S   give up on a run after waiting this long for the board to go idle (default 10800)
#   --idle-min PCT      overall CPU idle % needed (default 85)
#   --idle-proc-max PCT another user's process at or above this % of a core makes the board busy (default 20)
#   --workspace NAME    workspace directory on the board (compile_macro_bench.sh --riscv-workspace); the boards share one
#                       NFS home, so two series running at once each need their own (and their own checkout of this repo)
#   --direct            reach the boards directly instead of through the jump host
#   --riscv-user U --riscv-port P    (defaults jlei, 22)
#   --dry-run           print the plan and the first run's settings, then stop (no board call)
#
# Jump-host settings come from the environment: REMOTE_PASS (default empty), REMOTE_HOST (default fej),
# REMOTE_PORT (default 3322). Do not start other benchmarks, a wrapper's `--dry-run` or the test suite on the host
# meanwhile: they rewrite pj_gen/ under a running series.
#
# Test hooks: ISSUE55_LOGS, ISSUE55_RUN (stands for compile_macro_bench.sh), BOARD_TOP_CMD <board> (stands for the
# board's `top -bn2 -d1`), ISSUE55_PO (stands for the pack-only board work: `prepare <board>` and
# `run <board> M N K MR NR variant reps cpu`, output on stdout).
# ============================================================

# No `set -e`: a failed run must not end the series.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS="${ISSUE55_LOGS:-${HERE}/logs}"
BENCH_RUN="${ISSUE55_RUN:-${HERE}/compile_macro_bench.sh}"
LOCK="${LOGS}/.cron_bench.lock"
source "${HERE}/board_pin.sh"

ALL_VARIANTS="row row-intr scalar col-g1 col-g2 col-g4 col-gall col-g1-m4"
CONFIG="experiments/issue55/config_issue55_tiles.json"
ARMS="row,row-intr"
PACK_ONLY=false
PO_REPS=1000
BOARDS="jbpi,jbpi2"
LAUNCHES=3
START_AT=""
CPU_ARGS=(--cpu 4)
CPU_FLAG=false
NO_PIN=false
CPU=""
TMIN=10
IDLE_GAP=60
IDLE_POLL=300
IDLE_MAX_WAIT=10800
IDLE_MIN=85
IDLE_PROC_MAX=20
WORKSPACE=""
MODE_ARG="--jump"
RISCV_USER="jlei"
RISCV_PORT="22"
DRY_RUN=false

die() { echo "Error: $*" >&2; exit 1; }

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --variants)       ARMS="$2"; shift ;;
        --pack-only)      PACK_ONLY=true ;;
        --po-reps)        PO_REPS="$2"; shift ;;
        --boards)         BOARDS="$2"; shift ;;
        --launches)       LAUNCHES="$2"; shift ;;
        --start-at)       START_AT="$2"; shift ;;
        --tmin)           TMIN="$2"; shift ;;
        --cpu)            CPU_ARGS=(--cpu "$2"); CPU="$2"; CPU_FLAG=true; shift ;;
        --no-pin)         CPU_ARGS=(--no-pin); NO_PIN=true ;;
        --allow-core-0)   ALLOW_CORE0=true ;;
        --idle-gap)       IDLE_GAP="$2"; shift ;;
        --idle-poll)      IDLE_POLL="$2"; shift ;;
        --idle-max-wait)  IDLE_MAX_WAIT="$2"; shift ;;
        --idle-min)       IDLE_MIN="$2"; shift ;;
        --idle-proc-max)  IDLE_PROC_MAX="$2"; shift ;;
        --workspace)      WORKSPACE="$2"; shift ;;
        --direct)         MODE_ARG="--direct" ;;
        --riscv-user)     RISCV_USER="$2"; shift ;;
        --riscv-port)     RISCV_PORT="$2"; shift ;;
        --dry-run)        DRY_RUN=true ;;
        -h|--help)        sed -n '3,/^# ====*$/p' "$0" | sed '$d'; exit 0 ;;
        *) die "unknown parameter: $1" ;;
    esac
    shift
done

pin_check "${CPU_FLAG}" "${NO_PIN}" "${CPU}" "${ALLOW_CORE0:-false}" || exit 1
[ "${ALLOW_CORE0:-false}" = true ] && CPU_ARGS+=(--allow-core-0)
[[ "${LAUNCHES}" =~ ^[1-9][0-9]*$ ]] || die "--launches takes a whole number of at least 1 (got '${LAUNCHES}')"
[[ "${PO_REPS}" =~ ^[1-9][0-9]*$ ]] || die "--po-reps takes a whole number of at least 1 (got '${PO_REPS}')"
for v in "${IDLE_GAP}" "${IDLE_POLL}" "${IDLE_MAX_WAIT}" "${IDLE_MIN}" "${IDLE_PROC_MAX}"; do
    [[ "${v}" =~ ^[0-9]+$ ]] || die "the --idle-* flags take whole numbers (got '${v}')"
done
[[ -z "${START_AT}" || "${START_AT}" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || die "--start-at takes HH:MM (got '${START_AT}')"
IFS=',' read -ra BOARD_LIST <<< "${BOARDS}"
[ "${#BOARD_LIST[@]}" -ge 1 ] && [ -n "${BOARD_LIST[0]}" ] || die "--boards takes a comma-separated list"
IFS=',' read -ra ARM_LIST <<< "${ARMS}"
[ "${#ARM_LIST[@]}" -ge 1 ] && [ -n "${ARM_LIST[0]}" ] || die "--variants takes a comma-separated list"
for a in "${ARM_LIST[@]}"; do
    case " ${ALL_VARIANTS} " in *" ${a} "*) ;; *) die "--variants takes ${ALL_VARIANTS// /, } (got '${a}')" ;; esac
done
# The pack-only variants: the chosen ones except `row`.
PO_LIST=()
for a in "${ARM_LIST[@]}"; do [ "${a}" != row ] && PO_LIST+=("${a}"); done
if [ "${PACK_ONLY}" = true ] && [ "${#PO_LIST[@]}" -eq 0 ]; then
    die "--pack-only needs at least one variant other than row in --variants (row is not linked into the loop)"
fi

export REMOTE_PASS="${REMOTE_PASS-}" REMOTE_HOST="${REMOTE_HOST:-fej}" REMOTE_PORT="${REMOTE_PORT:-3322}"

# The pack-only cases: <name>:M:N:K and <name>:MR:NR
PO_SHAPES=("B3:64:384:384" "B3_K256:64:384:256")
PO_TILES=("16x8:16:8" "16x15:16:15")

# arm_rows -- how many tiles the config has: a finished run holds one per-rep file each
arm_rows() { grep -c '^[[:space:]]*\["fp32"' "${HERE}/${CONFIG}"; }

# order_of <launch> -- the arms of a launch, the list rotated left by launch-1
order_of() {
    local l="$1" n="${#ARM_LIST[@]}" i out=()
    for ((i = 0; i < n; i++)); do out+=("${ARM_LIST[$(( (i + l - 1) % n ))]}"); done
    echo "${out[*]}"
}
# po_order_of <launch> -- the same for the pack-only variants
po_order_of() {
    local l="$1" n="${#PO_LIST[@]}" i out=()
    for ((i = 0; i < n; i++)); do out+=("${PO_LIST[$(( (i + l - 1) % n ))]}"); done
    echo "${out[*]}"
}

CONN=("${MODE_ARG}" --riscv-user "${RISCV_USER}" --riscv-port "${RISCV_PORT}")

# build_cmd <arm> <board> -- sets CMD to the command line of one GEMM run (an array, so nothing is word-split)
build_cmd() {
    local arm="$1" board="$2"
    CMD=("${BENCH_RUN}" "${CONN[@]}" --riscv-ip "${board}" --config "${CONFIG}" --no-nc-overrides "${CPU_ARGS[@]}" --tmin "${TMIN}")
    case "${arm}" in
        row)    ;;
        scalar) CMD+=(--scalar-pack) ;;
        *)      CMD+=(--pack-b "${arm}") ;;
    esac
    [ -n "${WORKSPACE}" ] && CMD+=(--riscv-workspace "${WORKSPACE}")
    return 0
}

# ---- board access (the same forms compile_macro_bench.sh uses; no new connection path) ----
REMOTE_ROOT="${WORKSPACE:-xdsl_rvv_microkernel}/tests/pure_xdsl_blis_rvv"
# one_reading <board> <tag> <phase> -- takes a reading, records it; returns 0 if the board is idle, 1 if not
# (the reason is left in READ_WHY)
# ---- the pack-only loop on a board ----
po_prepare() {  # <board>
    if [ -n "${ISSUE55_PO:-}" ]; then ${ISSUE55_PO} prepare "$1"; return; fi
    local P="${HERE}"
    ${PYTHON:-python3} "${P}/pack_only/gen_scalar_inc.py" || return 1
    board_ssh "$1" "mkdir -p ${REMOTE_ROOT}/pack_only ${REMOTE_ROOT}/templates ${REMOTE_ROOT}/modelLevel" || return 1
    board_rsync "$1" --exclude=pack_b_only --exclude='*.o' "${P}/pack_only/" "@T@:${REMOTE_ROOT}/pack_only/" || return 1
    board_rsync "$1" "${P}/templates/pack_b_variants.h" "@T@:${REMOTE_ROOT}/templates/" || return 1
    board_rsync "$1" "${P}/modelLevel/" "@T@:${REMOTE_ROOT}/modelLevel/" || return 1
    board_ssh "$1" "cd ${REMOTE_ROOT}/pack_only && make clean && make" || return 1
}
po_run() {  # <board> M N K MR NR variant reps cpu -- the loop's output on stdout
    local board="$1"; shift
    if [ -n "${ISSUE55_PO:-}" ]; then ${ISSUE55_PO} run "${board}" "$@"; return; fi
    board_ssh "${board}" "cd ${REMOTE_ROOT}/pack_only && ./pack_b_only $*"
}

TOTAL_GEMM=$(( ${#BOARD_LIST[@]} * LAUNCHES * ${#ARM_LIST[@]} ))
PO_PER_LAUNCH=0
[ "${PACK_ONLY}" = true ] && PO_PER_LAUNCH=$(( ${#PO_LIST[@]} * ${#PO_SHAPES[@]} * ${#PO_TILES[@]} ))
TOTAL_PO=$(( ${#BOARD_LIST[@]} * LAUNCHES * PO_PER_LAUNCH ))
NOW="$(date +%s)"; T0="${NOW}"
if [ -n "${START_AT}" ]; then
    T0="$(date -d "today ${START_AT}" +%s)" || die "cannot read --start-at ${START_AT}"
    [ "${T0}" -lt "${NOW}" ] && T0="$(date -d "tomorrow ${START_AT}" +%s)"
fi

print_plan() {
    local runs_per_board=$(( LAUNCHES * (${#ARM_LIST[@]} + PO_PER_LAUNCH) ))
    echo "Issue #55 series: ${LAUNCHES} launches on each of ${BOARDS}, boards taking turns, variants ${ARMS} (${TOTAL_GEMM} GEMM runs, ${TOTAL_PO} pack-only runs)"
    echo "  Start     : $(date -d "@${T0}" '+%a %d %b %H:%M')"
    echo "  Settings  : ${CPU_ARGS[*]} --no-nc-overrides --tmin ${TMIN}, uninstrumented, reference protocol, config ${CONFIG} ($(arm_rows) tiles x 4 shapes)"
    echo "  Idle gate : before every run, 2 readings of top -bn2 -d1 (second sample) ${IDLE_GAP} s apart; idle = no other user's process >= ${IDLE_PROC_MAX}% of a core and overall idle >= ${IDLE_MIN}%;"
    echo "              busy -> wait ${IDLE_POLL} s and retry up to ${IDLE_MAX_WAIT} s, then stop the series; one reading after each run, a busy one discards it (re-queued once at the end of the launch)"
    local l
    for ((l = 1; l <= LAUNCHES; l++)); do echo "  Launch ${l}   : $(order_of "${l}")"; done
    if [ "${PACK_ONLY}" = true ]; then
        echo "  Pack-only : after the GEMM arms of each launch; ${PO_REPS} reps, core 4, shapes B3 (64 384 384) and B3_K256 (64 384 256), tiles 16x8 and 16x15"
        for ((l = 1; l <= LAUNCHES; l++)); do echo "  Pack-only launch ${l}: $(po_order_of "${l}")"; done
    else
        echo "  Pack-only : not run (add --pack-only)"
    fi
    local gate_min gemm_min po_min
    gate_min=$(awk -v g="${IDLE_GAP}" -v n="${runs_per_board}" 'BEGIN { printf "%.0f", n * (g + 10) / 60 }')
    gemm_min=$(awk -v a="${#ARM_LIST[@]}" -v t="$(arm_rows)" -v l="${LAUNCHES}" 'BEGIN { printf "%.0f", a * t * 2.4 * l }')
    po_min=$(( LAUNCHES * PO_PER_LAUNCH ))
    echo "  Board time per board (estimate, nothing measured): GEMM ${gemm_min} min (2.4 min per tile per arm) + pack-only ${po_min} min (1 min per variant per case) + idle gates ${gate_min} min"
    local per=$(( gemm_min + po_min + gate_min ))
    echo "    = about ${per} min ($(awk -v p="${per}" 'BEGIN { printf "%.1f", p / 60 }') h) per board; the boards take turns, so about $(( per * ${#BOARD_LIST[@]} )) min ($(awk -v p="$(( per * ${#BOARD_LIST[@]} ))" 'BEGIN { printf "%.1f", p / 60 }') h) for the series, plus any waiting for an idle board and re-queued runs"
}

if [ "${DRY_RUN}" = true ]; then
    print_plan
    echo ""; echo "=== the first launch's runs (${BOARD_LIST[0]}) ==="
    for a in $(order_of 1); do build_cmd "${a}" "${BOARD_LIST[0]}"; echo "  ${a}: ${CMD[*]}"; done
    if [ "${PACK_ONLY}" = true ]; then
        echo "  pack-only, per run: pack_b_only M N K MR NR <variant> ${PO_REPS} 4  (built on the board in ${REMOTE_ROOT}/pack_only)"
    fi
    echo ""; echo "=== the first run's own overview ==="
    build_cmd "$(order_of 1 | cut -d' ' -f1)" "${BOARD_LIST[0]}"
    "${CMD[@]}" --dry-run
    echo ""; echo "  --dry-run: nothing was started."
    exit 0
fi

mkdir -p "${LOGS}"
SERIES="${LOGS}/issue55_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${SERIES}"
print_plan | tee "${SERIES}/plan.txt"
printf 'board\tlaunch\tposition\tarm\tresult_dir\texit\tstatus\n' > "${SERIES}/runs.tsv"
printf 'board\tlaunch\tposition\tvariant\tshape\ttile\tfile\texit\tstatus\n' > "${SERIES}/packonly.tsv"

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

# complete <dir> -- a finished run holds one per-rep file for every tile of its config
complete() { [ "$(ls "$1"/*/reps_*.txt* 2>/dev/null | wc -l)" -eq "$(arm_rows)" ]; }

OK=0; FAILED=(); DISCARDED=(); GAVE_UP=""
# A unit is "g:<arm>" (a GEMM run) or "p:<variant>:<shape>:<tile>" (a pack-only run).
# run_unit <board> <launch> <pos> <unit> <retry 0|1> -- sets UNIT_STATUS (ok / failed / discarded-load) and UNIT_REQUEUE (1 = queue it again)
run_unit() {
    local board="$1" l="$2" pos="$3" unit="$4" retry="$5" kind arm tag log start rc dir status
    kind="${unit%%:*}"
    UNIT_REQUEUE=0
    rm -f "${SERIES}/.u_rc" "${SERIES}/.u_post" "${SERIES}/.u_gate"
    if [ "${kind}" = g ]; then
        arm="${unit#g:}"; tag="L${l}_${pos}_${arm}"; log="${SERIES}/${board}_${tag}.log"
        build_cmd "${arm}" "${board}"
        echo "[$(date +%H:%M)]   ${pos} ${arm}$([ "${retry}" = 1 ] && echo ' (re-queued)'): log ${log}"
        start="$(date +%Y-%m-%d_%H%M%S)"
        (
            flock -w 21600 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
            echo "$(date -Is) got the lock for ${board}"
            board_idle_gate "${board}" "${tag}" || { echo gave-up > "${SERIES}/.u_gate"; exit 98; }
            start="$(date +%Y-%m-%d_%H%M%S)"; echo "${start}" > "${SERIES}/.u_start"
            cd "${HERE}/../.." && "${CMD[@]}"
            echo $? > "${SERIES}/.u_rc"
            board_one_reading "${board}" "${tag}" post && echo ok > "${SERIES}/.u_post" || echo load > "${SERIES}/.u_post"
            exit 0
        ) 9>"${LOCK}" >> "${log}" 2>&1
        local lrc=$?
        [ -f "${SERIES}/.u_gate" ] && { GAVE_UP="${board} ${tag}"; UNIT_STATUS="gave-up"; return; }
        rc="$(cat "${SERIES}/.u_rc" 2>/dev/null || echo "${lrc}")"
        start="$(cat "${SERIES}/.u_start" 2>/dev/null || echo "${start}")"
        dir="$(newest_since "${start}")"
        if [ "${rc}" -ne 0 ] || [ "${lrc}" -ne 0 ] || [ -z "${dir}" ] || ! complete "${dir}"; then
            status=failed
            FAILED+=("launch ${l} ${board} ${arm} (exit ${rc})")
            echo "    FAILED or incomplete (exit ${rc}); see ${log}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "${arm}" "${dir:+$(basename "${dir}")}" "FAIL(${rc})" "${status}" | sed 's/\t\t/\t-\t/' >> "${SERIES}/runs.tsv"
        elif [ "$(cat "${SERIES}/.u_post" 2>/dev/null)" = load ]; then
            status=discarded-load
            DISCARDED+=("launch ${l} ${board} ${arm}")
            echo "    DISCARDED: the board was busy after the run (see idle_readings.txt)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "${arm}" "$(basename "${dir}")" 0 "${status}" >> "${SERIES}/runs.tsv"
            [ "${retry}" = 0 ] && UNIT_REQUEUE=1
        else
            status=ok; OK=$((OK + 1))
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "${arm}" "$(basename "${dir}")" 0 "${status}" >> "${SERIES}/runs.tsv"
            echo "    ok: $(basename "${dir}")"
        fi
        UNIT_STATUS="${status}"
    else
        local rest v shape tile sp tp M N K MR NR out
        rest="${unit#p:}"; v="${rest%%:*}"; rest="${rest#*:}"; shape="${rest%%:*}"; tile="${rest#*:}"
        for sp in "${PO_SHAPES[@]}"; do [ "${sp%%:*}" = "${shape}" ] && { IFS=: read -r _ M N K <<< "${sp}"; }; done
        for tp in "${PO_TILES[@]}"; do [ "${tp%%:*}" = "${tile}" ] && { IFS=: read -r _ MR NR <<< "${tp}"; }; done
        tag="L${l}_${pos}_po_${v}_${shape}_${tile}"; log="${SERIES}/${board}_${tag}.log"
        out="packonly_${board}_L${l}_${v}_${shape}_${tile}$([ "${retry}" = 1 ] && echo _retry).txt"
        echo "[$(date +%H:%M)]   ${pos} pack-only ${v} ${shape} ${tile}$([ "${retry}" = 1 ] && echo ' (re-queued)'): ${out}"
        (
            flock -w 21600 9 || { echo "$(date -Is) gave up waiting for the lock"; exit 99; }
            board_idle_gate "${board}" "${tag}" || { echo gave-up > "${SERIES}/.u_gate"; exit 98; }
            po_run "${board}" "${M}" "${N}" "${K}" "${MR}" "${NR}" "${v}" "${PO_REPS}" 4 > "${SERIES}/${out}" 2>> "${log}"
            echo $? > "${SERIES}/.u_rc"
            board_one_reading "${board}" "${tag}" post && echo ok > "${SERIES}/.u_post" || echo load > "${SERIES}/.u_post"
            exit 0
        ) 9>"${LOCK}" >> "${log}" 2>&1
        local lrc=$?
        [ -f "${SERIES}/.u_gate" ] && { GAVE_UP="${board} ${tag}"; UNIT_STATUS="gave-up"; return; }
        rc="$(cat "${SERIES}/.u_rc" 2>/dev/null || echo "${lrc}")"
        if [ "${rc}" -ne 0 ] || [ "${lrc}" -ne 0 ]; then
            status=failed
            FAILED+=("launch ${l} ${board} pack-only ${v} ${shape} ${tile} (exit ${rc})")
            echo "    FAILED (exit ${rc}); see ${log}"
        elif [ "$(cat "${SERIES}/.u_post" 2>/dev/null)" = load ]; then
            status=discarded-load
            DISCARDED+=("launch ${l} ${board} pack-only ${v} ${shape} ${tile}")
            echo "    DISCARDED: the board was busy after the run"
            [ "${retry}" = 0 ] && UNIT_REQUEUE=1
        else
            status=ok; OK=$((OK + 1)); echo "    ok: ${out}"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${board}" "${l}" "${pos}" "${v}" "${shape}" "${tile}" "${out}" "${rc}" "${status}" >> "${SERIES}/packonly.tsv"
        UNIT_STATUS="${status}"
    fi
}

stop_series() {
    echo ""; echo "STOPPING: the board never went idle (${GAVE_UP}); see ${SERIES}/idle_waits.txt. Nothing was run under load."
    echo "Series stopped: ${OK} runs ok so far. Run folders: ${SERIES}/runs.tsv"
    exit 2
}

for ((l = 1; l <= LAUNCHES; l++)); do
  for board in "${BOARD_LIST[@]}"; do
    echo ""; echo "[$(date +%H:%M)] launch ${l}/${LAUNCHES} on ${board}: $(order_of "${l}")"
    pos=0; QUEUE=()
    for a in $(order_of "${l}"); do
        pos=$((pos + 1))
        run_unit "${board}" "${l}" "${pos}" "g:${a}" 0
        [ "${UNIT_STATUS}" = gave-up ] && stop_series
        [ "${UNIT_REQUEUE}" = 1 ] && QUEUE+=("g:${a}")
    done
    if [ "${PACK_ONLY}" = true ]; then
        echo "[$(date +%H:%M)]   pack-only part: building on ${board}"
        prep_log="${SERIES}/${board}_L${l}_po_prepare.log"
        ( flock -w 21600 9 || exit 99; po_prepare "${board}" ) 9>"${LOCK}" >> "${prep_log}" 2>&1
        prc=$?
        if [ "${prc}" -ne 0 ]; then
            FAILED+=("launch ${l} ${board} pack-only build (exit ${prc})")
            echo "    pack-only build FAILED (exit ${prc}); see ${prep_log}; the pack-only runs of this launch are skipped"
        else
            for v in $(po_order_of "${l}"); do
                for sp in "${PO_SHAPES[@]}"; do
                    for tp in "${PO_TILES[@]}"; do
                        pos=$((pos + 1))
                        run_unit "${board}" "${l}" "${pos}" "p:${v}:${sp%%:*}:${tp%%:*}" 0
                        [ "${UNIT_STATUS}" = gave-up ] && stop_series
                        [ "${UNIT_REQUEUE}" = 1 ] && QUEUE+=("p:${v}:${sp%%:*}:${tp%%:*}")
                    done
                done
            done
        fi
    fi
    # re-queue once: the runs of this launch discarded for load
    for u in "${QUEUE[@]}"; do
        pos=$((pos + 1))
        run_unit "${board}" "${l}" "${pos}" "${u}" 1
        [ "${UNIT_STATUS}" = gave-up ] && stop_series
    done
  done
done

echo ""; echo "Series finished: ${OK} of $(( TOTAL_GEMM + TOTAL_PO )) planned runs ok (re-queued runs are extra). Run folders: ${SERIES}/runs.tsv"
[ "${#FAILED[@]}" -gt 0 ] && printf '  failed: %s\n' "${FAILED[@]}"
[ "${#DISCARDED[@]}" -gt 0 ] && printf '  discarded-load: %s\n' "${DISCARDED[@]}"
if [ -s "${SERIES}/load_warnings.txt" ]; then
    echo ""; echo "RED NOTE: another user's job was seen on a board during this series (${SERIES}/load_warnings.txt):"
    sed 's/^/  /' "${SERIES}/load_warnings.txt"
fi
[ "${#FAILED[@]}" -eq 0 ]
