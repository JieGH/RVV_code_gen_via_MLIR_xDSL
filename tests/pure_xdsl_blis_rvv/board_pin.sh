# board_pin.sh -- CPU pinning and run provenance, shared by compile_macro_bench.sh and
# ref_baseline/run_ref_baseline.sh (issue #21). Sourced, never run.
#
# Both harnesses have to time on the same core, under conditions that are written down,
# or a head-to-head percentage carries an unknown-sized variable: the board is shared and
# its thermal state drifts. compare_ref_baseline.py reads the `cpu pin` line this leaves in
# each side's provenance.txt.

# Not CPU0: the board routes every device interrupt there (/proc/interrupts).
BENCH_CPU_DEFAULT=4

# pin_cpu_valid <value> -- succeeds for one core number, nothing else. `taskset -c` would
# take a list or a range, but then "the pinned core" would not be one core.
pin_cpu_valid() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

# pin_check <cpu_flag> <no_pin> <cpu> [<allow_core0>] -- for the scripts' argument parsing:
# fails, saying why, when --cpu was given a bad core or together with --no-pin (rather than
# let the later flag win), or was given core 0 without --allow-core-0 (issue #39, house
# rule 1). The scripts pass "true"/"false" for whether each flag was given.
pin_check() {
    if [ "$1" = true ] && [ "$2" = true ]; then
        echo "Error: --cpu and --no-pin contradict each other."; return 1
    fi
    if [ "$1" = true ] && ! pin_cpu_valid "$3"; then
        echo "Error: --cpu takes one core number (got '$3')."; return 1
    fi
    if [ "$1" = true ] && [ "$3" = 0 ] && [ "${4:-false}" != true ]; then
        echo "Error: core 0 takes every device interrupt and times 5-10% slow (issue #39: never core 0)."
        echo "       Use core ${BENCH_CPU_DEFAULT}, or add --allow-core-0 to reproduce an old core-0 run on purpose."
        return 1
    fi
}

# pin_prefix <cpu> -- what goes in front of the board command; empty for an unpinned run.
pin_prefix() {
    if [ -n "$1" ]; then printf 'taskset -c %s ' "$1"; fi
}

# OTHER_USERS_AWK -- awk program (run with -v me=<our user>) over `ps -eo user:20,pcpu,psr,etime,comm`
# lines: names every process of another user (not root) using at least this share of a core, as
# "<user> <command> <pcpu>% on core <psr>, up <etime>; ". A single busy core of someone else's job
# still moves the chip's thermal state, so provenance.txt records it and reps_report.py flags it red.
OTHER_USERS_AWK='$1 != me && $1 != "root" && $2 + 0 >= 20 { printf "%s %s %s%% on core %s, up %s; ", $1, $5, $2, $3, $4 }'

# board_provenance <cpu> -- prints a shell script for the board (pipe it to `sh -s` there):
# the conditions this run's numbers were taken under. A probe the board cannot answer (no
# cpufreq node, no compiler on PATH) reads "unavailable" rather than a blank that looks
# like "recorded, and empty", and never aborts the run.
board_provenance() {
    local cpu="$1" pin shown="none"
    pin="$(pin_prefix "$cpu")"
    if [ -n "$cpu" ]; then shown="$cpu"; fi
    cat <<EOF
probe() { printf '  %-14s : %s\\n' "\$1" "\${2:-unavailable}"; }
probe "cpu pin" "${shown}"
probe "affinity" "\$(${pin}sh -c 'taskset -pc \$\$' 2>&1)"
probe "host" "\$(hostname) (\$(uname -srm))"
probe "compiler (gcc)" "\$(gcc --version 2>/dev/null | head -1)"
probe "compiler (g++)" "\$(g++ --version 2>/dev/null | head -1)"
probe "compiler (g++-14)" "\$(g++-14 --version 2>/dev/null | head -1) (builds f16f16f32 projects, issue #112)"
probe "governor" "\$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | tr -s ' \\n' ' ')"
probe "load average" "\$(cat /proc/loadavg)"
others="\$(ps -eo user:20,pcpu,psr,etime,comm --sort=-pcpu --no-headers 2>/dev/null | awk -v me="\$(id -un)" '${OTHER_USERS_AWK}')"
probe "other users" "\${others:-none}"
probe "taken" "\$(date -Is)"
echo "  /proc/cpuinfo  :"
cat /proc/cpuinfo
EOF
}

# Issue #42: was anything else running while we timed? B3's slow state may be the L2 that
# cores 4-7 share, so each project's timed run (`make run`, not its build) is bracketed by two
# readings of /proc/stat on the board, kept as cpu_times_before / cpu_times_after in its logs.
#
# board_cpu_times -- the board command that prints the per-core lines of /proc/stat.
board_cpu_times() {
    echo "grep '^cpu[0-9]' /proc/stat"
}

# busy_share <before> <after> -- two outputs of board_cpu_times; prints "cpu0 3% cpu1 0% ...",
# each core's non-idle share of its time between the two (idle and iowait count as idle), or
# "unavailable" when either reading is not a set of per-core lines (e.g. an ssh error).
busy_share() {
    printf '%s\n' "$1" "--" "$2" | awk '
        NF == 0 { next }
        $1 == "--" { after = 1; next }
        $1 !~ /^cpu[0-9]+$/ { bad = 1; next }
        {
            total = 0; for (i = 2; i <= 9 && i <= NF; i++) total += $i
            idle = $5 + $6
            if (after) { t1[$1] = total; i1[$1] = idle; order[n++] = $1 } else { t0[$1] = total; i0[$1] = idle }
        }
        END {
            if (bad || n == 0) { print "unavailable"; exit }
            line = ""
            for (k = 0; k < n; k++) {
                c = order[k]
                if (!(c in t0)) { print "unavailable"; exit }
                dt = t1[c] - t0[c]; di = i1[c] - i0[c]
                share = dt > 0 ? 100 * (dt - di) / dt : 0
                line = line (k ? " " : "") sprintf("%s %.0f%%", c, share)
            }
            print line
        }'
}

# launch_conditions <raw events> <run dir> -- the provenance.txt lines for what the launch
# counted and how busy each core was during each project's timed run ("<project>: cpu0 3% ...",
# joined by "; "), from the readings fetched into <run dir>/<project>/.
launch_conditions() {
    local shares="" d
    for d in "$2"/*/; do
        [ -f "${d}cpu_times_before" ] && [ -f "${d}cpu_times_after" ] || continue
        shares="${shares:+${shares}; }$(basename "$d"): $(busy_share "$(cat "${d}cpu_times_before")" "$(cat "${d}cpu_times_after")")"
    done
    printf '  %-14s : %s\n' "raw events" "${1:-none}"
    printf '  %-14s : %s\n' "busy share" "${shares:-unavailable}"
}

# ---------------------------------------------------------------------------------------------
# Board access module (issue #116). One place for how a run script reaches a board, reads who
# else is using it, and waits for it to be idle. Before, each run script carried its own copy.
#
# Interface (the globals a caller sets before calling any of these):
#   MODE_ARG                 "--direct" (the board is reachable from here) or anything else (jump host)
#   RISCV_USER, RISCV_PORT   the account and ssh port on the board
#   REMOTE_HOST, REMOTE_PORT, REMOTE_PASS   the jump host (unused with --direct)
#   SERIES                   folder for the idle gate's readings (board_idle_gate, board_one_reading only)
#   IDLE_GAP, IDLE_POLL, IDLE_MAX_WAIT, IDLE_MIN, IDLE_PROC_MAX   the idle gate's thresholds, in s / %
# Test hooks (one set for every script; each is an executable called as `<hook> <board> ...`):
#   BOARD_SSH_TTY  set to 1: the jump-host route asks for a tty on both hops
#   BOARD_SSH_CMD  stands for board_ssh itself: `<hook> <board> <command...>`, stdin passed through
#   BOARD_PS_CMD   stands for `ps` on the board        BOARD_TOP_CMD   stands for `top -bn2 -d1`
# ---------------------------------------------------------------------------------------------

# board_ssh <board> <command...> -- runs the command on the board, directly or through the jump host
board_ssh() {
    local board="$1"; shift
    if [ -n "${BOARD_SSH_CMD:-}" ]; then ${BOARD_SSH_CMD} "${board}" "$@"; return; fi
    local t=(); [ "${BOARD_SSH_TTY:-}" = 1 ] && t=(-t)   # a tty on every hop (compile_macro_bench.sh); direct mode never had one
    if [ "${MODE_ARG}" = "--direct" ]; then
        ssh -p "${RISCV_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${RISCV_USER}@${board}" "$@"
    else
        local escaped; escaped=$(printf '%q ' "$@")
        sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${t[@]}" \
            "${RISCV_USER}@${REMOTE_HOST}" "ssh -p ${RISCV_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15 ${t[*]:+${t[*]} }${board} ${escaped}"
    fi
}

# board_rsync <board> <rsync args...> -- the target is written as @T@:path and becomes the right host
board_rsync() {
    local board="$1"; shift
    local args=() a
    if [ "${MODE_ARG}" = "--direct" ]; then
        for a in "$@"; do args+=("${a//@T@/${RISCV_USER}@${board}}"); done
        rsync -av -e "ssh -p ${RISCV_PORT} -o StrictHostKeyChecking=no" "${args[@]}"
    else
        for a in "$@"; do args+=("${a//@T@/${RISCV_USER}@${REMOTE_HOST}}"); done
        sshpass -p "${REMOTE_PASS}" rsync -av -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" "${args[@]}"
    fi
}

# board_ps <board> -- the board's processes by CPU (user, %cpu, core, age, command), whoever's they are
board_ps() {
    if [ -n "${BOARD_PS_CMD:-}" ]; then ${BOARD_PS_CMD} "$1"; return; fi
    board_ssh "$1" 'ps -eo user:20,pcpu,psr,etime,comm --sort=-pcpu --no-headers | head -12' < /dev/null
}

# board_busy <board> <file> -- reads the board's processes into the file; prints another user's busy job, if any.
# A board that cannot be read counts as busy: no launch on a board we cannot see.
board_busy() {
    board_ps "$1" > "$2" 2>&1 || { echo "cannot read the board's processes"; return; }
    awk -v me="${RISCV_USER}" "${OTHER_USERS_AWK}" "$2"
}

# board_wait_idle <board> -- blocks, polling every 5 min, while another user's process is above 20% CPU.
# Used by the correctness-first scripts (#83, #94). Always the jump-host route, as before.
board_wait_idle() {
    local board="$1" b out
    while true; do
        if out=$(timeout 60 sshpass -p "${REMOTE_PASS}" ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            "jlei@${REMOTE_HOST}" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 ${board} 'ps -eo user=,pcpu=,comm='" 2>/dev/null); then
            b=$(echo "${out}" | awk '$1 != "jlei" && $1 != "root" && $2 > 20 {print $1 ":" $3 ":" $2}' | tr '\n' ' ')
        else
            b="unreadable"
        fi
        [ -n "${b}" ] || return 0
        echo "[$(date +%H:%M)] ${board} busy (${b}); waiting 5 min"; sleep 300
    done
}

# board_top_read <board> -- `top -bn2 -d1` on the board, raw
board_top_read() {
    if [ -n "${BOARD_TOP_CMD:-}" ]; then ${BOARD_TOP_CMD} "$1"; return; fi
    board_ssh "$1" "LC_ALL=C top -bn2 -d1"
}

# board_top_summary <file> -- from the LAST sample of a `top -b` output: "idle|heavy|root".
# idle = overall CPU idle %; heavy = processes of other users (not root, not us) at >= IDLE_PROC_MAX % of a core;
# root = root's processes at or above the same mark (recorded, never a block).
board_top_summary() {
    awk -v me="${RISCV_USER}" -v thr="${IDLE_PROC_MAX}" '
        /^top - / { idle = ""; heavy = ""; root = ""; hdr = 0; next }
        /Cpu\(s\)/ {
            if (match($0, /[0-9]+[.,]?[0-9]* *%?id/)) {
                s = substr($0, RSTART, RLENGTH); gsub(/[^0-9.,]/, "", s); gsub(",", ".", s); idle = s
            }
            next
        }
        /^ *PID +USER/ { for (i = 1; i <= NF; i++) { if ($i == "USER") uc = i; if ($i == "%CPU") cc = i } hdr = 1; next }
        hdr && NF >= cc && $uc != "" {
            c = $cc; gsub(",", ".", c)
            if (c + 0 >= thr) {
                item = sprintf("%s %s %s%%; ", $uc, $NF, c)
                if ($uc == "root") root = root item
                else if ($uc != me) heavy = heavy item
            }
        }
        END { printf "%s|%s|%s\n", idle, heavy, root }
    ' "$1"
}

# board_top_trim <file> -- the last sample of a top output, its first 30 lines
board_top_trim() {
    awk '/^top - / { n = 0; delete keep } { keep[n++] = $0 } END { for (i = 0; i < n && i < 30; i++) print keep[i] }' "$1"
}

# board_one_reading <board> <tag> <phase> -- takes a reading, records it in ${SERIES}; returns 0 if the board is
# idle, 1 if not (the reason is left in READ_WHY)
READ_N=0
board_one_reading() {
    local board="$1" tag="$2" phase="$3" raw sum idle heavy root verdict
    READ_N=$((READ_N + 1))
    raw="$(mktemp)"
    board_top_read "${board}" > "${raw}" 2>&1
    sum="$(board_top_summary "${raw}")"
    idle="${sum%%|*}"; sum="${sum#*|}"; heavy="${sum%%|*}"; root="${sum#*|}"
    board_top_trim "${raw}" > "${SERIES}/idle_${board}_${tag}_${phase}_${READ_N}.txt"
    rm -f "${raw}"
    READ_WHY=""
    if [ -z "${idle}" ]; then
        READ_WHY="no reading (could not read top)"
    else
        if [ -n "${heavy}" ]; then READ_WHY="other user busy: ${heavy}"; fi
        if awk -v i="${idle}" -v m="${IDLE_MIN}" 'BEGIN { exit !(i + 0 < m + 0) }'; then
            READ_WHY="${READ_WHY:+${READ_WHY} }overall idle ${idle}% is under ${IDLE_MIN}%"
        fi
    fi
    if [ -z "${READ_WHY}" ]; then verdict=idle; else verdict=BUSY; fi
    printf '%s\t%s\t%s\t%s\tidle=%s%%\tother_users=%s\troot=%s\t%s\n' "$(date -Is)" "${board}" "${tag}" "${phase}" \
        "${idle:-?}" "${heavy:-none}" "${root:-none}" "${verdict}" >> "${SERIES}/idle_readings.txt"
    [ -n "${heavy}" ] && echo "$(date -Is) ${board} ${tag} (${phase}): ${heavy}" >> "${SERIES}/load_warnings.txt"
    [ -z "${READ_WHY}" ]
}

# board_idle_gate <board> <tag> -- waits until two readings IDLE_GAP s apart both pass; 1 after IDLE_MAX_WAIT s of waiting
board_idle_gate() {
    local board="$1" tag="$2" waited=0
    while true; do
        if board_one_reading "${board}" "${tag}" pre; then
            [ "${IDLE_GAP}" -gt 0 ] && sleep "${IDLE_GAP}"
            if board_one_reading "${board}" "${tag}" pre; then return 0; fi
        fi
        if [ $((waited + IDLE_POLL)) -gt "${IDLE_MAX_WAIT}" ]; then
            echo "$(date -Is) ${board} ${tag}: still busy after ${waited} s (${READ_WHY}); giving up" >> "${SERIES}/idle_waits.txt"
            return 1
        fi
        echo "$(date -Is) ${board} ${tag}: busy: ${READ_WHY}; waiting ${IDLE_POLL} s (waited ${waited} of ${IDLE_MAX_WAIT} s)" >> "${SERIES}/idle_waits.txt"
        sleep "${IDLE_POLL}"
        waited=$((waited + IDLE_POLL))
    done
}
