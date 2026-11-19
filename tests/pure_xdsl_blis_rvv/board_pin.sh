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
