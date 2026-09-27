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
probe "governor" "\$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | tr -s ' \\n' ' ')"
probe "load average" "\$(cat /proc/loadavg)"
probe "taken" "\$(date -Is)"
echo "  /proc/cpuinfo  :"
cat /proc/cpuinfo
EOF
}
