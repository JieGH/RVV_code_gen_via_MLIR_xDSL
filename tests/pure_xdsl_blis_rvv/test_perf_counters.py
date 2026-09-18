"""
Issue #8: hardware perf counters as a supplement to wall-clock pack/compute timing.

`templates/perf_counters.h` is copied into every generated project and wraps a single
`perf_event_open()` event group (cycles, instructions, L1D read/write loads+misses,
dTLB read misses, stalled-cycles backend/frontend, branch misses). The generated
macrokernel snapshots that group around pack calls and the compute loop nest.

Why `perf_event_open()` and not the tools named in the issue (probed ad hoc on `jbpi`):
- `rdcycle`/`rdinstret` trap `SIGILL` in userspace: this board's kernel only delegates
  the `TM` bit of `scounteren`, not `CY`/`IR`.
- `perf stat` isn't installed and installing it needs an interactive `sudo`.
`perf_event_paranoid=2` on the board still permits self-process user-space counters.

Two seams:
- host-native tests of the header's counter logic (decode, region deltas, derived
  metrics). The host's `perf_event_paranoid` is typically too strict for real
  counters, so these feed synthetic kernel read buffers.
- an on-board test that the real event group opens, is never multiplexed, and gives
  stable, non-zero counts on `jbpi`. Skips if the board isn't reachable via `fej`.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _run_host_harness(
    body: str, tmp_path: Path, extra_flags: list[str] | None = None
) -> dict[str, float]:
    """Compiles `body` (statements inside main) against perf_counters.h and returns the
    `key=value` pairs it prints."""
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available for the host-side counter tests")

    src = tmp_path / "harness.cpp"
    src.write_text(
        '#include <cstdio>\n#include "perf_counters.h"\n'
        f"int main(void) {{\n{body}\nreturn 0;\n}}\n"
    )
    binary = tmp_path / "harness"
    compile_proc = subprocess.run(
        [compiler, "-O2", "-std=c++17", "-Wall", "-Werror", f"-I{TEMPLATES_DIR}",
         *(extra_flags or []), str(src), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    run_proc = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run_proc.returncode == 0, run_proc.stdout + run_proc.stderr

    values: dict[str, float] = {}
    for token in run_proc.stdout.split():
        key, _, value = token.partition("=")
        values[key] = float(value)
    return values


def test_group_read_decodes_only_complete_reads_from_the_full_group(tmp_path: Path) -> None:
    values = _run_host_harness(
        """
    // Kernel PERF_FORMAT_GROUP|TOTAL_TIME_ENABLED|TOTAL_TIME_RUNNING layout:
    // nr, time_enabled, time_running, value[0..nr-1].
    uint64_t buf[3 + PERF_NUM_EVENTS];
    buf[0] = PERF_NUM_EVENTS; buf[1] = 5000; buf[2] = 5000;
    for (int i = 0; i < PERF_NUM_EVENTS; i++) buf[3 + i] = 100 * (i + 1);
    perf_counts c;

    std::printf("ok_rc=%d ", perf_decode_group_read(buf, sizeof(buf), &c));
    std::printf("cycles=%llu branch_misses=%llu ",
                (unsigned long long) c.v[PERF_EV_CYCLES],
                (unsigned long long) c.v[PERF_EV_BRANCH_MISSES]);

    std::printf("short_rc=%d ", perf_decode_group_read(buf, sizeof(buf) - 8, &c));

    buf[2] = 4000;  // multiplexing is judged per window by the region, not per read
    int multiplexed_rc = perf_decode_group_read(buf, sizeof(buf), &c);
    std::printf("multiplexed_rc=%d time_enabled=%llu time_running=%llu ", multiplexed_rc,
                (unsigned long long) c.time_enabled, (unsigned long long) c.time_running);
    buf[2] = 5000;

    buf[0] = PERF_NUM_EVENTS - 1;  // a member event failed to join the group
    std::printf("wrong_nr_rc=%d ", perf_decode_group_read(buf, sizeof(buf), &c));
""",
        tmp_path,
    )

    assert values["ok_rc"] == 0
    assert values["cycles"] == 100
    assert values["branch_misses"] == 1000
    assert values["short_rc"] == -1
    assert values["multiplexed_rc"] == 0
    assert values["time_enabled"] == 5000
    assert values["time_running"] == 4000
    assert values["wrong_nr_rc"] == -1


def test_region_accumulates_only_windows_the_group_ran_for_in_full(tmp_path: Path) -> None:
    values = _run_host_harness(
        """
    perf_counts region = {}, before = {}, after = {};
    // Window 1: fully scheduled.
    before.time_enabled = before.time_running = 100;
    after.time_enabled = after.time_running = 200;
    before.v[PERF_EV_CYCLES] = 1000; after.v[PERF_EV_CYCLES] = 1250;
    before.v[PERF_EV_L1D_WRITE_MISS] = 7; after.v[PERF_EV_L1D_WRITE_MISS] = 9;
    std::printf("w1_rc=%d ", perf_counts_add_delta(&region, &before, &after));

    // Window 2: descheduled for 30 of its 100 time units -> rejected, region untouched.
    before.time_enabled = 200; before.time_running = 200;
    after.time_enabled = 300; after.time_running = 270;
    before.v[PERF_EV_CYCLES] = 2000; after.v[PERF_EV_CYCLES] = 9000;
    std::printf("w2_rc=%d ", perf_counts_add_delta(&region, &before, &after));

    // Window 3: fully scheduled again, even though lifetime totals still differ
    // because of window 2.
    before.time_enabled = 300; before.time_running = 270;
    after.time_enabled = 400; after.time_running = 370;
    before.v[PERF_EV_CYCLES] = 5000; after.v[PERF_EV_CYCLES] = 5100;
    before.v[PERF_EV_L1D_WRITE_MISS] = 9; after.v[PERF_EV_L1D_WRITE_MISS] = 9;
    std::printf("w3_rc=%d ", perf_counts_add_delta(&region, &before, &after));

    std::printf("cycles=%llu l1d_write_miss=%llu instructions=%llu",
                (unsigned long long) region.v[PERF_EV_CYCLES],
                (unsigned long long) region.v[PERF_EV_L1D_WRITE_MISS],
                (unsigned long long) region.v[PERF_EV_INSTRUCTIONS]);
""",
        tmp_path,
    )

    assert values["w1_rc"] == 0
    assert values["w2_rc"] == -1
    assert values["w3_rc"] == 0
    assert values["cycles"] == 350
    assert values["l1d_write_miss"] == 2
    assert values["instructions"] == 0


def test_per_rep_metrics_from_region_counts(tmp_path: Path) -> None:
    values = _run_host_harness(
        """
    perf_counts c = {};
    // Totals over 4 reps.
    c.v[PERF_EV_CYCLES] = 8000;
    c.v[PERF_EV_INSTRUCTIONS] = 2000;
    c.v[PERF_EV_L1D_READ] = 300;  c.v[PERF_EV_L1D_READ_MISS] = 3;
    c.v[PERF_EV_L1D_WRITE] = 100; c.v[PERF_EV_L1D_WRITE_MISS] = 37;
    c.v[PERF_EV_DTLB_READ_MISS] = 40;
    c.v[PERF_EV_STALLED_BACKEND] = 6000;
    c.v[PERF_EV_STALLED_FRONTEND] = 400;
    c.v[PERF_EV_BRANCH_MISSES] = 12;
    perf_metrics m = perf_metrics_per_rep(&c, 4);
    std::printf("cycles=%g instructions=%g ipc=%g l1d_read_miss_pct=%g l1d_write_miss_pct=%g "
                "dtlb_misses=%g stall_be_pct=%g stall_fe_pct=%g branch_misses=%g ",
                m.cycles, m.instructions, m.ipc, m.l1d_read_miss_pct, m.l1d_write_miss_pct, m.dtlb_misses,
                m.stall_be_pct, m.stall_fe_pct, m.branch_misses);

    perf_counts empty = {};
    perf_metrics e = perf_metrics_per_rep(&empty, 4);
    std::printf("empty_ipc_nan=%d empty_l1d_nan=%d empty_stall_nan=%d empty_cycles=%g",
                e.ipc != e.ipc, e.l1d_read_miss_pct != e.l1d_read_miss_pct,
                e.stall_be_pct != e.stall_be_pct, e.cycles);
""",
        tmp_path,
    )

    assert values["cycles"] == 2000
    assert values["instructions"] == 500
    assert values["ipc"] == 0.25
    assert values["l1d_read_miss_pct"] == 1
    assert values["l1d_write_miss_pct"] == 37
    assert values["dtlb_misses"] == 10
    assert values["stall_be_pct"] == 75
    assert values["stall_fe_pct"] == 5
    assert values["branch_misses"] == 3
    assert values["empty_ipc_nan"] == 1
    assert values["empty_l1d_nan"] == 1
    assert values["empty_stall_nan"] == 1
    assert values["empty_cycles"] == 0


def test_region_measurement_is_safe_whether_or_not_counters_open(tmp_path: Path) -> None:
    values = _run_host_harness(
        """
    perf_gemm_counters p;
    perf_gemm_open(&p);
    perf_gemm_reset(&p);
    perf_counts snap;
    volatile uint64_t sum = 0;
    perf_region_begin(&p, &snap);
    for (int i = 0; i < 1000000; i++) sum += i;
    perf_region_end(&p, &snap, &p.compute);
    std::printf("available=%d valid=%d compute_instructions=%llu pack_instructions=%llu",
                p.fd >= 0, p.valid,
                (unsigned long long) p.compute.v[PERF_EV_INSTRUCTIONS],
                (unsigned long long) p.pack.v[PERF_EV_INSTRUCTIONS]);
    perf_gemm_close(&p);
""",
        tmp_path,
    )

    assert values["pack_instructions"] == 0
    if values["available"]:
        assert values["valid"] == 1
        assert values["compute_instructions"] >= 1_000_000
    else:
        assert values["valid"] == 0
        assert values["compute_instructions"] == 0


JUMP_HOST = "fej"
BOARD_HOST = "jbpi"
BOARD_WORKSPACE = "xdsl_rvv_microkernel/perf_counter_probe"
BOARD_REPS = 50

# Pack-shaped workload: strided column copy from a large source into a contiguous
# buffer, so L1D writes/misses, dTLB misses, and backend stalls all get exercised.
BOARD_HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "perf_counters.h"

#define N_REPS %(reps)d

int main(void) {
    const size_t rows = 4096, cols = 1024, ld = rows;
    std::vector<float> src(rows * cols), dst(rows * cols);
    for (size_t i = 0; i < src.size(); i++) src[i] = (float) i;

    perf_gemm_counters p;
    perf_gemm_open(&p);
    if (p.fd < 0) { std::printf("PERF_EVENT_OPEN_FAILED\n"); return 1; }

    std::vector<perf_counts> reps(N_REPS);
    int all_valid = 1;
    for (int r = 0; r < N_REPS; r++) {
        perf_gemm_reset(&p);
        perf_counts snap;
        perf_region_begin(&p, &snap);
        size_t k = 0;
        for (size_t i = 0; i < rows; i += 16)
            for (size_t j = 0; j < cols; j++)
                for (size_t ii = 0; ii < 16; ii++) dst[k++] = src[j * ld + i + ii];
        perf_region_end(&p, &snap, &p.pack);
        all_valid &= p.valid;
        reps[r] = p.pack;
    }

    double sum = 0, sq = 0;
    for (int r = 0; r < N_REPS; r++) sum += (double) reps[r].v[PERF_EV_CYCLES];
    double mean = sum / N_REPS;
    for (int r = 0; r < N_REPS; r++) {
        double d = (double) reps[r].v[PERF_EV_CYCLES] - mean;
        sq += d * d;
    }
    perf_counts last = reps[N_REPS - 1];
    std::printf("all_valid=%%d cycles_cv_pct=%%f", all_valid, 100.0 * sqrt(sq / N_REPS) / mean);
    const char* names[PERF_NUM_EVENTS] = {"cycles", "instructions", "l1d_read", "l1d_read_miss",
        "l1d_write", "l1d_write_miss", "dtlb_read_miss", "stalled_backend", "stalled_frontend",
        "branch_misses"};
    for (int i = 0; i < PERF_NUM_EVENTS; i++)
        std::printf(" %%s=%%llu", names[i], (unsigned long long) last.v[i]);
    std::printf("\n");
    perf_gemm_close(&p);
    return 0;
}
""" % {"reps": BOARD_REPS}


def _ssh_via_jump(
    remote_cmd: str, connect_timeout: int, overall_timeout: int, batch_mode: bool = False
) -> subprocess.CompletedProcess[str]:
    """Runs `remote_cmd` on BOARD_HOST through JUMP_HOST (the board is only reachable
    from the jump host)."""
    jump_opts = ["-o", f"ConnectTimeout={connect_timeout}"]
    if batch_mode:
        jump_opts += ["-o", "BatchMode=yes"]
    return subprocess.run(
        [
            "ssh",
            *jump_opts,
            JUMP_HOST,
            f"ssh -o ConnectTimeout={connect_timeout} {BOARD_HOST} '{remote_cmd}'",
        ],
        capture_output=True,
        text=True,
        timeout=overall_timeout,
    )


def _board_reachable() -> bool:
    try:
        proc = _ssh_via_jump("echo OK", connect_timeout=6, overall_timeout=20, batch_mode=True)
        return proc.returncode == 0 and "OK" in proc.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def test_real_event_group_is_unmultiplexed_nonzero_and_stable_on_board(tmp_path: Path) -> None:
    if not _board_reachable():
        pytest.skip(f"RVV board not reachable via jump host '{JUMP_HOST}' -> '{BOARD_HOST}'")

    harness = tmp_path / "perf_group_board.cpp"
    harness.write_text(BOARD_HARNESS)

    mkdir_proc = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", JUMP_HOST, f"mkdir -p {BOARD_WORKSPACE}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert mkdir_proc.returncode == 0, mkdir_proc.stderr
    scp_proc = subprocess.run(
        ["scp", "-o", "ConnectTimeout=10", str(harness), str(TEMPLATES_DIR / "perf_counters.h"),
         f"{JUMP_HOST}:{BOARD_WORKSPACE}/"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert scp_proc.returncode == 0, scp_proc.stderr

    run_proc = _ssh_via_jump(
        f"cd {BOARD_WORKSPACE} && g++ -O2 -std=c++17 perf_group_board.cpp -o perf_group_board.x "
        "&& ./perf_group_board.x",
        connect_timeout=15,
        overall_timeout=120,
    )
    assert "PERF_EVENT_OPEN_FAILED" not in run_proc.stdout, (
        "the 10-event group was rejected on the board -- check "
        "/proc/sys/kernel/perf_event_paranoid on jbpi (2 when this test was written)"
    )
    assert run_proc.returncode == 0, run_proc.stdout + run_proc.stderr

    values = {
        key: float(value)
        for key, _, value in (tok.partition("=") for tok in run_proc.stdout.split())
    }
    assert values["all_valid"] == 1, f"group was multiplexed or a read failed: {values}"
    for headline in ("cycles", "instructions", "l1d_read", "l1d_write", "l1d_write_miss",
                     "dtlb_read_miss", "stalled_backend"):
        assert values[headline] > 0, f"{headline} read back zero on this PMU: {values}"
    assert values["cycles_cv_pct"] < 10.0, (
        f"cycles too noisy across {BOARD_REPS} reps of a fixed workload: {values}"
    )


# ---------------------------------------------------------------------------
# Issue #14: the instrumentation switch.
#
# Every timed region costs a pair of clock reads, and the per-tile pair is ~125-160 ns
# on jbpi -- roughly half charged to Compute_Time_ms and half to no timer at all.
# RVV_INSTRUMENTED=0 must compile the pairs and the counter reads away entirely, not
# merely branch around them, so the uninstrumented build's timed region has the same
# shape as the reference harness's: one wall-clock read per rep and nothing inside.
# ---------------------------------------------------------------------------

# Exercises both wrappers over a workload long enough for any clock to resolve, then
# reports what survived compilation.
_INSTRUMENTATION_PROBE = """
    double pack_seconds = 0.0;
    perf_gemm_counters perf;
    perf_gemm_open(&perf);
    perf_gemm_reset(&perf);
    volatile double sink = 0.0;
    {
        rvv_perf_region _region(&perf, &perf.pack);
        rvv_region_timer _timer(&pack_seconds);
        for (int i = 0; i < 4000000; i++) sink += (double) i;
    }
    perf_gemm_close(&perf);
    std::printf("macro=%d pack_seconds=%.9f pack_cycles=%llu timer_bytes=%zu ",
                (int) RVV_INSTRUMENTED, pack_seconds,
                (unsigned long long) perf.pack.v[PERF_EV_CYCLES],
                sizeof(rvv_region_timer));
"""


def test_instrumentation_is_on_by_default(tmp_path: Path) -> None:
    values = _run_host_harness(_INSTRUMENTATION_PROBE, tmp_path)

    assert values["macro"] == 1, "RVV_INSTRUMENTED must default to on"
    assert values["pack_seconds"] > 0.0, "the default build must still time its regions"


def test_uninstrumented_build_records_no_time_and_no_counters(tmp_path: Path) -> None:
    values = _run_host_harness(
        _INSTRUMENTATION_PROBE, tmp_path, extra_flags=["-DRVV_INSTRUMENTED=0"]
    )

    assert values["macro"] == 0
    assert values["pack_seconds"] == 0.0
    assert values["pack_cycles"] == 0


def test_uninstrumented_timer_carries_no_state_so_the_clock_reads_are_gone(
    tmp_path: Path,
) -> None:
    """A runtime `if` would still leave the timepoint field and the clock call behind.
    An empty object (sizeof 1, the C++ minimum) can only mean the pair was compiled out."""
    off = _run_host_harness(
        _INSTRUMENTATION_PROBE, tmp_path, extra_flags=["-DRVV_INSTRUMENTED=0"]
    )
    on = _run_host_harness(_INSTRUMENTATION_PROBE, tmp_path)

    assert off["timer_bytes"] == 1
    assert on["timer_bytes"] > 1


def test_uninstrumented_build_never_opens_the_counter_group(tmp_path: Path) -> None:
    """The host's perf_event_paranoid usually refuses the group anyway, so `valid` alone
    proves nothing here. `sizeof` does: an instrumented region carries a whole
    perf_counts snapshot, an uninstrumented one carries nothing to read into."""
    probe = """
    perf_gemm_counters perf;
    perf_gemm_open(&perf);
    std::printf("fd_negative=%d valid=%d region_bytes=%zu ",
                (int) (perf.fd < 0), perf.valid, sizeof(rvv_perf_region));
    perf_gemm_close(&perf);
"""
    off = _run_host_harness(probe, tmp_path, extra_flags=["-DRVV_INSTRUMENTED=0"])
    on = _run_host_harness(probe, tmp_path)

    assert off["fd_negative"] == 1
    assert off["valid"] == 0
    assert off["region_bytes"] == 1
    assert on["region_bytes"] > sizeof_perf_counts()


def sizeof_perf_counts() -> int:
    """(3 + PERF_NUM_EVENTS) uint64 fields: the snapshot an instrumented region holds."""
    return (2 + 10) * 8
