"""
Issue #27: the reference design's run protocol (test_gemm.c), and the run overview.

The reference times its GEMM with no warm-up, and keeps going until `tmin` seconds have
passed (`while (time <= tmin) {...}`), so its cold first rep sits inside the mean and its
rep count is whatever fits. The uninstrumented build now does the same by default. The
older protocol -- a warm-up and a fixed rep count, steadier on short shapes (issue #14) --
is `--no-ref-protocol`, and is the instrumented build's default, because the reference
protocol cannot run there. Switching protocol must change nothing but the warm-up, the
rep count and when correctness is checked.

Seam: the real `templates/driver_benchmark.cpp` and `modelLevel/model_level.c`, built on
the host against a stub `generated_kernels.h` whose kernel is a plain scalar GEMM. The
stub can sleep per call (so a rep has a known duration), can be told to compute a wrong
answer, and prints how many times it was called, and with which packer, when the process
exits. Nothing here needs the board: what is under test is the driver's control flow and
logging, not kernel speed.
"""

import csv
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
MODEL_LEVEL_DIR = HERE / "modelLevel"

CONFIG_ITERATIONS = 3

STUB_HEADER = """\
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "perf_counters.h"

#define CONFIG_DATATYPE "fp32"
#define CONFIG_METHOD "%(method)s"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "host"
#define CONFIG_ITERATIONS %(iterations)d
%(skip_ref)s

struct NcOverrideEntry { const char* label; int nc; };
%(nc_overrides)s

void stub_gemm(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf);

typedef void (*gemm_blis_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf);

struct KernelVariant { const char* name; int mr; int nr; gemm_blis_fn func; };
static const KernelVariant generated_variants[] = { {"MR4_NR4", 4, 4, stub_gemm} };
static const int num_variants = 1;
"""

# A correct column-major GEMM (C = alpha*A*B + beta*C, so re-running it is idempotent at
# beta = 0, like the real kernels), plus the knobs and counters described above.
STUB_KERNEL = """\
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <string>
#include <thread>
#include "generated_kernels.h"

static int calls_total = 0, calls_scalar_pack = 0;
// Issue #18: non-zero elements Ac/Bc held when a shape first reached the kernel.
static long fresh_nonzero = 0;
// Issue #19: the first shape's Ac/Bc as allocated, next to what the reference asks for.
static size_t ac_usable = 0, bc_usable = 0, ac_wanted = 0, bc_wanted = 0;
static size_t ac_needed = 0, bc_needed = 0;
struct StubReport {
    ~StubReport() {
        fprintf(stderr, "STUB_CALLS total=%d scalar_pack=%d\\n", calls_total, calls_scalar_pack);
        fprintf(stderr, "STUB_FRESH_BUFFERS nonzero=%ld\\n", fresh_nonzero);
        fprintf(stderr, "STUB_BUFFER_BYTES ac=%zu ac_wanted=%zu bc=%zu bc_wanted=%zu "
                "ac_needed=%zu bc_needed=%zu\\n",
                ac_usable, ac_wanted, bc_usable, bc_wanted, ac_needed, bc_needed);
    }
} stub_report;

void stub_gemm(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf) {
    calls_total++;
    if (use_scalar_pack) calls_scalar_pack++;
    if (ac_wanted == 0) {
        // test_gemm.c:271-272 with -DKR=1: (MR+mc)*(KR+kc) floats for Ac, (KR+kc)*(NR+nc) for Bc.
        ac_wanted = (size_t)(mr + mc) * (1 + kc) * sizeof(float);
        bc_wanted = (size_t)(1 + kc) * (nr + nc) * sizeof(float);
        // What the packed panels occupy: every panel is mr (nr) wide and kc_pad deep, and a
        // DOT kernel rounds kc up to a multiple of 4 (the generated dispatch's kc_pad).
        const size_t kc_pad = std::string(CONFIG_METHOD).find("DOT") != std::string::npos
            ? (size_t)((kc + 3) / 4) * 4 : (size_t)kc;
        ac_needed = (size_t)((mc + mr - 1) / mr) * mr * kc_pad * sizeof(float);
        bc_needed = (size_t)((nc + nr - 1) / nr) * nr * kc_pad * sizeof(float);
        ac_usable = malloc_usable_size(Ac);
        bc_usable = malloc_usable_size(Bc);
    }
    if (const char* ms = getenv("STUB_SLEEP_MS"))
        std::this_thread::sleep_for(std::chrono::milliseconds(atoi(ms)));
    if (getenv("STUB_DIRTY_BUFFERS")) {
        // The driver's own allocation sizes. When the shape changes, the driver has just
        // allocated Ac/Bc afresh; whatever the previous shape left in a freed chunk of the
        // same size is still there unless the driver cleared it. Leave them dirty again.
        static int seen_m = -1, seen_n = -1, seen_k = -1;
        const size_t ac_n = (size_t)(mr + mc) * (1 + kc), bc_n = (size_t)(1 + kc) * (nr + nc);
        if (m != seen_m || n != seen_n || k != seen_k) {
            for (size_t i = 0; i < ac_n; ++i) fresh_nonzero += Ac[i] != 0.0f;
            for (size_t i = 0; i < bc_n; ++i) fresh_nonzero += Bc[i] != 0.0f;
            seen_m = m; seen_n = n; seen_k = k;
        }
        for (size_t i = 0; i < ac_n; ++i) Ac[i] = 1.0f;
        for (size_t i = 0; i < bc_n; ++i) Bc[i] = 1.0f;
    }
    const bool broken = getenv("STUB_BREAK") != nullptr;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) acc += A[p * lda + i] * B[j * ldb + p];
            C[j * ldc + i] = alpha * acc + beta * C[j * ldc + i] + (broken ? 1.0f : 0.0f);
        }
    *pack_time = 1e-6; *kernel_time = 1e-6; *pack_bytes = 1;
}
"""


class Driver:
    """The driver compiled on the host for one value of RVV_INSTRUMENTED."""

    def __init__(
        self,
        root: Path,
        instrumented: int,
        skip_ref_run: bool = False,
        nc_overrides: dict[str, int] | None = None,
        method: str = "STUB",
    ):
        compiler = shutil.which("g++") or shutil.which("c++")
        c_compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None or c_compiler is None:
            pytest.skip("no native C/C++ compiler available for the host-side driver tests")
        self.cwd = root / f"run_instr{instrumented}{'_skipref' if skip_ref_run else ''}"
        self.method = method
        overrides = nc_overrides or {}
        if overrides:
            entries = ", ".join(f'{{"{label}", {nc}}}' for label, nc in overrides.items())
            nc_decl = (
                f"static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = {{ {entries} }};\n"
                f"static const size_t CONFIG_NC_OVERRIDES_COUNT = {len(overrides)};"
            )
        else:
            nc_decl = (
                'static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = { {"", 0} };\n'
                "static const size_t CONFIG_NC_OVERRIDES_COUNT = 0;"
            )
        proj = self.cwd / "pj" / "proj"
        proj.mkdir(parents=True)
        (proj / "generated_kernels.h").write_text(
            STUB_HEADER
            % {
                "iterations": CONFIG_ITERATIONS,
                "skip_ref": "#define SKIP_REF_RUN 1" if skip_ref_run else "",
                "method": method,
                "nc_overrides": nc_decl,
            }
        )
        (proj / "stub_kernel.cpp").write_text(STUB_KERNEL)
        shutil.copy(TEMPLATES_DIR / "driver_benchmark.cpp", proj)
        # driver_benchmark.cpp includes "../../modelLevel/model_level.h"
        shutil.copytree(MODEL_LEVEL_DIR, self.cwd / "modelLevel")
        self.binary = self.cwd / "driver"
        cxx = [
            compiler, "-O1", "-std=c++17", f"-DRVV_INSTRUMENTED={instrumented}",
            f"-I{TEMPLATES_DIR}", f"-I{proj}",
        ]
        model_obj = self.cwd / "model_level.o"
        subprocess.run(
            [c_compiler, "-O1", "-c", str(MODEL_LEVEL_DIR / "model_level.c"), "-o", str(model_obj)],
            check=True, capture_output=True, text=True,
        )
        proc = subprocess.run(
            [*cxx, str(proj / "driver_benchmark.cpp"), str(proj / "stub_kernel.cpp"),
             str(model_obj), "-lm", "-o", str(self.binary)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        (self.cwd / "logs").mkdir()
        (self.cwd / "tiny.dat").write_text("T1 32 32 32\n")
        # T10 shares T1's shape and label prefix, and must not share its override.
        (self.cwd / "twins.dat").write_text("T1 32 32 32\nT10 32 32 32\n")
        # Two shapes that allocate an Ac of the same size, one after the other.
        (self.cwd / "two_shapes.dat").write_text("T1 32 32 32\nT2 32 24 32\n")

    def run(self, *args: str, env: dict[str, str] | None = None, dataset: str = "tiny.dat"):
        for old in (self.cwd / "logs").iterdir():  # every run starts from empty logs
            old.unlink()
        return subprocess.run(
            [str(self.binary), dataset, *args],
            cwd=self.cwd, capture_output=True, text=True, timeout=120,
            env={"PATH": "/usr/bin:/bin", **(env or {})},
        )

    def raw_rows(self, name: str) -> list[dict[str, str]]:
        with (self.cwd / "logs" / name).open() as fh:
            return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def uninstrumented(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("uninstr"), instrumented=0)


@pytest.fixture(scope="module")
def uninstrumented_skip_ref(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("uninstr_skip"), instrumented=0, skip_ref_run=True)


@pytest.fixture(scope="module")
def with_nc_override(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("nc_override"), instrumented=0, nc_overrides={"T1": 16})


@pytest.fixture(scope="module")
def instrumented(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("instr"), instrumented=1)


def _stub_calls(stderr: str) -> tuple[int, int]:
    line = next(ln for ln in stderr.splitlines() if ln.startswith("STUB_CALLS"))
    fields = dict(tok.split("=") for tok in line.split()[1:])
    return int(fields["total"]), int(fields["scalar_pack"])


REFPROTO_RAW = "raw_fp32_STUB_noinstr_refproto.txt"
WARMUP_RAW = "raw_fp32_STUB_noinstr.txt"


def test_the_reference_protocol_is_the_default_and_runs_reps_until_tmin(
    uninstrumented: Driver,
):
    # No protocol flag at all: the uninstrumented build times the way the reference does.
    proc = uninstrumented.run("--tmin", "0.3", env={"STUB_SLEEP_MS": "20"})
    assert proc.returncode == 0, proc.stdout + proc.stderr

    (row,) = uninstrumented.raw_rows(REFPROTO_RAW)
    reps = int(row["N_Reps"])
    rep_s = float(row["RVV_Time_ms"]) / 1000.0
    total_s = rep_s * reps
    # The reference's `while (time <= tmin)`: it stops at the first rep that carries the
    # elapsed time past tmin, so the total is > tmin but one rep short of it is not.
    assert total_s > 0.3
    assert total_s - rep_s <= 0.3 + 0.005
    assert reps > CONFIG_ITERATIONS
    # the rep count is printed, not just logged
    assert re.search(rf"\breps\s*:\s*{reps}\s+\[until tmin=0\.3 s, reference protocol\]", proc.stdout)


def test_naming_the_flag_explicitly_changes_nothing(uninstrumented: Driver):
    uninstrumented.run("--ref-protocol", "--tmin", "0.05")
    assert uninstrumented.raw_rows(REFPROTO_RAW)


def test_the_reference_protocol_has_no_warm_up_and_no_scalar_pack_pass(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--tmin", "0.1", env={"STUB_SLEEP_MS": "10"})
    (row,) = uninstrumented.raw_rows(REFPROTO_RAW)
    total_calls, scalar_pack_calls = _stub_calls(proc.stderr)
    assert scalar_pack_calls == 0
    assert total_calls == int(row["N_Reps"])  # every kernel call was a timed rep


def test_no_ref_protocol_brings_back_the_warm_up_and_the_fixed_rep_count(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--no-ref-protocol")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Its own log name, a fixed rep count, and the warm-ups it always had: 5 warm-up
    # calls, N scalar-pack calls, N timed calls.
    (row,) = uninstrumented.raw_rows(WARMUP_RAW)
    assert int(row["N_Reps"]) == CONFIG_ITERATIONS
    assert _stub_calls(proc.stderr) == (5 + 2 * CONFIG_ITERATIONS, CONFIG_ITERATIONS)
    assert not (uninstrumented.cwd / "logs" / REFPROTO_RAW).exists()


def test_each_protocol_writes_only_its_own_logs(uninstrumented: Driver):
    uninstrumented.run("--tmin", "0.05")
    for kind in ("raw", "summary", "perf"):
        assert (uninstrumented.cwd / "logs" / f"{kind}_fp32_STUB_noinstr_refproto.txt").is_file()
    assert not list((uninstrumented.cwd / "logs").glob("*_noinstr.txt"))


def test_the_reference_protocol_still_checks_correctness_against_the_scalar_reference(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--tmin", "0.05")
    (row,) = uninstrumented.raw_rows(REFPROTO_RAW)
    assert row["Status"] == "PASS", proc.stdout

    # ... and the check is real: a kernel that computes the wrong answer fails it, even
    # though the check now runs after the timed loop.
    uninstrumented.run("--tmin", "0.05", env={"STUB_BREAK": "1"})
    (row,) = uninstrumented.raw_rows(REFPROTO_RAW)
    assert row["Status"] == "FAIL"


def test_it_warns_when_fewer_than_three_reps_fit_in_tmin(uninstrumented: Driver):
    slow = uninstrumented.run("--tmin", "0.3", env={"STUB_SLEEP_MS": "200"})
    (row,) = uninstrumented.raw_rows(REFPROTO_RAW)
    assert int(row["N_Reps"]) == 2
    assert "[WARN]" in slow.stdout and "only 2 rep(s)" in slow.stdout

    fast = uninstrumented.run("--tmin", "0.1", env={"STUB_SLEEP_MS": "10"})
    assert "[WARN]" not in fast.stdout


def test_the_instrumented_build_defaults_to_the_warm_up_protocol(instrumented: Driver):
    # The reference protocol cannot run here (its numbers carry the timer tax), so the
    # default for this build is the old protocol -- and it says so.
    proc = instrumented.run()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (row,) = instrumented.raw_rows("raw_fp32_STUB.txt")
    assert int(row["N_Reps"]) == CONFIG_ITERATIONS
    assert "Protocol       : WARM-UP" in proc.stdout
    assert "needs the uninstrumented build" in proc.stdout


def test_asking_the_instrumented_build_for_the_reference_protocol_is_refused(
    instrumented: Driver,
):
    proc = instrumented.run("--ref-protocol")
    assert proc.returncode != 0
    assert "--ref-protocol needs the uninstrumented build" in proc.stderr
    assert not list((instrumented.cwd / "logs").iterdir())  # it refused before logging


@pytest.mark.parametrize(
    "bad", [["--tmin"], ["--tmin", "0"], ["--tmin", "soon"], ["--tmin", "0.5abc"]]
)
def test_tmin_needs_a_positive_number_of_seconds(uninstrumented: Driver, bad: list[str]):
    proc = uninstrumented.run(*bad)
    assert proc.returncode != 0
    assert "--tmin needs a positive number" in proc.stderr


def test_tmin_is_refused_where_it_would_be_silently_ignored(
    uninstrumented: Driver, instrumented: Driver
):
    for driver, flags in ((uninstrumented, ["--no-ref-protocol"]), (instrumented, [])):
        proc = driver.run(*flags, "--tmin", "1")
        assert proc.returncode != 0
        assert "--tmin only applies to the reference protocol" in proc.stderr


# ---------------------------------------------------------------------------
# The overview printed at the start of every run: every setting in effect, including
# the ones nobody typed, each marked as a default or as set by the user.
# ---------------------------------------------------------------------------
def _overview(stdout: str) -> str:
    start = stdout.index("Run overview")
    return stdout[start : stdout.index("Benchmark Config", start)]


def test_the_overview_lists_every_setting_and_marks_the_defaults(uninstrumented: Driver):
    proc = uninstrumented.run()
    ov = _overview(proc.stdout)

    assert ov.index("Build") < ov.index("Protocol") < ov.index("Logs")  # printed up front
    assert "Build          : UNINSTRUMENTED" in ov
    assert "Protocol       : REFERENCE" in ov and "[default]" in ov.split("Protocol")[1].splitlines()[0]
    assert "Warm-up        : none" in ov
    assert "until tmin = 3 s" in ov and "[default]" in ov.split("Reps")[1].splitlines()[0]
    assert "AFTER timing" in ov
    assert "Packing        : RVV vector packing" in ov
    assert "Kernel family  : fp32 STUB" in ov
    assert "Variants       : MR4_NR4" in ov
    assert "Cache blocking : analytical model" in ov
    assert "RVV_FORCE_NC: not set" in ov
    assert "tiny.dat" in ov
    assert "logs/raw_fp32_STUB_noinstr_refproto.txt" in ov


def test_the_overview_marks_what_the_user_set_and_what_the_environment_set(
    uninstrumented: Driver,
):
    proc = uninstrumented.run(
        "--tmin", "0.05", "--scalar-pack", env={"RVV_FORCE_NC": "16"}
    )
    ov = _overview(proc.stdout)
    assert "until tmin = 0.05 s" in ov and "[--tmin]" in ov.split("Reps")[1].splitlines()[0]
    assert "scalar packing forced" in ov and "[--scalar-pack]" in ov
    assert "RVV_FORCE_NC=16" in ov

    warm = _overview(uninstrumented.run("--no-ref-protocol").stdout)
    assert "Protocol       : WARM-UP" in warm
    assert "[--no-ref-protocol]" in warm.split("Protocol")[1].splitlines()[0]
    assert f"{CONFIG_ITERATIONS} fixed reps" in warm
    assert "BEFORE timing" in warm


# ---------------------------------------------------------------------------
# The terminal report. It has to say what each number measures, in words, so nobody has to
# know that "Ker" once meant kernel-only. Bracketed notes explain what is inside a figure.
# ---------------------------------------------------------------------------
def _variant_block(stdout: str) -> str:
    start = stdout.index("  +- T1")
    return stdout[start : stdout.index("\u25b6", start)]


def _summary_line(stdout: str) -> str:
    return next(ln for ln in stdout.splitlines() if "\u25b6" in ln)


def test_the_report_says_the_speed_is_the_whole_gemm_call_including_packing(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--tmin", "0.05")
    assert "KerGOPs" not in proc.stdout  # the old label promised kernel-only speed
    block = _variant_block(proc.stdout)
    assert "GOPs = billions of operations per second" in block
    assert "the whole GEMM call per rep" in block
    assert "pack A and B + microkernel compute" in block
    assert "same region the reference design times" in block


def test_the_headline_figure_is_by_mean_time_like_the_reference_and_matches_the_log(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--tmin", "0.05")
    (row,) = uninstrumented.raw_rows(REFPROTO_RAW)
    by_mean = re.search(r"by mean time\s*:\s*([\d.]+) GOPs", proc.stdout)
    assert by_mean, proc.stdout
    expected = 2 * 32 * 32 * 32 / (float(row["RVV_Time_ms"]) * 1e6)
    assert float(by_mean.group(1)) == pytest.approx(expected, abs=0.006)
    assert "the reference design's own formula" in proc.stdout
    assert re.search(r"by median rep\s*:\s*[\d.]+ GOPs", proc.stdout)


def test_the_spread_is_spelled_out_not_abbreviated(uninstrumented: Driver):
    block = _variant_block(uninstrumented.run("--tmin", "0.05").stdout)
    for words in ("slowest", "5th percentile", "95th percentile", "fastest",
                  "middle-50% width", "std-dev / median", "fastest / median"):
        assert words in block
    for acronym in ("p5:", "p95:", "iqr:", "mx/md", "CV:"):
        assert acronym not in block


def test_the_scalar_reference_is_not_called_ref_or_delta(uninstrumented: Driver):
    block = _variant_block(uninstrumented.run("--tmin", "0.05").stdout)
    assert "naive scalar GEMM" in block
    assert "NOT the reference design" in block
    assert "times faster" in block
    assert "| Ref " not in block and "Delta" not in block


def test_a_skipped_scalar_reference_is_said_to_be_skipped(uninstrumented_skip_ref: Driver):
    block = _variant_block(uninstrumented_skip_ref.run("--tmin", "0.05").stdout)
    assert "naive scalar GEMM : not run" in block
    assert "skip_ref_run in config.json" in block
    assert "times faster" not in block


def test_the_uninstrumented_report_has_no_empty_breakdown_or_counter_lines(
    uninstrumented: Driver,
):
    stdout = uninstrumented.run("--tmin", "0.05").stdout
    block = _variant_block(stdout)
    for empty in ("nanms", "L1D", "Stall", "IPC", "Counters valid", "GB/s"):
        assert empty not in block and empty not in _summary_line(stdout)
    assert "pack vs compute" in block and "rerun with --instr" in block


def test_the_summary_line_names_the_best_variant_and_what_its_figures_mean(
    uninstrumented: Driver,
):
    stdout = uninstrumented.run("--tmin", "0.05").stdout
    line = _summary_line(stdout)
    assert "best variant 4x4 [4 rows x 4 columns" in line
    assert "GOPs by mean time [whole GEMM call, packing included]" in line
    assert "reps" in line and "\u2705" in line and "[sample entry" in line
    assert "cache blocking" in line
    # the same text goes to the summary log
    log = (uninstrumented.cwd / "logs" / "summary_fp32_STUB_noinstr_refproto.txt").read_text()
    assert line in log


def test_the_status_line_explains_the_cache_blocking_letters(uninstrumented: Driver):
    block = _variant_block(uninstrumented.run("--tmin", "0.05").stdout)
    assert "matches the naive scalar GEMM" in block
    assert "mc=" in block and "rows of A per block" in block
    assert "kc=" in block and "depth" in block


def test_the_instrumented_report_gives_the_split_and_says_it_is_a_debug_number(
    instrumented: Driver,
):
    stdout = instrumented.run().stdout
    block = _variant_block(stdout)
    assert "Time split per rep [pack = reorder A and B into contiguous buffers" in block
    assert "vector pack" in block and "scalar pack" in block and "compute" in block
    assert "DEBUG build, do not compare to the reference" in block
    assert "hardware counters" in block.lower()  # a table when valid, "unavailable" here on the host
    line = _summary_line(stdout)
    assert "GB/s [bytes packed / pack time]" in line


# ---------------------------------------------------------------------------
# Issue #16: the per-shape nc override in config.json (B3 -> 128) has no counterpart in
# the reference design, so a head-to-head run must be able to switch it off. The override
# also has to name a label exactly: a future B30 must not inherit B3's.
# ---------------------------------------------------------------------------
NO_OVERRIDE_RAW = "raw_fp32_STUB_noinstr_refproto_no_nc_override.txt"


def _ncs(stdout: str) -> list[int]:
    """The nc each row was run at, in row order, from its cache-blocking note."""
    return [int(n) for n in re.findall(r"\bnc=(\d+) columns of B per block", stdout)]


def test_a_config_override_applies_by_default_and_only_to_the_exact_label(
    with_nc_override: Driver,
):
    proc = with_nc_override.run("--tmin", "0.05", dataset="twins.dat")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    t1_nc, t10_nc = _ncs(proc.stdout)
    assert t1_nc == 16
    # T10 starts with the label "T1" but is a different shape: it keeps the heuristic's nc.
    assert t10_nc != 16


def test_no_nc_overrides_runs_every_row_at_the_heuristics_nc(with_nc_override: Driver):
    proc = with_nc_override.run("--tmin", "0.05", "--no-nc-overrides", dataset="twins.dat")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    t1_nc, t10_nc = _ncs(proc.stdout)
    assert t1_nc == t10_nc != 16


def test_no_nc_overrides_writes_its_own_logs_so_it_cannot_mix_with_the_shipped_run(
    with_nc_override: Driver,
):
    with_nc_override.run("--tmin", "0.05", "--no-nc-overrides")
    for kind in ("raw", "summary", "perf"):
        assert (
            with_nc_override.cwd / "logs" / f"{kind}_fp32_STUB_noinstr_refproto_no_nc_override.txt"
        ).is_file()
    assert not (with_nc_override.cwd / "logs" / REFPROTO_RAW).exists()

    with_nc_override.run("--tmin", "0.05")
    assert (with_nc_override.cwd / "logs" / REFPROTO_RAW).is_file()
    assert not (with_nc_override.cwd / "logs" / NO_OVERRIDE_RAW).exists()


def test_the_overview_says_whether_config_overrides_are_applied(with_nc_override: Driver):
    shipped = _overview(with_nc_override.run("--tmin", "0.05").stdout)
    assert "nc overrides: T1 -> nc=16" in shipped
    assert "[config.json]" in shipped.split("nc overrides")[1].splitlines()[0]

    off = _overview(with_nc_override.run("--tmin", "0.05", "--no-nc-overrides").stdout)
    assert "nc overrides: OFF" in off
    assert "T1 -> nc=16" in off  # what was switched off is still named
    assert "[--no-nc-overrides]" in off.split("nc overrides")[1].splitlines()[0]
    assert "logs/raw_fp32_STUB_noinstr_refproto_no_nc_override.txt" in off


def test_no_nc_overrides_with_no_overrides_configured_is_harmless(uninstrumented: Driver):
    proc = uninstrumented.run("--tmin", "0.05", "--no-nc-overrides")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nc overrides: OFF" in _overview(proc.stdout)


# ---------------------------------------------------------------------------
# Issue #26: a scalar-pack-live run is a different configuration from the vector-pack one
# (it is the experiment: which packer decides the end-to-end speed), so it must not share a
# log with it. Otherwise two runs in one directory append to each other.
# ---------------------------------------------------------------------------
SCALARPACK_RAW = "raw_fp32_STUB_noinstr_refproto_scalarpack.txt"


def test_a_scalar_pack_run_writes_its_own_logs_and_leaves_the_vector_ones_alone(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--tmin", "0.05", "--scalar-pack")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for kind in ("raw", "summary", "perf"):
        assert (
            uninstrumented.cwd / "logs" / f"{kind}_fp32_STUB_noinstr_refproto_scalarpack.txt"
        ).is_file()
    assert not (uninstrumented.cwd / "logs" / REFPROTO_RAW).exists()
    # ...and the overview names the file it is about to write
    assert "logs/raw_fp32_STUB_noinstr_refproto_scalarpack.txt" in _overview(proc.stdout)

    uninstrumented.run("--tmin", "0.05")
    assert (uninstrumented.cwd / "logs" / REFPROTO_RAW).is_file()
    assert not (uninstrumented.cwd / "logs" / SCALARPACK_RAW).exists()


def test_the_environment_variable_switches_the_log_name_like_the_flag(
    uninstrumented: Driver,
):
    uninstrumented.run("--tmin", "0.05", env={"USE_SCALAR_PACK": "1"})
    assert (uninstrumented.cwd / "logs" / SCALARPACK_RAW).is_file()


def test_a_scalar_pack_run_really_packs_with_the_scalar_packer_in_every_timed_rep(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--tmin", "0.1", "--scalar-pack", env={"STUB_SLEEP_MS": "10"})
    (row,) = uninstrumented.raw_rows(SCALARPACK_RAW)
    total_calls, scalar_pack_calls = _stub_calls(proc.stderr)
    assert scalar_pack_calls == total_calls == int(row["N_Reps"])


def test_the_scalar_pack_name_and_the_no_nc_override_name_combine(with_nc_override: Driver):
    with_nc_override.run("--tmin", "0.05", "--scalar-pack", "--no-nc-overrides")
    assert (
        with_nc_override.cwd
        / "logs"
        / "raw_fp32_STUB_noinstr_refproto_scalarpack_no_nc_override.txt"
    ).is_file()


def _results_file(tmp_path: Path, test_args: str) -> str:
    """The results_*.csv name `make run` tees the driver's output into, from a dry run."""
    proj = tmp_path / "pj" / "proj"
    proj.mkdir(parents=True)
    shutil.copy(TEMPLATES_DIR / "Makefile", proj)
    shutil.copytree(MODEL_LEVEL_DIR, tmp_path / "modelLevel")
    out = subprocess.run(
        ["make", "-n", "-C", str(proj), "run", "INSTRUMENTED=0", "DATASETS_PATTERN=d.dat",
         f"TEST_ARGS={test_args}"],
        capture_output=True, text=True, check=True,
    ).stdout
    # make -n leaves `$(basename $dataset .dat)` unexpanded, so return only what follows it
    return re.search(r"\.dat\)(\S*?)\.csv", out).group(1)


def test_the_results_file_of_a_no_nc_overrides_run_is_its_own(tmp_path: Path):
    """`make run` tees the driver's output into results_*.csv: like the logs, it must not be
    shared between the shipped configuration and a run without the overrides."""
    assert _results_file(tmp_path / "a", "") == "_noinstr_refproto"
    assert _results_file(tmp_path / "b", "--no-nc-overrides") == "_noinstr_refproto_no_nc_override"
    assert (
        _results_file(tmp_path / "c", "--no-ref-protocol --no-nc-overrides")
        == "_noinstr_no_nc_override"
    )


# ---------------------------------------------------------------------------
# compile_macro_bench.sh: the flags that reach the driver, and the refusals that must
# happen before the script touches the board (or regenerates pj_gen/).
# ---------------------------------------------------------------------------
SCRIPT = HERE / "compile_macro_bench.sh"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=HERE.parents[1], capture_output=True, text=True, timeout=30,
    )


def test_script_refuses_an_explicit_ref_protocol_with_the_instrumented_build():
    proc = _run_script("--ref-protocol", "--instr")
    assert proc.returncode != 0
    assert "--ref-protocol needs the uninstrumented build" in proc.stdout + proc.stderr
    assert "Dependency Check" not in proc.stdout  # refused during argument parsing


@pytest.mark.parametrize(
    "flags", [["--no-ref-protocol", "--tmin", "1.0"], ["--instr", "--tmin", "1.0"]]
)
def test_script_refuses_tmin_where_the_reference_protocol_is_not_running(flags: list[str]):
    proc = _run_script(*flags)
    assert proc.returncode != 0
    assert "--tmin only applies to the reference protocol" in proc.stdout + proc.stderr
    assert "Dependency Check" not in proc.stdout


def test_script_refuses_force_nc_with_no_nc_overrides():
    # RVV_FORCE_NC beats every other nc source, so the run would not be at the heuristic's
    # nc, yet would carry the head-to-head log name.
    proc = _run_script("--no-nc-overrides", "--force-nc", "128", "--dry-run")
    assert proc.returncode != 0
    assert "--force-nc cannot be combined with --no-nc-overrides" in proc.stdout + proc.stderr
    assert "Dependency Check" not in proc.stdout


def test_script_dry_run_prints_the_overview_of_a_default_run_and_stops():
    proc = _run_script("--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "Run overview" in out
    assert "Build          : UNINSTRUMENTED" in out
    assert "Protocol       : REFERENCE" in out and "[default]" in out.split("Protocol")[1].splitlines()[0]
    assert "tmin = 3.0 s" in out
    assert "raw_*_noinstr_refproto.txt" in out
    assert "Dependency Check" not in out  # it stops before touching anything


def test_script_dry_run_follows_the_flags_it_is_given():
    out = _run_script("--dry-run", "--tmin", "0.5", "--force-nc", "128").stdout
    assert "tmin = 0.5 s" in out and "[--tmin]" in out
    assert "RVV_FORCE_NC=128" in out

    out = _run_script("--dry-run", "--no-ref-protocol").stdout
    assert "Protocol       : WARM-UP" in out and "raw_*_noinstr.txt" in out

    out = _run_script("--dry-run", "--no-nc-overrides").stdout
    assert "nc overrides: OFF" in out and "[--no-nc-overrides]" in out
    assert "not consulted" not in out  # the config is still read: only its nc table is ignored
    assert "raw_*_noinstr_refproto_no_nc_override.txt" in out  # its own log name

    out = _run_script("--dry-run", "--instr").stdout
    assert "Build          : INSTRUMENTED" in out
    assert "Protocol       : WARM-UP" in out  # the reference protocol cannot run here


def test_the_pack_buffers_start_zeroed_like_the_references_calloc(
    uninstrumented: Driver,
):
    # Issue #18: the packers leave the pad slots of a partial panel alone and the dispatch
    # no longer zeroes them, so the driver must hand over Ac/Bc that are zero to begin
    # with, not whatever the previous shape left in the heap.
    proc = uninstrumented.run(
        "--tmin", "0.05", env={"STUB_DIRTY_BUFFERS": "1"}, dataset="two_shapes.dat"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    found = re.search(r"STUB_FRESH_BUFFERS nonzero=(\d+)", proc.stderr)
    assert found is not None, proc.stderr
    assert int(found.group(1)) == 0


BUFFER_BYTES = re.compile(
    r"STUB_BUFFER_BYTES ac=(\d+) ac_wanted=(\d+) bc=(\d+) bc_wanted=(\d+)"
    r" ac_needed=(\d+) bc_needed=(\d+)"
)


def _buffer_bytes(stderr: str) -> tuple[int, ...]:
    found = BUFFER_BYTES.search(stderr)
    assert found is not None, stderr
    return tuple(map(int, found.groups()))


def test_the_pack_buffers_are_the_size_the_reference_allocates(uninstrumented: Driver):
    # Issue #19: the driver sized Ac as (mr+mc)*(mr+kc), the reference
    # (MR+mc)*(KR+kc) with -DKR=1 -- 24% more at 32x6. A bigger buffer is a bigger
    # cache footprint, and it is ours alone. malloc_usable_size is the allocation
    # rounded up to the allocator's grain, so allow the rounding above the reference's
    # byte count and nothing more.
    proc = uninstrumented.run("--tmin", "0.05")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ac, ac_wanted, bc, bc_wanted, ac_needed, bc_needed = _buffer_bytes(proc.stderr)
    slack = 64 + 32  # aligned_alloc rounds the size up to 64; malloc's own grain is 16
    assert ac_wanted <= ac < ac_wanted + slack, (ac, ac_wanted)
    assert bc_wanted <= bc < bc_wanted + slack, (bc, bc_wanted)
    assert ac_needed <= ac and bc_needed <= bc  # and still holds every packed panel


@pytest.fixture(scope="module")
def dot_method(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("dot"), instrumented=0, method="DOT_STUB")


def test_a_dot_kernels_k_padding_still_fits_the_smaller_pack_buffers(dot_method: Driver):
    # Issue #19: a DOT kernel pads K to a multiple of 4, which can take a panel past the
    # reference's one spare row of K. Tall and shallow (m=256, k=5 -> kc_pad=8) is where
    # the panels outgrow (mr+mc)*(1+kc), so the allocation has to be floored at them.
    (dot_method.cwd / "tall.dat").write_text("T1 256 32 5\n")
    proc = dot_method.run("--tmin", "0.05", dataset="tall.dat")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ac, ac_wanted, bc, _, ac_needed, bc_needed = _buffer_bytes(proc.stderr)
    assert ac_needed > ac_wanted  # this shape does outgrow the reference's formula
    assert ac_needed <= ac and bc_needed <= bc


def test_the_results_file_of_a_scalar_pack_run_is_its_own(tmp_path: Path):
    """Same rule as the driver's logs, for the results_*.csv `make run` tees into."""
    assert _results_file(tmp_path / "a", "--scalar-pack") == "_noinstr_refproto_scalarpack"
    assert (
        _results_file(tmp_path / "b", "--scalar-pack --no-nc-overrides")
        == "_noinstr_refproto_scalarpack_no_nc_override"
    )
    assert (
        _results_file(tmp_path / "c", "--scalar-pack --no-ref-protocol")
        == "_noinstr_scalarpack"
    )
