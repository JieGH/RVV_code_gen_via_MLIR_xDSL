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
#define CONFIG_METHOD "STUB"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "host"
#define CONFIG_ITERATIONS %(iterations)d

struct NcOverrideEntry { const char* prefix; int nc; };
static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = { {"", 0} };
static const size_t CONFIG_NC_OVERRIDES_COUNT = 0;

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
#include <thread>
#include "generated_kernels.h"

static int calls_total = 0, calls_scalar_pack = 0;
struct StubReport {
    ~StubReport() {
        fprintf(stderr, "STUB_CALLS total=%d scalar_pack=%d\\n", calls_total, calls_scalar_pack);
    }
} stub_report;

void stub_gemm(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, perf_gemm_counters* perf) {
    calls_total++;
    if (use_scalar_pack) calls_scalar_pack++;
    if (const char* ms = getenv("STUB_SLEEP_MS"))
        std::this_thread::sleep_for(std::chrono::milliseconds(atoi(ms)));
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

    def __init__(self, root: Path, instrumented: int):
        compiler = shutil.which("g++") or shutil.which("c++")
        c_compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None or c_compiler is None:
            pytest.skip("no native C/C++ compiler available for the host-side driver tests")
        self.cwd = root / f"run_instr{instrumented}"
        proj = self.cwd / "pj" / "proj"
        proj.mkdir(parents=True)
        (proj / "generated_kernels.h").write_text(
            STUB_HEADER % {"iterations": CONFIG_ITERATIONS}
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

    def run(self, *args: str, env: dict[str, str] | None = None):
        for old in (self.cwd / "logs").iterdir():  # every run starts from empty logs
            old.unlink()
        return subprocess.run(
            [str(self.binary), "tiny.dat", *args],
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
    assert f"Reps:  {reps} " in proc.stdout  # the rep count is printed, not just logged


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

    out = _run_script("--dry-run", "--instr").stdout
    assert "Build          : INSTRUMENTED" in out
    assert "Protocol       : WARM-UP" in out  # the reference protocol cannot run here
