"""
Issue #27: `--ref-protocol`, the run protocol of the reference design's test_gemm.c.

The reference times its GEMM with no warm-up, and keeps going until `tmin` seconds have
passed (`while (time <= tmin) {...}`), so its cold first rep sits inside the mean and its
rep count is whatever fits. The default run keeps a warm-up and a fixed rep count, because
short shapes need them to give stable numbers (issue #14). `--ref-protocol` drops both,
for the like-for-like comparison, and must change nothing else.

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


def test_ref_protocol_keeps_running_reps_until_tmin_has_passed(uninstrumented: Driver):
    proc = uninstrumented.run(
        "--ref-protocol", "--tmin", "0.3", env={"STUB_SLEEP_MS": "20"}
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    (row,) = uninstrumented.raw_rows("raw_fp32_STUB_noinstr_refproto.txt")
    reps = int(row["N_Reps"])
    rep_s = float(row["RVV_Time_ms"]) / 1000.0
    total_s = rep_s * reps
    # The reference's `while (time <= tmin)`: it stops at the first rep that carries the
    # elapsed time past tmin, so the total is > tmin but one rep short of it is not.
    assert total_s > 0.3
    assert total_s - rep_s <= 0.3 + 0.005
    assert reps > CONFIG_ITERATIONS
    assert f"Reps:  {reps} " in proc.stdout  # the rep count is printed, not just logged


def test_ref_protocol_has_no_warm_up_and_no_scalar_pack_pass(uninstrumented: Driver):
    proc = uninstrumented.run(
        "--ref-protocol", "--tmin", "0.1", env={"STUB_SLEEP_MS": "10"}
    )
    (row,) = uninstrumented.raw_rows("raw_fp32_STUB_noinstr_refproto.txt")
    total_calls, scalar_pack_calls = _stub_calls(proc.stderr)
    assert scalar_pack_calls == 0
    assert total_calls == int(row["N_Reps"])  # every kernel call was a timed rep


def test_default_protocol_is_unchanged(uninstrumented: Driver):
    proc = uninstrumented.run()
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Its own log name, a fixed rep count, and the warm-ups it always had: 5 warm-up
    # calls, N scalar-pack calls, N timed calls.
    (row,) = uninstrumented.raw_rows("raw_fp32_STUB_noinstr.txt")
    assert int(row["N_Reps"]) == CONFIG_ITERATIONS
    assert _stub_calls(proc.stderr) == (5 + 2 * CONFIG_ITERATIONS, CONFIG_ITERATIONS)
    assert not (uninstrumented.cwd / "logs" / "raw_fp32_STUB_noinstr_refproto.txt").exists()
    assert "Protocol: DEFAULT" in proc.stdout


def test_ref_protocol_names_itself_in_the_banner_and_the_other_logs(uninstrumented: Driver):
    proc = uninstrumented.run("--ref-protocol", "--tmin", "0.05")
    assert "Protocol: REFERENCE" in proc.stdout
    for kind in ("raw", "summary", "perf"):
        assert (uninstrumented.cwd / "logs" / f"{kind}_fp32_STUB_noinstr_refproto.txt").is_file()
    assert not list((uninstrumented.cwd / "logs").glob("*_noinstr.txt"))


def test_ref_protocol_still_checks_correctness_against_the_scalar_reference(
    uninstrumented: Driver,
):
    proc = uninstrumented.run("--ref-protocol", "--tmin", "0.05")
    (row,) = uninstrumented.raw_rows("raw_fp32_STUB_noinstr_refproto.txt")
    assert row["Status"] == "PASS", proc.stdout

    # ... and the check is real: a kernel that computes the wrong answer fails it, even
    # though the check now runs after the timed loop.
    uninstrumented.run("--ref-protocol", "--tmin", "0.05", env={"STUB_BREAK": "1"})
    (row,) = uninstrumented.raw_rows("raw_fp32_STUB_noinstr_refproto.txt")
    assert row["Status"] == "FAIL"


def test_ref_protocol_warns_when_fewer_than_three_reps_fit_in_tmin(uninstrumented: Driver):
    slow = uninstrumented.run(
        "--ref-protocol", "--tmin", "0.3", env={"STUB_SLEEP_MS": "200"}
    )
    (row,) = uninstrumented.raw_rows("raw_fp32_STUB_noinstr_refproto.txt")
    assert int(row["N_Reps"]) == 2
    assert "[WARN]" in slow.stdout and "only 2 rep(s)" in slow.stdout

    fast = uninstrumented.run(
        "--ref-protocol", "--tmin", "0.1", env={"STUB_SLEEP_MS": "10"}
    )
    assert "[WARN]" not in fast.stdout


def test_ref_protocol_is_refused_by_the_instrumented_build(instrumented: Driver):
    proc = instrumented.run("--ref-protocol")
    assert proc.returncode != 0
    assert "--ref-protocol needs the uninstrumented build" in proc.stderr
    assert not list((instrumented.cwd / "logs").iterdir())  # it refused before logging


@pytest.mark.parametrize(
    "bad", [["--tmin"], ["--tmin", "0"], ["--tmin", "soon"], ["--tmin", "0.5abc"]]
)
def test_tmin_needs_a_positive_number_of_seconds(uninstrumented: Driver, bad: list[str]):
    proc = uninstrumented.run("--ref-protocol", *bad)
    assert proc.returncode != 0
    assert "--tmin needs a positive number" in proc.stderr


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


def test_script_refuses_ref_protocol_with_the_instrumented_build_before_doing_anything():
    proc = _run_script("--ref-protocol", "--instr")
    assert proc.returncode != 0
    assert "--ref-protocol needs the uninstrumented build" in proc.stdout + proc.stderr
    assert "Dependency Check" not in proc.stdout  # refused during argument parsing


def test_script_refuses_tmin_without_ref_protocol():
    proc = _run_script("--tmin", "1.0")
    assert proc.returncode != 0
    assert "--tmin only applies with --ref-protocol" in proc.stdout + proc.stderr
    assert "Dependency Check" not in proc.stdout
