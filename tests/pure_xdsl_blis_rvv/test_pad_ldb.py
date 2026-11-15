"""
Issue #49: pad B's leading dimension (`ldb = k + pad`, driver flag `--pad-ldb <elements>`)
to move where the B read streams land in memory, without changing k, the data, or the
answer. Before any board time is spent on the decisive test (padding ldb at fixed tiles and
reading the slow-rep share), this checks the one thing a host test can show: that the real
pack/dispatch code (`gemm_dispatch_xdsl.cpp`, as test_pad_zeroing.py and
test_zero_pack_lines.py build it) still computes the right GEMM when B's buffer is `ldb x n`
with `ldb > k`, for several pads, both packers and both betas.

Seam: same as test_zero_pack_lines.py -- the real generated dispatch, with the RVV kernels
and vector packers swapped for scalar stand-ins with the same packed-layout contract
(STUB_KERNELS), built on the host. `gemm_blis_generic` already takes `ldb` as a runtime
parameter (`generate_pure_blis_project.py`'s `offset_B = jc * ldb + pc`), so this test does
not touch generated code at all -- it is the pack path itself under a wider B stride.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from test_pad_zeroing import AXPY, MR, NR, STUB_KERNELS, generate_project

HERE = Path(__file__).resolve().parent

# Elements added to ldb beyond k: 0 (unpadded, must match today's behaviour exactly), two
# small pads in the spirit of "16 floats" / "8 floats" from the issue, and one big enough to
# push a whole extra page between columns at the shapes below.
LDB_PADS = (0, 8, 16, 128)

# Runs gemm_blis_generic with B allocated as ldb x n (ldb possibly > k): only the first k
# rows of each column hold real data, the rest is the gap the pad opens up. The reference
# reads B with the same ldb stride, since that is where the real data actually sits.
HARNESS = """\
#include <cmath>
#include <cstdio>
#include <vector>
#include "gemm_headers.h"

extern "C" void gemm_blis_generic(int32_t m, int32_t n, int32_t k, float alpha, float* A, int32_t lda, float* B, int32_t ldb, float beta, float* C, int32_t ldc, float* Ac, float* Bc, int32_t mc, int32_t nc, int32_t kc, int32_t mr, int32_t nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, bool zero_pack_lines, int pack_b_variant, perf_gemm_counters* perf);

static float rnd() {
    static unsigned s = 12345;
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 8) / (float)(1 << 24) * 2.0f - 1.0f;
}

static void run_case(int m, int n, int k, int mc, int nc, int kc, int ldb_pad, float beta, bool scalar) {
    const int mr = %(mr)d, nr = %(nr)d;
    const float alpha = 1.5f;
    const int ldb = k + ldb_pad;
    std::vector<float> A(m * k), B((size_t)ldb * n), C(m * n), C0(m * n);
    for (auto& x : A) x = rnd();
    // Only the real k x n block gets data; [k, ldb) of each column is the gap the pad opens.
    for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p) B[(size_t)j * ldb + p] = rnd();
    for (auto& x : C0) x = rnd();
    C = C0;
    std::vector<float> Ac((mr + mc) * (mr + kc), std::nanf(""));
    std::vector<float> Bc((nr + kc) * (nr + nc), std::nanf(""));
    double pt = 0, kt = 0;
    uint64_t pb = 0;
    gemm_blis_generic(m, n, k, alpha, A.data(), m, B.data(), ldb, beta, C.data(), m, Ac.data(), Bc.data(),
                      mc, nc, kc, mr, nr, &pt, &kt, &pb, scalar, false, 0, nullptr);

    double worst = 0;
    bool finite = true;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0;
            for (int p = 0; p < k; ++p) acc += (double)A[p * m + i] * (double)B[(size_t)j * ldb + p];
            double want = alpha * acc + (double)beta * C0[j * m + i];
            double got = C[j * m + i];
            if (!std::isfinite(got)) finite = false;
            worst = std::max(worst, std::fabs(got - want));
        }
    printf("CASE ldb_pad=%%d scalar_pack=%%d beta=%%g finite=%%d max_err=%%.3g\\n",
           ldb_pad, (int)scalar, beta, (int)finite, worst);
}

int main() {
    const int pads[] = {%(pads)s};
    for (int ldb_pad : pads)
        for (int scalar = 0; scalar < 2; ++scalar)
            for (float beta : {0.0f, 0.5f}) {
                // Fringe in M and N, several K blocks, K ending on a short block.
                run_case(37, 20, 50, 32, 12, 24, ldb_pad, beta, scalar);
                // Same, with K ending on a full block.
                run_case(37, 20, 48, 32, 12, 24, ldb_pad, beta, scalar);
                // Tiles exactly: nothing to pad in M or N.
                run_case(32, 12, 24, 32, 12, 24, ldb_pad, beta, scalar);
            }
    return 0;
}
"""


@pytest.fixture(scope="module")
def pad_ldb_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, str]]:
    root = tmp_path_factory.mktemp("pad_ldb")
    project = generate_project(root / "gen", AXPY, MR, NR)
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available for the host-side dispatch test")
    stub_inc = root / "stub_inc"
    stub_inc.mkdir()
    (stub_inc / "riscv_vector.h").write_text("")  # only the RVV kernels use it
    (root / "stub_kernels.cpp").write_text(STUB_KERNELS % {"mr": MR, "nr": NR})
    pads_src = ", ".join(str(p) for p in LDB_PADS)
    (root / "harness.cpp").write_text(HARNESS % {"mr": MR, "nr": NR, "pads": pads_src})
    binary = root / "dispatch_test"
    build = subprocess.run(
        [
            compiler,
            "-O1",
            "-std=c++17",
            "-DRVV_INSTRUMENTED=0",
            f"-I{stub_inc}",
            f"-I{project}",
            str(root / "harness.cpp"),
            str(root / "stub_kernels.cpp"),
            str(project / "gemm_dispatch_xdsl.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = [
        dict(tok.split("=") for tok in ln.split()[1:])
        for ln in run.stdout.splitlines()
        if ln.startswith("CASE")
    ]
    assert len(rows) == 12 * len(LDB_PADS), run.stdout
    return rows


def test_the_gemm_is_right_for_every_ldb_pad(pad_ldb_run: list[dict[str, str]]) -> None:
    for row in pad_ldb_run:
        assert row["finite"] == "1", row
        assert float(row["max_err"]) < 1e-4, row


def test_an_unpadded_ldb_matches_todays_behaviour(pad_ldb_run: list[dict[str, str]]) -> None:
    # ldb_pad=0 is exactly k: the same call every other test in this project makes.
    unpadded = [r for r in pad_ldb_run if r["ldb_pad"] == "0"]
    assert len(unpadded) == 12, pad_ldb_run
    for row in unpadded:
        assert float(row["max_err"]) < 1e-4, row
