"""
Issue #18: the dispatch used to `memset` the last partial panel of Ac and Bc before every
pack, outside both timers, so the cost showed up in wall clock and in neither
`Vector_Pack_Time_ms` nor `Compute_Time_ms`. The reference does no such thing: it
calloc's Ac/Bc once.

For the AXPY kernels the memsets are not needed. The packers leave the pad slots alone;
a pad row of Ac (or pad column of Bc) only feeds the pad rows/columns of the tile the
kernel computes, and the fringe path copies just `cur_mr x cur_nr` of that tile back to C.
So AXPY dispatch does no pad zeroing at all, and the driver hands it Ac/Bc that were zero
to begin with (see test_ref_protocol.py).

DOT kernels pad K too (`kc_pad`), and a K pad slot feeds *every* C entry, so their pad
zeroing is load-bearing. It stays, but inside the pack timer so the cost is counted.

Seam: the real `gemm_dispatch_xdsl.cpp` the generator emits, built on the host. The RVV
microkernels and vector packers are replaced by plain scalar stand-ins with the same
packed-layout contract, and `<riscv_vector.h>` by an empty header. Ac/Bc start filled
with NaN, so a pad slot that leaks into C shows up as a NaN. A pad slot then holds either
NaN (never written) or stale packed data from an earlier block; only a memset makes it an
exact 0.0.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

MLIR_TRANSLATE_CANDIDATES = [
    "/home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/llvm-project/build/bin/mlir-translate",
    "/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate",
]

MR, NR = 16, 6
AXPY = "AXPY_FP32_Pack"
DOT = "DOT_FP32_LANE_Pack"


def generate_project(root: Path, method: str, mr: int, nr: int) -> Path:
    """Runs the real project generator for one pinned kernel; returns the project dir."""
    mlir_translate = next(
        (p for p in MLIR_TRANSLATE_CANDIDATES if Path(p).exists()), None
    )
    if mlir_translate is None:
        pytest.skip("mlir-translate binary not found at any known location")
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        f'[["fp32", "{method}", [{mr}, {nr}], 256, "jbpi2", ["x.dat"], true, 5]]'
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(HERE / "generate_pure_blis_project.py"),
            "--config",
            "config.json",
            "--mlir-translate",
            mlir_translate,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (project,) = (root / "pj_gen").glob("fp32_*")
    return project


# A scalar microkernel over the packed layout the RVV one reads: Ac[k*lda + i], Bc[k*ldb + j],
# writing all MR x NR lanes (pad lanes included) like the real hardcoded kernel does.
STUB_KERNELS = """\
#include <stddef.h>
#include "gemm_headers.h"

static void kernel(size_t kc, float* alpha, float* Ac, size_t lda, float* Bc, size_t ldb,
                   float* beta, float* C, size_t ldc, bool accumulate) {
    for (size_t j = 0; j < %(nr)d; ++j)
        for (size_t i = 0; i < %(mr)d; ++i) {
            float acc = 0.0f;
            for (size_t p = 0; p < kc; ++p) acc += Ac[p * lda + i] * Bc[p * ldb + j];
            C[j * ldc + i] = *alpha * acc + (accumulate ? *beta * C[j * ldc + i] : 0.0f);
        }
}

extern "C" {
void gemm_microkernel_v_fp32_%(mr)dx%(nr)d_b0(size_t kc, float* a, float* Ac, size_t lda, float* Bc, size_t ldb, float* b, float* C, size_t ldc) {
    kernel(kc, a, Ac, lda, Bc, ldb, b, C, ldc, false);
}
void gemm_microkernel_v_fp32_%(mr)dx%(nr)d_b1(size_t kc, float* a, float* Ac, size_t lda, float* Bc, size_t ldb, float* b, float* C, size_t ldc) {
    kernel(kc, a, Ac, lda, Bc, ldb, b, C, ldc, true);
}

// Stand-ins for the RVV packers: the same layout, and -- like the real ones, which store
// with a tail-limited vl -- they write only the valid slots and leave the pad alone.
void pack_CB_v_if_AXPY_FP32_Pack_fp32_NR%(nr)d(float* M, float* Mc, size_t kc, size_t nc, size_t ldM) {
    for (size_t j = 0; j < nc; j += %(nr)d) {
        size_t w = std::min<size_t>(nc - j, %(nr)d);
        for (size_t p = 0; p < kc; ++p)
            for (size_t jj = 0; jj < w; ++jj) Mc[j * kc + p * %(nr)d + jj] = M[(j + jj) * ldM + p];
    }
}
void pack_RB_v_if_AXPY_FP32_Pack_fp32_MR%(mr)d(float* M, float* Mc, size_t mc, size_t kc, size_t ldM) {
    for (size_t i = 0; i < mc; i += %(mr)d) {
        size_t h = std::min<size_t>(mc - i, %(mr)d);
        for (size_t p = 0; p < kc; ++p)
            for (size_t ii = 0; ii < h; ++ii) Mc[i * kc + p * %(mr)d + ii] = M[p * ldM + i + ii];
    }
}
}
"""

# Runs gemm_blis_generic on NaN-filled Ac/Bc and prints, per case, the worst error against a
# double-precision GEMM and how many pad slots of the last block's last panels are exactly 0.0.
# Packed data is random in [-1, 1], so an exact zero there can only come from zeroing.
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

static void run_case(int m, int n, int k, int mc, int nc, int kc, float beta, bool scalar) {
    const int mr = %(mr)d, nr = %(nr)d;
    const float alpha = 1.5f;
    std::vector<float> A(m * k), B(k * n), C(m * n), C0(m * n);
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C0) x = rnd();
    C = C0;
    // Same sizes the driver allocates.
    std::vector<float> Ac((mr + mc) * (mr + kc), std::nanf(""));
    std::vector<float> Bc((nr + kc) * (nr + nc), std::nanf(""));
    double pt = 0, kt = 0;
    uint64_t pb = 0;
    gemm_blis_generic(m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m, Ac.data(), Bc.data(),
                      mc, nc, kc, mr, nr, &pt, &kt, &pb, scalar, false, 0, nullptr);

    double worst = 0;
    bool finite = true;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0;
            for (int p = 0; p < k; ++p) acc += (double)A[p * m + i] * B[j * k + p];
            double want = alpha * acc + (double)beta * C0[j * m + i];
            double got = C[j * m + i];
            if (!std::isfinite(got)) finite = false;
            worst = std::max(worst, std::fabs(got - want));
        }

    // The last block the dispatch packed: its partial panels' pad slots.
    int cur_nc = n - ((n - 1) / nc) * nc, cur_kc = k - ((k - 1) / kc) * kc, cur_mc = m - ((m - 1) / mc) * mc;
    int ac_zeroed = 0, bc_zeroed = 0;
    for (int r = cur_mc %% mr; cur_mc %% mr != 0 && r < mr; ++r)
        for (int p = 0; p < cur_kc; ++p)
            ac_zeroed += Ac[(cur_mc / mr) * mr * cur_kc + p * mr + r] == 0.0f;
    for (int c = cur_nc %% nr; cur_nc %% nr != 0 && c < nr; ++c)
        for (int p = 0; p < cur_kc; ++p)
            bc_zeroed += Bc[(cur_nc / nr) * nr * cur_kc + p * nr + c] == 0.0f;
    printf("CASE scalar_pack=%%d beta=%%g finite=%%d max_err=%%.3g ac_pad_zeroed=%%d bc_pad_zeroed=%%d\\n",
           (int)scalar, beta, (int)finite, worst, ac_zeroed, bc_zeroed);
}

int main() {
    for (int scalar = 0; scalar < 2; ++scalar)
        for (float beta : {0.0f, 0.5f}) {
            // Fringe in M and N, several blocks in every dimension, K ending on a short block.
            run_case(37, 20, 50, 32, 12, 24, beta, scalar);
            // Same, with K ending on a full block.
            run_case(37, 20, 48, 32, 12, 24, beta, scalar);
            // Tiles exactly: nothing to pad.
            run_case(32, 12, 24, 32, 12, 24, beta, scalar);
        }
    return 0;
}
"""


@pytest.fixture(scope="module")
def axpy_dispatch_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, str]]:
    root = tmp_path_factory.mktemp("axpy")
    project = generate_project(root / "gen", AXPY, MR, NR)
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available for the host-side dispatch test")
    stub_inc = root / "stub_inc"
    stub_inc.mkdir()
    (stub_inc / "riscv_vector.h").write_text("")  # only the RVV kernels use it
    (root / "stub_kernels.cpp").write_text(STUB_KERNELS % {"mr": MR, "nr": NR})
    (root / "harness.cpp").write_text(HARNESS % {"mr": MR, "nr": NR})
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
    assert len(rows) == 12, run.stdout
    return rows


def test_axpy_result_is_right_when_the_pad_slots_hold_nan(
    axpy_dispatch_run: list[dict[str, str]],
) -> None:
    for row in axpy_dispatch_run:
        assert row["finite"] == "1", row
        assert float(row["max_err"]) < 1e-4, row


def test_axpy_dispatch_does_no_pad_zeroing(
    axpy_dispatch_run: list[dict[str, str]],
) -> None:
    assert [
        r
        for r in axpy_dispatch_run
        if r["ac_pad_zeroed"] != "0" or r["bc_pad_zeroed"] != "0"
    ] == []


def _timed_pack_windows(source: str) -> list[str]:
    """What runs between each pack timer starting and its pack call: Bc's window, then Ac's."""
    windows: list[str] = []
    for part in source.split("rvv_region_timer _pack_clock(pack_time);")[1:]:
        windows.append(part[: part.index("_direct(")])
    return windows


def test_dot_dispatch_keeps_its_pad_zeroing_inside_the_pack_timer(
    tmp_path: Path,
) -> None:
    source = (
        generate_project(tmp_path / "gen", DOT, 8, 8) / "gemm_dispatch_xdsl.cpp"
    ).read_text()
    bc_window, ac_window = _timed_pack_windows(source)
    assert "memset(Bc + " in bc_window
    assert "memset(Ac + " in ac_window
