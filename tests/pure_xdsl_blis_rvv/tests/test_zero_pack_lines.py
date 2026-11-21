"""
Issue #43: clear each pack-buffer line with `cbo.zero` before packing into it.

On the board a store to a line that is not in the cache first reads that line from the L2,
even when the packer then overwrites all of it. `cbo.zero` (Zicboz, 64-byte lines on the X60)
puts a line of zeros in the cache without that read. Behind `--zero-pack-lines`, off by
default, the dispatch clears the whole lines of the Bc and Ac blocks it is about to pack.

Seams:
- `templates/pack_lines.h` on the host: which lines the helper clears for a byte range. The
  helper itself is a no-op on the host, so the test reads the pure "which lines" function.
- the real generated dispatch on the host (as in test_pad_zeroing.py), built with the helper
  switched to a plain memset of those lines (`-DRVV_ZERO_LINES_EMULATE`), so the clearing
  really happens: the GEMM result must still be right, and the lines must really be cleared
  (the pad slots of the last partial panel turn 0.0) only when the switch is on.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # P/tests, for sibling test modules

from test_pad_zeroing import AXPY, DOT, MR, NR, STUB_KERNELS, generate_project

HERE = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = HERE / "templates"


def _compiler() -> str:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available for the host-side tests")
    return compiler


def _lines(tmp_path: Path, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """For each (start offset, bytes) inside a 64-byte-aligned buffer: the first whole line's
    offset and the number of whole lines the helper would clear."""
    calls = "\n".join(
        f"    rvv_line_span s{i} = rvv_whole_lines(base + {off}, {n});\n"
        f'    std::printf("%ld %ld\\n", (long) (s{i}.first - (uintptr_t) base), (long) s{i}.count);'
        for i, (off, n) in enumerate(spans)
    )
    src = tmp_path / "lines.cpp"
    src.write_text(
        '#include <cstdio>\n#include "pack_lines.h"\n'
        "alignas(64) static char buf[4096];\n"
        f"int main(void) {{\n    char* base = buf;\n{calls}\n"
        "    rvv_zero_whole_lines(base + 3, 200);  // a no-op on the host: must build and run\n"
        "    return 0;\n}\n"
    )
    binary = tmp_path / "lines"
    build = subprocess.run(
        [_compiler(), "-O2", "-std=c++17", "-Wall", "-Werror", f"-I{TEMPLATES_DIR}", str(src), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    out: list[tuple[int, int]] = []
    for ln in run.stdout.splitlines():
        first, count = ln.split()
        out.append((int(first), int(count)))
    return out


def test_an_aligned_range_of_whole_lines_is_cleared_line_for_line(tmp_path: Path) -> None:
    assert _lines(tmp_path, [(0, 64), (0, 640), (128, 256)]) == [(0, 1), (0, 10), (128, 4)]


def test_a_partly_written_line_at_either_end_is_left_alone(tmp_path: Path) -> None:
    assert _lines(tmp_path, [(0, 100), (10, 118), (10, 200), (64, 127)]) == [
        (0, 1),  # [64, 100) is partial
        (64, 1),  # [10, 64) is partial, [64, 128) whole
        (64, 2),  # [10, 64) and [192, 210) partial
        (64, 1),  # [128, 191) partial
    ]


def test_a_range_with_no_whole_line_clears_nothing(tmp_path: Path) -> None:
    got = _lines(tmp_path, [(0, 0), (0, 63), (1, 64), (10, 100), (65, 62)])
    assert [count for _, count in got] == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------------------------
# Dispatch seam
# ---------------------------------------------------------------------------------------------

# Runs gemm_blis_generic on NaN-filled Ac/Bc with the switch off and on, and prints per case the
# worst error against a double GEMM and how many pad slots of the last block's partial panels
# hold exactly 0.0. Packed data is random in [-1, 1], so an exact zero comes only from clearing.
HARNESS = """\
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gemm_headers.h"

extern "C" void gemm_blis_generic(int32_t m, int32_t n, int32_t k, float alpha, float* A, int32_t lda, float* B, int32_t ldb, float beta, float* C, int32_t ldc, float* Ac, float* Bc, int32_t mc, int32_t nc, int32_t kc, int32_t mr, int32_t nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack, bool zero_pack_lines, int pack_b_variant, perf_gemm_counters* perf);

static float rnd() {
    static unsigned s = 12345;
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 8) / (float)(1 << 24) * 2.0f - 1.0f;
}

static void run_case(int m, int n, int k, int mc, int nc, int kc, float beta, bool scalar, bool zero_lines) {
    const int mr = %(mr)d, nr = %(nr)d;
    const float alpha = 1.5f;
    std::vector<float> A(m * k), B(k * n), C(m * n), C0(m * n);
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C0) x = rnd();
    C = C0;
    // Same sizes as the driver, 64-byte aligned as the driver allocates them.
    size_t ac_n = (mr + mc) * (mr + kc), bc_n = (nr + kc) * (nr + nc);
    float* Ac = (float*) std::aligned_alloc(64, (ac_n * sizeof(float) + 63) / 64 * 64);
    float* Bc = (float*) std::aligned_alloc(64, (bc_n * sizeof(float) + 63) / 64 * 64);
    for (size_t i = 0; i < ac_n; ++i) Ac[i] = std::nanf("");
    for (size_t i = 0; i < bc_n; ++i) Bc[i] = std::nanf("");
    double pt = 0, kt = 0;
    uint64_t pb = 0;
    gemm_blis_generic(m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m, Ac, Bc,
                      mc, nc, kc, mr, nr, &pt, &kt, &pb, scalar, zero_lines, 0, nullptr);

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

    int cur_nc = n - ((n - 1) / nc) * nc, cur_kc = k - ((k - 1) / kc) * kc, cur_mc = m - ((m - 1) / mc) * mc;
    int ac_zeroed = 0, bc_zeroed = 0;
    for (int r = cur_mc %% mr; cur_mc %% mr != 0 && r < mr; ++r)
        for (int p = 0; p < cur_kc; ++p)
            ac_zeroed += Ac[(cur_mc / mr) * mr * cur_kc + p * mr + r] == 0.0f;
    for (int c = cur_nc %% nr; cur_nc %% nr != 0 && c < nr; ++c)
        for (int p = 0; p < cur_kc; ++p)
            bc_zeroed += Bc[(cur_nc / nr) * nr * cur_kc + p * nr + c] == 0.0f;
    printf("CASE m=%%d zero_lines=%%d scalar_pack=%%d beta=%%g finite=%%d max_err=%%.3g ac_pad_zeroed=%%d bc_pad_zeroed=%%d\\n",
           m, (int)zero_lines, (int)scalar, beta, (int)finite, worst, ac_zeroed, bc_zeroed);
    std::free(Ac);
    std::free(Bc);
}

int main() {
    for (int zero_lines = 0; zero_lines < 2; ++zero_lines)
        for (int scalar = 0; scalar < 2; ++scalar)
            for (float beta : {0.0f, 0.5f}) {
                // Fringe in M and N, several blocks in every dimension, K ending on a short block.
                run_case(37, 20, 50, 32, 12, 24, beta, scalar, zero_lines);
                // Same, with K ending on a full block.
                run_case(37, 20, 48, 32, 12, 24, beta, scalar, zero_lines);
                // Tiles exactly: nothing to pad.
                run_case(32, 12, 24, 32, 12, 24, beta, scalar, zero_lines);
            }
    return 0;
}
"""


@pytest.fixture(scope="module")
def zero_lines_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, str]]:
    root = tmp_path_factory.mktemp("zero_lines")
    project = generate_project(root / "gen", AXPY, MR, NR)
    compiler = _compiler()
    stub_inc = root / "stub_inc"
    stub_inc.mkdir()
    (stub_inc / "riscv_vector.h").write_text("")
    (root / "stub_kernels.cpp").write_text(STUB_KERNELS % {"mr": MR, "nr": NR})
    (root / "harness.cpp").write_text(HARNESS % {"mr": MR, "nr": NR})
    binary = root / "dispatch_test"
    build = subprocess.run(
        [
            compiler,
            "-O1",
            "-std=c++17",
            "-DRVV_INSTRUMENTED=0",
            "-DRVV_ZERO_LINES_EMULATE",
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
    assert len(rows) == 24, run.stdout
    return rows


def test_the_gemm_is_right_with_the_switch_on_and_off(zero_lines_run: list[dict[str, str]]) -> None:
    for row in zero_lines_run:
        assert row["finite"] == "1", row
        assert float(row["max_err"]) < 1e-4, row


def test_the_lines_are_cleared_only_with_the_switch_on(zero_lines_run: list[dict[str, str]]) -> None:
    off = [r for r in zero_lines_run if r["zero_lines"] == "0"]
    on = [r for r in zero_lines_run if r["zero_lines"] == "1"]
    # Off: exactly today's dispatch, the pad slots are never written.
    assert [r for r in off if r["ac_pad_zeroed"] != "0" or r["bc_pad_zeroed"] != "0"] == []
    # On: the 37 x 20 cases end on partial panels in both Ac (5 of MR=16 rows) and Bc (8 = 6 + 2
    # of NR=6 columns), and the whole lines over their pad slots were cleared, for both packers.
    # Not every pad slot sits on a whole line, so only "some were cleared" is asserted.
    fringe = [r for r in on if r["m"] == "37"]
    assert len(fringe) == 8
    assert all(int(r["ac_pad_zeroed"]) > 0 and int(r["bc_pad_zeroed"]) > 0 for r in fringe), fringe


def test_the_dot_dispatch_does_not_clear_lines(tmp_path: Path) -> None:
    source = (generate_project(tmp_path / "gen", DOT, 8, 8) / "gemm_dispatch_xdsl.cpp").read_text()
    assert "rvv_zero_whole_lines" not in source


def test_the_axpy_dispatch_clears_inside_the_pack_timer(tmp_path: Path) -> None:
    source = (generate_project(tmp_path / "gen", AXPY, MR, NR) / "gemm_dispatch_xdsl.cpp").read_text()
    windows = [part[: part.index("_direct(")] for part in source.split("rvv_region_timer _pack_clock(pack_time);")[1:]]
    assert len(windows) == 2
    assert "rvv_zero_whole_lines(Bc," in windows[0]
    assert "rvv_zero_whole_lines(Ac," in windows[1]
