"""
Issues #17 and #24: the correctness pass.

Every row of the shipped `config.json` sets `skip_ref_run`, so every benchmark row logs
`Status=SKIP` and nothing on the board has been checked against the scalar oracle in the
shipped state. `config_correctness.json` is the pass that does the check: the same three
pinnings, `skip_ref_run` off, over `datasets/correctness_small.dat`.

The scalar oracle (`gemm_ref`) is O(m*n*k), so the pass has to stay small. Three seams:

- the oracle: the real `templates/driver_benchmark.cpp` built on the host, `gemm_ref` called
  directly and compared with a naive triple loop. Guards the `beta == 0` handling #17
  reported as broken (it was already right at HEAD).
- the config: `config_correctness.json` and its dataset are read the way the generator and
  the driver read them.
- the shapes: each pinning must reach the cases #24 says were never checked -- a partial
  panel that ends in the last vector chunk (the 3rd/4th chunk at MR=32), more than one K
  block (so the beta=1 kernel runs), a full tile and a fringe tile.

The board run itself (`Status=PASS` in the raw logs) is not something a host test can
show; see docs/agents/execution.md, "Correctness pass".
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cache_config import get_optim_mc_nc_kc  # noqa: E402
from config import BLISProjectConfig  # noqa: E402

TEMPLATES_DIR = HERE / "templates"
MODEL_LEVEL_DIR = HERE / "modelLevel"
DATASETS_DIR = HERE / "datasets"
SHIPPED_CONFIG = HERE / "config.json"
CORRECTNESS_CONFIG = HERE / "config_correctness.json"
YOLO_CORRECTNESS_CONFIG = HERE / "config_yolo_correctness.json"  # issue #60

# Elements per vector register at VLEN=256, fp32 (the generator's `vlen_elements`).
VLEN_ELEMENTS = 8
DATA_SIZE = 4  # fp32
# B2 (M=384 N=384 K=64), the largest shape #17 allows the scalar oracle to run on.
B2_MACS = 384 * 384 * 64


def load_shapes(dataset: Path) -> dict[str, tuple[int, int, int]]:
    """Label -> (m, n, k), read like the driver's `iss >> label >> m >> n >> k`."""
    shapes: dict[str, tuple[int, int, int]] = {}
    for line in dataset.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label, m, n, k, *_ = line.split()
        shapes[label] = (int(m), int(n), int(k))
    return shapes


def load_rows(config_path: Path) -> list[BLISProjectConfig]:
    return BLISProjectConfig.from_json(str(config_path))


def _pinning(row: BLISProjectConfig) -> tuple[int, int]:
    (mr_nr,) = row.mr_nr_family
    return mr_nr


EACH_PINNING = pytest.mark.parametrize(
    "row", load_rows(CORRECTNESS_CONFIG), ids=lambda r: "MR%dNR%d" % _pinning(r)
)


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

STUB_HEADER = """\
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "perf_counters.h"
#define CONFIG_DATATYPE "fp32"
#define CONFIG_METHOD "STUB"
#define CONFIG_VLEN 256
#define CONFIG_ARCH "host"
#define CONFIG_ITERATIONS 1
struct NcOverrideEntry { const char* label; int nc; };
static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = { {"", 0} };
static const size_t CONFIG_NC_OVERRIDES_COUNT = 0;
void stub_gemm(int, int, int, float, float*, int, float*, int, float, float*, int, float*, float*, int, int, int, int, int, double*, double*, uint64_t*, bool, bool, int, perf_gemm_counters*);
typedef void (*gemm_blis_fn)(int, int, int, float, float*, int, float*, int, float, float*, int, float*, float*, int, int, int, int, int, double*, double*, uint64_t*, bool, bool, int, perf_gemm_counters*);
struct KernelVariant { const char* name; int mr; int nr; gemm_blis_fn func; };
static const KernelVariant generated_variants[] = { {"MR4_NR4", 4, 4, stub_gemm} };
static const int num_variants = 1;
"""

# The driver's own translation unit, its `main` renamed, so `gemm_ref` is the very code
# the benchmark runs, not a copy of it.
ORACLE_HARNESS = """\
#define main driver_main
#include "driver_benchmark.cpp"
#undef main

void stub_gemm(int, int, int, float, float*, int, float*, int, float, float*, int, float*, float*, int, int, int, int, int, double*, double*, uint64_t*, bool, bool, int, perf_gemm_counters*) {}

// C = alpha*A*B + beta*C, one entry at a time. beta == 0 must not read C at all (the
// benchmark's C starts as anything), and alpha == 0 must not read A or B.
static float expected_entry(int m, int k, float alpha, const float* A, const float* B, float beta, float c0, int i, int j) {
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[p * m + i] * B[j * k + p];
    float base = (beta == 0.0f) ? 0.0f : beta * c0;
    return (alpha == 0.0f) ? base : alpha * acc + base;
}

int main() {
    const int m = 7, n = 5, k = 9;
    std::vector<float> A(m * k), B(k * n);
    for (int i = 0; i < m * k; ++i) A[i] = 0.25f * (float)((i * 7) % 11) - 1.0f;
    for (int i = 0; i < k * n; ++i) B[i] = 0.5f * (float)((i * 5) % 13) - 2.0f;

    struct Case { float alpha, beta; };
    const Case cases[] = {
        {1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.5f}, {2.0f, -1.5f}, {0.5f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.5f},
    };
    int failures = 0;
    for (const Case& c : cases) {
        std::vector<float> C0(m * n), C(m * n);
        for (int i = 0; i < m * n; ++i) C0[i] = 0.125f * (float)((i * 3) % 17) + 1.0f;
        // A beta == 0 call must ignore whatever C held.
        for (int i = 0; i < m * n; ++i) C[i] = (c.beta == 0.0f) ? std::nanf("") : C0[i];
        gemm_ref('C', 'C', 'C', 'N', 'N', m, n, k, c.alpha, A.data(), m, B.data(), k, c.beta, C.data(), m);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                float want = expected_entry(m, k, c.alpha, A.data(), B.data(), c.beta, C0[j * m + i], i, j);
                float got = C[j * m + i];
                if (!(std::fabs(got - want) <= 1e-4f)) {
                    std::printf("MISMATCH alpha=%g beta=%g C[%d,%d]: got %g want %g\\n", c.alpha, c.beta, i, j, got, want);
                    ++failures;
                }
            }
    }
    if (failures == 0) { std::printf("GEMM_REF OK\\n"); return 0; }
    std::printf("%d entries wrong\\n", failures);
    return 1;
}
"""


def test_gemm_ref_computes_alpha_ab_plus_beta_c_for_any_beta(tmp_path: Path):
    # #17 reported `beta` applied unconditionally. `alpha*tmp + beta*C` is the right formula
    # for any beta, so what the `beta == 0` guard buys is BLAS's rule that C is not read at
    # beta = 0: without it a C holding NaN/Inf poisons the result. The driver zeroes C before
    # every oracle call, so nothing else would notice if the guard came back off.
    compiler = shutil.which("g++") or shutil.which("c++")
    c_compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None or c_compiler is None:
        pytest.skip("no native C/C++ compiler available for the host-side oracle test")
    # The driver includes "../../modelLevel/model_level.h", so it sits two levels down, as
    # it does in a generated project (pj_gen/<project>/).
    proj = tmp_path / "pj" / "proj"
    proj.mkdir(parents=True)
    (proj / "generated_kernels.h").write_text(STUB_HEADER)
    (proj / "harness.cpp").write_text(ORACLE_HARNESS)
    shutil.copy(TEMPLATES_DIR / "driver_benchmark.cpp", proj)
    shutil.copy(TEMPLATES_DIR / "perf_counters.h", proj)
    shutil.copy(TEMPLATES_DIR / "pack_b_variants.h", proj)  # issue #55: the driver includes it
    shutil.copytree(MODEL_LEVEL_DIR, tmp_path / "modelLevel")
    model_obj = tmp_path / "model_level.o"
    subprocess.run(
        [c_compiler, "-O1", "-c", str(MODEL_LEVEL_DIR / "model_level.c"), "-o", str(model_obj)],
        check=True, capture_output=True, text=True,
    )
    binary = tmp_path / "oracle_check"
    build = subprocess.run(
        [compiler, "-O1", "-std=c++17", "-DRVV_INSTRUMENTED=0", f"-I{proj}",
         str(proj / "harness.cpp"), str(model_obj), "-lm", "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "GEMM_REF OK" in run.stdout


# ---------------------------------------------------------------------------
# The config and its shapes
# ---------------------------------------------------------------------------


def test_the_correctness_config_covers_every_shipped_pinning_with_the_check_on():
    shipped = {_pinning(row) for row in load_rows(SHIPPED_CONFIG)}
    rows = load_rows(CORRECTNESS_CONFIG)

    assert {_pinning(row) for row in rows} == shipped
    assert len(rows) == len(shipped)  # one project each, so no out_dir collision
    for row in rows:
        assert row.is_pinned
        assert not row.skip_ref_run, f"MR{_pinning(row)} still skips the check"
        assert row.datatype == "fp32" and row.method == "AXPY_FP32_Pack"


@pytest.mark.parametrize("config_path", [SHIPPED_CONFIG, CORRECTNESS_CONFIG, YOLO_CORRECTNESS_CONFIG], ids=lambda p: p.name)
def test_no_config_row_runs_the_scalar_oracle_on_a_shape_bigger_than_b2(config_path: Path):
    # The oracle is scalar and O(m*n*k): on S5 (5000^3) it is hours. A row may only turn
    # the check on if every shape in its datasets is B2-sized or smaller.
    for row in load_rows(config_path):
        if row.skip_ref_run:
            continue
        for dataset in row.datasets or []:
            for label, (m, n, k) in load_shapes(DATASETS_DIR / dataset).items():
                assert m * n * k <= B2_MACS, (
                    f"{config_path.name}: {dataset}:{label} is {m}x{n}x{k}, too big for the scalar oracle"
                )


def _correctness_shapes(row: BLISProjectConfig) -> dict[str, tuple[int, int, int]]:
    shapes: dict[str, tuple[int, int, int]] = {}
    for dataset in row.datasets or []:
        shapes.update(load_shapes(DATASETS_DIR / dataset))
    assert shapes, f"MR{_pinning(row)} has no shapes to check"
    return shapes


def _last_chunk_start(width: int) -> int:
    """Where the last vector chunk of a panel `width` wide begins: the panel packs in
    chunks of VLEN_ELEMENTS, the last one partial when width is not a multiple."""
    return ((width - 1) // VLEN_ELEMENTS) * VLEN_ELEMENTS


@EACH_PINNING
def test_each_pinning_is_checked_on_a_partial_panel_that_reaches_its_last_vector_chunk(
    row: BLISProjectConfig,
):
    # #24: at MR=32 the pack runs 4 vector chunks, and chunks 3 and 4 were never compared
    # with the oracle. A full panel runs all of them; the *partial* last panel is the one
    # that ends in the middle of the last chunk, so it needs a tail longer than the last
    # chunk's start. Both axes: A packs MR rows a panel, B packs NR columns.
    mr, nr = _pinning(row)
    shapes = _correctness_shapes(row).values()

    row_tails = [m % mr for m, _n, _k in shapes]
    col_tails = [n % nr for _m, n, _k in shapes]
    assert any(t > _last_chunk_start(mr) for t in row_tails), f"MR={mr}: row tails {row_tails}"
    assert any(t > _last_chunk_start(nr) for t in col_tails), f"NR={nr}: column tails {col_tails}"


@EACH_PINNING
def test_each_pinning_is_checked_on_full_tiles_and_on_more_than_one_k_block(
    row: BLISProjectConfig,
):
    # Full tiles take the kernel's direct path; only Fringe Tiles go through tmp_C. And
    # with k > kc the later K blocks run the beta=1 kernel (GET_BETA), a different kernel
    # from the first block's.
    mr, nr = _pinning(row)
    shapes = _correctness_shapes(row)

    def kc_of(m: int, n: int, k: int) -> int:
        return get_optim_mc_nc_kc(row.arch, DATA_SIZE, mr, nr, mmax=m, nmax=n, kmax=k)[2]

    assert any(m >= mr and n >= nr for m, n, _k in shapes.values()), "no shape holds a full tile"
    assert any(k > kc_of(m, n, k) for m, n, k in shapes.values()), "no shape needs a second K block"
    assert any(m < mr and n < nr for m, n, _k in shapes.values()), "no shape sits inside one fringe tile"
