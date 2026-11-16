"""
Issue #56: `--threads N`, the parallel GEMM (decisions in .scratch/issues/0040-parallel-gemm-design.md §5).

- Generator: `--threads 1` (and no flag) emits byte-identical projects; N > 1 adds the OpenMP
  dispatch and `-fopenmp`, and is refused for anything but fp32 AXPY.
- Dispatch: the real generated N-thread dispatch, built on the host with OpenMP against the scalar
  stand-in kernels and packers of test_pad_zeroing.py, on NaN-filled Ac/Bc. Cases hit the split's
  edges: fewer B panels than threads (k-chunks), k shorter than the chunk count, a last `ic` block
  that splits differently, several blocks in every dimension, beta 0 and not 0.
- Script: `compile_macro_bench.sh --dry-run --threads N`: cores, OpenMP env, refusals.
"""

import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_pad_zeroing import AXPY, MLIR_TRANSLATE_CANDIDATES, MR, NR, STUB_KERNELS

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "compile_macro_bench.sh"


def _generate(root: Path, *extra: str, method: str = AXPY, datatype: str = "fp32") -> subprocess.CompletedProcess[str]:
    mlir_translate = next((p for p in MLIR_TRANSLATE_CANDIDATES if Path(p).exists()), None)
    if mlir_translate is None:
        pytest.skip("mlir-translate binary not found at any known location")
    root.mkdir(parents=True)
    (root / "config.json").write_text(f'[["{datatype}", "{method}", [{MR}, {NR}], 256, "jbpi2", ["x.dat"], true, 5]]')
    return subprocess.run(
        [sys.executable, str(HERE / "generate_pure_blis_project.py"), "--config", "config.json",
         "--mlir-translate", mlir_translate, *extra],
        cwd=root, capture_output=True, text=True,
    )


def _project(root: Path, *extra: str) -> Path:
    proc = _generate(root, *extra)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (project,) = (root / "pj_gen").glob("fp32_*")
    return project


@pytest.fixture(scope="module")
def projects(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("par")
    return {
        "none": _project(root / "none"),
        "1": _project(root / "t1", "--threads", "1"),
        "3": _project(root / "t3", "--threads", "3"),
        "8": _project(root / "t8", "--threads", "8"),
    }


def _same_tree(a: Path, b: Path) -> list[str]:
    cmp = filecmp.dircmp(a, b)
    diff = cmp.left_only + cmp.right_only + cmp.diff_files + cmp.funny_files
    for sub in cmp.common_dirs:
        diff += _same_tree(a / sub, b / sub)
    return diff


def test_one_thread_is_byte_identical_to_no_flag(projects: dict[str, Path]) -> None:
    files = sorted(p.name for p in projects["none"].iterdir())
    assert files, "empty project"
    # filecmp.dircmp compares shallowly by stat; force a content compare on every file.
    for name in files:
        a, b = projects["none"] / name, projects["1"] / name
        if a.is_file():
            assert a.read_bytes() == b.read_bytes(), name
    assert _same_tree(projects["none"], projects["1"]) == []


def test_n_threads_adds_openmp_only_then(projects: dict[str, Path]) -> None:
    for key in ("none", "1"):
        assert "fopenmp" not in (projects[key] / "Makefile").read_text()
        src = (projects[key] / "gemm_dispatch_xdsl.cpp").read_text()
        assert "#pragma omp" not in src and "omp.h" not in src
    make8 = (projects["8"] / "Makefile").read_text()
    assert "-fopenmp" in make8
    src8 = (projects["8"] / "gemm_dispatch_xdsl.cpp").read_text()
    assert "#pragma omp parallel num_threads(8)" in src8
    assert "#pragma omp parallel num_threads(3)" in (projects["3"] / "gemm_dispatch_xdsl.cpp").read_text()


@pytest.mark.parametrize("method, datatype", [("DOT_FP32_LANE_Pack", "fp32"), ("AXPY_INT8_Pack", "int8")])
def test_n_threads_is_refused_outside_fp32_axpy(tmp_path: Path, method: str, datatype: str) -> None:
    proc = _generate(tmp_path / "g", "--threads", "2", method=method, datatype=datatype)
    assert proc.returncode != 0
    assert "--threads" in proc.stdout + proc.stderr


def test_bad_thread_count_is_refused(tmp_path: Path) -> None:
    proc = _generate(tmp_path / "g", "--threads", "0")
    assert proc.returncode != 0


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

static void run_case(int m, int n, int k, int mc, int nc, int kc, float beta) {
    const int mr = %(mr)d, nr = %(nr)d;
    const float alpha = 1.5f;
    std::vector<float> A(m * k), B(k * n), C(m * n), C0(m * n);
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C0) x = rnd();
    for (int rep = 0; rep < 20; ++rep) {  // repeat: a race shows up as a wrong or NaN entry in some rep
        C = C0;
        // The driver's sizes: (mr+mc)*(1+kc) and (1+kc)*(nr+nc).
        std::vector<float> Ac((mr + mc) * (1 + kc), std::nanf(""));
        std::vector<float> Bc((1 + kc) * (nr + nc), std::nanf(""));
        double pt = 0, kt = 0;
        uint64_t pb = 0;
        gemm_blis_generic(m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m, Ac.data(), Bc.data(),
                          mc, nc, kc, mr, nr, &pt, &kt, &pb, false, false, 0, nullptr);
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
        printf("CASE m=%%d n=%%d k=%%d beta=%%g rep=%%d finite=%%d max_err=%%.3g pack_bytes=%%llu\\n",
               m, n, k, beta, rep, (int)finite, worst, (unsigned long long)pb);
    }
}

int main() {
    for (float beta : {0.0f, 0.5f}) {
        run_case(37, 20, 50, 32, 12, 24, beta);   // blocks in every dimension, partial panels, short last K block
        run_case(100, 4, 30, 48, 4, 30, beta);    // one B panel, narrower than nr: B split by k only
        run_case(200, 16, 3, 96, 16, 3, beta);    // k shorter than the k-chunk count
        run_case(300, 16, 40, 128, 16, 40, beta); // last ic block (44 rows) splits differently from the full ones
        run_case(7, 6, 9, 16, 6, 9, beta);        // fewer A panels than threads
        run_case(400, 64, 70, 256, 32, 32, beta); // two jc blocks, three pc blocks
    }
    return 0;
}
"""


def _run_dispatch(project: Path, root: Path) -> list[dict[str, str]]:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available for the host-side dispatch test")
    stub_inc = root / "stub_inc"
    stub_inc.mkdir(parents=True)
    (stub_inc / "riscv_vector.h").write_text("")
    (root / "stub_kernels.cpp").write_text(STUB_KERNELS % {"mr": MR, "nr": NR})
    (root / "harness.cpp").write_text(HARNESS % {"mr": MR, "nr": NR})
    binary = root / "dispatch_test"
    build = subprocess.run(
        [compiler, "-O1", "-std=c++17", "-fopenmp", "-DRVV_INSTRUMENTED=0", f"-I{stub_inc}", f"-I{project}",
         str(root / "harness.cpp"), str(root / "stub_kernels.cpp"), str(project / "gemm_dispatch_xdsl.cpp"),
         "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = [dict(t.split("=") for t in ln.split()[1:]) for ln in run.stdout.splitlines() if ln.startswith("CASE")]
    assert len(rows) == 2 * 6 * 20, run.stdout
    return rows


@pytest.mark.parametrize("threads", ["3", "8"])
def test_the_parallel_gemm_is_right(projects: dict[str, Path], tmp_path: Path, threads: str) -> None:
    for row in _run_dispatch(projects[threads], tmp_path):
        assert row["finite"] == "1", row
        assert float(row["max_err"]) < 1e-4, row


def test_pack_bytes_match_one_thread(projects: dict[str, Path], tmp_path: Path) -> None:
    one = {(r["m"], r["n"], r["k"]): r["pack_bytes"] for r in _run_dispatch(projects["none"], tmp_path / "a")}
    eight = {(r["m"], r["n"], r["k"]): r["pack_bytes"] for r in _run_dispatch(projects["8"], tmp_path / "b")}
    assert one == eight


# ---------------------------------------------------------------------------------------------
# Script seam (--dry-run)
# ---------------------------------------------------------------------------------------------


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=HERE.parents[1], capture_output=True, text=True,
                          timeout=30, env={k: v for k, v in os.environ.items() if k != "USE_SCALAR_PACK"})


def _line(out: str, key: str) -> str:
    return next(ln for ln in out.splitlines() if key in ln)


def test_script_default_is_one_thread_on_core_4() -> None:
    out = _run_script("--dry-run").stdout
    assert "1 (one core)" in _line(out, "Threads")
    assert "taskset -c 4 " in _line(out, "Pinning")
    assert "OMP_" not in out


def test_script_8_threads_run_on_cores_0_to_7_with_openmp_binding() -> None:
    proc = _run_script("--dry-run", "--threads", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "8" in _line(out, "Threads") and "[--threads]" in _line(out, "Threads")
    pin = _line(out, "Pinning")
    assert "OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-7 " in pin
    assert "core 0" in out  # the exception is named


@pytest.mark.parametrize(
    "args, words",
    [
        (["--threads", "0"], ["--threads"]),
        (["--threads", "9"], ["--threads"]),
        (["--threads", "x"], ["--threads"]),
        (["--threads", "8", "--instr"], ["--threads", "--instr"]),
        (["--threads", "8", "--no-ref-protocol"], ["--threads", "--no-ref-protocol"]),
        (["--threads", "8", "--scalar-pack"], ["--threads", "--scalar-pack"]),
        (["--threads", "8", "--pack-b", "col-g1"], ["--threads", "--pack-b"]),
        (["--threads", "8", "--zero-pack-lines"], ["--threads", "--zero-pack-lines"]),
        (["--threads", "8", "--cpu", "4"], ["--threads", "--cpu"]),
        (["--threads", "8", "--no-pin"], ["--threads", "--no-pin"]),
    ],
)
def test_script_refuses_at_argument_time(args: list[str], words: list[str]) -> None:
    proc = _run_script("--dry-run", *args)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert all(w in out for w in words), out
    assert "Run overview" not in out


def test_the_provenance_block_names_threads_and_cores(tmp_path: Path) -> None:
    text = SCRIPT.read_text()
    start = text.index('{\n    echo "Run on the board')
    end = text.index('} > "${RUN_DIR}/provenance.txt"') + len('} > "${RUN_DIR}/provenance.txt"')
    proc = subprocess.run(
        ["bash", "-c", text[start:end]], capture_output=True, text=True, cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "RUN_DIR": str(tmp_path), "THREADS": "8", "PIN_CPU": "0-7"},
    )
    assert proc.returncode == 0, proc.stderr
    lines = (tmp_path / "provenance.txt").read_text().splitlines()
    assert any(ln.split(":")[0].strip() == "threads" and "8 on cores 0-7" in ln for ln in lines), lines
