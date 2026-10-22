"""
Issue #55: the `--pack-b <name>` switch (generator dispatch, driver, compile_macro_bench.sh).

Phase 2 added `templates/pack_b_variants.h` (the B packer variants; test_pack_b_variants.py checks
their layout). This file tests the plumbing around them:

- Dispatch: the real generated fp32 AXPY dispatch, built on the host (kernels and shipped
  packers replaced by scalar stand-ins, as in test_pad_zeroing.py). The variant branch sits in the
  B pack timer after today's call, the A block has none, the GEMM is right with every variant, and
  the int8 and DOT dispatches carry no variant call but still compile without unused-parameter
  warnings.
- Driver: the real `templates/driver_benchmark.cpp` on the host against the stub kernel of
  test_ref_protocol.py: the flag, its refusals, the log names, the overview line, which variant
  each kernel call gets (the warm-up scalar pass gets 0).
- Script: `compile_macro_bench.sh --dry-run` (no board is touched) for pass-through, refusals,
  log names and the overview line; the provenance block is run on its own.

Nothing here needs a board. The `__riscv` intrinsic branch of the header is NOT exercised here:
it is first compiled on the board (phase 5).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_pad_zeroing import (
    AXPY,
    DOT,
    MLIR_TRANSLATE_CANDIDATES,
    MR,
    NR,
    STUB_KERNELS,
    generate_project,
)
from test_ref_protocol import Driver

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
SCRIPT = HERE / "compile_macro_bench.sh"

NAMES = ["row", "row-intr", "col-g1", "col-g2", "col-g4", "col-gall", "col-g1-m4"]


def _compiler() -> str:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available for the host-side tests")
    return compiler


def _generate(root: Path, datatype: str, method: str, mr: int, nr: int) -> Path:
    """Like test_pad_zeroing.generate_project, for any datatype."""
    mlir_translate = next((p for p in MLIR_TRANSLATE_CANDIDATES if Path(p).exists()), None)
    if mlir_translate is None:
        pytest.skip("mlir-translate binary not found at any known location")
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        f'[["{datatype}", "{method}", [{mr}, {nr}], 256, "jbpi2", ["x.dat"], true, 5]]'
    )
    proc = subprocess.run(
        [sys.executable, str(HERE / "generate_pure_blis_project.py"), "--config", "config.json",
         "--mlir-translate", mlir_translate],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (project,) = (root / "pj_gen").glob(f"{datatype}_*")
    return project


# ---------------------------------------------------------------------------------------------
# Dispatch seam
# ---------------------------------------------------------------------------------------------

# Runs gemm_blis_generic once per variant 0..6 (scalar pack off) and prints the worst error against
# a double GEMM. Packed Ac/Bc start as NaN, so a pad slot that leaks into C shows as a NaN. The
# cases end on partial panels in N (20 = 3*6 + 2, and 7 = 6 + 1) and K, and span several blocks.
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

static void run_case(int variant, int m, int n, int k, int mc, int nc, int kc, float beta) {
    const int mr = %(mr)d, nr = %(nr)d;
    const float alpha = 1.5f;
    std::vector<float> A(m * k), B(k * n), C(m * n), C0(m * n);
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C0) x = rnd();
    C = C0;
    std::vector<float> Ac((mr + mc) * (mr + kc), std::nanf(""));
    std::vector<float> Bc((nr + kc) * (nr + nc), std::nanf(""));
    double pt = 0, kt = 0;
    uint64_t pb = 0;
    gemm_blis_generic(m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m, Ac.data(), Bc.data(),
                      mc, nc, kc, mr, nr, &pt, &kt, &pb, false, false, variant, nullptr);
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
    printf("CASE variant=%%d n=%%d k=%%d beta=%%g finite=%%d max_err=%%.3g pack_bytes=%%llu\\n",
           variant, n, k, beta, (int)finite, worst, (unsigned long long)pb);
}

int main() {
    for (int variant = 0; variant < 7; ++variant)
        for (float beta : {0.0f, 0.5f}) {
            run_case(variant, 37, 20, 50, 32, 12, 24, beta);  // partial panel in N, short last K block
            run_case(variant, 37, 20, 48, 32, 12, 24, beta);  // full last K block
            run_case(variant, 32, 12, 24, 32, 12, 24, beta);  // tiles exactly
            run_case(variant, 16, 7, 30, 16, 12, 30, beta);   // last panel of one column
        }
    return 0;
}
"""


@pytest.fixture(scope="module")
def axpy_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_project(tmp_path_factory.mktemp("axpy") / "gen", AXPY, MR, NR)


@pytest.fixture(scope="module")
def variant_run(tmp_path_factory: pytest.TempPathFactory, axpy_project: Path) -> list[dict[str, str]]:
    root = tmp_path_factory.mktemp("variant_run")
    stub_inc = root / "stub_inc"
    stub_inc.mkdir()
    (stub_inc / "riscv_vector.h").write_text("")  # host: the header's plain-C fallback runs
    (root / "stub_kernels.cpp").write_text(STUB_KERNELS % {"mr": MR, "nr": NR})
    (root / "harness.cpp").write_text(HARNESS % {"mr": MR, "nr": NR})
    binary = root / "dispatch_test"
    build = subprocess.run(
        [_compiler(), "-O1", "-std=c++17", "-DRVV_INSTRUMENTED=0", f"-I{stub_inc}", f"-I{axpy_project}",
         str(root / "harness.cpp"), str(root / "stub_kernels.cpp"),
         str(axpy_project / "gemm_dispatch_xdsl.cpp"), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = [dict(tok.split("=") for tok in ln.split()[1:]) for ln in run.stdout.splitlines() if ln.startswith("CASE")]
    assert len(rows) == 7 * 2 * 4, run.stdout
    return rows


def test_the_gemm_is_right_with_every_variant_and_a_partial_last_panel(
    variant_run: list[dict[str, str]],
) -> None:
    assert sorted({int(r["variant"]) for r in variant_run}) == list(range(7))
    for row in variant_run:
        assert row["finite"] == "1", row
        assert float(row["max_err"]) < 1e-4, row
    # the partial-panel cases are really in the run
    assert {r["n"] for r in variant_run} == {"20", "12", "7"}


def test_every_variant_packs_the_same_bytes_as_the_default(variant_run: list[dict[str, str]]) -> None:
    by_case: dict[tuple[str, str, str], set[str]] = {}
    for r in variant_run:
        by_case.setdefault((r["n"], r["k"], r["beta"]), set()).add(r["pack_bytes"])
    assert all(len(v) == 1 for v in by_case.values()), by_case


def _b_window(source: str) -> str:
    """Everything from the B pack timer to the pack_bytes line that follows it."""
    start = source.index("rvv_region_timer _pack_clock(pack_time);")
    return source[start : source.index("*pack_bytes +=", start)]


def test_the_variant_branch_is_in_the_b_pack_timer_after_todays_call(axpy_project: Path) -> None:
    source = (axpy_project / "gemm_dispatch_xdsl.cpp").read_text()
    window = _b_window(source)
    assert window.count("pack_CB_variant(") == 1
    default = window.index("if (pack_b_variant == 0) { pack_CB_direct(")
    variant = window.index("pack_CB_variant(")
    assert default < variant
    # today's call comes first: this is the cut test_zero_pack_lines.py relies on
    assert window.index("_direct(") == window.index("pack_CB_direct(") + len("pack_CB")
    assert "pack_CB_variant(pack_b_variant, b_ptr, Bc, cur_kc, cur_nc, ldb, nr);" in window
    # the scalar branch is untouched, and the call sits after the timer starts
    assert "pack_CB_scalar(cur_kc, cur_nc, b_ptr, ldb, Bc, nr);" in window
    assert window.index("_pack_clock") < default


def test_the_a_pack_block_has_no_variant_branch(axpy_project: Path) -> None:
    source = (axpy_project / "gemm_dispatch_xdsl.cpp").read_text()
    a_part = source.split("rvv_region_timer _pack_clock(pack_time);")[2]
    a_window = a_part[: a_part.index("pack_RB_direct(")]
    assert "pack_b_variant" not in a_window and "pack_CB_variant" not in a_window
    # exactly one use of the parameter in the function body, and it is the B block's
    assert len(re.findall(r"pack_b_variant\s*==", source)) == 1
    assert source.count("pack_CB_variant(") == 1


def test_the_dispatch_header_includes_and_ships_the_variant_header(axpy_project: Path) -> None:
    assert '#include "pack_b_variants.h"' in (axpy_project / "gemm_headers.h").read_text()
    assert (axpy_project / "pack_b_variants.h").read_bytes() == (TEMPLATES_DIR / "pack_b_variants.h").read_bytes()
    for name in ("generated_kernels.h", "gemm_dispatch_xdsl.cpp"):
        assert "int pack_b_variant, perf_gemm_counters* perf)" in (axpy_project / name).read_text()


def _compiles_clean(project: Path, root: Path) -> None:
    stub_inc = root / "stub_inc"
    stub_inc.mkdir()
    (stub_inc / "riscv_vector.h").write_text("")
    build = subprocess.run(
        [_compiler(), "-O1", "-std=c++17", "-DRVV_INSTRUMENTED=0", "-Wunused-parameter",
         f"-I{stub_inc}", f"-I{project}", "-c", str(project / "gemm_dispatch_xdsl.cpp"), "-o", str(root / "d.o")],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    # other generated functions have unused parameters of their own (cur_mr, cur_nr, and zero_pack_lines in DOT): only ours counts
    assert "unused parameter ‘pack_b_variant’" not in build.stderr, build.stderr


def test_the_dot_dispatch_has_no_variant_call_and_still_compiles(tmp_path: Path) -> None:
    project = generate_project(tmp_path / "gen", DOT, 8, 8)
    source = (project / "gemm_dispatch_xdsl.cpp").read_text()
    assert "pack_CB_variant(" not in source and "pack_b_variant ==" not in source
    assert "(void) pack_b_variant;" in source
    _compiles_clean(project, tmp_path)


def test_the_int8_dispatch_has_no_variant_call_and_still_compiles(tmp_path: Path) -> None:
    project = _generate(tmp_path / "gen", "int8", "AXPY_INT8_Pack", 32, 32)
    source = (project / "gemm_dispatch_xdsl.cpp").read_text()
    assert "pack_CB_variant(" not in source and "pack_b_variant ==" not in source
    assert "(void) pack_b_variant;" in source
    _compiles_clean(project, tmp_path)


def test_the_fp32_axpy_dispatch_has_no_unused_cast_and_compiles_clean(
    tmp_path: Path, axpy_project: Path
) -> None:
    assert "(void) pack_b_variant;" not in (axpy_project / "gemm_dispatch_xdsl.cpp").read_text()
    _compiles_clean(axpy_project, tmp_path)


# ---------------------------------------------------------------------------------------------
# Driver seam
# ---------------------------------------------------------------------------------------------

RAW = "raw_fp32_STUB_noinstr_refproto.txt"


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("pb_driver"), instrumented=0)


@pytest.fixture(scope="module")
def dot_driver(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("pb_dot"), instrumented=0, method="DOT_STUB")


@pytest.fixture(scope="module")
def int8_driver(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    return Driver(tmp_path_factory.mktemp("pb_int8"), instrumented=0, datatype="int8")


def _counts(stderr: str) -> dict[str, int]:
    line = next(ln for ln in stderr.splitlines() if ln.startswith("STUB_CALLS"))
    return {k: int(v) for k, v in (tok.split("=") for tok in line.split()[1:])}


def _overview(stdout: str) -> str:
    start = stdout.index("Run overview")
    return stdout[start : stdout.index("Benchmark Config", start)]


def _logs(d: Driver) -> list[str]:
    return sorted(p.name for p in (d.cwd / "logs").glob("raw_*"))


def test_the_driver_accepts_a_variant_and_names_its_logs(driver: Driver) -> None:
    proc = driver.run("--tmin", "0.05", "--pack-b", "col-g1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _logs(driver) == ["raw_fp32_STUB_noinstr_refproto_packb-col-g1.txt"]
    for kind in ("summary", "perf"):
        assert (driver.cwd / "logs" / f"{kind}_fp32_STUB_noinstr_refproto_packb-col-g1.txt").is_file()
    counts = _counts(proc.stderr)
    # every timed call carries variant 2 (col-g1); the stub never sees a scalar call with a variant
    assert counts["variant_calls"] == counts["total"] and counts["last_variant"] == 2
    assert counts["variant_scalar"] == 0


def test_the_suffix_sits_after_padldb_and_before_no_nc_override(driver: Driver) -> None:
    proc = driver.run("--tmin", "0.05", "--zero-pack-lines", "--pad-ldb", "8", "--pack-b", "col-g2", "--no-nc-overrides")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _logs(driver) == ["raw_fp32_STUB_noinstr_refproto_zeroline_padldb8_packb-col-g2_no_nc_override.txt"]


@pytest.mark.parametrize("name", NAMES[1:])
def test_every_name_runs_and_passes_its_correctness_check(driver: Driver, name: str) -> None:
    proc = driver.run("--tmin", "0.02", "--pack-b", name)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (row,) = driver.raw_rows(f"raw_fp32_STUB_noinstr_refproto_packb-{name}.txt")
    assert row["Status"] == "PASS"
    assert _counts(proc.stderr)["last_variant"] == NAMES.index(name)


def test_pack_b_row_behaves_like_no_flag(driver: Driver) -> None:
    proc = driver.run("--tmin", "0.05", "--pack-b", "row")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _logs(driver) == [RAW]
    counts = _counts(proc.stderr)
    assert counts["variant_calls"] == 0 and counts["last_variant"] == 0
    # ... also next to --scalar-pack, which only a non-row variant refuses
    again = driver.run("--tmin", "0.05", "--pack-b", "row", "--scalar-pack")
    assert again.returncode == 0, again.stdout + again.stderr


def test_without_the_flag_the_default_is_variant_zero(driver: Driver) -> None:
    proc = driver.run("--tmin", "0.05")
    assert _logs(driver) == [RAW]
    assert _counts(proc.stderr)["variant_calls"] == 0


def test_the_warm_up_scalar_pass_passes_variant_zero(driver: Driver) -> None:
    proc = driver.run("--no-ref-protocol", "--pack-b", "col-g4")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    counts = _counts(proc.stderr)
    # warm-up protocol: 5 warm-up calls, N scalar-pack calls (variant 0), N timed calls
    assert counts["scalar_pack"] > 0
    assert counts["variant_scalar"] == 0
    assert counts["variant_calls"] == counts["total"] - counts["scalar_pack"]
    assert counts["last_variant"] == 4


def test_the_overview_names_the_packer(driver: Driver) -> None:
    ov = _overview(driver.run("--tmin", "0.05").stdout)
    line = next(ln for ln in ov.splitlines() if "Pack B" in ln)
    assert "row (shipped xDSL packer)" in line and "[default]" in line

    ov = _overview(driver.run("--tmin", "0.05", "--pack-b", "col-gall").stdout)
    line = next(ln for ln in ov.splitlines() if "Pack B" in ln)
    assert "col-gall" in line and "[--pack-b]" in line


def test_the_flag_may_come_before_the_dataset_name(driver: Driver) -> None:
    proc = subprocess.run(
        [str(driver.binary), "--pack-b", "col-g1", "--tmin", "0.05", "tiny.dat"],
        cwd=driver.cwd, capture_output=True, text=True, timeout=120, env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Dataset        : tiny.dat" in proc.stdout


@pytest.mark.parametrize(
    "args, env, words",
    [
        (["--pack-b", "nonsense"], {}, ["--pack-b", "nonsense"]),
        (["--pack-b"], {}, ["--pack-b needs one of"]),
        (["--pack-b", "col-g1", "--scalar-pack"], {}, ["--pack-b col-g1", "scalar"]),
        (["--scalar-pack", "--pack-b", "col-g1"], {}, ["--pack-b col-g1", "scalar"]),
        (["--pack-b", "col-g1"], {"USE_SCALAR_PACK": "1"}, ["--pack-b col-g1", "USE_SCALAR_PACK"]),
    ],
)
def test_the_driver_refuses_what_it_cannot_run(
    driver: Driver, args: list[str], env: dict[str, str], words: list[str]
) -> None:
    proc = driver.run("--tmin", "0.05", *args, env=env)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert all(w in proc.stderr for w in words), proc.stderr
    assert _logs(driver) == []  # refused before any log was opened


def test_the_driver_refuses_a_variant_for_a_dot_kernel(dot_driver: Driver) -> None:
    proc = dot_driver.run("--tmin", "0.05", "--pack-b", "col-g1")
    assert proc.returncode == 1
    assert "--pack-b" in proc.stderr and "DOT" in proc.stderr
    assert dot_driver.run("--tmin", "0.05", "--pack-b", "row").returncode == 0


def test_the_driver_refuses_a_variant_for_int8(int8_driver: Driver) -> None:
    proc = int8_driver.run("--tmin", "0.05", "--pack-b", "col-g1")
    assert proc.returncode == 1
    assert "--pack-b" in proc.stderr and "int8" in proc.stderr


# ---------------------------------------------------------------------------------------------
# Script seam (--dry-run: nothing is generated, synced or run, no board contact)
# ---------------------------------------------------------------------------------------------


def _run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=HERE.parents[1], capture_output=True, text=True, timeout=30,
        env={**os.environ, **(env or {})},
    )


def _line(out: str, key: str) -> str:
    return next(ln for ln in out.splitlines() if key in ln)


@pytest.mark.parametrize("name", NAMES[1:])
def test_script_passes_pack_b_to_the_driver_and_names_its_logs(name: str) -> None:
    proc = _run_script("--dry-run", "--pack-b", name)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    line = _line(out, "Pack B")
    assert name in line and "[--pack-b]" in line
    assert f"raw_*_noinstr_refproto_packb-{name}.txt" in out
    args = _line(out, "Driver args").split()
    assert args[args.index("--pack-b") + 1] == name


def test_script_puts_the_suffix_where_the_driver_does() -> None:
    out = _run_script("--dry-run", "--zero-pack-lines", "--pack-b", "col-g1", "--no-nc-overrides").stdout
    assert "raw_*_noinstr_refproto_zeroline_packb-col-g1_no_nc_override.txt" in out
    out = _run_script("--dry-run", "--instr", "--pack-b", "col-g1").stdout
    assert "raw_*_packb-col-g1.txt" in out


def test_script_default_and_row_are_the_shipped_packer_with_no_suffix() -> None:
    out = _run_script("--dry-run").stdout
    line = _line(out, "Pack B")
    assert "row (shipped xDSL packer)" in line and "[default]" in line
    assert "packb" not in out and "--pack-b" not in _line(out, "Driver args")

    out = _run_script("--dry-run", "--pack-b", "row").stdout
    assert "row (shipped xDSL packer)" in _line(out, "Pack B")
    assert "packb" not in out


@pytest.mark.parametrize(
    "args, env, words",
    [
        (["--pack-b", "nonsense"], {}, ["unknown --pack-b name 'nonsense'", "col-g1-m4"]),
        (["--pack-b"], {}, ["--pack-b needs a name"]),
        (["--pack-b", "col-g1", "--scalar-pack"], {}, ["--pack-b col-g1", "--scalar-pack"]),
        (["--scalar-pack", "--pack-b", "col-g1"], {}, ["--pack-b col-g1", "--scalar-pack"]),
        (["--pack-b", "col-g1"], {"USE_SCALAR_PACK": "1"}, ["--pack-b col-g1", "USE_SCALAR_PACK"]),
    ],
)
def test_script_refuses_at_argument_time(args: list[str], env: dict[str, str], words: list[str]) -> None:
    proc = _run_script("--dry-run", *args, env=env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert all(w in out for w in words), out
    assert "Dependency Check" not in out and "Run overview" not in out  # before any other work


def test_script_lets_row_sit_beside_the_scalar_packer() -> None:
    proc = _run_script("--dry-run", "--pack-b", "row", "--scalar-pack")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_script_and_driver_know_the_same_names() -> None:
    header = (TEMPLATES_DIR / "pack_b_variants.h").read_text()
    script = SCRIPT.read_text()
    listed = re.search(r'PACK_B_NAMES="([^"]+)"', script)
    assert listed is not None
    assert listed.group(1).split() == NAMES
    for name in NAMES:
        assert f'"{name}"' in header


def test_the_provenance_block_names_the_packer(tmp_path: Path) -> None:
    # The block runs after the board run, so it is cut out of the script and run on its own.
    text = SCRIPT.read_text()
    start = text.index('{\n    echo "Run on the board')
    end = text.index('} > "${RUN_DIR}/provenance.txt"') + len('} > "${RUN_DIR}/provenance.txt"')
    block = text[start:end]
    for name, expect in (("col-g4", "col-g4"), ("row", "row")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        proc = subprocess.run(
            ["bash", "-c", block],
            capture_output=True, text=True, cwd=tmp_path,
            env={
                "PATH": "/usr/bin:/bin", "RUN_DIR": str(run_dir), "RISCV_USER": "u", "RISCV_REMOTE_IP": "b",
                "BUILD_BANNER": "bb", "PROTOCOL_BANNER": "pb", "ZERO_LINES_BANNER": "zb", "PACK_B_NAME": name,
                "BUILD_LOGS": "raw_x.txt", "PROVENANCE": "prov",
            },
        )
        assert proc.returncode == 0, proc.stderr
        lines = (run_dir / "provenance.txt").read_text().splitlines()
        pack_b = next(ln for ln in lines if ln.startswith("  pack B"))
        assert pack_b == f"  pack B         : {expect}"
        # next to "pack lines"
        assert lines.index(pack_b) == next(i for i, ln in enumerate(lines) if ln.startswith("  pack lines")) + 1
