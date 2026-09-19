"""
Issue #7: prove pack_RB_scalar/pack_CB_scalar and the generated RVV
pack_RB_v_if/pack_CB_v_if/_else functions produce the same packed layout as the
canonical BLIS reference design (`pack_RB`/`pack_CB` in
`ext/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c`), before any pack-time
timing comparison between scalar and vector packing is trusted.

Two tests, two seams:

- `test_scalar_pack_matches_canonical_blis_reference`: host-native, no RISC-V hardware
  required. Compiles `pack_RB_scalar`/`pack_CB_scalar` (the exact same source the
  benchmark pipeline generates, via `render_scalar_pack_functions`) against a
  self-contained reference oracle transcribed directly from `gemm_blis.c`'s
  `pack_RB`/`pack_CB` (both the `orderM=='C'` and the `else` branch), and runs it
  locally with a native compiler.
- `test_rvv_vector_pack_matches_scalar_reference_on_board`: builds the real RVV
  `pack_CB_v_if`/`pack_CB_v_else`/`pack_RB_v_if`/`pack_RB_v_else` functions through the
  same xDSL/MLIR lowering pipeline `generate_pure_blis_project.py` uses, with an `RR`
  that is *not* a multiple of `vlen` (RR=15, vlen=8) so the vector tail path is
  actually exercised, and checks them against `pack_RB_scalar`/`pack_CB_scalar`
  (verified correct by the first test). Requires the RISC-V board reachable via the
  `fej` jump host; skips with a clear reason if that hardware isn't reachable.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# This file's own directory too: `generate_pure_blis_project` lives there, and without it the
# file only imports when another test module has already put the directory on the path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../gemm_blis_xdsl"))
)

from generate_pure_blis_project import (
    merge_modules,
    rename_symbols,
    render_scalar_pack_functions,
)
from config import BLISProjectConfig
from pack_builder import build_pack_module

MLIR_TRANSLATE_CANDIDATES = [
    "/home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/llvm-project/build/bin/mlir-translate",
    "/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate",
]

JUMP_HOST = "fej"
BOARD_HOST = "jbpi"
BOARD_WORKSPACE = "xdsl_rvv_microkernel/pack_correctness_test"

# ----------------------------------------------------------------------------
# Reference oracle: transcribed directly from
# ext/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c, pack_RB (lines
# 168-201) and pack_CB (lines 203-235). Mcol(a1,a2) == M[a2*ldM + a1].
#
# This is the canonical BLIS layout both pack_RB_scalar/pack_CB_scalar and the RVV
# pack_RB_v_if/pack_CB_v_if/_else builders are supposed to match. It is written
# independently here (not re-derived from pack_builder.py or
# generate_pure_blis_project.py) so it can actually catch a bug in either.
# ----------------------------------------------------------------------------
REFERENCE_ORACLE_CPP = """
// Reference oracle, transcribed from gemm_blis.c's pack_RB/pack_CB (both branches).
static void ref_pack_RB(bool order_c, int32_t mc, int32_t nc, float* M, int32_t ldM, float* Mc, int32_t RR) {
    int32_t i, j, ii, k, rr;
    if (order_c) {
        // if ((transM=='N')&&(orderM=='C')): Mc[k] = Mcol(i+ii, j) = M[j*ldM + (i+ii)]
        for (i = 0; i < mc; i += RR) {
            k = i * nc;
            rr = std::min(mc - i, RR);
            for (j = 0; j < nc; j++) {
                for (ii = 0; ii < rr; ii++) {
                    Mc[k] = M[j * ldM + (i + ii)];
                    k++;
                }
                k += (RR - rr);
            }
        }
    } else {
        // else: Mc[k] = Mcol(j, i+ii) = M[(i+ii)*ldM + j]
        for (i = 0; i < mc; i += RR) {
            k = i * nc;
            rr = std::min(mc - i, RR);
            for (j = 0; j < nc; j++) {
                for (ii = 0; ii < rr; ii++) {
                    Mc[k] = M[(i + ii) * ldM + j];
                    k++;
                }
                k += (RR - rr);
            }
        }
    }
}

static void ref_pack_CB(bool order_c, int32_t mc, int32_t nc, float* M, int32_t ldM, float* Mc, int32_t RR) {
    int32_t i, j, jj, k, nr;
    if (order_c) {
        // if ((transM=='N')&&(orderM=='C')): Mc[k] = Mcol(i, j+jj) = M[(j+jj)*ldM + i]
        for (j = 0; j < nc; j += RR) {
            k = j * mc;
            nr = std::min(nc - j, RR);
            for (i = 0; i < mc; i++) {
                for (jj = 0; jj < nr; jj++) {
                    Mc[k] = M[(j + jj) * ldM + i];
                    k++;
                }
                k += (RR - nr);
            }
        }
    } else {
        // else: Mc[k] = Mcol(j+jj, i) = M[i*ldM + (j+jj)]
        for (j = 0; j < nc; j += RR) {
            k = j * mc;
            nr = std::min(nc - j, RR);
            for (i = 0; i < mc; i++) {
                for (jj = 0; jj < nr; jj++) {
                    Mc[k] = M[i * ldM + (j + jj)];
                    k++;
                }
                k += (RR - nr);
            }
        }
    }
}
"""


def _fill_matrix_source():
    # Encodes (row, col) into the value so a transposition/stride bug shows up as a
    # mismatch instead of silently comparing equal (a symmetric fill wouldn't).
    return """
static int32_t round_up_to_multiple(int32_t x, int32_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

static void fill_known(float* M, int32_t ldM, int32_t rows, int32_t cols) {
    for (int32_t c = 0; c < cols; c++)
        for (int32_t r = 0; r < rows; r++)
            M[c * ldM + r] = (float)r + (float)c * 1000.0f;
}

static int compare_buffers(const char* name, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "[MISMATCH] %s at index %zu: got %f, expected %f\\n", name, i, a[i], b[i]);
            return 1;
        }
    }
    return 0;
}
"""


def _mlir_translate_path() -> str:
    for candidate in MLIR_TRANSLATE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    pytest.skip("mlir-translate binary not found at any known location")


def _compile_and_run_native(
    source: str, tmp_path: Path, binary_name: str
) -> subprocess.CompletedProcess[str]:
    cpp_path = tmp_path / f"{binary_name}.cpp"
    cpp_path.write_text(source)
    bin_path = tmp_path / binary_name

    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no native C++ compiler available to run the host-side pack test")

    compile_proc = subprocess.run(
        [compiler, "-O2", "-std=c++17", str(cpp_path), "-o", str(bin_path)],
        capture_output=True,
        text=True,
    )
    assert compile_proc.returncode == 0, (
        f"failed to compile host-side pack correctness harness:\n{compile_proc.stderr}"
    )

    return subprocess.run([str(bin_path)], capture_output=True, text=True)


def test_scalar_pack_matches_canonical_blis_reference(tmp_path: Path) -> None:
    source = f"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

{render_scalar_pack_functions("float", "float")}
{REFERENCE_ORACLE_CPP}
{_fill_matrix_source()}

int main() {{
    int failures = 0;

    // (mc, kc/nc, RR, ldM padding) — deliberately not multiples of each other, and
    // ldM != mc/kc, so a stride/leading-dimension mixup would show up as a mismatch.
    struct Case {{ int32_t mc; int32_t nc; int32_t RR; int32_t ld_pad; }};
    Case cases[] = {{
        {{37, 22, 15, 3}},   // RR doesn't divide mc or nc; tail block on both axes
        {{16, 15, 15, 2}},   // exact multiple on one axis (mc), tail on the other
        {{8, 8, 3, 5}},      // small, RR doesn't divide either dimension
        {{1, 1, 4, 0}},      // degenerate 1x1
    }};

    for (auto& c : cases) {{
        int32_t ldM_A = c.mc + c.ld_pad;
        int32_t ldM_B = c.nc + c.ld_pad;

        // Both pack_RB and pack_CB pad their destination out to a multiple of RR
        // along the blocked dimension (`k += (RR - rr)` / `k += (RR - nr)`), so the
        // destination buffer must be sized on the padded dimension, not mc*nc
        // directly, or the last block overflows it. Padding slots are never written,
        // so zero-initializing both buffers keeps the comparison meaningful there.
        int32_t mc_pad = round_up_to_multiple(c.mc, c.RR);
        int32_t nc_pad = round_up_to_multiple(c.nc, c.RR);

        float* A = new float[ldM_A * c.nc];
        float* Ac_scalar = new float[(size_t)mc_pad * c.nc]();
        float* Ac_ref = new float[(size_t)mc_pad * c.nc]();
        fill_known(A, ldM_A, c.mc, c.nc);

        pack_RB_scalar(c.mc, c.nc, A, ldM_A, Ac_scalar, c.RR);
        ref_pack_RB(true, c.mc, c.nc, A, ldM_A, Ac_ref, c.RR);
        failures += compare_buffers("pack_RB_scalar", Ac_scalar, Ac_ref, (size_t)mc_pad * c.nc);

        float* B = new float[ldM_B * c.mc];
        float* Bc_scalar = new float[(size_t)c.mc * nc_pad]();
        float* Bc_ref = new float[(size_t)c.mc * nc_pad]();
        fill_known(B, ldM_B, c.nc, c.mc);

        pack_CB_scalar(c.mc, c.nc, B, ldM_B, Bc_scalar, c.RR);
        ref_pack_CB(true, c.mc, c.nc, B, ldM_B, Bc_ref, c.RR);
        failures += compare_buffers("pack_CB_scalar", Bc_scalar, Bc_ref, (size_t)c.mc * nc_pad);

        delete[] A; delete[] Ac_scalar; delete[] Ac_ref;
        delete[] B; delete[] Bc_scalar; delete[] Bc_ref;
    }}

    if (failures == 0) {{
        std::printf("ALL PACK CORRECTNESS CHECKS PASSED\\n");
        return 0;
    }}
    std::printf("%d PACK CORRECTNESS CHECK(S) FAILED\\n", failures);
    return 1;
}}
"""
    result = _compile_and_run_native(source, tmp_path, "scalar_pack_correctness")
    assert result.returncode == 0, (
        f"scalar pack vs. canonical BLIS reference mismatch:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "ALL PACK CORRECTNESS CHECKS PASSED" in result.stdout


def _lower_rvv_pack_functions_to_cpp(rr: int, vlen: int, tmp_path: Path) -> str:
    """
    Builds pack_CB_v_if/_else and pack_RB_v_if/_else for a single RR via the same
    xDSL -> EmitC -> mlir-translate pipeline generate_pure_blis_project.py uses, and
    returns the resulting C++ source (function bodies only, no wrapping extern "C").
    """
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, memref, scf
    from xdsl.dialects import func as func_dialect
    from xdsl.dialects.builtin import Float32Type, IndexType, MemRefType
    from xdsl.printer import Printer

    from xdsltemplate.dialects.rvv import RVV
    from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
    from xdsltemplate.transforms.gemm_to_arith import GemmToArithPass
    from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
    from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
    from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
    from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass
    from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func_dialect.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(RVV)

    pass_pipeline = [
        GemmToArithPass(),
        MemRefToEmitCPass(),
        ArithToEmitCPass(),
        MemrefLoadToEmitcPass(),
        MemrefStoreToEmitcPass(),
        SCFToEmitCPass(),
        RVVToEmitCPass(),
    ]

    idx = IndexType()
    mem_type = MemRefType(Float32Type(), [-1])
    method = "AXPY_FP32_Pack"
    suffix = "fp32"

    pack_mod = build_pack_module(
        idx,
        mem_type,
        NR=rr,
        MR=rr,
        elem_bytes=4,
        suffix=suffix,
        vlen=vlen,
        method=method,
    )
    rename_symbols(
        pack_mod,
        {
            f"pack_CB_v_if_{method}_{suffix}_RR{rr}": f"pack_CB_v_if_{method}_{suffix}_NR{rr}",
            f"pack_CB_v_else_{method}_{suffix}_RR{rr}": f"pack_CB_v_else_{method}_{suffix}_NR{rr}",
            f"pack_RB_v_if_{method}_{suffix}_RR{rr}": f"pack_RB_v_if_{method}_{suffix}_MR{rr}",
            f"pack_RB_v_else_{method}_{suffix}_RR{rr}": f"pack_RB_v_else_{method}_{suffix}_MR{rr}",
        },
    )

    merged_mod = merge_modules(pack_mod)
    for p in pass_pipeline:
        p.apply(ctx, merged_mod)

    mlir_path = tmp_path / "pack_correctness.mlir"
    with open(mlir_path, "w") as f:
        Printer(stream=f, print_generic_format=True).print_op(merged_mod)

    cpp_path = tmp_path / "pack_correctness_lowered.cpp"
    subprocess.run(
        [
            _mlir_translate_path(),
            "-allow-unregistered-dialect",
            "-mlir-to-cpp",
            str(mlir_path),
            "-o",
            str(cpp_path),
        ],
        check=True,
    )
    return cpp_path.read_text()


def _ssh_via_jump(
    remote_cmd: str, connect_timeout: int, overall_timeout: int, batch_mode: bool = False
) -> subprocess.CompletedProcess[str]:
    """Runs `remote_cmd` on BOARD_HOST, reached through JUMP_HOST (double-hop SSH:
    the board is only reachable from the jump host, not directly)."""
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


# Every RR the shipped pinnings pack with (pj_gen/): NR=6, NR=7, MR=NR=15 (16x15), MR=16,
# MR=32. Issue #24: the test used to hardcode 15, which is two vector chunks (8 + 7), so
# the 3rd and 4th chunks that only MR=32 needs were never compared with the oracle.
SHIPPED_RR = [6, 7, 15, 16, 32]


def test_shipped_rr_is_every_mr_and_nr_config_json_pins() -> None:
    # SHIPPED_RR is a copy of the pinnings in config.json; this is what stops it drifting.
    config = Path(__file__).parent / "config.json"
    pinned = {dim for row in BLISProjectConfig.from_json(str(config)) for dim in row.mr_nr_family[0]}
    assert pinned == set(SHIPPED_RR)


@pytest.mark.parametrize("rr", SHIPPED_RR, ids=lambda rr: f"RR{rr}")
def test_rvv_vector_pack_matches_scalar_reference_on_board(
    tmp_path: Path, rr: int
) -> None:
    if not _board_reachable():
        pytest.skip(
            f"RVV board not reachable via jump host '{JUMP_HOST}' -> '{BOARD_HOST}'"
        )

    vlen = 8  # elements per vector register at VLEN=256, fp32
    lowered_cpp = _lower_rvv_pack_functions_to_cpp(rr, vlen, tmp_path)

    harness = f"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <riscv_vector.h>

extern "C" {{
{lowered_cpp}
}}

{render_scalar_pack_functions("float", "float")}
{REFERENCE_ORACLE_CPP}
{_fill_matrix_source()}

int main() {{
    int failures = 0;
    const int32_t RR = {rr};

    // (mc, kc) chosen so RR doesn't divide mc: exercises the partial-final-block tail
    // (rr < RR) and, when RR > vlen=8, the within-block vector tail chunk (chunk_avl <
    // vlen on the last chunk). Issue #24: the 3rd and 4th chunks only exist at RR=32,
    // so the cases also cover a partial block that ends in each of them, and full blocks
    // (all four chunks whole). CB's RR-blocked dimension is its second argument, `mc`.
    struct Shape {{ int32_t mc; int32_t kc; }};
    const Shape shapes[] = {{
        {{37, 22}},   // tail 5 at RR=32 (1st chunk), 5 at 16, 7 at 15, 2 at 7, 1 at 6
        {{59, 22}},   // tail 27 at RR=32 (3 whole chunks + 3 in the 4th), 11 at 16, 14 at 15
        {{51, 9}},    // tail 19 at RR=32 (2 whole chunks + 3 in the 3rd), 3 at 16, 6 at 15
        {{96, 7}},    // exact multiple of 32, 16 and 6: full blocks only, no tail
    }};

    for (const Shape& sh : shapes) {{
        const int32_t mc = sh.mc, kc = sh.kc;
        const int failures_before = failures;
        int32_t ldM_A = mc + 3;
        int32_t ldM_B = kc + 3;

        // pack_*_v_if implements the same orderM=='C' branch as pack_*_scalar (the only
        // branch the benchmark dispatcher ever calls); pack_*_v_else implements the
        // *other* branch of the reference's if/else (row-major source access) — it is
        // generated but never wired into the dispatcher, so its oracle is the reference
        // function's else-branch directly, not pack_*_scalar.
        int32_t mc_pad = round_up_to_multiple(mc, RR);
        // CB's RR-blocked dimension is its second argument (N-tile width, `mc` here),
        // not its first (`kc`) — see pack_CB_scalar(mc_param=kc, nc_param=mc, ...) below.
        int32_t nc_pad = round_up_to_multiple(mc, RR);

        // The `_else` packers read their source row-major, M[(i+ii)*ldM + j], which runs past
        // an ldM*cols column-major buffer whenever rows > cols (it did at 37x22 too, and passed
        // only because whatever sat beyond the buffer read the same both times). Size the
        // sources for that read, and zero them so what it reads is the same every time.
        float* A = new float[(size_t)ldM_A * std::max(kc, mc)]();
        float* Ac_vec_if = new float[(size_t)mc_pad * kc]();
        float* Ac_vec_else = new float[(size_t)mc_pad * kc]();
        float* Ac_scalar = new float[(size_t)mc_pad * kc]();
        float* Ac_ref_else = new float[(size_t)mc_pad * kc]();
        fill_known(A, ldM_A, mc, kc);

        pack_RB_v_if_AXPY_FP32_Pack_fp32_MR{rr}(A, Ac_vec_if, mc, kc, ldM_A);
        pack_RB_v_else_AXPY_FP32_Pack_fp32_MR{rr}(A, Ac_vec_else, mc, kc, ldM_A);
        pack_RB_scalar(mc, kc, A, ldM_A, Ac_scalar, RR);
        ref_pack_RB(false, mc, kc, A, ldM_A, Ac_ref_else, RR);
        failures += compare_buffers("pack_RB_v_if", Ac_vec_if, Ac_scalar, (size_t)mc_pad * kc);
        failures += compare_buffers("pack_RB_v_else", Ac_vec_else, Ac_ref_else, (size_t)mc_pad * kc);

        float* B = new float[(size_t)ldM_B * std::max(kc, mc)]();
        float* Bc_vec_if = new float[(size_t)kc * nc_pad]();
        float* Bc_vec_else = new float[(size_t)kc * nc_pad]();
        float* Bc_scalar = new float[(size_t)kc * nc_pad]();
        float* Bc_ref_else = new float[(size_t)kc * nc_pad]();
        fill_known(B, ldM_B, kc, mc);

        pack_CB_v_if_AXPY_FP32_Pack_fp32_NR{rr}(B, Bc_vec_if, kc, mc, ldM_B);
        pack_CB_v_else_AXPY_FP32_Pack_fp32_NR{rr}(B, Bc_vec_else, kc, mc, ldM_B);
        pack_CB_scalar(kc, mc, B, ldM_B, Bc_scalar, RR);
        ref_pack_CB(false, kc, mc, B, ldM_B, Bc_ref_else, RR);
        failures += compare_buffers("pack_CB_v_if", Bc_vec_if, Bc_scalar, (size_t)kc * nc_pad);
        failures += compare_buffers("pack_CB_v_else", Bc_vec_else, Bc_ref_else, (size_t)kc * nc_pad);

        delete[] A; delete[] Ac_vec_if; delete[] Ac_vec_else; delete[] Ac_scalar; delete[] Ac_ref_else;
        delete[] B; delete[] Bc_vec_if; delete[] Bc_vec_else; delete[] Bc_scalar; delete[] Bc_ref_else;
        if (failures != failures_before) std::fprintf(stderr, "  ^ RR=%d mc=%d kc=%d\\n", RR, mc, kc);
    }}

    if (failures == 0) {{
        std::printf("ALL PACK CORRECTNESS CHECKS PASSED\\n");
        return 0;
    }}
    std::printf("%d PACK CORRECTNESS CHECK(S) FAILED\\n", failures);
    return 1;
}}
"""
    harness_path = tmp_path / "rvv_pack_correctness.cpp"
    harness_path.write_text(harness)

    remote_mkdir = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", JUMP_HOST, f"mkdir -p {BOARD_WORKSPACE}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert remote_mkdir.returncode == 0, (
        f"could not create workspace on {JUMP_HOST}: {remote_mkdir.stderr}"
    )

    scp_proc = subprocess.run(
        [
            "scp",
            "-o",
            "ConnectTimeout=10",
            str(harness_path),
            f"{JUMP_HOST}:{BOARD_WORKSPACE}/rvv_pack_correctness.cpp",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert scp_proc.returncode == 0, (
        f"failed to copy harness to {JUMP_HOST}: {scp_proc.stderr}"
    )

    remote_cmd = (
        f"cd {BOARD_WORKSPACE} && "
        "g++ -O2 -std=c++17 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing "
        "rvv_pack_correctness.cpp -o rvv_pack_correctness.x && "
        "./rvv_pack_correctness.x"
    )
    run_proc = _ssh_via_jump(remote_cmd, connect_timeout=15, overall_timeout=120)

    assert run_proc.returncode == 0, (
        f"RVV pack correctness check failed on board:\n"
        f"stdout:\n{run_proc.stdout}\nstderr:\n{run_proc.stderr}"
    )
    assert "ALL PACK CORRECTNESS CHECKS PASSED" in run_proc.stdout
