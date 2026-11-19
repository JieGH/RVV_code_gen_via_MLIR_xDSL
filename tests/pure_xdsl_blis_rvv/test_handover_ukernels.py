"""
Issue #112: the hand-over bundle (gen_handover_ukernels.py), checked on the host.

The kernels themselves are checked on the board (run_issue112_handover.sh: every kernel against the
naive reference). Here: the bundle has its files, every wrapper has Héctor's argument list, only the
wrappers are exported, and the mixed kernels use the FP16 load and the widening multiply-add.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_handover_ukernels import DEFAULT_TRANSLATE  # noqa: E402

TRANSLATE = Path(DEFAULT_TRANSLATE)

# GEMMLib_yolo b292c92, src/GEMM/gemm_blis.h, with fp16_t = _Float16 and the parameter names of his kernels.
HIS_ARGS = {
    "fp32": "int mr, int nr, int kc, const float *Ar, const float *Br, float *Cr, float beta, int ldA, int ldB, int Clda",
    "f16f16f32": "int mr, int nr, int kc, const _Float16 *Ar, const _Float16 *Br, float *Cr, float beta, int ldA, int ldB, int Clda",
}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    if not TRANSLATE.exists():
        pytest.skip("mlir-translate not built")
    out = tmp_path_factory.mktemp("handover")
    subprocess.run(
        [sys.executable, str(HERE / "gen_handover_ukernels.py"), "--out", str(out), "--mr", "8,24", "--nr-max", "2"],
        check=True,
        cwd=HERE,
    )
    return out


def test_bundle_files(bundle):
    names = {p.name for p in bundle.iterdir()}
    assert names == {
        "xdsl_ukernels_fp32.c", "xdsl_ukernels_fp32.h", "xdsl_ukernels_f16f16f32.c", "xdsl_ukernels_f16f16f32.h",
        "ref_ukernels.c", "ref_ukernels.h", "test_xdsl_ukernels.c", "Makefile", "README.md",
    }


@pytest.mark.parametrize("dt", ["fp32", "f16f16f32"])
def test_wrappers_take_his_arguments_and_only_they_are_exported(bundle, dt):
    src = (bundle / f"xdsl_ukernels_{dt}.c").read_text()
    hdr = (bundle / f"xdsl_ukernels_{dt}.h").read_text()
    tiles = [(mr, nr) for mr in (8, 24) for nr in (1, 2)]
    for mr, nr in tiles:
        proto = f"void xdsl_ukernel_{mr}x{nr}_{dt}({HIS_ARGS[dt]})"
        assert proto + ";" in hdr
        assert proto + " {" in src
        # beta == 0 -> our b0 kernel, otherwise b1; packed strides MR and NR.
        assert f"if (beta == 0.0f) gemm_microkernel_v_{dt}_{mr}x{nr}_b0(kc, &one, " in src
        assert f"else              gemm_microkernel_v_{dt}_{mr}x{nr}_b1(kc, &one, " in src
        assert f"*)Ar, {mr}, ({'float' if dt == 'fp32' else '_Float16'} *)Br, {nr}, &one, Cr, Clda);" in src
    exported = re.findall(r"^void (\w+)\(", src, re.M)
    assert sorted(exported) == sorted(f"xdsl_ukernel_{mr}x{nr}_{dt}" for mr, nr in tiles)
    assert len(re.findall(r"^static void gemm_microkernel_v_", src, re.M)) == 2 * len(tiles)


def test_mixed_kernels_widen_fp16_into_fp32(bundle):
    src = (bundle / "xdsl_ukernels_f16f16f32.c").read_text()
    assert "__riscv_vle16_v_f16mf2" in src and "__riscv_vfwmacc_vf_f32m1" in src
    assert "__riscv_vle32_v_f32m1(" in src  # C loads (beta != 0) stay FP32
    assert "__riscv_vfmacc_vf_f32m1" not in src
    assert "vfwmacc" not in (bundle / "xdsl_ukernels_fp32.c").read_text()


def test_check_program_uses_his_types_and_the_reference(bundle):
    test = (bundle / "test_xdsl_ukernels.c").read_text()
    assert "/*@TABLES@*/" not in test
    assert "{24, 2, xdsl_ukernel_24x2_f16f16f32}," in test
    assert "static const uk_intrinsic_fp16 ref_f16 = ref_ukernel_f16f16f32;" in test
    assert "CC     = gcc-14" in (bundle / "Makefile").read_text()
