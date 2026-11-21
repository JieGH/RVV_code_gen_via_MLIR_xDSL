"""
Issue #54: the reference design with our one-column B packer (`ref_baseline/ref_pack_fix.py`).

What is checked here: the patch renames the reference's `pack_CB` once and appends the one-column
packer, refuses `ext/` and a second patch; the two functions taken from the PATCHED real gemm_blis.c
write the same Bc byte for byte (host fallback path, many sizes, pad slots untouched); and the RISC-V
path compiles with the cross compiler to one vle32 (LMUL 4) and one vsse32 per strip.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.append(os.path.join(HERE, "ref_baseline"))

import ref_pack_fix as rpf  # noqa: E402

REF_SRC = HERE.parents[2] / "ext" / "RVV_ukernels_benchmark" / "gemm_blis_family" / "src"
CROSS_CC = Path("/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc")
CROSS_OBJDUMP = Path("/opt/riscv/bin/riscv64-unknown-linux-gnu-objdump")

needs_ref = pytest.mark.skipif(not (REF_SRC / "gemm_blis.c").exists(), reason="reference tree not next to the repository")


def _function(src: str, signature: str) -> str:
    """The text of one C function, from its signature to its matching closing brace."""
    start = src.index(signature)
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


def _patched_functions() -> str:
    patched = rpf.patch_gemm_blis((REF_SRC / "gemm_blis.c").read_text())
    orig = _function(patched, rpf.RENAMED)
    new = patched[patched.index("/* " + rpf.MARK):]  # the appended block, with its #includes
    return orig + "\n" + new


PRELUDE = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#define FP32
#define DTYPE float
#define min(a,b) (((a)<(b))?(a):(b))
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
"""

HARNESS = r"""
static int run(int mc, int nc, int ldM, int RR) {
  int panels = (nc + RR - 1) / RR, i, bad = 0;
  size_t n = (size_t) panels * RR * mc;
  float *B = malloc(sizeof(float) * ((size_t) nc * ldM + 8));
  float *ref = malloc(sizeof(float) * n), *got = malloc(sizeof(float) * n);
  for (i = 0; i < nc * ldM + 8; i++) B[i] = 1.0f + (float) i;
  for (size_t e = 0; e < n; e++) { ref[e] = -12345.0f; got[e] = -12345.0f; }
  pack_CB_orig('C', 'N', mc, nc, B, ldM, ref, RR);
  pack_CB('C', 'N', mc, nc, B, ldM, got, RR);
  if (memcmp(ref, got, sizeof(float) * n) != 0) bad = 1;
  free(B); free(ref); free(got);
  return bad;
}
int main(void) {
  int mcs[] = {1, 7, 31, 32, 33, 64, 128, 130, 256}, ncs[] = {1, 5, 14, 15, 16, 30, 47, 384};
  int rrs[] = {1, 6, 8, 14, 15, 16, 24}, cases = 0, bad = 0;
  for (int a = 0; a < 9; a++) for (int b = 0; b < 8; b++) for (int c = 0; c < 7; c++) for (int pad = 0; pad < 2; pad++) {
    int mc = mcs[a], ldM = mc + 3 * pad;
    cases++; bad += run(mc, ncs[b], ldM, rrs[c]);
  }
  printf("cases=%d bad=%d\n", cases, bad);
  return bad != 0;
}
"""


def test_patch_renames_once_and_appends_the_one_column_packer():
    src = "/* head */\n" + rpf.ANCHOR + "\n  body();\n}\n"
    out = rpf.patch_gemm_blis(src)
    assert out.count(rpf.RENAMED) == 1
    assert out.count(rpf.ANCHOR) == 1  # the new pack_CB
    assert out.index(rpf.RENAMED) < out.index(rpf.ANCHOR)
    assert "pack_CB_orig( orderM, transM, mc, nc, M, ldM, Mc, RR );" in out
    assert "__riscv_vle32_v_f32m4" in out and "__riscv_vsse32_v_f32m4" in out


def test_patch_refuses_a_second_patch_and_a_changed_reference():
    out = rpf.patch_gemm_blis(rpf.ANCHOR + "}\n")
    with pytest.raises(rpf.PatchError, match="already patched"):
        rpf.patch_gemm_blis(out)
    with pytest.raises(rpf.PatchError, match="found 0 times"):
        rpf.patch_gemm_blis("void pack_CB(void){}\n")


def test_patch_refuses_the_ext_tree(tmp_path):
    target = tmp_path / "ext" / "gemm_blis.c"
    target.parent.mkdir()
    target.write_text(rpf.ANCHOR + "}\n")
    r = subprocess.run([sys.executable, str(HERE / "ref_baseline" / "ref_pack_fix.py"), "patch", str(target)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "REFUSING" in r.stderr
    assert target.read_text() == rpf.ANCHOR + "}\n"


@needs_ref
@pytest.mark.skipif(shutil.which("gcc") is None, reason="no host gcc")
def test_patched_packer_writes_the_same_bc_as_the_reference_packer(tmp_path):
    c = tmp_path / "h.c"
    c.write_text(PRELUDE + _patched_functions() + HARNESS)
    exe = tmp_path / "h"
    subprocess.run(["gcc", "-O2", "-std=gnu11", "-Wall", "-o", str(exe), str(c)], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "cases=1008 bad=0"


@needs_ref
@pytest.mark.skipif(not (CROSS_CC.exists() and CROSS_OBJDUMP.exists()), reason="no RISC-V cross compiler")
def test_riscv_path_compiles_to_one_column_vector_loads_and_strided_stores(tmp_path):
    c = tmp_path / "p.c"
    c.write_text(PRELUDE + _patched_functions())
    obj = tmp_path / "p.o"
    subprocess.run([str(CROSS_CC), "-O3", "-march=rv64gcvzfh", "-mabi=lp64d", "-c", "-o", str(obj), str(c)], check=True)
    dis = subprocess.run([str(CROSS_OBJDUMP), "-d", str(obj)], capture_output=True, text=True, check=True).stdout
    new = dis[dis.index("<pack_CB>:"):]  # the appended packer (pack_CB_orig comes first in the object)
    # GCC may encode e32/m4 as e8/m1 + vle32 (same 32 lanes), so check the memory instructions, not the vtype text.
    assert "vle32.v" in new and "vsse32.v" in new
