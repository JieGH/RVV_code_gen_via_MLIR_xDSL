"""Issue #55: host layout tests for templates/pack_b_variants.h (plain C fallback path).

Every variant 1..6 must give byte-for-byte the Bc of the canonical pack_CB_scalar, and
must leave pad slots as allocated. The RVV intrinsic branch cannot be compiled on the host;
it is first compiled at the board gate.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from generate_pure_blis_project import render_scalar_pack_functions

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"

RRS = [1, 4, 6, 7, 8, 9, 15, 16, 24, 31]
KCS = [1, 7, 128, 130, 256]

HARNESS = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "pack_b_variants.h"

%(scalar)s

static int run(int variant, int32_t kc, int32_t nc, int32_t ldb, int32_t RR) {
    std::vector<float> B((size_t) nc * ldb + 16);
    for (size_t i = 0; i < B.size(); ++i) B[i] = 1.0f + (float) i;
    const int32_t panels = (nc + RR - 1) / RR;
    const size_t n = (size_t) panels * RR * kc;
    const float SENT = -12345.0f;
    std::vector<float> ref(n, SENT), got(n, SENT);
    pack_CB_scalar(kc, nc, B.data(), ldb, ref.data(), RR);
    pack_CB_variant(variant, B.data(), got.data(), kc, nc, ldb, RR);
    if (std::memcmp(ref.data(), got.data(), n * sizeof(float)) != 0) {
        for (size_t i = 0; i < n; ++i)
            if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "MISMATCH v=%%d kc=%%d nc=%%d ldb=%%d RR=%%d idx=%%zu got=%%f want=%%f\n",
                             variant, kc, nc, ldb, RR, i, got[i], ref[i]);
                break;
            }
        return 1;
    }
    return 0;
}

int main() {
    const int RRs[] = {%(rrs)s};
    const int kcs[] = {%(kcs)s};
    int bad = 0, cases = 0;
    for (int v = 1; v <= 6; ++v)
        for (int RR : RRs)
            for (int kc : kcs) {
                const int ncs[] = {384, 383, RR * 3, RR * 3 + 1, RR, 1};
                const int ldbs[] = {kc, kc + 8, kc + 128};
                for (int nc : ncs)
                    for (int ldb : ldbs) { bad += run(v, kc, nc, ldb, RR); ++cases; }
            }
    std::printf("cases=%%d bad=%%d\n", cases, bad);
    return bad ? 1 : 0;
}
"""


def _compiler() -> str:
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        pytest.skip("no native C++ compiler available for the host pack test")
    return cc


def _source() -> str:
    return HARNESS % {
        "scalar": render_scalar_pack_functions("float", "float"),
        "rrs": ", ".join(map(str, RRS)),
        "kcs": ", ".join(map(str, KCS)),
    }


def _build_and_run(tmp_path: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    cpp = tmp_path / "h.cpp"
    cpp.write_text(_source())
    exe = tmp_path / "h"
    comp = subprocess.run(
        [_compiler(), "-O1", "-std=c++17", "-Wall", "-Wextra", "-I", str(TEMPLATES), *extra, str(cpp), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert comp.returncode == 0, comp.stderr
    return subprocess.run([str(exe)], capture_output=True, text=True)


def test_all_variants_match_canonical_scalar_pack(tmp_path: Path) -> None:
    proc = _build_and_run(tmp_path, [])
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "bad=0" in proc.stdout


def test_header_does_not_need_real_riscv_vector_h(tmp_path: Path) -> None:
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "riscv_vector.h").write_text("")
    proc = _build_and_run(tmp_path, ["-I", str(stub)])
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "bad=0" in proc.stdout


def test_variant_names_round_trip_and_unknown(tmp_path: Path) -> None:
    src = r"""
#include <cstdio>
#include <cstring>
#include "pack_b_variants.h"
int main() {
    const char* want[] = {"row", "row-intr", "col-g1", "col-g2", "col-g4", "col-gall", "col-g1-m4"};
    for (int v = 0; v < 7; ++v) {
        if (std::strcmp(pack_b_variant_name(v), want[v]) != 0) return 1;
        if (pack_b_variant_from_name(want[v]) != v) return 2;
    }
    if (pack_b_variant_from_name("nope") != -1) return 3;
    if (pack_b_variant_from_name("") != -1) return 4;
    if (pack_b_variant_from_name("col-g3") != -1) return 5;
    return 0;
}
"""
    cpp = tmp_path / "n.cpp"
    cpp.write_text(src)
    exe = tmp_path / "n"
    comp = subprocess.run(
        [_compiler(), "-O1", "-std=c++17", "-I", str(TEMPLATES), str(cpp), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert comp.returncode == 0, comp.stderr
    assert subprocess.run([str(exe)]).returncode == 0


def test_variant_zero_aborts(tmp_path: Path) -> None:
    src = r"""
#include "pack_b_variants.h"
int main() { float b[4] = {0}, c[4] = {0}; pack_CB_variant(0, b, c, 1, 1, 1, 1); return 0; }
"""
    cpp = tmp_path / "z.cpp"
    cpp.write_text(src)
    exe = tmp_path / "z"
    comp = subprocess.run(
        [_compiler(), "-O1", "-std=c++17", "-I", str(TEMPLATES), str(cpp), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert comp.returncode == 0, comp.stderr
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "not implemented" in proc.stderr
