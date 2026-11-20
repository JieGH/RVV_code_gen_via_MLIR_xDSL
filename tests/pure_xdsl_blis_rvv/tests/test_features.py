"""Issue #128: one host smoke test per feature the Project Generator offers (see README.md).

Each test generates a project from a small config in a temp folder and checks the files the
bench builds from are there. The deeper checks of each feature live in the tests the README names.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from test_pad_zeroing import MLIR_TRANSLATE_CANDIDATES  # noqa: E402

FILES = ("Makefile", "driver_benchmark.cpp", "gemm_dispatch_xdsl.cpp", "gemm_pack_xdsl.cpp",
         "gemm_b0_xdsl.cpp", "gemm_b1_xdsl.cpp", "generated_kernels.h")


def _generate(root: Path, rows: str) -> list[Path]:
    mlir_translate = next((p for p in MLIR_TRANSLATE_CANDIDATES if Path(p).exists()), None)
    if mlir_translate is None:
        pytest.skip("mlir-translate binary not found at any known location")
    (root / "config.json").write_text(f"[{rows}]")
    proc = subprocess.run(
        [sys.executable, str(HERE / "generate_pure_blis_project.py"), "--config", "config.json",
         "--mlir-translate", mlir_translate],
        cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    order = (root / "pj_gen" / "compile_order.txt").read_text().split()
    projects = [root / "pj_gen" / name for name in order]
    for project in projects:
        missing = [f for f in FILES if not (project / f).is_file()]
        assert not missing, f"{project.name} lacks {missing}"
    return projects


@pytest.mark.parametrize(
    "datatype, method, expect",
    [
        ("fp32", "AXPY_FP32_Pack", "fp32_AXPY_FP32_Pack_vlen256_MR16NR15"),
        ("fp32", "DOT_FP32_LANE_Pack", "fp32_DOT_FP32_LANE_Pack_vlen256_MR16NR15"),
        ("f16f16f32", "AXPY_FP32_Pack", "f16f16f32_AXPY_FP32_Pack_vlen256_MR16NR15"),
        ("int8", "AXPY_INT8_Pack", None),
    ],
)
def test_kernel_generation_per_data_type(tmp_path: Path, datatype: str, method: str, expect: str | None) -> None:
    (project,) = _generate(tmp_path, f'["{datatype}", "{method}", [16, 15], 256, "jbpi", ["x.dat"], true, 5]')
    if expect:
        assert project.name == expect


def test_pinned_rows_give_one_project_per_tile(tmp_path: Path) -> None:
    # Tile Search: one pinned row (8th value present) per Candidate Tile, one project each, in order.
    rows = ", ".join(f'["fp32", "AXPY_FP32_Pack", [{mr}, {nr}], 256, "jbpi", ["x.dat"], true, 5]'
                     for mr, nr in ((16, 15), (32, 6), (24, 8)))
    projects = _generate(tmp_path, rows)
    assert [p.name.rsplit("_", 1)[1] for p in projects] == ["MR16NR15", "MR32NR6", "MR24NR8"]


def test_pinned_row_with_a_family_uses_its_first_tile(tmp_path: Path) -> None:
    # config.py: an 8th value means pinned, and a pinned row keeps only the first MRxNR of its list.
    (project,) = _generate(tmp_path, '["fp32", "AXPY_FP32_Pack", [[16, 15], [32, 6]], 256, "jbpi", ["x.dat"], true, 5]')
    assert project.name.endswith("_MR16NR15")


def test_search_row_builds_every_tile_up_to_the_largest(tmp_path: Path) -> None:
    # No 8th value: search mode, one project holding every kernel MRxNR up to the family's max.
    (project,) = _generate(tmp_path, '["fp32", "AXPY_FP32_Pack", [[16, 15], [32, 6]], 256, "jbpi", ["x.dat"], true]')
    assert project.name == "fp32_AXPY_FP32_Pack_vlen256"
