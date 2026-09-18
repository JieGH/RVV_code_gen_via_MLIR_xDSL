"""
Issue #14, part 1: `e2e` must be 2mnk / `RVV_Time_ms`, not 2mnk / (pack + compute).

`Vector_Pack_Time_ms + Compute_Time_ms` omits the time that falls *between* the
per-tile `chrono::now()` pairs — no timer is running there, so it is charged to
neither column. `RVV_Time_ms` is the outer wall clock around the whole N_REPS loop
and cannot lose that time, which is what makes it comparable to the reference
design's own `2mnk / wall(gemm_blis_B3A2C0)`.

Ground truth is the published B1 row of `logs/run_2026-09-18_044110`, the same run
the issue's evidence table was computed from: RVV_Time_ms 120.481, pack+compute
117.407, and a reference of 7.02 GFLOPS. The issue states the corrected figure is
6.68 (-4.8% vs reference), against the 6.86 (-2.3%) the old denominator printed.
"""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_baseline"))

from compare_ref_baseline import load_ours  # noqa: E402

RAW_HEADER = (
    "Project,Layer,M,N,K,Variant,Status,KerGOPs_Harmonic,KerGOPs_Min,KerGOPs_P5,"
    "KerGOPs_P95,KerGOPs_Max,KerGOPs_IQR,KerGOPs_Mean,KerGOPs_Std,KerGOPs_Median,"
    "RVV_Pack_GBs,Scl_Pack_GBs,RVV_Time_ms,Ref_Time_ms,Vector_Pack_Time_ms,"
    "Scalar_Pack_Time_ms,Compute_Time_ms"
)

# Verbatim from logs/run_2026-09-18_044110/all_raw_fp32_AXPY_FP32_Pack.txt.
B1_ROW = (
    "fp32_AXPY_FP32_Pack_vlen256_MR32NR6,B1,1024,384,1024,MR32_NR6,SKIP,"
    "8.15863,8.05363,8.11358,8.20547,8.26268,0.0392234,8.15873,0.0282286,8.15873,"
    "0.308391,0.277866,120.481,0,18.7008,20.7552,98.7061"
)

# The same row as an uninstrumented (RVV_INSTRUMENTED=0) build writes it: the inner
# pack/compute timers are compiled out, so those columns are nan and only the outer
# wall clock survives.
B1_ROW_NOINSTR = (
    "fp32_AXPY_FP32_Pack_vlen256_MR32NR6,B1,1024,384,1024,MR32_NR6,SKIP,"
    "8.15863,8.05363,8.11358,8.20547,8.26268,0.0392234,8.15873,0.0282286,8.15873,"
    "nan,nan,120.481,0,nan,nan,nan"
)

B1_OPS = 2.0 * 1024 * 384 * 1024
RAW_NAME = "all_raw_fp32_AXPY_FP32_Pack.txt"


def _write_run(tmp_path: Path, *rows: str, name: str = RAW_NAME) -> Path:
    run = tmp_path / "run_2026-09-18_044110"
    run.mkdir(exist_ok=True)
    (run / name).write_text(RAW_HEADER + "\n" + "\n".join(rows) + "\n")
    return run


def test_e2e_uses_the_outer_wall_clock_not_the_sum_of_the_inner_timers(tmp_path: Path):
    run = _write_run(tmp_path, B1_ROW)

    ours = load_ours([run])["B1"]

    # 2mnk / 120.481 ms == 6.68 GFLOPS, the issue's corrected figure.
    assert ours["e2e"] == pytest.approx(B1_OPS / (120.481 * 1e6))
    assert round(ours["e2e"], 2) == 6.68
    # and specifically NOT the old denominator, which printed 6.86.
    assert round(B1_OPS / (117.4069 * 1e6), 2) == 6.86


def test_e2e_against_the_reference_is_the_full_gap_not_the_understated_one(tmp_path: Path):
    run = _write_run(tmp_path, B1_ROW)
    ref_gflops = 7.02  # reference design's B1, from the issue's evidence table

    ours = load_ours([run])["B1"]

    assert round(100 * (ours["e2e"] / ref_gflops - 1), 1) == -4.8


def test_pack_fraction_still_comes_from_the_instrumented_breakdown(tmp_path: Path):
    run = _write_run(tmp_path, B1_ROW)

    ours = load_ours([run])["B1"]

    assert ours["vpack_ms"] == pytest.approx(18.7008)
    assert ours["compute_ms"] == pytest.approx(98.7061)
    assert ours["pack_frac"] == pytest.approx(18.7008 / (18.7008 + 98.7061))


def test_uninstrumented_rows_still_yield_e2e_and_report_no_pack_breakdown(tmp_path: Path):
    run = _write_run(tmp_path, B1_ROW_NOINSTR)

    ours = load_ours([run])["B1"]

    assert ours["e2e"] == pytest.approx(B1_OPS / (120.481 * 1e6))
    assert math.isnan(ours["pack_frac"])
    assert math.isnan(ours["vpack_ms"])


def test_a_named_raw_file_lets_the_uninstrumented_run_be_compared(tmp_path: Path):
    name = "all_raw_fp32_AXPY_FP32_Pack_noinstr.txt"
    run = _write_run(tmp_path, B1_ROW_NOINSTR, name=name)

    assert load_ours([run]) == {}
    assert "B1" in load_ours([run], raw_name=name)
