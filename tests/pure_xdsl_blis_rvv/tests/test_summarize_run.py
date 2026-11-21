"""
`summarize_run.py` turns a run's merged `all_raw_<dtype>_<method>.txt` CSV (written by
`compile_macro_bench.sh`'s log-fetch step) into a per-case Markdown comparison table:
Case, Shape, MRxNR, pack/compute ms, pack % of total, KerGOPs, Ref GOPs, speedup vs
ref, and Vector Pack Gain %.

Ground truth for the percentage/gain math comes from real, already-published rows,
not values derived the same way the code under test derives them:
- S1 and B3's full-precision Scalar/Vector_Pack_Time_ms and Compute_Time_ms are from
  `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_MR{32NR7,16NR15}_2026-09-16_151844/raw_fp32_AXPY_FP32_Pack.txt`.
  Their Gain %/pack % are hand-computed here from those figures, and B3's -12.7%
  independently matches `.scratch/issues/0003-scalar-vs-vector-pack-timing.md`'s
  published table (rounded from the same underlying run).
- Ref GOPs/speedup use the standard GEMM FLOP-count formula (2*M*N*K FLOPs) with
  clean round numbers chosen to be hand-verifiable, not copied from the run logs
  (skip_ref_run was true throughout that run, so Ref_Time_ms is 0 in every real row).
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "readers"))

from summarize_run import build_comparison_rows

RAW_HEADER = (
    "Layer,M,N,K,Variant,Status,KerGOPs_Harmonic,KerGOPs_Min,KerGOPs_P5,KerGOPs_P95,"
    "KerGOPs_Max,KerGOPs_IQR,KerGOPs_Mean,KerGOPs_Std,KerGOPs_Median,RVV_Pack_GBs,"
    "Scl_Pack_GBs,RVV_Time_ms,Ref_Time_ms,Vector_Pack_Time_ms,Scalar_Pack_Time_ms,"
    "Compute_Time_ms"
)

# Real rows from the raw log cited above (skip_ref_run=true -> Ref_Time_ms=0).
S1_ROW = "S1,1000,1000,1000,MR32_NR7,SKIP,8.82503,8.81371,8.81371,8.84751,8.84751,0.0114747,8.82505,0.0122616,8.81856,0.374711,0.347694,252.448,0,21.3498,23.0088,226.628"
B3_ROW = "B3,64,384,384,MR16_NR15,SKIP,10.371,8.42218,9.18505,12.4373,12.5792,2.00097,10.4967,1.15838,11.0232,0.747423,0.842598,3.05567,0,1.18371,1.05001,1.81992"
# Recovered live from a real jbpi board run during this session (see incident notes),
# not derived by this script: tests/pure_xdsl_blis_rvv/logs/RECOVERED_20260916/.
B1_ROW = "B1,1024,384,1024,MR32_NR6,SKIP,8.56053,8.47258,8.5133,8.61107,8.64598,0.0395438,8.56063,0.029337,8.56019,0.315192,0.285137,115.435,0,18.2973,20.2259,94.072"

PERF_HEADER = (
    "Layer,M,N,K,Variant,Status,Counters_Valid,Vector_Pack_Cycles,Vector_Pack_Instructions,"
    "Vector_Pack_IPC,Vector_Pack_L1D_Read_Miss_Pct,Vector_Pack_L1D_Write_Miss_Pct,"
    "Vector_Pack_dTLB_Misses,Vector_Pack_Stall_BE_Pct,Vector_Pack_Stall_FE_Pct,"
    "Vector_Pack_Branch_Misses,Scalar_Pack_Cycles,Scalar_Pack_Instructions,Scalar_Pack_IPC,"
    "Scalar_Pack_L1D_Read_Miss_Pct,Scalar_Pack_L1D_Write_Miss_Pct,Scalar_Pack_dTLB_Misses,"
    "Scalar_Pack_Stall_BE_Pct,Scalar_Pack_Stall_FE_Pct,Scalar_Pack_Branch_Misses,Compute_Cycles,"
    "Compute_Instructions,Compute_IPC,Compute_L1D_Read_Miss_Pct,Compute_L1D_Write_Miss_Pct,"
    "Compute_dTLB_Misses,Compute_Stall_BE_Pct,Compute_Stall_FE_Pct,Compute_Branch_Misses,"
    "Vector_Pack_Cycles_CV_Pct,Compute_Cycles_CV_Pct,Compute_FLOPs_per_Cycle,"
    "Arith_Intensity_FLOPs_per_Byte"
)
# Real perf row recovered from the same live jbpi run as B1_ROW above.
B1_PERF_ROW = (
    "B1,1024,384,1024,MR32_NR6,SKIP,1,2.97415e+07,1.26141e+06,0.0424124,10.6618,0.131116,"
    "36758.9,94.5061,0.244764,417.387,3.28527e+07,8.67161e+06,0.263955,5.62793,2.00684,"
    "36892.7,81.6448,0.604587,982.568,1.54258e+08,9.25975e+07,0.600279,1.32121,0.0483824,"
    "262126,48.2264,0.260135,4132.3,0.643585,0.31776,5.22053,139.636"
)
# Same case, but a multiplexed/failed counter read (Counters_Valid=0) -- driver writes
# NaN into every other perf column in this case, matching write_perf_metrics' nan_v.
B1_PERF_ROW_INVALID = (
    "B1,1024,384,1024,MR32_NR6,SKIP,0,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,"
    "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan"
)


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    csv_path = tmp_path / "all_raw_fp32_AXPY_FP32_Pack.txt"
    csv_path.write_text(RAW_HEADER + "\n" + "\n".join(rows) + "\n")
    return csv_path


def _write_perf_csv(tmp_path: Path, rows: list[str]) -> Path:
    csv_path = tmp_path / "all_perf_fp32_AXPY_FP32_Pack.txt"
    csv_path.write_text(PERF_HEADER + "\n" + "\n".join(rows) + "\n")
    return csv_path


def test_pack_percentages_and_gain_match_a_real_published_run(tmp_path: Path) -> None:
    rows = build_comparison_rows(str(_write_csv(tmp_path, [S1_ROW, B3_ROW])))
    by_case = {r["Case"]: r for r in rows}

    s1 = by_case["S1"]
    assert s1["Shape (m×n×k)"] == "1000×1000×1000"
    assert s1["MRxNR"] == "32x7"
    assert s1["Scalar Pack % of total"] == "9.2%"
    assert s1["Vector Pack % of total"] == "8.6%"
    assert s1["Vector Pack Gain %"] == "+7.2%"

    b3 = by_case["B3"]
    # -12.7% independently matches the published write-up's table for this exact case.
    assert b3["Vector Pack Gain %"] == "-12.7%"
    assert b3["Scalar Pack % of total"] == "36.6%"
    assert b3["Vector Pack % of total"] == "39.4%"


def test_kergops_jitter_bracket_matches_the_console_table_format(tmp_path: Path) -> None:
    # Format and rounding mirror driver_benchmark.cpp's console summary line exactly
    # (min/p5/p95/max/iqr/s: 1 decimal; CV%: integer; mx/md: 2 decimals).
    rows = build_comparison_rows(str(_write_csv(tmp_path, [B1_ROW])))
    b1 = rows[0]
    assert (
        b1["KerGOPs Jitter [min, p5, p95, max, iqr, s, CV%, mx/md]"]
        == "[min:8.5, p5:8.5, p95:8.6, max:8.6, iqr:0.0, s:0.0, CV:0%, mx/md:1.01]"
    )


def test_perf_counters_joined_across_all_three_regions_for_a_real_run(tmp_path: Path) -> None:
    raw = _write_csv(tmp_path, [B1_ROW])
    perf = _write_perf_csv(tmp_path, [B1_PERF_ROW])
    rows = build_comparison_rows(str(raw), str(perf))
    b1 = rows[0]

    assert b1["Counters Valid"] == "yes"
    assert b1["Vector Pack IPC"] == "0.04"
    assert b1["Vector Pack L1D Read Miss %"] == "10.7%"
    assert b1["Vector Pack L1D Write Miss %"] == "0.1%"
    assert b1["Vector Pack Stall BE %"] == "94.5%"
    assert b1["Scalar Pack IPC"] == "0.26"
    assert b1["Scalar Pack L1D Read Miss %"] == "5.6%"
    assert b1["Scalar Pack L1D Write Miss %"] == "2.0%"
    assert b1["Scalar Pack Stall BE %"] == "81.6%"
    assert b1["Compute IPC"] == "0.60"
    assert b1["Compute L1D Read Miss %"] == "1.3%"
    assert b1["Compute L1D Write Miss %"] == "0.0%"
    assert b1["Compute Stall BE %"] == "48.2%"
    assert b1["Compute FLOPs/Cycle"] == "5.22"
    # perf columns sit before the trailing catch-all, matching the user's original ask
    # that "additional info" stay last.
    assert list(b1.keys())[-1] == "additional info"


def test_perf_columns_are_na_when_counters_were_invalid(tmp_path: Path) -> None:
    raw = _write_csv(tmp_path, [B1_ROW])
    perf = _write_perf_csv(tmp_path, [B1_PERF_ROW_INVALID])
    rows = build_comparison_rows(str(raw), str(perf))
    b1 = rows[0]

    assert b1["Counters Valid"] == "no"
    assert b1["Compute IPC"] == "N/A"
    assert b1["Vector Pack L1D Read Miss %"] == "N/A"


def test_perf_columns_are_na_when_no_perf_file_is_given(tmp_path: Path) -> None:
    rows = build_comparison_rows(str(_write_csv(tmp_path, [S1_ROW])))
    s1 = rows[0]
    assert s1["Counters Valid"] == "N/A"
    assert s1["Compute IPC"] == "N/A"


def test_ref_gops_and_speedup_are_na_when_ref_run_was_skipped(tmp_path: Path) -> None:
    rows = build_comparison_rows(str(_write_csv(tmp_path, [S1_ROW])))
    s1 = rows[0]
    assert s1["Ref GOPs"] == "N/A"
    assert s1["Δ vs ref"] == "N/A"
    assert s1["additional info"] == "SKIP (skip_ref_run)"


def test_ref_gops_and_speedup_computed_from_the_standard_flop_count_formula(
    tmp_path: Path,
) -> None:
    # 2*M*N*K = 2e9 FLOPs; Ref_Time_ms=1000 (1s) -> Ref GOPs = 2e9 / 1 / 1e9 = 2.0
    # exactly. KerGOPs_Median=20.0 -> 10.0x the reference by construction.
    row = (
        "Custom,1000,1000,1000,MR32_NR7,PASS,20,20,20,20,20,0,20,0,20.0,"
        "1,1,100,1000,10,10,50"
    )
    rows = build_comparison_rows(str(_write_csv(tmp_path, [row])))
    r = rows[0]
    assert r["Ref GOPs"] == "2.00"
    assert r["Δ vs ref"] == "10.0×"
    assert r["additional info"] == "PASS"


# Issue #14: an uninstrumented run (`make INSTRUMENTED=0`) writes the same schema with
# the breakdown columns nan, and compile_macro_bench.sh merges it into its own
# all_raw_*_noinstr.txt, which this script's glob picks up like any other. Real row,
# from a jbpi A/B run of the 32x6 project on 2026-09-18.
B1_ROW_NOINSTR = (
    "B1,1024,384,1024,MR32_NR6,SKIP,8.16463,8.14996,8.15054,8.19485,8.19587,"
    "0.0104934,8.16465,0.010476,8.16372,nan,nan,98.6342,0,nan,nan,nan"
)


def test_uninstrumented_rows_render_as_na_not_the_string_nan(tmp_path: Path) -> None:
    rows = build_comparison_rows(str(_write_csv(tmp_path, [B1_ROW_NOINSTR])))

    row = rows[0]
    assert row["Case"] == "B1"
    assert row["KerGOPs (this run)"] == "8.16"  # the one figure this build does produce
    for column in ("Scalar Pack ms", "Vector Pack ms", "Compute ms",
                   "Scalar Pack % of total", "Vector Pack % of total",
                   "Vector Pack Gain %"):
        assert row[column] == "N/A", f"{column} leaked a nan into the table"
