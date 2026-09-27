"""
Issues #35/#36: reading the per-rep logs that `--save-reps` writes (reps_*.txt).

Seam: `reps_report.py`'s functions over hand-made rep rows, and its command line over
hand-made run directories laid out as compile_macro_bench.sh leaves them. No board needed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from reps_report import case_stats, launch_table, read_reps  # noqa: E402

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles\n"
OPS = 2 * 64 * 384 * 384  # B3


def _rows(layer: str, nc: int, seconds: list[float], m: int = 64, n: int = 384, k: int = 384) -> str:
    return "".join(
        f"{layer},{m},{n},{k},MR16_NR15,64,{nc},128,{r},{s},nan,nan,nan,nan\n"
        for r, s in enumerate(seconds)
    )


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + body)
    return path


FAST, SLOW = OPS / 10e9, OPS / 4e9  # 10 and 4 GOPs


def test_read_reps_groups_rows_by_case_in_rep_order(tmp_path: Path):
    f = _write(tmp_path / "reps.txt", _rows("B3_nc128", 128, [FAST, SLOW]) + _rows("B3_nc384", 384, [SLOW]))
    cases = read_reps(f)
    assert list(cases) == [("B3_nc128", 128), ("B3_nc384", 384)]
    assert cases[("B3_nc128", 128)].seconds == [FAST, SLOW]
    assert cases[("B3_nc128", 128)].ops == OPS


def test_gops_by_mean_time_is_the_reference_formula_and_median_is_the_median_rep():
    st = case_stats(OPS, [FAST, FAST, SLOW])
    assert st.gops_mean_time == pytest.approx(3 * OPS / ((2 * FAST + SLOW) * 1e9))
    assert st.gops_median == pytest.approx(10.0)
    assert st.best == pytest.approx(10.0)
    assert st.n == 3


def test_a_steady_case_has_no_slow_reps_and_a_small_cv():
    st = case_stats(OPS, [FAST, FAST * 1.01, FAST * 0.99, FAST])
    assert st.slow_share == 0.0
    assert st.cv_pct < 1.5


def test_slow_reps_are_those_well_below_the_fast_mode():
    st = case_stats(OPS, [FAST] * 7 + [SLOW] * 4)  # rep 0 is left out (see below): 6 fast, 4 slow
    assert st.slow_share == pytest.approx(0.4)


def test_a_case_that_is_almost_always_slow_is_still_seen_as_two_speed():
    # B3_K256 on 2026-09-27: 4 fast reps, then thousands of slow ones
    st = case_stats(OPS, [FAST] * 5 + [SLOW] * 200)
    assert st.slow_share == pytest.approx(200 / 204)


def test_two_speeds_close_together_are_one_speed():
    # 3% apart is noise, not a slow mode
    st = case_stats(OPS, [FAST, FAST * 1.03] * 20)
    assert st.slow_share == 0.0


def test_fast_start_is_how_long_a_case_stays_fast_before_its_first_slow_rep():
    st = case_stats(OPS, [FAST] * 11 + [SLOW] * 5 + [FAST] * 5)
    assert st.fast_start_reps == 11
    assert st.fast_start_s == pytest.approx(11 * FAST)
    steady = case_stats(OPS, [FAST] * 8)
    assert steady.fast_start_reps == 8  # never went slow: the whole case


def test_slow_reps_in_one_block_are_reported_as_clumped():
    clumped = case_stats(OPS, [FAST] * 10 + [SLOW] * 10)
    assert clumped.slow_runs == 1
    assert clumped.clumping < 0.5  # far fewer runs than chance would give


def test_slow_reps_alternating_are_reported_as_scattered():
    scattered = case_stats(OPS, [FAST, SLOW] * 10)
    assert scattered.slow_runs == 10
    assert scattered.clumping > 1.5


def test_the_cold_first_rep_is_left_out_of_the_slow_mode_counts():
    # Rep 0 under the reference protocol is cold; one slow rep 0 is not a slow mode.
    st = case_stats(OPS, [SLOW] + [FAST] * 9)
    assert st.slow_share == 0.0
    assert st.n == 10  # ...but it is still inside the mean, as in the reference's formula
    assert st.gops_mean_time < 10.0


def _run_dir(root: Path, name: str, scalar: bool, body: str) -> Path:
    d = root / name
    suffix = "_scalarpack" if scalar else ""
    _write(d / "fp32_AXPY_FP32_Pack_vlen256_MR16NR15" / f"reps_fp32_AXPY_FP32_Pack_noinstr_refproto{suffix}.txt", body)
    return d


def test_launch_table_gives_one_row_per_packer_and_case_across_launches(tmp_path: Path):
    a = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    b = _run_dir(tmp_path, "run_b", False, _rows("B3_nc128", 128, [FAST] * 2 + [SLOW] * 2))
    c = _run_dir(tmp_path, "run_c", True, _rows("B3_nc128", 128, [FAST] * 4))
    rows = launch_table([a, b, c])
    assert [(r.packer, r.layer, r.nc, r.launches) for r in rows] == [
        ("vector", "B3_nc128", 128, 2),
        ("scalar", "B3_nc128", 128, 1),
    ]
    vec = rows[0]
    assert vec.gops_mean_time_lo < vec.gops_mean_time_avg < vec.gops_mean_time_hi == pytest.approx(10.0)


def test_command_line_prints_a_markdown_table(tmp_path: Path):
    a = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    out = subprocess.run(
        [sys.executable, str(HERE / "reps_report.py"), str(a)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "| packer | layer | nc |" in out
    assert "| vector | B3_nc128 | 128 |" in out


def test_a_gzipped_per_rep_log_reads_like_the_plain_one(tmp_path: Path):
    # compile_macro_bench.sh gzips reps_*.txt when it fetches them (they run to ~10 MB a launch)
    import gzip

    body = HEADER + _rows("B3_nc128", 128, [FAST, SLOW])
    d = tmp_path / "run_gz" / "fp32_AXPY_FP32_Pack_vlen256_MR16NR15"
    d.mkdir(parents=True)
    gz = d / "reps_fp32_AXPY_FP32_Pack_noinstr_refproto_scalarpack.txt.gz"
    gz.write_bytes(gzip.compress(body.encode()))
    assert read_reps(gz)[("B3_nc128", 128)].seconds == [FAST, SLOW]
    (row,) = launch_table([tmp_path / "run_gz"])
    assert (row.packer, row.layer) == ("scalar", "B3_nc128")
