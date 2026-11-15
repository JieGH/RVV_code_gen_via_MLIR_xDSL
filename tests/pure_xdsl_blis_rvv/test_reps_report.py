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

from reps_report import CounterRow, case_stats, counter_table, launch_table, read_reps, run_conditions  # noqa: E402

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


def test_read_reps_keeps_the_shape_and_the_blocking_of_each_case(tmp_path: Path):
    # Issue #47: the survey needs M, N, K, MC, KC and the tile next to the times.
    f = _write(tmp_path / "reps.txt", _rows("B3", 384, [FAST, SLOW]))
    case = read_reps(f)[("B3", 384)]
    assert (case.m, case.n, case.k, case.mc, case.kc, case.variant) == (64, 384, 384, 64, 128, "MR16_NR15")


def test_read_reps_keeps_each_reps_pack_and_compute_seconds_and_nan_for_an_uninstrumented_log(tmp_path: Path):
    # Issue #48: the tile sweep splits every rep into pack and compute time (the instrumented build's columns).
    instr = "B3,64,384,384,MR16_NR15,64,384,128,{r},0.0030,{p},{c},1,2\n"
    f = _write(tmp_path / "instr.txt", instr.format(r=0, p=0.0010, c=0.0018) + instr.format(r=1, p=0.0015, c=0.0012))
    plain = _write(tmp_path / "plain.txt", _rows("B3", 384, [FAST]))

    case = read_reps(f)[("B3", 384)]
    assert case.pack_seconds == [0.0010, 0.0015] and case.compute_seconds == [0.0018, 0.0012]
    assert all(x != x for x in read_reps(plain)[("B3", 384)].pack_seconds)  # nan


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


def test_a_zero_lines_launch_is_its_own_packer_row(tmp_path: Path):
    # Issue #43: --zero-pack-lines on and off is the A-B comparison, so the two never pool.
    def run(name: str, suffix: str) -> Path:
        d = tmp_path / name
        _write(d / "p" / f"reps_fp32_AXPY_FP32_Pack_noinstr_refproto{suffix}.txt", _rows("B3", 128, [FAST] * 4))
        return d

    runs = [run("a", ""), run("b", "_zeroline"), run("c", "_scalarpack_zeroline_no_nc_override")]
    assert [r.packer for r in launch_table(runs)] == ["vector", "vector+zeroline", "scalar+zeroline"]


def test_the_reference_designs_per_rep_files_are_their_own_rows_per_built_tile(tmp_path: Path):
    # Issue #46: reps_ref_<tile>_<variant>.txt.gz come from the reference design, not from our
    # vector packer, and the 16x14 and 16x16 builds are two different kernels: never pooled.
    def run(name: str, fname: str) -> Path:
        d = tmp_path / name
        _write(d / fname, _rows("B3", 384, [FAST] * 4))
        return d

    runs = [run("a", "reps_fp32_AXPY_FP32_Pack_noinstr_refproto_no_nc_override.txt"),
            run("b", "reps_ref_16x14_macc.txt"), run("c", "reps_ref_16x16_macc.txt"), run("d", "reps_ref_16x14_macc.txt")]
    assert [(r.packer, r.launches) for r in launch_table(runs)] == [
        ("vector", 1), ("reference 16x14", 2), ("reference 16x16", 1)]


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


# Issue #39, house rules 2 and 4: a case seen in fewer than 3 launches is not a result, and
# every run folder says which build, protocol and core it was timed on.
PROVENANCE = """Run on the board, generated 2026-09-27T10:43:38+02:00 by compile_macro_bench.sh
  board          : jlei@jbpi
  build          : UNINSTRUMENTED (INSTRUMENTED=0) -- comparable to the reference design
  protocol       : REFERENCE (tmin=10 s) -- no warm-up, reps until tmin
  logs           : raw_*_noinstr_refproto.txt
  cpu pin        : {pin}
  governor       :  8 performance
"""


def test_run_conditions_come_from_the_run_folders_provenance(tmp_path: Path):
    d = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    (d / "provenance.txt").write_text(PROVENANCE.format(pin="4"))
    cond = run_conditions(d)
    assert cond.board == "jlei@jbpi" and cond.core == "4"
    assert cond.build.startswith("UNINSTRUMENTED") and cond.protocol.startswith("REFERENCE")
    assert run_conditions(tmp_path / "nowhere").core == "unknown"


def _cli(*args: Path) -> str:
    return subprocess.run(
        [sys.executable, str(HERE / "reps_report.py"), *map(str, args)],
        capture_output=True, text=True, check=True,
    ).stdout


def test_the_report_names_build_protocol_and_core_of_every_run(tmp_path: Path):
    d = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    (d / "provenance.txt").write_text(PROVENANCE.format(pin="4"))
    out = _cli(d)
    line = next(ln for ln in out.splitlines() if "run_a" in ln)
    assert "UNINSTRUMENTED" in line and "REFERENCE" in line and "core 4" in line


def test_the_report_flags_a_run_on_core_0(tmp_path: Path):
    d = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    (d / "provenance.txt").write_text(PROVENANCE.format(pin="0"))
    assert "WARNING" in _cli(d) and "core 0" in _cli(d)


def test_the_report_warns_when_a_case_has_fewer_than_three_launches(tmp_path: Path):
    runs = [_run_dir(tmp_path, f"run_{i}", False, _rows("B3_nc128", 128, [FAST] * 4)) for i in "abc"]
    assert "fewer than 3 launches" in _cli(runs[0], runs[1])
    assert "fewer than 3 launches" not in _cli(*runs)


# Issue #42: fast reps beside slow reps, for every counter column the per-rep log carries
# (the cycles, and the raw events asked for with --raw-events).
RAW_HEADER = HEADER.rstrip("\n") + ",Pack_l2_load_miss,Compute_l2_load_miss\n"


def _counted_rows(layer: str, reps: list[tuple[float, int, int]]) -> str:
    """reps: (seconds, pack cycles, pack l2 misses); compute l2 misses are 10x the pack ones."""
    return "".join(
        f"{layer},64,384,384,MR16_NR15,64,384,128,{r},{s},nan,nan,{pc},nan,{pm},{10 * pm}\n"
        for r, (s, pc, pm) in enumerate(reps)
    )


def _counted_run(root: Path, name: str, body: str, scalar: bool = False) -> Path:
    d = root / name
    suffix = "_scalarpack" if scalar else ""
    f = d / "fp32_AXPY_FP32_Pack_vlen256_MR16NR15" / f"reps_fp32_AXPY_FP32_Pack{suffix}.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(RAW_HEADER + body)
    return d


def _by_column(rows: list[CounterRow]) -> dict[str, CounterRow]:
    return {r.column: r for r in rows}


def test_counters_are_averaged_over_fast_reps_and_over_slow_reps(tmp_path: Path):
    # Rep 0 is cold and left out, as it is left out of the fast/slow split.
    reps = [(SLOW, 9999, 9999)] + [(FAST, 1000, 5)] * 3 + [(SLOW, 2500, 20)] * 2
    rows = _by_column(counter_table([_counted_run(tmp_path, "run_a", _counted_rows("B3", reps))]))
    assert set(rows) == {"Pack_Cycles", "Pack_l2_load_miss", "Compute_l2_load_miss"}
    miss = rows["Pack_l2_load_miss"]
    assert (miss.layer, miss.nc, miss.launches, miss.n_fast, miss.n_slow) == ("B3", 384, 1, 3, 2)
    assert (miss.fast_mean, miss.slow_mean) == (5, 20)
    assert miss.ratio == pytest.approx(4.0)
    assert rows["Compute_l2_load_miss"].slow_mean == 200
    assert rows["Pack_Cycles"].ratio == pytest.approx(2.5)


def test_fast_and_slow_reps_are_pooled_over_launches(tmp_path: Path):
    a = _counted_run(tmp_path, "run_a", _counted_rows("B3", [(FAST, 1, 4)] * 3 + [(SLOW, 1, 40)]))
    b = _counted_run(tmp_path, "run_b", _counted_rows("B3", [(FAST, 1, 8)] * 3 + [(SLOW, 1, 20)] * 3))
    miss = _by_column(counter_table([a, b]))["Pack_l2_load_miss"]
    assert miss.launches == 2
    assert (miss.n_fast, miss.n_slow) == (4, 4)  # rep 0 of each launch left out
    assert miss.fast_mean == pytest.approx((4 * 2 + 8 * 2) / 4)
    assert miss.slow_mean == pytest.approx((40 + 20 * 3) / 4)


def test_a_one_speed_case_has_no_slow_reps_to_compare(tmp_path: Path):
    d = _counted_run(tmp_path, "run_a", _counted_rows("B2", [(FAST, 1000, 5)] * 5))
    miss = _by_column(counter_table([d]))["Pack_l2_load_miss"]
    assert miss.n_slow == 0 and miss.slow_mean != miss.slow_mean  # nan
    assert "one speed" in _cli(d)


def test_missing_counts_are_left_out_and_an_all_nan_column_is_dropped(tmp_path: Path):
    body = _counted_rows("B3", [(FAST, 1000, 5)] * 3 + [(SLOW, 2500, 20)] * 2).replace(",20,200", ",nan,nan", 1)
    rows = _by_column(counter_table([_counted_run(tmp_path, "run_a", body)]))
    assert rows["Pack_l2_load_miss"].n_slow == 1 and rows["Pack_l2_load_miss"].slow_mean == 20
    assert "Compute_Cycles" not in rows  # nan in every rep
    # an uninstrumented log has no counts at all: no counter table
    plain = _run_dir(tmp_path, "run_p", False, _rows("B3_nc128", 128, [FAST] * 3 + [SLOW] * 2))
    assert counter_table([plain]) == []
    assert "slow/fast" not in _cli(plain)


def test_the_command_line_prints_the_counter_table_after_the_launch_table(tmp_path: Path):
    reps = [(FAST, 1000, 5)] * 4 + [(SLOW, 2500, 20)] * 2
    out = _cli(_counted_run(tmp_path, "run_a", _counted_rows("B3", reps)))
    assert out.index("| packer | layer | nc | launches | GOPs") < out.index("slow/fast")
    line = next(ln for ln in out.splitlines() if "Pack_l2_load_miss" in ln)
    assert "| vector | B3 | 384 | 1 |" in line and "4.00" in line
    assert "fewer than 3 launches" in out  # the house-rule warning covers these rows too


def test_the_conditions_line_names_the_raw_events_and_the_other_cores_busy_share(tmp_path: Path):
    d = _counted_run(tmp_path, "run_a", _counted_rows("B3", [(FAST, 1, 1)] * 3))
    (d / "provenance.txt").write_text(
        PROVENANCE.format(pin="4")
        + "  raw events     : x60-l2\n  busy share     : cpu0 3% cpu1 0% cpu4 100% cpu5 97%\n"
    )
    cond = run_conditions(d)
    assert cond.raw_events == "x60-l2" and cond.busy_share.startswith("cpu0 3%")
    line = next(ln for ln in _cli(d).splitlines() if "run_a" in ln)
    assert "raw events x60-l2" in line and "busy cpu0 3% cpu1 0% cpu4 100% cpu5 97%" in line
    # a run from before #42 says so rather than leave a blank
    old = _run_dir(tmp_path, "run_old", False, _rows("B3", 384, [FAST] * 3))
    (old / "provenance.txt").write_text(PROVENANCE.format(pin="4"))
    assert run_conditions(old).raw_events == "unknown" and run_conditions(old).busy_share == "unknown"


# 2026-09-29: another user's job on the board moves its thermal state even from another core, so
# a run made under one carries a red note, and a clean or unrecorded run carries none.
def test_a_run_under_another_users_load_gets_a_red_note(tmp_path: Path):
    d = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    (d / "provenance.txt").write_text(
        PROVENANCE.format(pin="4") + "  other users    : psiwins spmv 92.0% on core 0, up 01:21:45; \n")
    out = _cli(d)
    note = next(ln for ln in out.splitlines() if "RED NOTE" in ln)
    assert "run_a" in note and "psiwins spmv 92.0% on core 0" in note and "quiet window" in note
    assert "RED NOTE" in out.splitlines()[-1] or any("RED NOTE" in ln for ln in out.splitlines()[-3:])  # also in the summary at the end


def test_a_run_with_no_other_users_or_no_record_has_no_red_note(tmp_path: Path):
    a = _run_dir(tmp_path, "run_a", False, _rows("B3_nc128", 128, [FAST] * 4))
    (a / "provenance.txt").write_text(PROVENANCE.format(pin="4") + "  other users    : none\n")
    b = _run_dir(tmp_path, "run_b", False, _rows("B3_nc128", 128, [FAST] * 4))
    (b / "provenance.txt").write_text(PROVENANCE.format(pin="4"))  # made before the probe existed
    assert "RED NOTE" not in _cli(a, b)
