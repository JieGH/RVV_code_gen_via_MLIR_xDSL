"""
Issue #29: `compare_ref_sweep.py` puts the reference design's runs beside our two packers.

The logs here are made up but shaped like the real ones: `refbase_<stamp>/run_plan.tsv` and
`result_<tag>_macc.csv` for the reference, `run_<stamp>/all_raw_*_noinstr_refproto[_scalarpack]_
no_nc_override.txt` for ours. What is checked is what the table claims: the mean, range and run
count per row, `bracket` rows for a layer the reference has no kernel for, the sign of `vector/ref` and `scalar/ref`,
and when a difference is called faster, slower or "within noise".
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_baseline"))

from compare_ref_sweep import board_of, load_ref_runs, main, runs_in_window, runs_of_sweep  # noqa: E402

PLAN_HEADER = "# case\tnumid\tM\tN\tK\tpinned_MR\tpinned_NR\tbuilt_MR\tbuilt_NR\tkind\tpred_MC\tpred_NC\tpred_KC\ttag\n"
# B1 is pinned 32x6 and the reference has it; S1 is pinned 32x7, which it does not, so it is
# built at 32x6 and 32x8.
PLAN_ROWS = [
    "B1\t1\t1024\t384\t1024\t32\t6\t32\t6\texact\t768\t384\t128\t32x6",
    "S1\t11\t1000\t1000\t1000\t32\t7\t32\t6\tbracket\t768\t1000\t128\t32x6",
    "S1\t11\t1000\t1000\t1000\t32\t7\t32\t8\tbracket\t768\t1000\t128\t32x8",
]

RAW_HEADER = "Project,Layer,M,N,K,Variant,Status,RVV_Time_ms"


def make_ref_run(
    logs: Path, stamp: str, b1: float | None, s1_6: float, s1_8: float, board: str = "jbpi", pin: str = "0"
) -> Path:
    """A refbase dir. `b1=None` leaves B1's row out, as a run that failed on 32x6 would."""
    d = logs / f"refbase_{stamp}"
    d.mkdir(parents=True)
    (d / "run_plan.tsv").write_text(PLAN_HEADER + "\n".join(PLAN_ROWS) + "\n")
    rows6 = ["#l;m;n;k;Gflops"]
    if b1 is not None:
        rows6.append(f"1;1024;384;1024;{b1:.2e}")
    rows6.append(f"11;1000;1000;1000;{s1_6:.2e}")
    (d / "result_32x6_macc.csv").write_text("\n".join(rows6) + "\n")
    (d / "result_32x8_macc.csv").write_text(f"#l;m;n;k;Gflops\n11;1000;1000;1000;{s1_8:.2e}\n")
    (d / "provenance.txt").write_text(f"Reference-design baseline\n  board          : {board} (user jlei)\n  cpu pin        : {pin}\n")
    return d


def make_our_run(logs: Path, stamp: str, glob_name: str, b1_ms: float, s1_ms: float) -> Path:
    d = logs / f"run_{stamp}"
    d.mkdir(parents=True)
    (d / glob_name).write_text(
        RAW_HEADER + "\n"
        + f"p,B1,1024,384,1024,MR32_NR6,SKIP,{b1_ms}\n"
        + f"p,S1,1000,1000,1000,MR32_NR7,SKIP,{s1_ms}\n"
    )
    (d / "provenance.txt").write_text("Run on the board\n  board          : jlei@jbpi\n  cpu pin        : 0\n")
    return d


VECTOR_LOG = "all_raw_p_noinstr_refproto_no_nc_override.txt"
SCALAR_LOG = "all_raw_p_noinstr_refproto_scalarpack_no_nc_override.txt"

B1_OPS = 2.0 * 1024 * 384 * 1024
S1_OPS = 2.0 * 1000**3


def ms_for(ops: float, gops: float) -> float:
    return ops / (gops * 1e6)


def test_the_reference_rows_carry_mean_range_and_run_count(tmp_path: Path):
    a = make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0)
    b = make_ref_run(tmp_path, "2026-09-21_103000", 7.2, 6.2, 8.2)

    rows = load_ref_runs([a, b])

    assert rows[("B1", "32x6")].gflops == pytest.approx([7.0, 7.2])
    assert rows[("B1", "32x6")].kind == "exact"
    assert rows[("S1", "32x6")].gflops == pytest.approx([6.0, 6.2])
    assert rows[("S1", "32x8")].gflops == pytest.approx([8.0, 8.2])


def test_a_layer_without_a_kernel_at_its_pinning_is_two_bracket_rows(tmp_path: Path):
    a = make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0)

    rows = load_ref_runs([a])

    assert {b for (layer, b), r in rows.items() if layer == "S1" and r.kind == "bracket"} == {"32x6", "32x8"}
    assert rows[("S1", "32x6")].pinned == "32x7"


def test_a_run_that_left_no_row_leaves_that_row_one_run_short(tmp_path: Path):
    a = make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0)
    b = make_ref_run(tmp_path, "2026-09-21_103000", None, 6.2, 8.2)

    rows = load_ref_runs([a, b])

    assert len(rows[("B1", "32x6")].gflops) == 1
    assert len(rows[("S1", "32x8")].gflops) == 2


def test_a_dir_that_is_not_a_reference_result_is_refused(tmp_path: Path):
    (tmp_path / "refbase_2026-09-21_100000").mkdir()

    with pytest.raises(SystemExit, match="run_plan.tsv"):
        load_ref_runs([tmp_path / "refbase_2026-09-21_100000"])


def test_the_window_takes_only_runs_inside_it_that_have_the_log(tmp_path: Path):
    inside = make_our_run(tmp_path, "2026-09-20_010000", VECTOR_LOG, 100, 100)
    make_our_run(tmp_path, "2026-09-19_120000", VECTOR_LOG, 100, 100)  # before the window
    make_our_run(tmp_path, "2026-09-20_020000", SCALAR_LOG, 100, 100)  # right time, other packer

    got = runs_in_window(tmp_path, "2026-09-19_230000", "2026-09-20_070000", "all_raw_*_noinstr_refproto_no_nc_override.txt")

    assert got == [inside]


def test_the_board_is_read_from_either_provenance_format(tmp_path: Path):
    ref = make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0, board="jbpi2")
    ours = make_our_run(tmp_path, "2026-09-20_010000", VECTOR_LOG, 100, 100)

    assert board_of(ref) == "jbpi2"
    assert board_of(ours) == "jbpi"
    assert board_of(tmp_path) == "?"


def _sweep(tmp_path: Path, name: str, runs: list[tuple[str, Path]]) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "runs.txt").write_text("".join(f"{board}\t{run.name}\n" for board, run in runs))
    return d


def test_a_sweeps_run_list_names_its_dirs(tmp_path: Path):
    a = make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0)
    sweep = _sweep(tmp_path, "sweep_ref_2026-09-21_095900", [("jbpi", a)])

    assert runs_of_sweep(tmp_path, sweep) == [a]


def _table_row(out: str, layer: str, built: str) -> list[str]:
    for line in out.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("|") and cells[0] == layer and cells[3] == built:
            return cells
    raise AssertionError(f"no {layer}/{built} row in:\n{out}")


def _two_ref_runs(tmp_path: Path, pin: str = "0") -> list[Path]:
    return [
        make_ref_run(tmp_path, "2026-09-21_100000", 7.00, 6.0, 8.0, pin=pin),
        make_ref_run(tmp_path, "2026-09-21_103000", 7.02, 6.0, 8.0, pin=pin),
    ]


def _our_runs(tmp_path: Path, vector_b1: float, scalar_b1: float, scalar_stamp: str = "1") -> Path:
    """Two vector runs and two scalar runs; returns the scalar sweep dir."""
    for i in (1, 2):
        make_our_run(tmp_path, f"2026-09-20_0{i}0000", VECTOR_LOG, ms_for(B1_OPS, vector_b1), ms_for(S1_OPS, 7.0))
    scalar = [
        make_our_run(tmp_path, f"2026-09-20_{scalar_stamp}{i}0000", SCALAR_LOG, ms_for(B1_OPS, scalar_b1), ms_for(S1_OPS, 7.0))
        for i in (4, 5)
    ]
    return _sweep(tmp_path, f"sweep_scalar_2026-09-20_{scalar_stamp}33000", [("jbpi", d) for d in scalar])


def test_the_table_reads_ours_over_reference_positive_when_we_are_faster(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = _two_ref_runs(tmp_path)
    # Vector pack at 8.0 GOPs on B1 (well above the reference's 7.01), scalar pack at 6.0 (below).
    sweep = _our_runs(tmp_path, vector_b1=8.0, scalar_b1=6.0)

    rc = main(["--ref", *map(str, refs), "--logs", str(tmp_path), "--scalar-sweep", str(sweep)])

    out = capsys.readouterr().out
    assert rc == 0
    b1 = _table_row(out, "B1", "32x6")
    assert b1[4] == "exact"
    assert b1[5].startswith("7.01 (7.00-7.02, n=2)")
    assert b1[8].startswith("+14.1% (faster")  # 8.0 / 7.01 - 1
    assert b1[9].startswith("-14.4% (slower")  # 6.0 / 7.01 - 1
    assert b1[10] == "0.3%"  # the reference's own spread: (7.02 - 7.00) / 7.01
    assert "reference   : 2 runs" in out and "vector pack : 2 runs" in out and "scalar pack : 2 runs" in out


def test_a_bracket_row_is_compared_too_and_carries_no_published_figure(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = _two_ref_runs(tmp_path)
    sweep = _our_runs(tmp_path, vector_b1=8.0, scalar_b1=6.0)

    main(["--ref", *map(str, refs), "--logs", str(tmp_path), "--scalar-sweep", str(sweep)])

    out = capsys.readouterr().out
    low, high = _table_row(out, "S1", "32x6"), _table_row(out, "S1", "32x8")
    assert low[8].startswith("+16.7% (faster")  # ours 7.0 against the reference's 6.0 at 32x6
    assert high[8].startswith("-12.5% (slower")  # ours 7.0 against the reference's 8.0 at 32x8
    assert low[11] == "8.5" and low[12] == ""  # S1's published figure is for 32x7, not a neighbour
    assert _table_row(out, "B1", "32x6")[12] == "-12.4%"  # 7.01 against the published 8.0, an exact row


def test_a_gap_no_bigger_than_the_spread_is_called_within_noise(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = [
        make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0),
        make_ref_run(tmp_path, "2026-09-21_103000", 7.4, 6.0, 8.0),  # spread 5.6% of the mean 7.2
    ]
    make_our_run(tmp_path, "2026-09-20_010000", VECTOR_LOG, ms_for(B1_OPS, 7.3), ms_for(S1_OPS, 7.0))
    make_our_run(tmp_path, "2026-09-20_020000", VECTOR_LOG, ms_for(B1_OPS, 7.3), ms_for(S1_OPS, 7.0))

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    b1 = _table_row(capsys.readouterr().out, "B1", "32x6")
    assert b1[8] == "+1.4% (within noise, noise 5.6%)"


def test_bracket_rows_are_labelled_and_the_note_says_not_to_quote_them(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = [make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0)]
    make_our_run(tmp_path, "2026-09-20_010000", VECTOR_LOG, ms_for(B1_OPS, 7.0), ms_for(S1_OPS, 7.0))

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    out = capsys.readouterr().out
    assert _table_row(out, "S1", "32x6")[4] == "bracket"
    assert _table_row(out, "S1", "32x8")[4] == "bracket"
    assert "do not quote one as 'the reference number'" in out


def test_one_run_on_each_side_says_it_has_no_noise_estimate(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = [make_ref_run(tmp_path, "2026-09-21_100000", 7.0, 6.0, 8.0)]
    make_our_run(tmp_path, "2026-09-20_010000", VECTOR_LOG, ms_for(B1_OPS, 7.0), ms_for(S1_OPS, 7.0))

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    assert "one run each: no noise estimate" in _table_row(capsys.readouterr().out, "B1", "32x6")[8]


def test_the_output_names_the_limits_of_the_comparison(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = _two_ref_runs(tmp_path)

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    out = capsys.readouterr().out
    assert "not made in the same hours" in out
    assert "S3-S5 are about one rep per run" in out
    assert "3 significant digits" in out


def test_a_sweep_dir_is_read_through_its_run_list_and_the_newest_scalar_sweep_is_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    refs = _two_ref_runs(tmp_path)
    ref_sweep = _sweep(tmp_path, "sweep_ref_2026-09-21_095900", [("jbpi", refs[0]), ("jbpi2", refs[1])])
    _our_runs(tmp_path, vector_b1=8.0, scalar_b1=5.0, scalar_stamp="1")  # an older scalar sweep, at 5.0
    _our_runs_newer = [
        make_our_run(tmp_path, f"2026-09-20_2{i}0000", SCALAR_LOG, ms_for(B1_OPS, 6.0), ms_for(S1_OPS, 7.0)) for i in (0, 1)
    ]
    _sweep(tmp_path, "sweep_scalar_2026-09-20_200000", [("jbpi", d) for d in _our_runs_newer])

    main(["--sweep", str(ref_sweep), "--logs", str(tmp_path)])

    b1 = _table_row(capsys.readouterr().out, "B1", "32x6")
    assert b1[5].startswith("7.01 (7.00-7.02, n=2)")
    assert b1[7].startswith("6.00 (6.00-6.00, n=2)")  # the newest sweep's runs, not the older one's 5.0


def test_it_warns_when_the_sets_were_not_all_pinned_to_one_core(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = _two_ref_runs(tmp_path, pin="4")  # ours ran on core 0
    _our_runs(tmp_path, vector_b1=8.0, scalar_b1=6.0)

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    assert "WARNING: the sets were pinned to different cores." in capsys.readouterr().out


def test_it_warns_when_a_reference_run_was_not_pinned(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = _two_ref_runs(tmp_path, pin="none")

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    assert "WARNING: some runs were not pinned" in capsys.readouterr().out


def test_it_says_nothing_of_pinning_when_every_set_ran_on_core_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    refs = _two_ref_runs(tmp_path)
    _our_runs(tmp_path, vector_b1=8.0, scalar_b1=6.0)

    main(["--ref", *map(str, refs), "--logs", str(tmp_path)])

    assert "WARNING" not in capsys.readouterr().out


def test_it_needs_exactly_one_of_sweep_and_ref(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--logs", str(tmp_path)])
