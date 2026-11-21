"""Issue #55: `issue55_report.py` reads a `run_issue55.sh` series: one row per variant per board, tile and shape (GOPs by mean
time and by median rep, slow-rep share, fast start, CV, launches, kc), only runs with status ok, tiles never merged, and the
pack-only table (median, fast-rep and slow-rep mean, slow share).

Seam: hand-made series folders (runs.tsv, packonly.tsv, per-rep files in the run folders' project subfolders, provenance.txt).
No board.
"""

import gzip
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))

import issue55_report as rep  # noqa: E402

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds\n"
OPS = 2.0 * 64 * 384 * 384
FAST, SLOW = OPS / 10e9, OPS / 4e9  # 10 and 4 GOPs
STEADY = [FAST] * 60
TWO_SPEED = [FAST] * 10 + [SLOW] * 50


def reps_text(tile: str, kc: int, seconds: list[float], layer: str) -> str:
    return HEADER + "".join(f"{layer},64,384,384,MR{tile.replace('x', '_NR')},64,384,{kc},{r},{s},nan,nan\n" for r, s in enumerate(seconds))


def run_folder(logs: Path, name: str, tiles: dict[str, tuple[int, list[float]]], layer: str = "B3", others: str = "none") -> None:
    d = logs / name
    d.mkdir(parents=True)
    (d / "provenance.txt").write_text(f"  board          : jlei@jbpi\n  cpu pin        : 4\n  other users    : {others}\n")
    for tile, (kc, seconds) in tiles.items():
        f = d / f"fp32_AXPY_FP32_Pack_vlen256_MR{tile.replace('x', 'NR')}" / "reps_fp32_noinstr_refproto_no_nc_override.txt.gz"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(gzip.compress(reps_text(tile, kc, seconds, layer).encode()))


def series(tmp_path: Path, runs: list[tuple[int, str, str, dict[str, tuple[int, list[float]]], str]], status_column: bool = True,
           others: str = "none") -> Path:
    """One run per (launch, board, arm, tiles, status): runs.tsv rows in the order given."""
    logs = tmp_path / "logs"
    sd = logs / "issue55_2026-10-23_000000"
    sd.mkdir(parents=True)
    rows = ["board\tlaunch\tposition\tarm\tresult_dir\texit" + ("\tstatus" if status_column else "")]
    for i, (launch, board, arm, tiles, status) in enumerate(runs):
        name = f"run_2026-10-23_00{i:02d}00"
        run_folder(logs, name, tiles, others=others)
        code = "FAIL(1)" if status == "failed" else "0"
        rows.append(f"{board}\t{launch}\t{i}\t{arm}\t{name}\t{code}" + (f"\t{status}" if status_column else ""))
    (sd / "runs.tsv").write_text("\n".join(rows) + "\n")
    return sd


TILES = {"16x15": (128, TWO_SPEED), "16x8": (128, STEADY)}


def test_one_row_per_variant_per_tile_and_shape_with_the_blocking_and_tiles_are_not_merged(tmp_path: Path):
    runs = [(i, "jbpi", v, TILES, "ok") for i in (1, 2, 3) for v in ("row", "row-intr")]
    text = rep.report([series(tmp_path, runs)])

    assert "### jbpi, 16x15, B3" in text and "### jbpi, 16x8, B3" in text
    t15 = text.split("### jbpi, 16x15, B3")[1].split("###")[0]
    t8 = text.split("### jbpi, 16x8, B3")[1].split("###")[0]
    assert t15.count("| row |") == 1 and t15.count("| row-intr |") == 1
    assert t15.index("| row |") < t15.index("| row-intr |")
    # 16x15 is two-speed (4 and 10 GOPs), 16x8 steady at 10: the two tiles carry their own numbers
    assert "| row | 128 | 3 |" in t15
    assert "| 85 |" in t15 and "| 0 |" in t8 and "| 85 |" not in t8
    row15 = next(r for r in rep.rows_of(rep.collect([series(tmp_path / "x", runs)])[0]) if r.tile == (16, 15))
    row8 = next(r for r in rep.rows_of(rep.collect([series(tmp_path / "y", runs)])[0]) if r.tile == (16, 8))
    assert row15.slow_share > 0.8 and row8.slow_share == 0.0
    assert 4.0 < row15.gops_mean_avg < 5.0 and abs(row8.gops_mean_avg - 10.0) < 0.01
    assert row15.launches == 3 and row15.kc == 128


def test_the_speed_columns_carry_mean_time_and_median_rep_and_the_lowest_highest_launch(tmp_path: Path):
    fast_start = {"16x15": (128, [FAST] * 40 + [SLOW] * 20)}
    runs = [(1, "jbpi", "col-g1", fast_start, "ok"), (2, "jbpi", "col-g1", {"16x15": (128, STEADY)}, "ok"),
            (3, "jbpi", "col-g1", fast_start, "ok")]
    rows = rep.rows_of(rep.collect([series(tmp_path, runs)])[0])

    (r,) = rows
    assert r.gops_mean_lo < r.gops_mean_hi and abs(r.gops_mean_hi - 10.0) < 0.01
    assert r.gops_median > r.gops_mean_avg  # median rep is a fast rep; the mean carries the slow ones
    text = rep.report([tmp_path / "logs" / "issue55_2026-10-23_000000"])
    assert "GOPs by mean time (avg, lo-hi)" in text and "GOPs by median rep" in text and "fast start (s)" in text


def test_discarded_and_failed_runs_are_left_out_named_and_flagged(tmp_path: Path):
    runs = [(1, "jbpi", "row", TILES, "ok"), (2, "jbpi", "row", TILES, "discarded-load"), (3, "jbpi", "row", TILES, "ok"),
            (1, "jbpi", "row-intr", TILES, "failed"), (2, "jbpi", "row-intr", TILES, "ok"), (3, "jbpi", "row-intr", TILES, "ok"),
            (4, "jbpi", "row", TILES, "ok")]
    text = rep.report([series(tmp_path, runs)])

    assert "Runs left out (status is not ok): jbpi L2 row (discarded-load); jbpi L1 row-intr (failed)" in text
    t = text.split("### jbpi, 16x15, B3")[1].split("###")[0]
    row_line = next(ln for ln in t.splitlines() if ln.startswith("| row |"))
    assert "| 3 |" in row_line and "1 run(s) of this arm discarded" in row_line
    intr = next(ln for ln in t.splitlines() if ln.startswith("| row-intr |"))
    assert "| 2 |" in intr and "only 2 launch(es)" in intr
    assert "RED NOTE: 1 run(s) were discarded" in text
    assert "WARNING: " in text and "fewer than 3 launches" in text


def test_a_series_without_the_status_column_counts_exit_0_only(tmp_path: Path):
    runs = [(1, "jbpi", "row", TILES, "ok"), (2, "jbpi", "row", TILES, "failed"), (3, "jbpi", "row", TILES, "ok")]
    samples, excluded = rep.collect([series(tmp_path, runs, status_column=False)])

    assert sorted({s.launch for s in samples}) == [1, 3] and [r.launch for r in excluded] == [2]


def test_a_run_made_under_another_users_load_is_flagged_red(tmp_path: Path):
    runs = [(i, "jbpi", "row", TILES, "ok") for i in (1, 2, 3)]
    text = rep.report([series(tmp_path, runs, others="psiwins spmv 92.0% on core 1, up 01:21:45; ")])

    assert "RED NOTE: 3 sample(s) under another user's load" in text


def test_the_header_names_the_build_protocol_core_and_nc_setting(tmp_path: Path):
    text = rep.report([series(tmp_path, [(1, "jbpi", "row", TILES, "ok")])])

    first = text.splitlines()[2]
    assert "uninstrumented" in first and "reference" in first and "--tmin 10" in first and "core 4" in first and "--no-nc-overrides" in first


# ---- pack-only ----

def pack_file(path: Path, variant: str, seconds: list[float], check: str = "ok", kc: int = 128) -> None:
    lines = ["# board: jbpi", "# cpu: 4", f"# variant: {variant}", "# shape: M=64 N=384 K=384", "# tile: MR=16 NR=15",
             f"# blocking: mc=64 nc=384 kc={kc}", f"# reps: {len(seconds)} (warm-up 5, not printed)", "# blocks: 3",
             "# block: jc=0 pc=0 nc=384 kc=128", "Rep,Pack_Seconds"]
    lines += [f"{i},{s}" for i, s in enumerate(seconds)]
    lines.append(f"# check: {check}")
    path.write_text("\n".join(lines) + "\n")


def test_pack_stats_split_slow_reps_by_the_reps_report_rule():
    two = rep.pack_stats([0.001] * 20 + [0.0025] * 20)
    one = rep.pack_stats([0.001 + 1e-6 * (i % 3) for i in range(40)])

    assert two.slow_share > 0.4 and abs(two.fast_ms - 1.0) < 0.05 and abs(two.slow_ms - 2.5) < 0.1
    assert one.slow_share == 0.0 and one.slow_ms != one.slow_ms  # nan: no slow reps
    assert abs(one.median_ms - 1.0) < 0.01


def test_the_pack_only_table_has_median_fast_slow_and_slow_share_per_variant(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "row", TILES, "ok")])
    rows = ["board\tlaunch\tposition\tvariant\tshape\ttile\tfile\texit\tstatus"]
    for launch in (1, 2, 3):
        for variant, secs in (("row-intr", [0.001] * 40), ("col-g1", [0.0008] * 20 + [0.002] * 20)):
            name = f"packonly_jbpi_L{launch}_{variant}_B3_16x15.txt"
            pack_file(sd / name, variant, secs)
            rows.append(f"jbpi\t{launch}\t{launch}\t{variant}\tB3\t16x15\t{name}\t0\tok")
    pack_file(sd / "packonly_bad.txt", "col-g2", [0.001] * 5, check="MISMATCH")
    rows.append("jbpi\t1\t9\tcol-g2\tB3\t16x15\tpackonly_bad.txt\t1\tfailed")
    rows.append("jbpi\t1\t10\tcol-g4\tB3\t16x15\tpackonly_bad.txt\t0\tdiscarded-load")
    (sd / "packonly.tsv").write_text("\n".join(rows) + "\n")

    text = rep.report([sd])
    block = text.split("## 2. Pack-only")[1]
    assert "### jbpi, 16x15, B3" in block
    intr = next(ln for ln in block.splitlines() if ln.startswith("| row-intr |"))
    col = next(ln for ln in block.splitlines() if ln.startswith("| col-g1 |"))
    assert "| 128 | 3 | 1.0000 | 1.0000 | no slow reps | 0 |" in intr
    assert "| 128 | 3 |" in col and "| 0.8000 | 2.0000 | 51 |" in col  # 20 slow of the 39 reps after rep 0
    assert "col-g2" not in "".join(ln for ln in block.splitlines() if ln.startswith("| "))  # failed run not in a table
    assert "Pack-only runs left out:" in block and "col-g2 B3 16x15: failed" in block and "col-g4 B3 16x15: discarded-load" in block


def test_a_failed_bc_check_is_flagged_in_the_pack_only_table(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "row", TILES, "ok")])
    pack_file(sd / "p.txt", "col-g1", [0.001] * 10, check="MISMATCH")
    (sd / "packonly.tsv").write_text("board\tlaunch\tposition\tvariant\tshape\ttile\tfile\texit\tstatus\n"
                                     "jbpi\t1\t1\tcol-g1\tB3\t16x15\tp.txt\t0\tok\n")

    assert "CHECK FAILED in 1 run(s)" in rep.report([sd])


def test_the_script_prints_markdown_and_writes_out_files(tmp_path: Path):
    runs = [(i, "jbpi", "row", TILES, "ok") for i in (1, 2, 3)]
    sd = series(tmp_path, runs)
    printed = subprocess.run([sys.executable, str(HERE / "experiments/issue55/issue55_report.py"), str(sd)], capture_output=True, text=True)
    out = tmp_path / "r.md"
    wrote = subprocess.run([sys.executable, str(HERE / "experiments/issue55/issue55_report.py"), str(sd), "--out", str(out)], capture_output=True, text=True)
    usage = subprocess.run([sys.executable, str(HERE / "experiments/issue55/issue55_report.py")], capture_output=True, text=True)

    assert printed.returncode == 0 and "| variant | kc | launches |" in printed.stdout
    assert wrote.returncode == 0 and out.read_text() == printed.stdout
    assert usage.returncode == 2 and "Usage" in usage.stderr
