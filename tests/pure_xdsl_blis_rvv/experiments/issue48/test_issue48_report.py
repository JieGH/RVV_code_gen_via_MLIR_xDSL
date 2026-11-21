"""
Issue #48: `issue48_report.py` reads a `run_issue48.sh` series: speed and slow-rep share per tile (the
uninstrumented arms) and pack time against compute time, in fast reps and in slow reps, per tile (the
instrumented arms).

Seam: hand-made series folders (runs.tsv, per-rep files in the run folders' project subfolders, provenance.txt).
No board.
"""

import gzip
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))

import issue48_report as rep  # noqa: E402

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles\n"
OPS_B3 = 2.0 * 64 * 384 * 384
FAST, SLOW = OPS_B3 / 10e9, OPS_B3 / 4e9  # 10 and 4 GOPs


def reps_text(tile: str, kc: int, seconds: list[float], pack: float | None = None, layer: str = "B3") -> str:
    """Rows of one tile's per-rep file. With `pack`, the build is instrumented: pack time is that share of a
    fast rep and 2.5 times as long (in seconds) in a slow rep, the compute time the rest."""
    rows: list[str] = []
    for r, s in enumerate(seconds):
        if pack is None:
            p = c = "nan"
        else:
            slow = s > FAST * 1.5
            p_s = pack * FAST * (2.5 if slow else 1.0)
            p, c = f"{p_s}", f"{s - p_s}"
        rows.append(f"{layer},64,384,384,MR{tile.replace('x', '_NR')},64,384,{kc},{r},{s},{p},{c},nan,nan\n")
    return HEADER + "".join(rows)


def run_folder(logs: Path, name: str, tiles: dict[str, tuple[int, list[float]]], pack: float | None = None,
               others: str = "none", layer: str = "B3") -> None:
    d = logs / name
    (d / "x").mkdir(parents=True)
    (d / "provenance.txt").write_text(f"  board          : jlei@jbpi\n  cpu pin        : 4\n  other users    : {others}\n")
    for tile, (kc, seconds) in tiles.items():
        suffix = "noinstr_refproto" if pack is None else ""
        f = d / f"fp32_AXPY_FP32_Pack_vlen256_MR{tile.replace('x', 'NR')}" / f"reps_fp32_AXPY_FP32_Pack_{suffix}_no_nc_override.txt.gz"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(gzip.compress(reps_text(tile, kc, seconds, pack, layer).encode()))


STEADY = [FAST] * 60
TWO_SPEED = [FAST] * 10 + [SLOW] * 50  # 50 of the 59 reps after rep 0 are slow


def series(tmp_path: Path, arms: list[tuple[int, str, str, dict[str, tuple[int, list[float]]], float | None]],
           failed: tuple[str, ...] = (), others: str = "none") -> Path:
    """One run per (launch, board, arm, tiles, pack): runs.tsv rows in the order given."""
    logs = tmp_path / "logs"
    sd = logs / "issue48_2026-10-01_000000"
    sd.mkdir(parents=True)
    rows = ["board\tlaunch\tposition\tarm\tresult_dir\texit"]
    for i, (launch, board, arm, tiles, pack) in enumerate(arms):
        name = f"run_2026-10-01_00{i:02d}00"
        run_folder(logs, name, tiles, pack, others)
        code = "FAIL(1)" if arm in failed else "0"
        rows.append(f"{board}\t{launch}\t{i}\t{arm}\t{name}\t{code}")
    (sd / "runs.tsv").write_text("\n".join(rows) + "\n")
    return sd


TILES = {"16x14": (128, STEADY), "16x15": (128, TWO_SPEED)}


def test_the_speed_table_has_a_row_per_tile_with_its_slow_share_and_the_blocking(tmp_path: Path):
    sd = series(tmp_path, [(i, "jbpi", "vec", TILES, None) for i in (1, 2, 3)])
    rows = rep.speed_rows(rep.collect([sd]))

    by_tile = {(r.tile, r.layer): r for r in rows}
    steady, slow = by_tile[((16, 14), "B3")], by_tile[((16, 15), "B3")]
    assert steady.launches == slow.launches == 3
    assert steady.slow_share == 0.0 and slow.slow_share > 0.8
    assert steady.kc == 128 and steady.board == "jbpi" and steady.packer == "vector"
    assert abs(steady.gops_mean_time_avg - 10.0) < 1e-9 and slow.gops_mean_time_avg < 6.0


def test_a_tile_is_averaged_over_launches_with_the_lowest_and_highest_launch(tmp_path: Path):
    faster = [FAST * 0.9] * 60
    tiles2 = {"16x14": (128, faster), "16x15": (128, TWO_SPEED)}
    sd = series(tmp_path, [(1, "jbpi", "vec", TILES, None), (2, "jbpi", "vec", tiles2, None), (3, "jbpi", "vec", TILES, None)])
    (r,) = [r for r in rep.speed_rows(rep.collect([sd])) if r.tile == (16, 14)]

    assert r.launches == 3
    assert abs(r.gops_mean_time_lo - 10.0) < 1e-9 and round(r.gops_mean_time_hi, 2) == 11.11
    assert 10.0 < r.gops_mean_time_avg < 11.11


def test_the_scalar_packer_and_the_two_boards_are_separate_rows(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "vec", TILES, None), (1, "jbpi", "scalar", TILES, None), (1, "jbpi2", "vec", TILES, None)])
    keys = {(r.board, r.packer, r.tile) for r in rep.speed_rows(rep.collect([sd]))}

    assert keys == {(b, p, t) for b, p in (("jbpi", "vector"), ("jbpi", "scalar"), ("jbpi2", "vector")) for t in ((16, 14), (16, 15))}


def test_a_run_that_failed_is_left_out(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "vec", TILES, None), (1, "jbpi", "scalar", TILES, None)], failed=("scalar",))

    assert {r.packer for r in rep.speed_rows(rep.collect([sd]))} == {"vector"}


def test_the_instrumented_arms_feed_the_split_table_and_not_the_speed_table(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "vec", TILES, None), (1, "jbpi", "vec-instr", TILES, 0.4)])
    samples = rep.collect([sd])

    assert all(r.packer == "vector" for r in rep.speed_rows(samples))
    assert len(rep.speed_rows(samples)) == 2 and len(rep.split_rows(samples)) == 2


def test_the_split_reads_pack_and_compute_apart_in_fast_reps_and_in_slow_reps(tmp_path: Path):
    sd = series(tmp_path, [(i, "jbpi", "vec-instr", TILES, 0.4) for i in (1, 2, 3)])
    rows = {r.tile: r for r in rep.split_rows(rep.collect([sd]))}
    steady, slow = rows[(16, 14)], rows[(16, 15)]

    fast_pack = 0.4 * FAST * 1e3  # ms
    assert steady.n_slow == 0 and steady.slow_pack_ms != steady.slow_pack_ms  # nan: no slow reps
    assert abs(steady.fast_pack_ms - fast_pack) < 1e-9
    assert abs(steady.fast_compute_ms - 0.6 * FAST * 1e3) < 1e-9
    assert slow.n_slow > 100 and slow.n_fast > 10
    assert abs(slow.slow_pack_ms - 2.5 * fast_pack) < 1e-9  # pack 2.5 times as long in a slow rep
    assert slow.slow_compute_ms > slow.fast_compute_ms
    assert abs(steady.pack_share_pct - 40.0) < 1e-6
    assert abs(slow.pack_share_pct - 40.0) < 1e-6  # pack is 40% of a fast rep and of a slow rep in this data


def test_the_report_lists_b3_first_the_tiles_in_order_and_the_run_conditions(tmp_path: Path):
    controls = {"16x15": (128, [FAST * 0.2] * 60)}
    sd = series(tmp_path, [(i, "jbpi", "vec", TILES, None) for i in (1, 2, 3)])
    run_folder(tmp_path / "logs", "run_2026-10-01_009900", controls, None, layer="B2")
    with (sd / "runs.tsv").open("a") as f:
        f.write("jbpi\t9\t9\tvec\trun_2026-10-01_009900\t0\n")
    text = rep.report([sd])

    assert text.index("B3") < text.index("B2")
    assert text.index("| 16x14 |") < text.index("| 16x15 |")
    assert "core 4" in text and "🔴" not in text


def test_another_users_load_is_a_red_note(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "vec", TILES, None)], others="psiwins 92%")

    assert "RED NOTE" in rep.report([sd])


def test_fewer_than_three_launches_is_warned_about(tmp_path: Path):
    sd = series(tmp_path, [(1, "jbpi", "vec", TILES, None)])

    assert "fewer than 3 launches" in rep.report([sd])


def test_the_command_line_prints_the_report(tmp_path: Path):
    sd = series(tmp_path, [(i, "jbpi", "vec", TILES, None) for i in (1, 2, 3)])
    done = subprocess.run([sys.executable, str(HERE / "experiments/issue48/issue48_report.py"), str(sd)], capture_output=True, text=True, check=False)

    assert done.returncode == 0, done.stderr
    assert "| 16x15 |" in done.stdout
