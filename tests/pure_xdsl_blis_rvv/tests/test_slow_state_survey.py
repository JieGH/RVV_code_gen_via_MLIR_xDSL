"""
Issue #47: the slow state across every shape, from the per-rep logs already committed.

Seam: `slow_state_survey.py`'s functions over hand-made run folders laid out as
compile_macro_bench.sh leaves them (provenance.txt plus <project>/reps_*.txt), and its command line.
No board, no real logs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "readers"))

import slow_state_survey as sss  # noqa: E402

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles\n"
PROV = """Run on the board, generated 2026-09-27T10:43:38+02:00 by compile_macro_bench.sh
  board          : jlei@{board}
  build          : {build}
  protocol       : {protocol}
  cpu pin        : {core}
"""
UNINSTR = "UNINSTRUMENTED (INSTRUMENTED=0) -- comparable to the reference design"
REFPROTO = "REFERENCE (tmin=10 s) -- no warm-up, reps until tmin"


def ops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def gops_to_s(g: float, m: int = 64, n: int = 384, k: int = 384) -> float:
    return ops(m, n, k) / (g * 1e9)


def make_run(root: Path, name: str, layer: str, gops: list[float], *, shape: tuple[int, int, int] = (64, 384, 384),
             nc: int = 384, mc: int = 64, kc: int = 128, board: str = "jbpi", core: str = "4",
             build: str = UNINSTR, protocol: str = REFPROTO, suffix: str = "_noinstr_refproto",
             tile: str = "MR16NR15") -> Path:
    m, n, k = shape
    d = root / name
    (d / f"fp32_AXPY_FP32_Pack_vlen256_{tile}").mkdir(parents=True)
    (d / "provenance.txt").write_text(PROV.format(board=board, build=build, protocol=protocol, core=core))
    rows = "".join(
        f"{layer},{m},{n},{k},{tile[:4]}_{tile[4:]},{mc},{nc},{kc},{r},{gops_to_s(g, m, n, k)},nan,nan,nan,nan\n"
        for r, g in enumerate(gops)
    )
    (d / f"fp32_AXPY_FP32_Pack_vlen256_{tile}" / f"reps_fp32_AXPY_FP32_Pack{suffix}.txt").write_text(HEADER + rows)
    return d


TWO_SPEEDS = [10.0] * 30 + [4.5] * 70  # falls slow after 30 fast reps
ONE_SPEED = [10.0] * 100


# ---- the packed-bytes formula -------------------------------------------------------------------
def test_packed_bytes_per_rep_follow_the_loop_nest():
    # jc over N by nc, pc over K by kc: B is packed once per element (K*N); A block (mc x kc) is
    # packed inside the jc loop, so every A element goes through the packer ceil(N/nc) times.
    assert sss.bytes_per_rep(64, 384, 384, 384) == 4 * (384 * 384 + 64 * 384 * 1)
    assert sss.bytes_per_rep(64, 384, 384, 128) == 4 * (384 * 384 + 64 * 384 * 3)
    assert sss.bytes_per_rep(1000, 1000, 1000, 1000) == 4 * (1000 * 1000 + 1000 * 1000 * 1)
    assert sss.bytes_per_rep(64, 384, 384, 120) == 4 * (384 * 384 + 64 * 384 * 4)  # 384/120 rounds up


# ---- inventory and the #39 rules ----------------------------------------------------------------
def test_the_inventory_has_a_row_per_reps_file_and_says_why_it_is_dropped(tmp_path: Path):
    make_run(tmp_path, "run_1", "B3", ONE_SPEED)
    make_run(tmp_path, "run_2", "B3", ONE_SPEED, core="0")
    make_run(tmp_path, "run_3", "B3", ONE_SPEED, build="INSTRUMENTED (DEBUG)", protocol="WARM-UP (5 warm-up calls)", suffix="")
    make_run(tmp_path, "run_4", "B3", ONE_SPEED, core="none")
    launches, inventory = sss.collect(tmp_path)

    assert [i.run for i in inventory] == ["run_1", "run_2", "run_3", "run_4"]
    assert [i.used for i in inventory] == ["yes", "no: core 0", "no: instrumented build (counts only)", "no: not pinned"]
    assert [ln.run for ln in launches] == ["run_1"]


def test_a_run_folder_without_provenance_is_dropped_as_unknown_core(tmp_path: Path):
    d = make_run(tmp_path, "run_1", "B3", ONE_SPEED)
    (d / "provenance.txt").unlink()
    launches, inventory = sss.collect(tmp_path)

    assert launches == [] and inventory[0].used == "no: core unknown"


def test_packer_nc_override_and_zeroline_come_from_the_file_name(tmp_path: Path):
    make_run(tmp_path, "run_1", "B3", ONE_SPEED, suffix="_noinstr_refproto_scalarpack_zeroline_no_nc_override")
    (inv,) = sss.collect(tmp_path)[1]

    assert (inv.packer, inv.nc_override) == ("scalar+zeroline", True)


def test_runs_after_the_until_stamp_are_left_out(tmp_path: Path):
    # The survey of an issue has to be repeatable on the logs it read, however many runs come after.
    make_run(tmp_path, "run_2026-09-28_235900", "B3", ONE_SPEED)
    make_run(tmp_path, "run_2026-09-29_000100", "B3", ONE_SPEED)
    launches, inventory = sss.collect(tmp_path, until="2026-09-28_235959")

    assert [ln.run for ln in launches] == ["run_2026-09-28_235900"] and len(inventory) == 1


# ---- the slow-state table -----------------------------------------------------------------------
def three(root: Path, layer: str, gops: list[float], tag: str = "", **kw: object) -> None:
    for i in range(3):
        make_run(root, f"run_{layer}_{tag}_{i}", layer, gops, **kw)  # type: ignore[arg-type]


def test_a_case_in_three_launches_gets_its_slow_share_and_fast_start(tmp_path: Path):
    three(tmp_path, "B3", TWO_SPEEDS)
    launches, _ = sss.collect(tmp_path)
    (row,) = sss.slow_table(launches).rows

    assert (row.layer, row.launches, row.two_speed_launches) == ("B3", 3, 3)
    assert row.slow_share_avg == pytest.approx(0.70, abs=0.02)
    assert row.fast_start_reps_avg == pytest.approx(30, abs=1)
    assert row.gops_median_avg == pytest.approx(4.5)                      # both metrics: by mean time and by median rep
    assert row.gops_avg == pytest.approx(len(TWO_SPEEDS) / (30 / 10 + 70 / 4.5), rel=0.01)
    assert row.fast_start_s_avg == pytest.approx(30 * gops_to_s(10.0), rel=0.05)


def test_a_case_in_fewer_than_three_launches_is_no_result_not_a_row(tmp_path: Path):
    make_run(tmp_path, "run_a", "B3", TWO_SPEEDS)
    make_run(tmp_path, "run_b", "B3", TWO_SPEEDS)
    launches, _ = sss.collect(tmp_path)
    table = sss.slow_table(launches)

    assert table.rows == [] and [(g.layer, g.launches) for g in table.no_result] == [("B3", 2)]


def test_boards_packers_and_nc_are_never_pooled(tmp_path: Path):
    three(tmp_path, "B3", TWO_SPEEDS, "a", board="jbpi")
    three(tmp_path, "B3", ONE_SPEED, "b", board="jbpi2")
    three(tmp_path, "B3", ONE_SPEED, "c", nc=128, suffix="_noinstr_refproto_scalarpack")
    table = sss.slow_table(sss.collect(tmp_path)[0])

    assert sorted((r.board, r.packer, r.nc) for r in table.rows) == [
        ("jbpi", "scalar", 128), ("jbpi", "vector", 384), ("jbpi2", "vector", 384)]


def test_a_steady_case_has_no_two_speed_launch_and_a_single_slow_rep_is_not_a_slow_state(tmp_path: Path):
    three(tmp_path, "B2", ONE_SPEED, shape=(384, 384, 64), mc=384, kc=64)
    three(tmp_path, "B1", [10.0] * 60 + [3.0] + [10.0] * 39, shape=(1024, 384, 1024))  # 1 of 100 reps: an interrupt
    rows = {r.layer: r for r in sss.slow_table(sss.collect(tmp_path)[0]).rows}

    assert rows["B2"].two_speed_launches == 0 and rows["B1"].two_speed_launches == 0
    assert rows["B1"].slow_share_avg > 0  # still reported, just not a state


# ---- time, reps or bytes ------------------------------------------------------------------------
def test_the_switch_table_says_which_of_time_reps_bytes_is_closest_to_constant_across_shapes(tmp_path: Path):
    # Shape X switches after 30 reps and shape Y after 60 reps, but both after the same seconds.
    x, y = (64, 384, 384), (128, 384, 384)
    t = 3.0  # seconds fast
    sx = ops(*x) / (10e9)
    sy = ops(*y) / (10e9)
    gx = [10.0] * round(t / sx) + [4.5] * round(t / sx)  # as many slow reps as fast: a state, not a blip
    gy = [10.0] * round(t / sy) + [4.5] * round(t / sy)
    three(tmp_path, "X", gx, shape=x)
    three(tmp_path, "Y", gy, shape=y, mc=128)
    sw = sss.switch_table(sss.collect(tmp_path)[0])

    assert {r.layer for r in sw.rows} == {"X", "Y"}
    assert sw.across["seconds"] < sw.across["reps"] and sw.across["seconds"] < sw.across["bytes"]
    assert sw.closest == "seconds"


def test_the_switch_table_gives_the_lowest_median_and_highest_launch(tmp_path: Path):
    for i, fast in enumerate([10, 30, 90]):
        make_run(tmp_path, f"run_{i}", "B3", [10.0] * fast + [4.5] * 100)
    (row,) = sss.switch_table(sss.collect(tmp_path)[0]).rows

    assert (row.reps_min, row.reps_median, row.reps_max) == (10, 30, 90)
    assert row.seconds_min < row.seconds_median < row.seconds_max


def test_a_constant_number_of_reps_is_named_reps(tmp_path: Path):
    three(tmp_path, "X", [10.0] * 30 + [4.5] * 50, shape=(64, 384, 384))
    three(tmp_path, "Y", [10.0] * 30 + [4.5] * 50, shape=(256, 1024, 1024), mc=256, nc=384, kc=128)
    assert sss.switch_table(sss.collect(tmp_path)[0]).closest == "reps"


def test_with_fewer_than_two_shapes_switching_nothing_is_named_closest(tmp_path: Path):
    three(tmp_path, "X", TWO_SPEEDS)
    three(tmp_path, "Y", ONE_SPEED, shape=(384, 384, 64), mc=384, kc=64)

    assert sss.switch_table(sss.collect(tmp_path)[0]).closest is None


# ---- shape features -----------------------------------------------------------------------------
def test_the_feature_table_lists_block_sizes_and_which_leading_dimensions_are_powers_of_two(tmp_path: Path):
    three(tmp_path, "B3_K256", TWO_SPEEDS, shape=(64, 384, 256))
    three(tmp_path, "B3", ONE_SPEED, shape=(64, 384, 384))
    feats = {f.layer: f for f in sss.feature_table(sss.collect(tmp_path)[0])}

    k256, b3 = feats["B3_K256"], feats["B3"]
    assert (k256.b_block_bytes, k256.a_block_bytes) == (128 * 384 * 4, 64 * 128 * 4)   # kc=128 <= K=256
    assert k256.ldb_pow2 and not b3.ldb_pow2
    assert k256.lda == 64 and k256.lda_pow2 and b3.ldb == 384 and not b3.ldb_pow2   # column-major: lda=m, ldb=k
    assert k256.slow_share > b3.slow_share
    assert [f.layer for f in sss.feature_table(sss.collect(tmp_path)[0])][0] == "B3_K256"  # sorted, slowest first


def test_the_feature_table_counts_the_loop_blocks_and_the_packing_per_flop(tmp_path: Path):
    three(tmp_path, "B3", ONE_SPEED, shape=(64, 384, 384), nc=128)   # jc: 3 blocks, pc: 3 of kc=128, ic: 1 of mc=64
    (f,) = sss.feature_table(sss.collect(tmp_path)[0])

    assert (f.jc_blocks, f.pc_blocks, f.ic_blocks) == (3, 3, 1)
    assert f.pack_bytes_per_flop == pytest.approx(sss.bytes_per_rep(64, 384, 384, 128) / ops(64, 384, 384))


# ---- command line -------------------------------------------------------------------------------
def test_the_command_line_prints_every_section_and_can_write_the_file(tmp_path: Path):
    three(tmp_path, "B3", TWO_SPEEDS)
    out = tmp_path / "survey.md"
    done = subprocess.run([sys.executable, str(HERE / "readers" / "slow_state_survey.py"), str(tmp_path), "--out", str(out), "--until", "2026-12-31_000000"],
                          capture_output=True, text=True, check=True)

    for heading in ("## 1. Inventory", "## 2. Slow state per case", "## 3. Time, reps or bytes", "## 4. Shape features"):
        assert heading in done.stdout
    assert out.read_text() == done.stdout
