"""Issue #60: the Tile Search. A search config is plain pinned rows, one per Candidate Tile (no new row kind), and
`search_report.py` reads the per-rep logs of its run folders and names the fastest tile per shape.

Seams: the configs, read with the project's own config reader; `search_report.py` on hand-made run folders. No board.
"""

import gzip
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from config import BLISProjectConfig  # noqa: E402

# The 19 Candidate Tiles agreed in issue #59 §3.2.
S1_TILES = [(8, nr) for nr in (4, 8, 12, 16, 24)] + [(16, nr) for nr in (4, 6, 8, 10, 12, 14, 15)] \
    + [(24, nr) for nr in (4, 6, 8, 9)] + [(32, nr) for nr in (4, 6, 7)]


def test_the_yolo_search_config_is_one_pinned_row_per_s1_tile_on_the_yolo_dataset():
    cfgs = BLISProjectConfig.from_json(str(HERE / "config_yolo_search.json"))
    assert [c.mr_nr_family[0] for c in cfgs] == S1_TILES
    assert all(c.is_pinned and len(c.mr_nr_family) == 1 for c in cfgs)
    assert all(c.datasets == ["yolo26_k1.dat"] and c.skip_ref_run for c in cfgs)
    assert len({c.out_dir() for c in cfgs}) == 19  # one project folder each


def test_the_yolo_correctness_config_checks_the_same_tiles_on_small_yolo_like_shapes():
    cfgs = BLISProjectConfig.from_json(str(HERE / "config_yolo_correctness.json"))
    assert [c.mr_nr_family[0] for c in cfgs] == S1_TILES
    assert all(c.is_pinned and not c.skip_ref_run for c in cfgs)
    assert all(c.datasets == ["yolo_correctness_small.dat"] for c in cfgs)
    shapes = [ln.split() for ln in (HERE / "datasets/yolo_correctness_small.dat").read_text().splitlines()
              if ln.strip() and not ln.startswith("#")]
    assert {int(n) for _, _, n, _ in shapes} == {4, 16}
    assert {int(m) for _, m, _, _ in shapes} == {300}
    # the scalar oracle's B2 limit: test_correctness_pass.py::test_no_config_row_runs_the_scalar_oracle_on_a_shape_bigger_than_b2


# --- search_report.py on hand-made run folders ------------------------------------------------------------------

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds\n"
SHAPES = {"Y00": (300, 16, 27), "Y01": (300, 4, 9)}


def secs(layer: str, gops: float, reps: int = 20) -> list[float]:
    m, n, k = SHAPES[layer]
    return [2.0 * m * n * k / (gops * 1e9)] * reps


def run_folder(logs: Path, name: str, gops: dict[tuple[str, str], float], others: str = "none") -> Path:
    """One launch: per (tile, layer) a steady speed in GOPs, one per-rep file per tile's project folder."""
    d = logs / name
    d.mkdir(parents=True)
    (d / "provenance.txt").write_text(f"  board          : jlei@jbpi\n  cpu pin        : 4\n  other users    : {others}\n")
    for tile in sorted({t for t, _ in gops}):
        mr, nr = tile.split("x")
        rows = "".join(f"{layer},{m},{n},{k},MR{mr}_NR{nr},64,{n},{k},{r},{s},nan,nan\n"
                       for (t, layer), g in gops.items() if t == tile
                       for (m, n, k) in [SHAPES[layer]] for r, s in enumerate(secs(layer, g)))
        f = d / f"fp32_AXPY_FP32_Pack_vlen256_MR{mr}NR{nr}" / "logs" / "reps_fp32_noinstr_refproto.txt.gz"
        f.parent.mkdir(parents=True)
        f.write_bytes(gzip.compress((HEADER + rows).encode()))
    return d


# 16x8 wins Y00 (n = 16), 16x4 wins Y01 (n = 4); over both shapes 16x8 takes less time in total.
SPEEDS = {("16x8", "Y00"): 6.0, ("16x8", "Y01"): 2.0, ("16x4", "Y00"): 4.0, ("16x4", "Y01"): 3.0}


def run_report(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(HERE / "search_report.py"), *args], capture_output=True, text=True)


def table_row(text: str, first_cell: str) -> list[str]:
    line = next(ln for ln in text.splitlines() if ln.startswith(f"| {first_cell} |"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_the_report_names_the_fastest_tile_per_shape_by_both_metrics_with_the_compare_tile(tmp_path: Path):
    runs = [run_folder(tmp_path, f"run_{i}", SPEEDS) for i in range(3)]
    out = run_report(*map(str, runs), "--compare", "16x8", "--compare", "n=4:16x4")
    assert out.returncode == 0, out.stderr

    y00, y01 = table_row(out.stdout, "Y00"), table_row(out.stdout, "Y01")
    # shape, m, n, k, launches, best by mean time, its GOPs, best by median rep, its GOPs, compare tile, its GOPs
    assert y00 == ["Y00", "300", "16", "27", "3", "16x8", "6.00 (6.00-6.00)", "16x8", "6.00 (6.00-6.00)", "16x8", "6.00"]
    assert y01 == ["Y01", "300", "4", "9", "3", "16x4", "3.00 (3.00-3.00)", "16x4", "3.00 (3.00-3.00)", "16x4", "3.00"]
    assert "WARNING" not in out.stdout and "core 4" in out.stdout


def test_the_summary_ranks_tiles_by_total_time_over_the_dataset_and_per_n(tmp_path: Path):
    out = run_report(*(str(run_folder(tmp_path, f"run_{i}", SPEEDS)) for i in range(3))).stdout
    summary = out.split("## Best single tile")[1].split("##")[0]
    # 16x8: 259200 / 6 + 21600 / 2 = 54000 ns; 16x4: 64800 + 7200 = 72000 ns (ops / GOPs = ns)
    assert summary.index("| 16x8 |") < summary.index("| 16x4 |")
    assert table_row(summary, "16x8")[1:3] == ["0.0540", "5.20"]  # ms per pass, GOPs over the pass
    per_n = out.split("## Best tile per n")[1]
    assert table_row(per_n, "4")[1:3] == ["1", "16x4"] and table_row(per_n, "16")[1:3] == ["1", "16x8"]


def test_fewer_than_three_launches_warns(tmp_path: Path):
    out = run_report(str(run_folder(tmp_path, "run_0", SPEEDS))).stdout
    assert "WARNING" in out and "fewer than 3 launches" in out


def test_a_tile_missing_from_one_launch_warns_that_the_shape_is_not_compared_like_for_like(tmp_path: Path):
    runs = [run_folder(tmp_path, f"run_{i}", SPEEDS) for i in range(3)]
    runs.append(run_folder(tmp_path, "run_3", {k: v for k, v in SPEEDS.items() if k[0] == "16x8"}))
    out = run_report(*map(str, runs)).stdout
    assert "different number of launches" in out and "Y00, Y01" in out


def test_one_label_with_two_shapes_is_an_error(tmp_path: Path):
    first = run_folder(tmp_path, "run_0", SPEEDS)
    SHAPES["Y00"] = (300, 16, 99)  # the same label, another dataset
    try:
        second = run_folder(tmp_path, "run_1", SPEEDS)
    finally:
        SHAPES["Y00"] = (300, 16, 27)
    out = run_report(str(first), str(second))
    assert out.returncode != 0 and "Y00" in out.stderr and "one dataset" in out.stderr


def test_a_run_folder_without_per_rep_logs_is_an_error(tmp_path: Path):
    (tmp_path / "run_0").mkdir()
    out = run_report(str(tmp_path / "run_0"))
    assert out.returncode != 0 and "--no-save-reps" in out.stderr


def test_the_grid_shows_every_tile_per_shape_sorted_by_n_with_the_best_in_bold(tmp_path: Path):
    out = run_report(*(str(run_folder(tmp_path, f"run_{i}", SPEEDS)) for i in range(3))).stdout
    grid = out.split("## GOPs grid")[1].split("##")[0]
    assert "| shape | m | n | k | 16x4 | 16x8 |" in grid
    assert grid.index("| Y01 |") < grid.index("| Y00 |")  # n = 4 before n = 16
    assert table_row(grid, "Y00")[4:] == ["4.00", "**6.00**"]


def test_heatmap_writes_a_png(tmp_path: Path):
    runs = [str(run_folder(tmp_path, f"run_{i}", SPEEDS)) for i in range(3)]
    png = tmp_path / "heat.png"
    out = run_report(*runs, "--heatmap", str(png))
    assert out.returncode == 0, out.stderr
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
