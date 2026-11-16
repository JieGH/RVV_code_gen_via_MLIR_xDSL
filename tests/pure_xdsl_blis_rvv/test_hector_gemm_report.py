"""
`hector_gemm_report.py` turns a `run_hector_gemm.sh` series folder (runs.tsv + one CSV per launch) into a Markdown
report: per shape, his GOPs by mean time at 1 and 8 threads (avg over launches, lowest and highest launch), the
8-thread speedup, and one pass over the dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hector_gemm_report import render  # noqa: E402

HEAD = "#l;m;n;k;Gflops:time\n"


def series(tmp_path: Path, launches: list[tuple[int, int, list[float]]]) -> Path:
    """launches: (launch, threads, seconds per shape) over two shapes."""
    d = tmp_path / "hector_gemm_2026-11-16_000000"
    d.mkdir()
    (d / "plan.txt").write_text("Issue #62: plan line\n")
    (d / "side_notes.txt").write_text("")
    rows = ["launch\tthreads\tcpus\tboard\texit"]
    for l, th, secs in launches:
        rows.append(f"{l}\t{th}\t{'4' if th == 1 else '0-7'}\tjbpi\t0")
        shapes = [(100, 16, 50), (200, 32, 50)]
        lines = [f"{i};{m};{n};{k};{2 * m * n * k / s / 1e9:.2e};{s:.4e}" for i, ((m, n, k), s) in enumerate(zip(shapes, secs))]
        (d / f"L{l}_T{th}_jbpi.csv").write_text(HEAD + "\n".join(lines) + "\n")
    (d / "runs.tsv").write_text("\n".join(rows) + "\n")
    return d


def test_per_shape_avg_low_high_and_speedup_over_launches(tmp_path: Path):
    d = series(tmp_path, [(1, 1, [1e-3, 2e-3]), (2, 8, [2.5e-4, 5e-4]), (3, 8, [2.5e-4, 5e-4]), (4, 1, [1e-3, 2e-3])])
    md = render(d)
    # shape 1: 2*100*16*50 = 160000 flop -> 0.16 GOPs at 1 ms, 0.64 at 0.25 ms; speedup 4.00
    assert "| 100 | 16 | 50 | 0.16 (0.16-0.16) | 0.64 (0.64-0.64) | 4.00 |" in md
    # one pass: 3 ms at 1T, 0.75 ms at 8T
    assert "| 1 thread | 3.000 ms |" in md and "| 8 threads | 0.750 ms |" in md
    assert "Issue #62: plan line" in md
    assert "WARNING" in md  # 2 launches per thread count, under 3


def test_no_warning_with_three_launches_each(tmp_path: Path):
    order = [1, 8, 8, 1, 1, 8]
    d = series(tmp_path, [(i + 1, th, [1e-3, 2e-3]) for i, th in enumerate(order)])
    assert "WARNING" not in render(d)


def test_plot_writes_a_png(tmp_path: Path):
    from hector_gemm_report import plot  # noqa: E402

    d = series(tmp_path, [(1, 1, [1e-3, 2e-3]), (2, 8, [2.5e-4, 5e-4])])
    png = tmp_path / "fig.png"
    plot(d, png)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
