"""
Issue #46: `issue46_report.py` reads a `run_issue46.sh` series and answers its plan's step 2 (does the
per-rep reference agree with the unchanged one?) and step 4 (slow-rep share of the reference next to ours).

Seam: hand-made series folders (runs.tsv, result CSVs, per-rep files, provenance.txt). No board.
"""

import gzip
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import issue46_report as rep  # noqa: E402

CASES = "# case\tnumid\tM\tN\tK\tMR\tNR\nB3\t3\t64\t384\t384\t16\t15\nB2\t2\t384\t384\t64\t16\t15\n"
OPS_B3 = 2.0 * 64 * 384 * 384


def ref_dir(logs: Path, name: str, gflops: dict[int, float], per_rep: list[float] | None = None) -> None:
    d = logs / name
    d.mkdir(parents=True)
    (d / "cases.tsv").write_text(CASES)
    (d / "provenance.txt").write_text("  board          : jbpi (user jlei)\n  cpu pin        : 4\n  other users    : none\n")
    for tag in ("16x14", "16x16"):
        rows = "".join(f"{i};64;384;384;{g:.2e}\n" for i, g in gflops.items())
        (d / f"result_{tag}_macc.csv").write_text("#l;m;n;k;Gflops\n" + rows)
    if per_rep is not None:
        text = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles\n" + "".join(
            f"B3,64,384,384,MR16_NR14,64,384,128,{r},{OPS_B3 / (g * 1e9)},nan,nan,nan,nan\n" for r, g in enumerate(per_rep))
        (d / "reps_ref_16x14_macc.txt.gz").write_bytes(gzip.compress(text.encode()))


def series(tmp_path: Path, pairs: list[tuple[float, float]]) -> Path:
    """One launch per pair (plain reference GOPs for B3, per-rep reference GOPs for B3), on jbpi."""
    logs = tmp_path / "logs"
    sd = logs / "issue46_2026-09-29_000000"
    sd.mkdir(parents=True)
    rows = ["board\tlaunch\tposition\tstep\tresult_dir\texit"]
    for i, (p, r) in enumerate(pairs, 1):
        ref_dir(logs, f"refbase_2026-09-29_00{i}000", {3: p, 2: 12.0})
        ref_dir(logs, f"refbase_2026-09-29_00{i}500", {3: r, 2: 12.0}, per_rep=[8.0] * 30 + [4.0] * 70)
        rows += [f"jbpi\t{i}\t1\tref\trefbase_2026-09-29_00{i}000\t0", f"jbpi\t{i}\t2\tref-per-rep\trefbase_2026-09-29_00{i}500\t0"]
    (sd / "runs.tsv").write_text("\n".join(rows) + "\n")
    return sd


def test_the_control_compares_the_two_reference_runs_per_case_and_launch(tmp_path: Path):
    sd = series(tmp_path, [(8.00, 8.10), (8.10, 8.05), (8.05, 8.10)])
    (c,) = [c for c in rep.control(sd) if c.case == "B3" and c.tile == "16x14"]

    assert c.launches == 3
    assert [round(d, 1) for d in c.diff_pct] == [1.2, -0.6, 0.6]
    assert c.verdict == "agree"


def test_a_gap_over_three_percent_fails_the_control(tmp_path: Path):
    sd = series(tmp_path, [(8.00, 8.10), (8.10, 8.60), (8.05, 7.90)])
    (c,) = [c for c in rep.control(sd) if c.case == "B3" and c.tile == "16x14"]

    assert c.verdict.startswith("DISAGREE") and "3%" in c.verdict


def test_the_same_sign_in_every_launch_is_a_systematic_gap_not_noise(tmp_path: Path):
    sd = series(tmp_path, [(8.00, 8.13), (8.00, 8.12), (8.00, 8.14)])   # +1.4% each time
    (c,) = [c for c in rep.control(sd) if c.case == "B3" and c.tile == "16x14"]

    assert c.verdict.startswith("SYSTEMATIC")


def test_a_gap_inside_the_cases_own_launch_to_launch_noise_is_not_a_failure(tmp_path: Path):
    # A two-speed case moves ~10% between launches of the plain reference itself; a gap of that size is not the patch.
    sd = series(tmp_path, [(7.0, 7.6), (7.8, 7.5), (7.4, 7.3)])   # gaps +8.6, -3.8, -1.4 %; plain spread (7.8-7.0)/7.4 = 10.8%
    (c,) = [c for c in rep.control(sd) if c.case == "B3" and c.tile == "16x14"]

    assert c.verdict.startswith("agree within the case's own noise") and "11%" in c.verdict
    assert round(c.plain_spread_pct) == 11


def test_fewer_than_three_launches_is_no_verdict(tmp_path: Path):
    sd = series(tmp_path, [(8.00, 8.10), (8.10, 8.05)])
    (c,) = [c for c in rep.control(sd) if c.case == "B3" and c.tile == "16x14"]

    assert c.verdict.startswith("no verdict")


def test_a_failed_run_is_left_out_of_the_control(tmp_path: Path):
    sd = series(tmp_path, [(8.00, 8.10), (8.10, 8.05), (8.05, 8.10)])
    text = (sd / "runs.tsv").read_text().replace("refbase_2026-09-29_003500\t0", "-\tFAIL(1)")
    (sd / "runs.tsv").write_text(text)
    (c,) = [c for c in rep.control(sd) if c.case == "B3" and c.tile == "16x14"]

    assert c.launches == 2


def test_the_report_prints_the_control_and_the_reference_slow_share(tmp_path: Path):
    sd = series(tmp_path, [(8.00, 8.10), (8.10, 8.05), (8.05, 8.10)])
    done = subprocess.run([sys.executable, str(HERE / "issue46_report.py"), str(sd)], capture_output=True, text=True, check=True)

    assert "## 1. Control" in done.stdout and "## 2. Per-rep" in done.stdout
    assert "reference 16x14" in done.stdout and "| B3 |" in done.stdout
    assert "| 71 |" in done.stdout  # 70 of the 99 reps after rep 0 are slow in the made-up file


