"""Issue #56: parallel_gemm_report.py on a made-up series (two shapes, 1 launch per thread count) and a made-up #62 series."""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readers"))

import parallel_gemm_report as r  # noqa: E402

HEADER = "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles\n"


def _run(logs: Path, name: str, secs: dict[str, float], status: str) -> None:
    d = logs / name / "proj"
    d.mkdir(parents=True)
    rows = [f"{lb},{m},{n},{k},MR24_NR8,{m},{n},{k},{rep},{s},nan,nan,nan,nan\n"
            for lb, (m, n, k) in (("Y00", (100, 16, 10)), ("Y01", (200, 64, 10))) for rep in range(3)
            for s in [secs[lb]]]
    (d / "reps_fp32_noinstr.txt").write_text(HEADER + "".join(rows))
    (logs / name / "all_raw_x.txt").write_text("Project,Layer,M,N,K,Variant,Status\np,Y00,1,1,1,v,PASS\np,Y01,1,1,1,v," + status + "\n")
    (logs / name / "provenance.txt").write_text("  board          : jlei@jbpi\n  cpu pin        : 4\n  other users    : none\n")


def test_report(tmp_path: Path) -> None:
    logs, series, hec = tmp_path / "logs", tmp_path / "series", tmp_path / "hec"
    _run(logs, "run_a", {"Y00": 4e-6, "Y01": 8e-6}, "PASS")   # 1T
    _run(logs, "run_b", {"Y00": 1e-6, "Y01": 2e-6}, "FAIL")   # 8T: 4x faster
    series.mkdir()
    (series / "plan.txt").write_text("plan")
    (series / "runs.tsv").write_text("launch\tthreads\tcores\tboard\tresult_dir\texit\n"
                                     "1\t1\t4\tjbpi\trun_a\t0\n2\t8\t0-7\tjbpi\trun_b\t0\n3\t1\t4\tjbpi\trun_c\tFAIL(x)\n")
    hec.mkdir()
    (hec / "runs.tsv").write_text("launch\tthreads\tcpus\tboard\texit\n1\t1\t4\tjbpi\t0\n2\t8\t0-7\tjbpi\t0\n")
    (hec / "L1_T1_jbpi.csv").write_text("#l;m;n;k;Gflops:time\n0;100;16;10;1;8e-6\n1;200;64;10;1;16e-6\n")
    (hec / "L2_T8_jbpi.csv").write_text("#l;m;n;k;Gflops:time\n0;100;16;10;1;2e-6\n1;200;64;10;1;2e-6\n")

    text = r.render(series, logs, hec)
    assert "1 at 1 thread" in text and "1 at 8 threads" in text  # the failed launch is left out
    assert "WARNING: 1 launch(es) at 1 thread(s)" in text
    assert "L1 1T: 2 PASS, 0 FAIL" in text and "L2 8T: 1 PASS, 1 FAIL" in text
    # Y00: 32000 ops / 4 us = 8 GOPs at 1T, 32 at 8T; Héctor 4 and 16
    assert "| Y00 | 100x16x10 | 8.00 (8.00-8.00) | 8.00 | 32.00 (32.00-32.00) | 32.00 | 4.00x | 4.00 (4.00-4.00) | 16.00 (16.00-16.00) | 2.00 |" in text
    # one pass: 288000 ops in 4 + 8 us at 1T (24 GOPs), 1 + 2 us at 8T (96 GOPs)
    assert "| ours, by mean time | 0.0 ms · 24.00 GOPs | 0.0 ms · 96.00 GOPs | 4.00x |" in text
    # n = 64 (Y01, 256000 ops): ours 8 us / 2 us, Héctor 16 us / 2 us
    assert "| 64 | 1 | 32.00 | 128.00 | 4.00x | 16.00 | 128.00 | 8.00x |" in text
    r.plot(series, logs, hec, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0
