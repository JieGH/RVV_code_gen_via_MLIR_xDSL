"""Issue #85: run_issue85.sh, the vector-against-scalar packing sweep driver. --dry-run only: order, flags, refusals."""

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_issue85.sh"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=30)


def _steps(out: str) -> list[tuple[str, str]]:
    # "[HH:MM] M1-launch2: scalar packer, 1t, --tmin 2 on jbpi" -> ("M1-launch2", "scalar")
    return [(ln.split("] ")[1].split(":")[0], ln.split(": ")[1].split(" packer")[0])
            for ln in out.splitlines() if " packer, " in ln]


def test_one_thread_series_checks_correctness_then_m1_abba_then_m2(tmp_path: Path):
    done = run("--board", "jbpi", "--threads", "1", "--report-dir", str(tmp_path), "--dry-run")
    assert done.returncode == 0, done.stderr
    v, s = "vector", "scalar"
    assert _steps(done.stderr) == [
        ("correctness", v), ("correctness", s),
        ("M1-launch1", v), ("M1-launch2", s), ("M1-launch3", s), ("M1-launch4", v), ("M1-launch5", v), ("M1-launch6", s),
        ("M2-launch1", v), ("M2-launch2", s), ("M2-launch3", s), ("M2-launch4", v),
    ]
    assert "1t, --tmin 0.2" in done.stderr and "1t, --tmin 3 " in done.stderr and "1t, --instr" in done.stderr


def test_eight_thread_series_names_its_split_m3(tmp_path: Path):
    done = run("--board", "jbpi2", "--threads", "8", "--report-dir", str(tmp_path), "--workspace", "w2", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert [st for st, _ in _steps(done.stderr)][-4:] == ["M3-launch1", "M3-launch2", "M3-launch3", "M3-launch4"]
    assert "8t, --instr" in done.stderr


def test_refusals(tmp_path: Path):
    assert "--board is required" in run("--threads", "1", "--report-dir", str(tmp_path)).stderr
    assert "--report-dir is required" in run("--board", "a", "--threads", "1").stderr
    assert "--threads takes 1 or 8" in run("--board", "a", "--threads", "4", "--report-dir", str(tmp_path)).stderr
