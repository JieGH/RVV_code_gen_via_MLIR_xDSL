"""Issue #83: run_issue83.sh, the scalar-packing rerun driver. --dry-run only: the order and the refusals."""

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_issue83.sh"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=30)


def test_yolo_part_checks_correctness_at_1_and_8_threads_before_timing():
    done = run("--part", "yolo", "--board", "jbpi", "--src", "/his", "--dry-run")
    assert done.returncode == 0, done.stderr
    steps = [ln.split("] ")[1].split(":")[0] for ln in done.stdout.splitlines() if ln.startswith("[")]
    assert steps == ["C1", "C2", "T2", "T1", "issue #83 yolo part finished."]
    assert "config_yolo_correctness.json" in done.stdout and "--scalar-pack" in done.stdout


def test_paper_part_uses_the_oracle_sized_config():
    done = run("--part", "paper", "--board", "jbpi2", "--workspace", "w2", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "config_paper_correctness.json" in done.stdout and "T3: run_paper_sweep.sh --scalar-pack" in done.stdout


def test_refusals():
    assert "--part takes yolo or paper" in run("--part", "x", "--board", "a").stderr
    assert "--board is required" in run("--part", "paper").stderr
    assert "needs --src" in run("--part", "yolo", "--board", "a").stderr
    assert "--workspace is for the paper part" in run("--part", "yolo", "--board", "a", "--src", "/s", "--workspace", "w").stderr
