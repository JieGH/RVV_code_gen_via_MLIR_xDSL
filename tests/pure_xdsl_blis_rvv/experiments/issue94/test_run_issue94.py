"""Issue #94: run_issue94.sh (V, S, H mixed in one series) and run_parallel_gemm.sh --threads/--scalar-pack. No board."""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
SCRIPT = HERE / "experiments/issue94/run_issue94.sh"
ROUNDS = ["V 1", "S 1", "H 1", "V 8", "S 8", "H 8",
          "S 8", "H 8", "V 8", "S 1", "H 1", "V 1",
          "H 1", "V 1", "S 1", "H 8", "V 8", "S 8"]


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, **(env or {})})


def test_order_rotates_setups_and_alternates_thread_counts_with_correctness_first():
    done = run("--src", "/his", "--dry-run")
    assert done.returncode == 0, done.stderr
    steps = [ln.split(": ", 1)[1].rstrip("T") for ln in done.stdout.splitlines() if ln.startswith("  step ")]
    assert steps == ROUNDS
    for s in "VSH":  # 3 launches per setup per thread count
        for t in "18":
            assert steps.count(f"{s} {t}") == 3
    c = done.stdout.index("C1: config_issue94_correctness.json")
    assert c < done.stdout.index("C2: config_issue94_correctness.json") and "--scalar-pack, oracle on" in done.stdout


def test_refusals():
    assert "--src is required" in run("--dry-run").stderr
    assert "unknown parameter" in run("--src", "/s", "--bogus").stderr


def test_series_calls_each_runner_with_its_setup(tmp_path: Path):
    calls = tmp_path / "calls.txt"
    stub = tmp_path / "stub.sh"
    stub.write_text(f'#!/bin/bash\necho "$(basename "$0") $*" >> {calls}\n')
    stub.chmod(0o755)
    ours, his = tmp_path / "ours.sh", tmp_path / "his.sh"
    ours.symlink_to(stub)
    his.symlink_to(stub)
    done = run("--src", "/his", "--skip-correctness",
               env={"ISSUE94_OURS": str(ours), "ISSUE94_HECTOR": str(his), "ISSUE94_LOGS": str(tmp_path / "logs"),
                    "HECTOR_YOLO_DIR": str(tmp_path / "reports")})
    assert done.returncode == 0, done.stdout + done.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 18
    for line, step in zip(lines, ROUNDS):
        s, t = step.split()
        assert f"--threads {t} --launches 1" in line and "--issue 94" in line
        assert line.startswith("his.sh" if s == "H" else "ours.sh")
        assert ("--scalar-pack" in line) == (s == "S")
        assert ("--omp-bind" in line) == (s == "H")  # same run settings as ours (user, 2026-11-18)
    his_calls = [ln for ln in lines if ln.startswith("his.sh")]
    assert "--no-build" not in his_calls[0] and all("--no-build" in ln for ln in his_calls[1:])
    series = next((tmp_path / "logs").glob("issue94_*"))
    assert len((series / "steps.tsv").read_text().splitlines()) == 19
    assert (tmp_path / "reports" / "issue-94" / series.name / "plan.txt").exists()


def test_parallel_gemm_runs_one_thread_count_with_the_scalar_packer():
    pg = HERE / "run_parallel_gemm.sh"
    done = subprocess.run(["bash", str(pg), "--threads", "8", "--launches", "1", "--scalar-pack", "--issue", "94", "--dry-run"],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert "scalar packer" in done.stdout and "Order    : 8 (threads" in done.stdout
    assert "Issue #94:" in done.stdout and "Issue #56:" not in done.stdout
    default = subprocess.run(["bash", str(pg), "--dry-run"], capture_output=True, text=True, timeout=30).stdout
    assert "vector packer" in default and "Order    : 1 8 8 1 1 8" in default and "Issue #56:" in default
    bad = subprocess.run(["bash", str(pg), "--threads", "4", "--dry-run"], capture_output=True, text=True, timeout=30)
    assert "--threads takes 1 or 8" in bad.stderr


def test_each_thread_count_runs_its_own_config(tmp_path: Path):
    """#109: ours at 1T runs --config-1t, at 8T --config-8t; Héctor's steps get neither; C1/C2 name their own configs."""
    calls = tmp_path / "calls.txt"
    stub = tmp_path / "stub.sh"
    stub.write_text(f'#!/bin/bash\necho "$(basename "$0") $*" >> {calls}\n')
    stub.chmod(0o755)
    ours, his = tmp_path / "ours.sh", tmp_path / "his.sh"
    ours.symlink_to(stub)
    his.symlink_to(stub)
    c1, c8 = HERE / "configs/benchmark/config_yolo53_24x8.json", HERE / "configs/tile_search/config_yolo53_search_8t.json"  # any two configs with distinct tiles
    flags = ("--src", "/his", "--config-1t", str(c1), "--config-8t", str(c8), "--issue", "109")
    done = run(*flags, "--skip-correctness",
               env={"ISSUE94_OURS": str(ours), "ISSUE94_HECTOR": str(his), "ISSUE94_LOGS": str(tmp_path / "logs"),
                    "HECTOR_YOLO_DIR": str(tmp_path / "reports")})
    assert done.returncode == 0, done.stdout + done.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 18
    for line, step in zip(lines, ROUNDS):
        s, t = step.split()
        assert "--issue 109" in line
        want = "" if s == "H" else f"--config {c1 if t == '1' else c8}"
        assert ("--config" in line) == bool(want) and want in line
    assert (tmp_path / "reports" / "issue-109").is_dir()
    plan = run(*flags, "--check-1t", str(HERE / "configs/correctness/config_yolo53_correctness_24x8.json"), "--dry-run").stdout
    assert "Ours 1T  : config_yolo53_24x8.json (tile 24x8)" in plan and "(tile 16x12)" in plan
    assert "C1: config_yolo53_correctness_24x8.json" in plan and "C2: config_issue94_correctness.json" in plan
    pg = subprocess.run(["bash", str(HERE / "run_parallel_gemm.sh"), "--config", str(c1), "--dry-run"],
                        capture_output=True, text=True, timeout=30).stdout
    assert "(tile 24x8)" in pg and "datasets/yolo26_k1_53.dat (53 shapes" in pg
