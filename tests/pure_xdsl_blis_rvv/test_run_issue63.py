"""
Issue #63: `run_issue63.sh` alternates our 8-thread Tile Search and Héctor's 8-thread gemm.x, one launch at a time,
ours his his ours ours his (A-B-B-A), on one board. Seam: ISSUE63_SWEEP / ISSUE63_HECTOR stand in for
run_yolo_sweep.sh / run_hector_gemm.sh and record their arguments. No board is touched.
"""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_issue63.sh"

# Records "<side> <args>"; fails on the FAIL_AT-th call of any side.
STUB = r"""#!/bin/bash
echo "$(basename "$0" .sh) $*" >> "${ISSUE63_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE63_LOGS}/.n" 2>/dev/null || echo 0) + 1 )); echo $n > "${ISSUE63_LOGS}/.n"
[ "$n" = "${FAIL_AT:-0}" ] && exit 1
exit 0
"""


def run(tmp_path: Path, *extra: str, **env_extra: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    for side in ("ours", "his"):
        (tmp_path / f"{side}.sh").write_text(STUB)
        (tmp_path / f"{side}.sh").chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "ISSUE63_LOGS": str(logs), "ISSUE63_SWEEP": str(tmp_path / "ours.sh"),
           "ISSUE63_HECTOR": str(tmp_path / "his.sh"), **env_extra}
    done = subprocess.run([str(SERIES), "--src", "/his/tree", *extra], env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    calls = (logs / "calls.txt").read_text().splitlines() if (logs / "calls.txt").exists() else []
    return done, calls


def test_three_launches_each_alternate_abba_on_one_board_at_8_threads(tmp_path: Path):
    done, calls = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [c.split()[0] for c in calls] == ["ours", "his", "his", "ours", "ours", "his"]
    ours = [c for c in calls if c.startswith("ours")]
    assert [c.split("--launch ")[1].split()[0] for c in ours] == ["1", "2", "3"]
    assert all("--threads 8" in c and "--launches 3" in c and "--boards jbpi " in c + " " and "--tmin 2" in c
               and "--issue 63" in c for c in ours)
    his = [c for c in calls if c.startswith("his")]
    assert all("--src /his/tree" in c and "--threads 8" in c and "--launches 1" in c and "--boards jbpi" in c
               and "--timin 2" in c and "--issue 63" in c for c in his)
    # one build: the first of his launches builds, the others reuse it
    assert ["--no-build" in c for c in his] == [False, True, True]


def test_a_failed_launch_stops_the_series(tmp_path: Path):
    done, calls = run(tmp_path, FAIL_AT="2")
    assert done.returncode != 0 and len(calls) == 2
    assert "step 2 (his) failed" in done.stdout


def test_board_and_tmin_are_passed_through(tmp_path: Path):
    done, calls = run(tmp_path, "--board", "jbpi2", "--tmin", "10", "--launches", "1")
    assert done.returncode == 0, done.stdout + done.stderr
    assert [c.split()[0] for c in calls] == ["ours", "his"]
    assert all("--boards jbpi2" in c for c in calls) and "--tmin 10" in calls[0] and "--timin 10" in calls[1]


def test_dry_run_prints_the_order_and_starts_nothing(tmp_path: Path):
    done, calls = run(tmp_path, "--dry-run")
    assert done.returncode == 0 and calls == []
    assert "ours L1, his, his, ours L2, ours L3, his" in done.stdout
