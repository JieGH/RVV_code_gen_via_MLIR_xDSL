"""
`run_paper_sweep.sh` alternates 1-thread and 8-thread Tile Search launches on the paper's shapes, 1T 8T 8T 1T 1T 8T,
after waiting for another series' process. Seam: PAPER_SWEEP stands in for run_yolo_sweep.sh. No board is touched.
"""

import os
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_paper_sweep.sh"

STUB = r"""#!/bin/bash
echo "$*" >> "${PAPER_LOGS}/calls.txt"
n=$(( $(cat "${PAPER_LOGS}/.n" 2>/dev/null || echo 0) + 1 )); echo $n > "${PAPER_LOGS}/.n"
[ "$n" = "${FAIL_AT:-0}" ] && exit 1
exit 0
"""


def run(tmp_path: Path, *extra: str, **env_extra: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    (tmp_path / "sweep.sh").write_text(STUB)
    (tmp_path / "sweep.sh").chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "PAPER_LOGS": str(logs), "PAPER_SWEEP": str(tmp_path / "sweep.sh"), **env_extra}
    done = subprocess.run([str(SERIES), "--issue", "71", *extra], env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    calls = (logs / "calls.txt").read_text().splitlines() if (logs / "calls.txt").exists() else []
    return done, calls


def arg(call: str, flag: str) -> str:
    parts = call.split()
    return parts[parts.index(flag) + 1]


def test_three_launches_each_alternate_1t_and_8t_abba_on_the_paper_config(tmp_path: Path):
    done, calls = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [(arg(c, "--threads"), arg(c, "--launch")) for c in calls] == [
        ("1", "1"), ("8", "1"), ("8", "2"), ("1", "2"), ("1", "3"), ("8", "3")]
    assert all(arg(c, "--config").endswith("config_paper_search.json") and arg(c, "--boards") == "jbpi"
               and arg(c, "--launches") == "3" and arg(c, "--issue") == "71" for c in calls)


def test_it_waits_for_the_given_process_before_the_first_step(tmp_path: Path):
    blocker = subprocess.Popen(["sleep", "2"])
    threading.Thread(target=blocker.wait).start()  # reap it, or kill -0 still sees the zombie
    done, calls = run(tmp_path, "--launches", "1", "--after-pid", str(blocker.pid), PAPER_POLL_S="0.2")
    assert blocker.poll() is not None and done.returncode == 0 and len(calls) == 2
    assert f"waiting for process {blocker.pid}" in done.stdout


def test_a_failed_step_stops_the_series(tmp_path: Path):
    done, calls = run(tmp_path, FAIL_AT="2")
    assert done.returncode != 0 and len(calls) == 2 and "step 2 (8T L1) failed" in done.stdout


def test_dry_run_starts_nothing(tmp_path: Path):
    done, calls = run(tmp_path, "--dry-run")
    assert done.returncode == 0 and calls == [] and "1T L1, 8T L1, 8T L2, 1T L2, 1T L3, 8T L3" in done.stdout


def test_scalar_pack_and_workspace_reach_every_step(tmp_path: Path):
    """Issue #83: the scalar-packing rerun, possibly on the second board's own workspace."""
    done, calls = run(tmp_path, "--launches", "1", "--issue", "83", "--scalar-pack", "--workspace", "ws2")
    assert done.returncode == 0 and len(calls) == 2
    assert all("--scalar-pack" in c and "--workspace ws2" in c and arg(c, "--issue") == "83" for c in calls)
