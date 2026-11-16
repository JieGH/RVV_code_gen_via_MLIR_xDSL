"""
Issue #62: `run_hector_gemm.sh` builds Héctor's GEMMLib (unedited) on the board and runs his `gemm.x` with the
`gemm.bin` command line: 1 thread on core 4, 8 threads on cores 0-7, in the order 1T 8T 8T 1T 1T 8T, each launch
on an idle board of jbpi/jbpi2, with a side note for a busy one.

Seam: `HECTOR_GEMM_ON_BOARD` stands in for "run this shell command on board $1" (stdin passed through), so the
board's process list, the copy, the build and the runs are all canned. No board is touched.
"""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_hector_gemm.sh"

# $1 board, $2 command. Records each call; answers ps with a busy line when the board is in BUSY_BOARDS or this
# ps read's number is in BUSY_AT; prints a gemm.x result table and the CSV for a run (ERROR on the RUN_ERROR-th run).
STUB = r"""#!/bin/bash
b="$1"; cmd="$2"; L="${HECTOR_GEMM_LOGS}"
cat > /dev/null
case "${cmd}" in
  ps\ *)
    n=$(( $(cat "$L/.ps" 2>/dev/null || echo 0) + 1 )); echo $n > "$L/.ps"
    if [[ ",${BUSY_BOARDS}," == *",$b,"* ]] || [[ ",${BUSY_AT}," == *",$n,"* ]]; then
        echo "someone              99.0   5    01:00:00 python"
    fi
    echo "jlei                  1.0   3       00:00 bash" ;;
  *gemm.x*)
    echo "$b $cmd" >> "$L/runs.txt"
    r=$(( $(cat "$L/.runs" 2>/dev/null || echo 0) + 1 )); echo $r > "$L/.runs"
    [ "$r" = "${RUN_FAIL:-0}" ] && exit 3
    printf ' |      300      16    144 |  1.00e-03   1.38e+00   0.00e+00   |'
    if [ "$r" = "${RUN_ERROR:-0}" ]; then echo "  ERROR  |"; else echo "    OK   |"; fi
    echo "===CSV==="; echo "#l;m;n;k;Gflops:time"; echo "0;300;16;144;1.38e+00;1.0000e-03" ;;
  *) echo "$b $cmd" >> "$L/other.txt" ;;
esac
"""


def run_series(tmp_path: Path, *extra: str, **env_extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    (tmp_path / "stub.sh").write_text(STUB)
    (tmp_path / "stub.sh").chmod(0o755)
    src = tmp_path / "GEMMLib_yolo"
    (src / "gemm_study").mkdir(parents=True)
    (src / "gemm_study" / "yolo.dat").write_text("0\t300\t16\t144\n")
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "HECTOR_GEMM_LOGS": str(logs), "HECTOR_GEMM_ON_BOARD": str(tmp_path / "stub.sh"),
           "HECTOR_GEMM_COMMIT": "b292c92",
           "HECTOR_YOLO_DIR": str(tmp_path / "HectorYolo"), **env_extra}
    done = subprocess.run([str(SERIES), "--src", str(src), "--boards", "a,b", "--wait-min", "0", *extra],
                          env=env, capture_output=True, text=True, timeout=120, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("hector_gemm_*"))[-1]


def test_six_launches_run_his_command_1t_on_core_4_and_8t_on_cores_0_to_7_in_abba_order(tmp_path: Path):
    done, logs = run_series(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    runs = (logs / "runs.txt").read_text().splitlines()
    assert len(runs) == 6
    threads = ["1" if "taskset -c 4 " in r else "8" if "taskset -c 0-7 " in r else "?" for r in runs]
    assert threads == ["1", "8", "8", "1", "1", "8"]
    assert all("OMP_NUM_THREADS=1 " in r for r, t in zip(runs, threads) if t == "1")
    assert all('./build/gemm.x "" C C C N N 1.0 0.0 2.0 T gemm_study/yolo.dat' in r for r in runs)
    # gemm.x NEEDs libGEMM.so by name; his LD_PRELOAD alone does not satisfy it (2026-11-16)
    assert all("LD_LIBRARY_PATH=$(pwd)/install/lib " in r for r in runs)
    sd = series_dir(logs)
    assert (sd / "L1_T1_a.csv").read_text().startswith("#l;m;n;k")
    assert "Series finished: 6 of 6 launches ok" in done.stdout
    # his README's serial build: under -j, gemm.x links against a half-written libGEMM.so (2026-11-16)
    assert "make clean && make && make install" in (logs / "other.txt").read_text()


def test_a_busy_board_gets_a_side_note_and_the_launch_runs_on_the_idle_one(tmp_path: Path):
    done, logs = run_series(tmp_path, BUSY_BOARDS="a")
    assert done.returncode == 0, done.stdout + done.stderr
    assert all(r.startswith("b ") for r in (logs / "runs.txt").read_text().splitlines())
    notes = (series_dir(logs) / "side_notes.txt").read_text()
    assert "a busy: someone python 99.0% on core 5" in notes


def test_with_both_boards_busy_the_launch_waits_and_never_starts_under_load(tmp_path: Path):
    # reads 1-2: launch 1 finds a and b busy; reads 3-4: both idle
    done, logs = run_series(tmp_path, "--launches", "1", BUSY_AT="1,2")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no idle board; reading again" in done.stdout
    assert len((logs / "runs.txt").read_text().splitlines()) == 2  # 1T and 8T


def test_a_launch_with_a_failed_check_or_exit_stops_the_series(tmp_path: Path):
    done, logs = run_series(tmp_path, RUN_ERROR="2")
    assert done.returncode != 0
    assert "launch 2" in done.stdout and "Stopping the series." in done.stdout
    assert len((logs / "runs.txt").read_text().splitlines()) == 2


def test_the_report_and_logs_land_in_hector_yolo_per_issue(tmp_path: Path):
    done, logs = run_series(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    dest = tmp_path / "HectorYolo" / "issue-62" / series_dir(logs).name
    report = (dest / "hector_gemm_report.md").read_text()
    assert "| 0 | 300 | 16 | 144 |" in report and "WARNING" not in report
    assert (dest / "L1_T1_a.csv").exists() and (dest / "runs.tsv").exists()


def test_a_stopped_series_still_dumps_what_it_has(tmp_path: Path):
    _, logs = run_series(tmp_path, "--issue", "70", RUN_ERROR="2")
    dest = tmp_path / "HectorYolo" / "issue-70" / series_dir(logs).name
    assert (dest / "L1_T1_a.log").exists() and (dest / "hector_gemm_report.md").exists()
