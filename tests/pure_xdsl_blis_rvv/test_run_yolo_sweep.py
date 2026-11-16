"""
Issue #61: `run_yolo_sweep.sh` runs the Tile Search on the YOLO shapes as launches of
config_yolo_search.json, each launch with the rows rotated, on the first idle board, and waits
while another user's job is busy on it.

Same seam as test_run_issue51.py: the bench script is a stub (`YOLO_SWEEP_RUN`), the board's
process list a canned one (`YOLO_SWEEP_PS_CMD`), the logs go to a temp dir. No board is touched.
"""

import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_yolo_sweep.sh"

# Records its arguments; leaves one per-rep file per row of the config it is given (none on exit1).
STUB_RUN = r"""#!/bin/bash
echo "RUN $*" >> "${YOLO_SWEEP_LOGS}/calls.txt"
n=$(( $(cat "${YOLO_SWEEP_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${YOLO_SWEEP_LOGS}/.calls"
what="$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")"
[ "${what}" = exit1 ] && exit 1
cfg="$(echo "$*" | sed -E 's/.*--config ([^ ]+).*/\1/')"
sleep 1.1
d="${YOLO_SWEEP_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"
shapes=$(grep -c '^[A-Za-z0-9_]' "$(dirname "$0")/dataset.dat"); [ "${what}" = short ] && shapes=$((shapes - 1))
for ((i = 0; i < $(grep -c '^ *\[\"' "${cfg}"); i++)); do
    mkdir -p "${d}/p${i}"; { echo "Layer,M"; for ((j = 0; j < shapes; j++)); do echo "Y${j},1"; done; } > "${d}/p${i}/reps_x.txt"
done
exit 0
"""

# Board $1 is busy when its name is in BUSY_BOARDS, or when this read's number (1, 2, ...) is in BUSY_AT.
STUB_PS = r"""#!/bin/bash
n=$(( $(cat "${YOLO_SWEEP_LOGS}/.ps" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${YOLO_SWEEP_LOGS}/.ps"
if [[ ",${BUSY_BOARDS}," == *",$1,"* ]] || [[ ",${BUSY_AT}," == *",${n},"* ]]; then
    echo "someone              99.0   4    01:00:00 gemm"
fi
echo "jlei                 50.0   3       00:00 bash"
"""


def run_series(tmp_path: Path, *extra: str, behaviour: str = "ok,ok,ok", **env_extra: str
               ) -> tuple[subprocess.CompletedProcess[str], Path]:
    for name, text in (("run.sh", STUB_RUN), ("ps.sh", STUB_PS)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    (tmp_path / "dataset.dat").write_text((HERE / "datasets/yolo26_k1.dat").read_text())
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "YOLO_SWEEP_LOGS": str(logs), "YOLO_SWEEP_RUN": str(tmp_path / "run.sh"),
           "YOLO_SWEEP_PS_CMD": str(tmp_path / "ps.sh"), "STUB_BEHAVIOUR": behaviour, **env_extra}
    done = subprocess.run([str(SERIES), "--boards", "a,b", "--wait-min", "0", *extra],
                          env=env, capture_output=True, text=True, timeout=180, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("yolo_sweep_*"))[-1]


def tiles(config: Path) -> list[str]:
    rows = json.loads(re.sub(r"^\s*//.*$", "", config.read_text(), flags=re.M))
    return [f"{r[2][0]}x{r[2][1]}" for r in rows]


def test_three_launches_rotate_the_19_tiles_and_keep_the_approved_settings(tmp_path: Path):
    done, logs = run_series(tmp_path, "--tmin", "2")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 3 of 3 launches ok on a." in done.stdout

    sd = series_dir(logs)
    base = tiles(HERE / "config_yolo_search.json")
    orders = [tiles(sd / f"config_L{l}.json") for l in (1, 2, 3)]
    assert [o[0] for o in orders] == ["8x4", "16x8", "24x8"]  # offsets 0, 7, 14 of 19
    assert all(sorted(o) == sorted(base) and len(o) == 19 for o in orders)
    calls = (logs / "calls.txt").read_text().splitlines()
    assert len(calls) == 3
    assert all("--riscv-ip a" in c and "--cpu 4" in c and "--tmin 2" in c and "--no-nc-overrides" in c for c in calls)
    assert [ln.split("\t")[2] for ln in (sd / "runs.tsv").read_text().splitlines()[1:]] == ["8x4", "16x8", "24x8"]


def test_a_busy_board_is_passed_over_for_an_idle_one_and_kept_for_every_launch(tmp_path: Path):
    done, logs = run_series(tmp_path, BUSY_BOARDS="a")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "a is busy: someone gemm 99.0% on core 4" in done.stdout
    assert all("--riscv-ip b" in c for c in (logs / "calls.txt").read_text().splitlines())


def test_with_no_idle_board_the_series_waits_before_picking_one(tmp_path: Path):
    done, logs = run_series(tmp_path, "--boards", "a", "--launches", "1", BUSY_AT="1,2")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("no idle board; reading again") == 2
    assert len((logs / "calls.txt").read_text().splitlines()) == 1


def test_a_launch_waits_while_another_users_job_is_busy_and_never_starts_under_it(tmp_path: Path):
    # read 1 picks the board, read 2 clears launch 1, read 3 is after launch 1, read 4 finds launch 2's board busy
    done, logs = run_series(tmp_path, "--boards", "a", "--launches", "2", BUSY_AT="4")
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert "launch 2: a is busy (someone gemm 99.0% on core 4" in out
    assert out.index("launch 2: a is busy") < out.index("launch 2/2 on a")
    assert len((logs / "calls.txt").read_text().splitlines()) == 2


def test_a_board_that_cannot_be_read_counts_as_busy(tmp_path: Path):
    (tmp_path / "fail.sh").write_text("#!/bin/bash\nn=$(( $(cat \"${YOLO_SWEEP_LOGS}/.f\" 2>/dev/null || echo 0) + 1 ))\n"
                                      "echo $n > \"${YOLO_SWEEP_LOGS}/.f\"\n[ $n -le 1 ] && exit 255\nexit 0\n")
    (tmp_path / "fail.sh").chmod(0o755)
    (tmp_path / "logs").mkdir()
    env = {**os.environ, "YOLO_SWEEP_LOGS": str(tmp_path / "logs"), "YOLO_SWEEP_RUN": "/bin/false",
           "YOLO_SWEEP_PS_CMD": str(tmp_path / "fail.sh")}
    done = subprocess.run([str(SERIES), "--boards", "a", "--launches", "1", "--wait-min", "0"],
                          env=env, capture_output=True, text=True, timeout=60, check=False)
    assert "a is busy: cannot read the board's processes" in done.stdout


def test_a_failed_launch_stops_the_series_and_is_named(tmp_path: Path):
    done, logs = run_series(tmp_path, behaviour="ok,exit1,ok")
    assert done.returncode != 0
    assert "Series finished: 1 of 3 launches ok" in done.stdout and "launch 2 (exit 1" in done.stdout
    assert "Stopping the series." in done.stdout
    assert len((logs / "calls.txt").read_text().splitlines()) == 2


def test_a_per_rep_log_missing_a_shape_makes_the_launch_incomplete(tmp_path: Path):
    done, _ = run_series(tmp_path, behaviour="short")
    assert done.returncode != 0 and "19 missing shapes" in done.stdout


def test_load_that_comes_up_during_a_launch_is_a_red_note(tmp_path: Path):
    # read 1 picks, read 2 clears launch 1, read 3 is the read after launch 1
    done, logs = run_series(tmp_path, "--boards", "a", "--launches", "1", BUSY_AT="3")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "RED NOTE: another user's job was on a at the end of launch 1" in done.stdout
    assert "someone gemm" in (series_dir(logs) / "load_warnings.txt").read_text()


def test_dry_run_prints_the_plan_and_starts_nothing(tmp_path: Path):
    done, logs = run_series(tmp_path, "--dry-run")
    assert done.returncode == 0 and "nothing was started" in done.stdout
    assert "Launch 2  : starts at 16x8" in done.stdout
    assert not (logs / "calls.txt").exists()
