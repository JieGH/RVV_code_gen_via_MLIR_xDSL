"""
Issue #61: `run_yolo_sweep.sh` runs the Tile Search on the YOLO shapes as launches of
config_yolo_search.json, each launch with the rows rotated, on the first idle board, and waits
while another user's job is busy on it.

Same seam as test_run_issue51.py: the bench script is a stub (`YOLO_SWEEP_RUN`), the board's
process list a canned one (`BOARD_PS_CMD`), the logs go to a temp dir. No board is touched.
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

# Stands in for search_report.py: prints its arguments as the report, touches the heatmap it is asked for.
STUB_REPORT = r"""#!/bin/bash
echo "REPORT $*"
while [ "$#" -gt 0 ]; do [ "$1" = --heatmap ] && touch "$2"; shift; done
"""


def run_series(tmp_path: Path, *extra: str, behaviour: str = "ok,ok,ok", dataset: str = "yolo26_k1.dat",
               **env_extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    for name, text in (("run.sh", STUB_RUN), ("ps.sh", STUB_PS), ("report.sh", STUB_REPORT)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    (tmp_path / "dataset.dat").write_text((HERE / "datasets" / dataset).read_text())
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "YOLO_SWEEP_LOGS": str(logs), "YOLO_SWEEP_RUN": str(tmp_path / "run.sh"),
           "BOARD_PS_CMD": str(tmp_path / "ps.sh"), "STUB_BEHAVIOUR": behaviour,
           "YOLO_SWEEP_REPORT": str(tmp_path / "report.sh"), "HECTOR_YOLO_DIR": str(tmp_path / "HectorYolo"), **env_extra}
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
           "BOARD_PS_CMD": str(tmp_path / "fail.sh"), "HECTOR_YOLO_DIR": str(tmp_path / "HectorYolo"),
           "YOLO_SWEEP_REPORT": "/bin/true"}
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


def test_the_report_and_logs_land_in_hector_yolo_per_issue_after_every_launch(tmp_path: Path):
    done, logs = run_series(tmp_path, "--issue", "63", behaviour="ok,ok,exit1")
    sd = series_dir(logs)
    dest = tmp_path / "HectorYolo" / "issue-63" / sd.name
    report = (dest / "tile_search_report.md").read_text()
    # the report covers the two launches that finished, with Héctor's tiles to compare against
    assert report.count("run_") == 2 and "--compare 16x8 --compare n=4:16x4" in report
    assert (dest / "tile_search_heatmap.png").exists()
    assert (dest / "runs.tsv").read_text() == (sd / "runs.tsv").read_text()
    assert f"Report copy: {dest}" in done.stdout


def test_the_issue_folder_defaults_to_61(tmp_path: Path):
    done, _ = run_series(tmp_path, "--launches", "1", behaviour="ok")
    assert done.returncode == 0, done.stdout + done.stderr
    assert list((tmp_path / "HectorYolo" / "issue-61").glob("yolo_sweep_*/tile_search_report.md"))


# Issue #63: the same Tile Search at 8 threads, one launch at a time so another program's launches can go between.
def test_threads_8_replaces_the_core_and_tells_the_bench_to_run_8_threads(tmp_path: Path):
    done, logs = run_series(tmp_path, "--threads", "8", "--launches", "1", behaviour="ok")
    assert done.returncode == 0, done.stdout + done.stderr
    call = (logs / "calls.txt").read_text()
    assert "--threads 8" in call and "--cpu" not in call
    assert "Threads   : 8 on cores 0-7" in (series_dir(logs) / "plan.txt").read_text()


def test_threads_with_cpu_is_refused(tmp_path: Path):
    done, logs = run_series(tmp_path, "--threads", "8", "--cpu", "5")
    assert done.returncode != 0 and "--threads fixes the cores" in done.stderr
    assert not (logs / "calls.txt").exists()


def test_launch_runs_only_that_launch_with_its_rotation(tmp_path: Path):
    done, logs = run_series(tmp_path, "--launches", "3", "--launch", "2", behaviour="ok")
    assert done.returncode == 0, done.stdout + done.stderr
    assert len((logs / "calls.txt").read_text().splitlines()) == 1
    sd = series_dir(logs)
    assert tiles(sd / "config_L2.json")[0] == "16x8" and not (sd / "config_L1.json").exists()
    assert "Series finished: 1 of 1 launches ok" in done.stdout


def test_launch_beyond_launches_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, "--launches", "3", "--launch", "4")
    assert done.returncode != 0 and "--launch takes 1 to 3" in done.stderr


def test_threads_other_than_1_or_8_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, "--threads", "4")
    assert done.returncode != 0 and "--threads takes 1 or 8" in done.stderr


# The paper's BERT and square shapes go through the same sweep with their own config.
def test_config_runs_the_paper_shapes_on_the_19_tiles(tmp_path: Path):
    done, logs = run_series(tmp_path, "--config", str(HERE / "config_paper_search.json"), "--launches", "1",
                            behaviour="ok", dataset="paper_bert_square.dat")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "19 tiles x datasets/paper_bert_square.dat" in done.stdout
    cfg = series_dir(logs) / "config_L1.json"
    assert len(tiles(cfg)) == 19 and "yolo26_k1.dat" not in cfg.read_text()


def test_a_config_mixing_datasets_is_refused(tmp_path: Path):
    mixed = tmp_path / "mixed.json"
    rows = [r for r in (HERE / "config_paper_search.json").read_text().splitlines() if r.lstrip().startswith('["')]
    mixed.write_text("[\n" + rows[0] + "\n" + rows[1].replace("paper_bert_square.dat", "yolo26_k1.dat") + "\n]\n")
    done, _ = run_series(tmp_path, "--config", str(mixed))
    assert done.returncode != 0 and "must run paper_bert_square.dat" in done.stderr


def test_scalar_pack_reaches_every_launch_and_the_plan(tmp_path: Path):
    """Issue #83: --scalar-pack is passed to compile_macro_bench.sh and named in plan.txt."""
    done, logs = run_series(tmp_path, "--threads", "8", "--launches", "1", "--scalar-pack", behaviour="ok")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--scalar-pack" in (logs / "calls.txt").read_text()
    assert "scalar packer" in (series_dir(logs) / "plan.txt").read_text()
