"""
Issue #50: `run_issue50.sh` runs the prefetcher-counters series as repeated instrumented
launches (no setting to rotate: every launch is the same --instr --raw-events x60-l2 run of
the four-tile config), letting the boards take turns.

Same seam as test_run_issue48.py: the bench script is replaced by a stub (`ISSUE50_RUN`), the
board's process list by a canned one (`BOARD_PS_CMD`) and the logs go to a temp dir, so no
board is touched. What is checked is the loop: the boards take turns, every run gets
--instr --raw-events, a failed or incomplete run is named and the series goes on, another
user's heavy job is a RED NOTE without stopping anything, and the sweep config holds the four
tiles the issue names.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
SERIES = HERE / "experiments/issue50/run_issue50.sh"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "readers"))

from pack_dims_table import sweep_tiles  # noqa: E402

STUB_RUN = r"""#!/bin/bash
echo "RUN $*" >> "${ISSUE50_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE50_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE50_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
cfg=""; while [ $# -gt 0 ]; do [ "$1" = --config ] && cfg="$2"; shift; done
rows=$(grep -c '^[[:space:]]*\["fp32"' "${STUB_CONFIG_DIR}/${cfg}")
[ "${what}" = short ] && rows=$((rows - 1))
sleep 1.1
d="${ISSUE50_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"
for ((i = 0; i < rows; i++)); do mkdir -p "${d}/proj${i}"; : > "${d}/proj${i}/reps_x.txt.gz"; done
exit 0
"""


def run_series(tmp_path: Path, behaviour: str, *extra: str, ps: str = "") -> tuple[subprocess.CompletedProcess[str], Path]:
    (tmp_path / "run.sh").write_text(STUB_RUN)
    (tmp_path / "run.sh").chmod(0o755)
    (tmp_path / "ps.sh").write_text(f"#!/bin/bash\ncat <<'EOF'\n{ps}EOF\n")
    (tmp_path / "ps.sh").chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "run_zz_old").mkdir(exist_ok=True)  # sorts after every stamped folder: must not be taken for a result
    env = {**os.environ, "ISSUE50_LOGS": str(logs), "ISSUE50_RUN": str(tmp_path / "run.sh"),
           "BOARD_PS_CMD": str(tmp_path / "ps.sh"), "STUB_BEHAVIOUR": behaviour, "STUB_CONFIG_DIR": str(HERE)}
    done = subprocess.run([str(SERIES), "--boards", "a,b", "--launches", "2", *extra],
                          env=env, capture_output=True, text=True, timeout=180, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("issue50_*"))[-1]


def steps(logs: Path) -> list[str]:
    """`<board><launch>` for every run, from runs.tsv."""
    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    return [f"{r[0]}{r[1]}" for r in rows]


ALL_OK = ",".join(["ok"] * 4)


def test_the_boards_take_turns_launch_by_launch(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 4 of 4 runs ok." in done.stdout
    assert steps(logs) == ["a1", "b1", "a2", "b2"]


def test_every_run_gets_instr_and_the_default_raw_events(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK)

    for line in (logs / "calls.txt").read_text().splitlines():
        assert "--config experiments/issue50/config_issue50_sweep.json" in line and "--no-nc-overrides" in line
        assert "--cpu 4" in line and "--instr" in line and "--raw-events x60-l2" in line
        assert "--tmin" not in line  # the instrumented build runs the warm-up protocol, which refuses --tmin


def test_raw_events_can_be_overridden(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK, "--raw-events", "name=code,0x1")

    assert all("--raw-events name=code,0x1" in ln for ln in (logs / "calls.txt").read_text().splitlines())


def test_empty_raw_events_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--raw-events", "")

    assert done.returncode != 0
    assert "--raw-events needs a value" in done.stderr


def test_a_workspace_is_passed_on_to_every_run_and_left_out_by_default(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, logs = run_series(tmp_path / "a", ALL_OK, "--workspace", "ws_b")
    _, plain = run_series(tmp_path / "b", ALL_OK)

    assert all("--riscv-workspace ws_b" in ln for ln in (logs / "calls.txt").read_text().splitlines())
    assert all("--riscv-workspace" not in ln for ln in (plain / "calls.txt").read_text().splitlines())


def test_core_0_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--cpu", "0")

    assert done.returncode != 0
    assert "core 0" in (done.stdout + done.stderr)


def test_a_run_that_exits_non_zero_is_named_and_the_series_goes_on(tmp_path: Path):
    done, logs = run_series(tmp_path, "ok,exit1,ok,ok")

    assert done.returncode == 1
    assert "Series finished: 3 of 4 runs ok." in done.stdout
    assert "failed: launch 1 b (exit 1)" in done.stdout
    assert "FAIL(1)" in (series_dir(logs) / "runs.tsv").read_text()


def test_a_run_that_leaves_a_tile_out_is_failed_though_it_exited_zero(tmp_path: Path):
    done, _ = run_series(tmp_path, "short,ok,ok,ok")

    assert done.returncode == 1
    assert "failed: launch 1 a (exit 0)" in done.stdout


PS_BUSY = "psiwins        92.0   0 01:21:45 spmv\njlei           20.0   1 00:00:00 bash\nroot            9.5   4 00:00:01 sshd\n"


def test_another_users_heavy_job_is_a_red_note_and_does_not_stop_the_series(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, ps=PS_BUSY)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "RED NOTE" in done.stdout
    warnings = (series_dir(logs) / "load_warnings.txt").read_text()
    assert "psiwins" in warnings and "jlei" not in warnings


def test_the_dry_run_prints_the_plan_and_starts_nothing(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--dry-run")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "4 tiles x 3 shapes" in done.stdout
    assert not list(logs.glob("issue50_*"))


def test_the_sweep_config_holds_the_four_tiles_that_flip():
    tiles = set(sweep_tiles(HERE / "experiments/issue50/config_issue50_sweep.json"))

    assert tiles == {(16, 6), (16, 7), (16, 8), (16, 15)}
