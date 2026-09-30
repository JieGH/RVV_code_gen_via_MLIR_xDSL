"""
Issue #48: `run_issue48.sh` runs the B3 tile sweep as launches of arms (vector, scalar, instrumented),
rotating the arms between launches and letting the boards take turns.

The bench script is replaced by a stub (`ISSUE48_RUN`), the board's process list by a canned one
(`ISSUE48_PS_CMD`) and the logs go to a temp dir, so no board is touched. What is checked is the loop:
the order of the arms and the boards, the flags each arm is given, that a run which exits non-zero or
leaves a tile out is named as failed and the series goes on, that another user's heavy job is written
down as a RED NOTE without stopping anything, and that the arm configs hold the tiles the issue names.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_issue48.sh"

sys.path.insert(0, str(HERE))

from pack_dims_table import pack_plan, sweep_tiles  # noqa: E402

# Leaves one per-rep file per row of the config it is given (the real config files), unless told not to.
STUB_RUN = r"""#!/bin/bash
echo "RUN $*" >> "${ISSUE48_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE48_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE48_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
cfg=""; while [ $# -gt 0 ]; do [ "$1" = --config ] && cfg="$2"; shift; done
rows=$(grep -c '^[[:space:]]*\["fp32"' "${STUB_CONFIG_DIR}/${cfg}")
[ "${what}" = short ] && rows=$((rows - 1))
sleep 1.1
d="${ISSUE48_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"
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
    env = {**os.environ, "ISSUE48_LOGS": str(logs), "ISSUE48_RUN": str(tmp_path / "run.sh"),
           "ISSUE48_PS_CMD": str(tmp_path / "ps.sh"), "STUB_BEHAVIOUR": behaviour, "STUB_CONFIG_DIR": str(HERE)}
    done = subprocess.run([str(SERIES), "--boards", "a,b", "--launches", "2", *extra],
                          env=env, capture_output=True, text=True, timeout=180, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("issue48_*"))[-1]


def steps(logs: Path) -> list[str]:
    """`<board><launch>:<arm>` for every run, from runs.tsv."""
    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    return [f"{r[0]}{r[1]}:{r[3]}" for r in rows]


ALL_OK = ",".join(["ok"] * 8)


def test_the_boards_take_turns_and_the_arms_swap_places_between_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 8 of 8 runs ok." in done.stdout
    assert steps(logs) == ["a1:vec", "a1:scalar", "b1:vec", "b1:scalar", "a2:scalar", "a2:vec", "b2:scalar", "b2:vec"]


def test_three_arms_rotate_so_each_is_first_once_in_three_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ",".join(["ok"] * 12), "--arms", "vec,scalar,vec-instr")

    assert done.returncode == 0, done.stdout + done.stderr
    order = [s.split(":")[1] for s in steps(logs)]
    assert order[0:3] == ["vec", "scalar", "vec-instr"]  # launch 1 on a
    assert order[6:9] == ["scalar", "vec-instr", "vec"]  # launch 2 on a


def test_every_arm_gets_its_config_its_packer_its_build_and_the_same_pin(tmp_path: Path):
    _, logs = run_series(tmp_path, ",".join(["ok"] * 16), "--arms", "vec,scalar,vec-instr,scalar-instr")

    calls = (logs / "calls.txt").read_text().splitlines()[:4]
    vec, scalar, vec_i, scalar_i = calls
    for line in calls:
        assert "--no-nc-overrides" in line and "--cpu 4" in line and "--riscv-ip a" in line
    assert "--config config_issue48_sweep.json" in vec and "--scalar-pack" not in vec and "--instr" not in vec and "--tmin 10" in vec
    assert "--config config_issue48_sweep_scalar.json" in scalar and "--scalar-pack" in scalar and "--tmin 10" in scalar
    assert "--config config_issue48_sweep.json" in vec_i and "--instr" in vec_i and "--scalar-pack" not in vec_i
    assert "--config config_issue48_sweep_scalar.json" in scalar_i and "--instr" in scalar_i and "--scalar-pack" in scalar_i
    assert "--tmin" not in vec_i and "--tmin" not in scalar_i  # the instrumented build runs the warm-up protocol, which refuses --tmin


def test_raw_events_go_only_to_the_instrumented_arms(tmp_path: Path):
    _, logs = run_series(tmp_path, ",".join(["ok"] * 8), "--arms", "vec,vec-instr", "--raw-events", "x60-l2")

    vec, vec_i = (logs / "calls.txt").read_text().splitlines()[:2]
    assert "--raw-events" not in vec and "--raw-events x60-l2" in vec_i


def test_raw_events_without_an_instrumented_arm_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--raw-events", "x60-l2")

    assert done.returncode == 1 and "--raw-events needs an instrumented arm" in done.stderr


def test_core_0_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--cpu", "0")

    assert done.returncode != 0
    assert "core 0" in (done.stdout + done.stderr)


def test_a_run_that_exits_non_zero_is_named_and_the_series_goes_on(tmp_path: Path):
    done, logs = run_series(tmp_path, "ok,exit1," + ",".join(["ok"] * 6))

    assert done.returncode == 1
    assert "Series finished: 7 of 8 runs ok." in done.stdout
    assert "failed: launch 1 a scalar (exit 1)" in done.stdout
    assert "FAIL(1)" in (series_dir(logs) / "runs.tsv").read_text()


def test_a_run_that_leaves_a_tile_out_is_failed_though_it_exited_zero(tmp_path: Path):
    done, _ = run_series(tmp_path, "short," + ",".join(["ok"] * 7))

    assert done.returncode == 1
    assert "failed: launch 1 a vec (exit 0)" in done.stdout


def test_the_result_folder_is_the_stamped_one_not_an_older_folder_that_sorts_after_it(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK)

    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    assert all(r[5] == "0" and r[4].startswith("run_2") for r in rows), rows


PS_BUSY = "psiwins        92.0   0 01:21:45 spmv\njlei           20.0   1 00:00:00 bash\nroot            9.5   4 00:00:01 sshd\n"


def test_another_users_heavy_job_is_a_red_note_and_does_not_stop_the_series(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, ps=PS_BUSY)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 8 of 8 runs ok." in done.stdout
    assert "RED NOTE" in done.stdout
    warnings = (series_dir(logs) / "load_warnings.txt").read_text()
    assert "psiwins" in warnings and "jlei" not in warnings


def test_a_quiet_board_leaves_no_note(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, ps="root            0.3   2 00:00:01 sshd\n")

    assert "RED NOTE" not in done.stdout
    assert not (series_dir(logs) / "load_warnings.txt").exists()


def test_the_dry_run_prints_the_plan_and_starts_nothing(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--dry-run")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Launch 1   : vec scalar" in done.stdout and "Launch 2   : scalar vec" in done.stdout
    assert "25 tiles x 4 shapes" in done.stdout and "7 tiles x 4 shapes" in done.stdout
    assert not list(logs.glob("issue48_*"))


def test_the_sweep_config_holds_the_tiles_the_issue_names_and_the_scalar_arm_is_a_subset():
    full = set(sweep_tiles(HERE / "config_issue48_sweep.json"))
    scalar = set(sweep_tiles(HERE / "config_issue48_sweep_scalar.json"))

    assert {8, 16, 24, 32} <= {mr for mr, _ in full}
    assert {(16, 14), (16, 15), (16, 16)} <= full  # #46's pair and the over-budget reference
    assert {(4, 4), (4, 8), (4, 16), (8, 4), (16, 4), (12, 8), (16, 8), (8, 16), (8, 8)} <= full  # the small and odd tiles asked for on 2026-09-30
    over = [t for t in full if pack_plan(t[0], t[1], 64, 384, 128).over_register_budget]
    assert over == [(16, 16)]  # the one known-bad reference; every other tile fits the 32 registers
    assert scalar < full
    assert {(16, 14), (16, 15)} <= scalar
