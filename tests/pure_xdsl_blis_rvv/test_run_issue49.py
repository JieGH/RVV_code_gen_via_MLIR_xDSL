"""
Issue #49: `run_issue49.sh` runs the decisive ldb-padding test as launches of pads (0, 8, 16,
128 elements), rotating the pads between launches and letting the boards take turns.

Same seam as test_run_issue48.py: the bench script is replaced by a stub (`ISSUE49_RUN`), the
board's process list by a canned one (`BOARD_PS_CMD`) and the logs go to a temp dir, so no
board is touched. What is checked is the loop: the order of the pads and the boards, that
--pad-ldb is given for every pad but 0 (today's behaviour), that a failed or incomplete run is
named and the series goes on, that another user's heavy job is a RED NOTE without stopping
anything, and that the sweep config holds the two tiles the issue names.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_issue49.sh"

sys.path.insert(0, str(HERE))

from pack_dims_table import sweep_tiles  # noqa: E402

# Leaves one per-rep file per row of the config it is given (the real config file), unless told not to.
STUB_RUN = r"""#!/bin/bash
echo "RUN $*" >> "${ISSUE49_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE49_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE49_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
cfg=""; while [ $# -gt 0 ]; do [ "$1" = --config ] && cfg="$2"; shift; done
rows=$(grep -c '^[[:space:]]*\["fp32"' "${STUB_CONFIG_DIR}/${cfg}")
[ "${what}" = short ] && rows=$((rows - 1))
sleep 1.1
d="${ISSUE49_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"
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
    env = {**os.environ, "ISSUE49_LOGS": str(logs), "ISSUE49_RUN": str(tmp_path / "run.sh"),
           "BOARD_PS_CMD": str(tmp_path / "ps.sh"), "STUB_BEHAVIOUR": behaviour, "STUB_CONFIG_DIR": str(HERE)}
    done = subprocess.run([str(SERIES), "--boards", "a,b", "--launches", "2", *extra],
                          env=env, capture_output=True, text=True, timeout=180, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("issue49_*"))[-1]


def steps(logs: Path) -> list[str]:
    """`<board><launch>:<pad>` for every run, from runs.tsv."""
    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    return [f"{r[0]}{r[1]}:{r[3]}" for r in rows]


ALL_OK = ",".join(["ok"] * 16)  # 2 boards x 2 launches x 4 pads (the default --pads)


def test_the_boards_take_turns_and_the_pads_rotate_between_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 16 of 16 runs ok." in done.stdout
    assert steps(logs) == [
        "a1:pad0", "a1:pad8", "a1:pad16", "a1:pad128",
        "b1:pad0", "b1:pad8", "b1:pad16", "b1:pad128",
        "a2:pad8", "a2:pad16", "a2:pad128", "a2:pad0",
        "b2:pad8", "b2:pad16", "b2:pad128", "b2:pad0",
    ]


def test_three_pads_rotate_so_each_is_first_once_in_three_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ",".join(["ok"] * 18), "--launches", "3", "--pads", "0,16,512")

    assert done.returncode == 0, done.stdout + done.stderr
    order = [s.split(":")[1] for s in steps(logs)]
    assert order[0:3] == ["pad0", "pad16", "pad512"]  # launch 1 on a
    assert order[6:9] == ["pad16", "pad512", "pad0"]  # launch 2 on a


def test_pad_ldb_is_given_for_every_pad_but_zero(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK)

    pad0, pad8, pad16, pad128 = (logs / "calls.txt").read_text().splitlines()[:4]
    for line in (pad0, pad8, pad16, pad128):
        assert "--config config_issue49_sweep.json" in line and "--no-nc-overrides" in line
        assert "--cpu 4" in line and "--riscv-ip a" in line and "--tmin 10" in line
    assert "--pad-ldb" not in pad0
    assert "--pad-ldb 8" in pad8 and "--pad-ldb 16" in pad16 and "--pad-ldb 128" in pad128


def test_bad_pads_are_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--pads", "0,-8")

    assert done.returncode != 0
    assert "--pads takes non-negative integers" in done.stderr


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
    done, logs = run_series(tmp_path, "ok,exit1," + ",".join(["ok"] * 14))

    assert done.returncode == 1
    assert "Series finished: 15 of 16 runs ok." in done.stdout
    assert "failed: launch 1 a pad8 (exit 1)" in done.stdout
    assert "FAIL(1)" in (series_dir(logs) / "runs.tsv").read_text()


def test_a_run_that_leaves_a_tile_out_is_failed_though_it_exited_zero(tmp_path: Path):
    done, _ = run_series(tmp_path, "short," + ",".join(["ok"] * 7))

    assert done.returncode == 1
    assert "failed: launch 1 a pad0 (exit 0)" in done.stdout


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
    assert "Launch 1   : 0 8 16 128" in done.stdout and "Launch 2   : 8 16 128 0" in done.stdout
    assert "2 tiles x 4 shapes" in done.stdout
    assert not list(logs.glob("issue49_*"))


def test_the_sweep_config_holds_the_two_tiles_that_flip_in_opposite_directions():
    tiles = set(sweep_tiles(HERE / "config_issue49_sweep.json"))

    assert tiles == {(16, 6), (16, 8)}
