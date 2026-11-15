"""
Issue #51: `run_issue51.sh` runs the M sweep as launches of tiles (16x6, 32x6, 16x8, 16x15,
32x7), rotating the tiles between launches and letting the boards take turns. Each tile is its
own single-row config (`config_issue51_<tile>.json`), so the rotation is real rather than
bundled into one multi-row sweep (the gap #50's code review found).

Same seam as test_run_issue49.py: the bench script is replaced by a stub (`ISSUE51_RUN`), the
board's process list by a canned one (`ISSUE51_PS_CMD`) and the logs go to a temp dir, so no
board is touched. What is checked is the loop: the order of the tiles and the boards, that
each tile gets its own config, that a failed or incomplete run is named and the series goes
on, that another user's heavy job is a RED NOTE without stopping anything, and that a config
is required for every named tile.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_issue51.sh"
DEFAULT_TILES = ("16x6", "32x6", "16x8", "16x15", "32x7")

sys.path.insert(0, str(HERE))

from pack_dims_table import sweep_tiles  # noqa: E402

# Leaves one per-rep file for the (single-row) config it is given.
STUB_RUN = r"""#!/bin/bash
echo "RUN $*" >> "${ISSUE51_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE51_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE51_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
rows=1
[ "${what}" = short ] && rows=0
sleep 1.1
d="${ISSUE51_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"
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
    env = {**os.environ, "ISSUE51_LOGS": str(logs), "ISSUE51_RUN": str(tmp_path / "run.sh"),
           "ISSUE51_PS_CMD": str(tmp_path / "ps.sh"), "STUB_BEHAVIOUR": behaviour}
    done = subprocess.run([str(SERIES), "--boards", "a,b", "--launches", "2", *extra],
                          env=env, capture_output=True, text=True, timeout=180, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("issue51_*"))[-1]


def steps(logs: Path) -> list[str]:
    """`<board><launch>:<tile>` for every run, from runs.tsv."""
    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    return [f"{r[0]}{r[1]}:{r[3]}" for r in rows]


ALL_OK = ",".join(["ok"] * 20)  # 2 boards x 2 launches x 5 tiles (the default --tiles)


def test_the_boards_take_turns_and_the_tiles_rotate_between_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 20 of 20 runs ok." in done.stdout
    assert steps(logs) == [
        "a1:16x6", "a1:32x6", "a1:16x8", "a1:16x15", "a1:32x7",
        "b1:16x6", "b1:32x6", "b1:16x8", "b1:16x15", "b1:32x7",
        "a2:32x6", "a2:16x8", "a2:16x15", "a2:32x7", "a2:16x6",
        "b2:32x6", "b2:16x8", "b2:16x15", "b2:32x7", "b2:16x6",
    ]


def test_the_default_five_launches_put_every_tile_first_and_last_once(tmp_path: Path):
    # The reason the default is 5, not house rule 2's bare minimum of 3 (code review, 2026-10-14):
    # with 5 tiles and only 3 launches, cyclic rotation never puts two of them -- including
    # 16x15 and 32x7, the two this issue most wants isolated -- in the first slot, and a
    # different two never in the last slot. Five launches closes a full cycle.
    done, logs = run_series(tmp_path, ",".join(["ok"] * 50), "--launches", "5")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 50 of 50 runs ok." in done.stdout
    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    a_rows = [r for r in rows if r[0] == "a"]  # board, launch, position, tile, result_dir, exit
    firsts = {r[3] for r in a_rows if r[2] == "1"}
    lasts = {r[3] for r in a_rows if r[2] == "5"}
    assert firsts == set(DEFAULT_TILES), firsts
    assert lasts == set(DEFAULT_TILES), lasts


def test_three_tiles_rotate_so_each_is_first_once_in_three_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ",".join(["ok"] * 18), "--launches", "3", "--tiles", "16x6,32x6,16x8")

    assert done.returncode == 0, done.stdout + done.stderr
    order = [s.split(":")[1] for s in steps(logs)]
    assert order[0:3] == ["16x6", "32x6", "16x8"]  # launch 1 on a
    assert order[6:9] == ["32x6", "16x8", "16x6"]  # launch 2 on a


def test_each_tile_gets_its_own_config(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK)

    calls = (logs / "calls.txt").read_text().splitlines()[:5]
    for line, tile in zip(calls, ("16x6", "32x6", "16x8", "16x15", "32x7")):
        assert f"--config config_issue51_{tile}.json" in line
        assert "--no-nc-overrides" in line and "--cpu 4" in line and "--riscv-ip a" in line and "--tmin 10" in line


def test_an_unknown_tile_with_no_config_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--tiles", "16x6,99x99")

    assert done.returncode != 0
    assert "no config_issue51_99x99.json" in done.stderr


def test_a_malformed_tile_is_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--tiles", "16x6,bogus")

    assert done.returncode != 0
    assert "--tiles takes MRxNR entries" in done.stderr


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
    done, logs = run_series(tmp_path, "ok,exit1," + ",".join(["ok"] * 18))

    assert done.returncode == 1
    assert "Series finished: 19 of 20 runs ok." in done.stdout
    assert "failed: launch 1 a 32x6 (exit 1)" in done.stdout
    assert "FAIL(1)" in (series_dir(logs) / "runs.tsv").read_text()


def test_a_run_that_leaves_the_project_out_is_failed_though_it_exited_zero(tmp_path: Path):
    done, _ = run_series(tmp_path, "short," + ",".join(["ok"] * 19))

    assert done.returncode == 1
    assert "failed: launch 1 a 16x6 (exit 0)" in done.stdout


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
    assert "Launch 1   : 16x6 32x6 16x8 16x15 32x7" in done.stdout
    assert "Launch 2   : 32x6 16x8 16x15 32x7 16x6" in done.stdout
    assert "15 shapes" in done.stdout
    assert not list(logs.glob("issue51_*"))


def test_every_default_tiles_config_pins_the_tile_its_filename_claims():
    # A typo'd pin (e.g. config_issue51_16x6.json actually pinning (16, 60)) would run silently
    # against the wrong tile; this ties the filename to the pin it must hold.
    for tile in DEFAULT_TILES:
        mr, nr = (int(x) for x in tile.split("x"))
        (pinned,) = sweep_tiles(HERE / f"config_issue51_{tile}.json")
        assert pinned == (mr, nr), tile
